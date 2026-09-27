"""Run a local instruct model as an extraction-validation judge and collect
last-prompt-token representations.

``RepresentationLM`` builds one chat prompt per data point —
``## INSTRUCTIONS:`` / ``## CONTEXT:`` / ``## QUERY:`` wrapped in the model's
chat template with a generation prompt — and runs a single prefill pass. At the
final prompt position (the position whose next-token distribution *is* the
verdict) it reads:

- the residual stream at each requested layer (see ``prompts.resolve_layers``
  for the layer convention), and
- the logits, from which a casing-marginalised ``P(true)`` / ``P(false)`` is
  computed over the ``true``/``True`` and ``false``/``False`` tokens.

The representation and the probability therefore come from the *same* position.
No text is generated: the verdict is read as ``argmax`` of the same logits.

``collect`` shards its output one ``.npz`` per ``document_id`` under a cache
directory and skips documents whose shard is already complete, so a large run
resumes after a failure.

Fail-loud, per this repo's ``CLAUDE.md``: torch / nnterp are imported at module
top (an install problem is a hard error, not a silent CPU fallback), and the
device and dtype are required arguments with no inferred default.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch
from nnterp import StandardizedTransformer

from judge_interp.prompts import resolve_layers, row_key

_TORCH_DTYPES: dict[str, "torch.dtype"] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

_PROMPT_TEMPLATE = (
    "## INSTRUCTIONS:\n{instructions}\n\n"
    "## CONTEXT:\n{context}\n\n"
    "## QUERY:\n{query}"
)


class RepresentationLM:
    """Judge a data point and capture its last-prompt-token representations.

    Args:
        model_name: HF repo id of an instruction-tuned model with a chat template.
        layers: Config layer list (ints and/or the string ``"last"``); validated
            against the loaded model and stored sorted+deduped as ``self.layers``.
        device: Passed to ``StandardizedTransformer`` as ``device_map`` (e.g.
            ``"cuda"``, ``"cuda:0"``, ``"cpu"``). Required — no auto-detect.
        dtype: One of ``"float32"``, ``"float16"``, ``"bfloat16"``. Required.
        hf_cache_dir: HuggingFace cache dir. ``None`` uses the environment
            (``HF_HOME`` / ``HF_HUB_CACHE``), which is how the cluster is set up.
        verbose: Print model / device setup.
    """

    def __init__(
        self,
        model_name: str,
        layers: list,
        device: str,
        dtype: str,
        hf_cache_dir: str | None = None,
        verbose: bool = False,
    ):
        if dtype not in _TORCH_DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_TORCH_DTYPES)}, got {dtype!r}")

        self.model_name = model_name
        self.device = device
        self.dtype = _TORCH_DTYPES[dtype]
        self.verbose = verbose

        model_kwargs: dict = {"dtype": self.dtype}
        if hf_cache_dir is not None:
            model_kwargs["cache_dir"] = hf_cache_dir

        self.llm = StandardizedTransformer(
            model_name,
            enable_attention_probs=False,
            device_map=device,
            **model_kwargs,
        )
        self.tokenizer = self.llm.tokenizer
        if self.tokenizer.chat_template is None:
            raise ValueError(
                f"{model_name!r} has no chat template; RepresentationLM expects an "
                "instruction-tuned model."
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.n_layers = len(self.llm.model.layers)
        self.hidden_size = int(self.llm.config.hidden_size)
        self.max_position_embeddings = int(self.llm.config.max_position_embeddings)
        self.layers = resolve_layers(layers, self.n_layers)

        self._binary_token_ids = self._init_binary_token_ids()

        if self.verbose:
            print(
                f"RepresentationLM: {model_name} on {device} ({dtype}), "
                f"{self.n_layers} blocks, hidden {self.hidden_size}, "
                f"collecting layers {self.layers}"
            )

    # ------------------------------------------------------------------

    def _init_binary_token_ids(self) -> dict[str, int]:
        """Single-token ids for 'true', 'True', 'false', 'False'. Hard error on a
        multi-token verdict word — the P(true) computation assumes one token."""
        ids: dict[str, int] = {}
        for s in ("true", "True", "false", "False"):
            enc = self.tokenizer.encode(s, add_special_tokens=False)
            if len(enc) != 1:
                raise ValueError(
                    f"verdict word {s!r} tokenizes to {len(enc)} tokens {enc}; "
                    "P(true) assumes a single token."
                )
            ids[s] = enc[0]
        return ids

    def _build_prompt(self, instructions: str, context: str, query: str) -> list[int]:
        """Chat-templated prompt token ids, with a generation prompt appended.

        Renders the template to a string first (``tokenize=False``) and tokenizes
        separately, rather than ``apply_chat_template(tokenize=True)`` — on this
        transformers version that returns a ``BatchEncoding`` (a dict of
        ``input_ids``/``attention_mask``), not a bare id list, and would silently
        hand the trace a 2-element garbage sequence instead of the prompt.
        """
        content = _PROMPT_TEMPLATE.format(
            instructions=instructions, context=context, query=query
        )
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        assert isinstance(formatted, str), (
            f"apply_chat_template(tokenize=False) returned {type(formatted)}, not str"
        )
        input_ids = self.tokenizer(formatted, add_special_tokens=False)["input_ids"]
        return list(input_ids)

    def _read_point(self, layer: int):
        """Trace-time tensor for a layer. Call in ascending ``layer`` order."""
        if layer == 0:
            return self.llm.token_embeddings
        if layer == self.n_layers:
            return self.llm.ln_final.output
        return self.llm.model.layers[layer - 1].output[0]

    def _p_true(self, logits: np.ndarray) -> dict[str, float]:
        """Casing-marginalised P(true) / P(false) from verdict-position logits."""
        bt = self._binary_token_ids
        true_ids = {bt["true"], bt["True"]}
        false_ids = {bt["false"], bt["False"]}
        log_p_true = torch.logsumexp(torch.tensor([logits[i] for i in true_ids]), dim=0)
        log_p_false = torch.logsumexp(torch.tensor([logits[i] for i in false_ids]), dim=0)
        probs = torch.softmax(torch.stack([log_p_true, log_p_false]), dim=0)
        return {
            "p_true": float(probs[0]),
            "p_false": float(probs[1]),
            "logit_p_true": float(log_p_true),
            "logit_p_false": float(log_p_false),
        }

    # ------------------------------------------------------------------

    def judge_one(self, instructions: str, context: str, query: str) -> dict:
        """Judge a single data point.

        Returns a dict with:
            ``representations``: ``{layer: float32 [hidden_size]}`` — last prompt
                token, one entry per ``self.layers``.
            ``p_true`` / ``p_false`` / ``logit_p_true`` / ``logit_p_false``:
                casing-marginalised verdict probabilities and their logits,
                computed from the *unspaced* true/True/false/False token ids.
            ``verdict_recognised``: whether ``argmax`` landed on one of those
                same four ids — the thing ``p_true`` actually describes. A
                space-prefixed ``" true"`` (a different id after a chat
                template's trailing ``\n``) is NOT recognised, so
                ``verdict_recognised`` and ``p_true`` never silently disagree.
            ``verdict_true``: ``argmax`` id is ``true``/``True`` (only
                meaningful when ``verdict_recognised``).
            ``verdict``: the decoded, stripped ``argmax`` token — for eyeballing
                only, never for accuracy (use ``verdict_true`` /
                ``verdict_recognised`` for that).
            ``verdict_token_id``: the raw argmax id.
            ``prompt_n_tokens``: prompt length.

        Raises:
            ValueError: the prompt exceeds ``max_position_embeddings``. Silently
                truncating would drop the ``## QUERY:`` block and/or the
                generation prompt — the read position would then sit mid-context
                and the representation/verdict would be meaningless while
                looking like ordinary output.
        """
        input_ids = self._build_prompt(instructions, context, query)
        prompt_n_tokens = len(input_ids)
        if prompt_n_tokens > self.max_position_embeddings:
            raise ValueError(
                f"prompt is {prompt_n_tokens} tokens > max_position_embeddings "
                f"{self.max_position_embeddings}; truncation would cut into the "
                "query, not spare context — fix the input rather than truncate."
            )

        saved: dict[int, "torch.Tensor"] = {}
        with torch.no_grad(), self.llm.trace(input_ids, logits_to_keep=1):
            for layer in self.layers:  # ascending — resolve_layers sorted it
                h = self._read_point(layer)
                seq = h[0] if h.ndim == 3 else h
                saved[layer] = seq[-1, :].detach().to(torch.float32).save()
            logits_saved = self.llm.logits[0, -1, :].detach().to(torch.float32).save()

        representations = {
            layer: np.asarray(t.cpu().numpy(), dtype=np.float32) for layer, t in saved.items()
        }
        logits = np.asarray(logits_saved.cpu().numpy(), dtype=np.float32)

        for layer, arr in representations.items():
            assert arr.shape == (self.hidden_size,), (layer, arr.shape)
            assert np.isfinite(arr).all(), f"non-finite representation at layer {layer}"
        assert logits.shape == (int(self.llm.config.vocab_size),), logits.shape

        verdict_token_id = int(logits.argmax())
        bt = self._binary_token_ids
        verdict_recognised = verdict_token_id in bt.values()
        verdict_true = verdict_token_id in (bt["true"], bt["True"])

        out = {
            "representations": representations,
            "verdict": self.tokenizer.decode([verdict_token_id]).strip(),
            "verdict_token_id": verdict_token_id,
            "verdict_recognised": verdict_recognised,
            "verdict_true": verdict_true,
            "prompt_n_tokens": prompt_n_tokens,
            **self._p_true(logits),
        }

        del saved, logits_saved
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return out

    # ------------------------------------------------------------------

    def verify_read_point(self, instructions: str, context: str, query: str) -> None:
        """Smoke-only gate on the per-layer read points and determinism.

        On one data point, at the last prompt token, checks:

        1. **Determinism.** Two identical prefill passes are bitwise equal at
           every requested layer. bf16 GPU kernels do not guarantee this; if it
           fails, fix it (float32, or ``torch.use_deterministic_algorithms``)
           before interpreting any run.
        2. **Adjacent requested layers distinct** — catches an off-by-one or a
           read point that returns the same module twice.
        3. **Final layer is post-final-norm** (only if ``n_layers`` is
           requested): ``ln_final.output`` differs from the last block's output,
           satisfies the RMSNorm identity ``rms(y / weight) ≈ 1`` per row, and
           matches a numpy recomputation of RMSNorm from the last block's output.
        """
        input_ids = self._build_prompt(instructions, context, query)
        if len(input_ids) > self.max_position_embeddings:
            input_ids = input_ids[: self.max_position_embeddings]

        pre_final_block = self.n_layers - 1  # 0-indexed; always read, backs check 3
        want_lnf = self.n_layers in self.layers

        def _run() -> dict:
            saved: dict = {}
            with torch.no_grad(), self.llm.trace(input_ids, logits_to_keep=1):
                if 0 in self.layers:
                    e = self.llm.token_embeddings
                    e = e[0] if e.ndim == 3 else e
                    saved[0] = e[-1, :].detach().to(torch.float32).save()
                block_reads = sorted(
                    {L - 1 for L in self.layers if 1 <= L <= self.n_layers - 1}
                    | {pre_final_block}
                )
                for bi in block_reads:
                    h = self.llm.model.layers[bi].output[0]
                    h = h[0] if h.ndim == 3 else h
                    saved[("block", bi)] = h[-1, :].detach().to(torch.float32).save()
                if want_lnf:
                    lo = self.llm.ln_final.output
                    lo = lo[0] if lo.ndim == 3 else lo
                    saved[self.n_layers] = lo[-1, :].detach().to(torch.float32).save()
            out = {k: np.asarray(v.cpu().numpy(), dtype=np.float32) for k, v in saved.items()}
            del saved
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return out

        r1, r2 = _run(), _run()

        def _layer_arr(run: dict, L: int) -> np.ndarray:
            if L == 0:
                return run[0]
            if L == self.n_layers:
                return run[self.n_layers]
            return run[("block", L - 1)]

        per1 = {L: _layer_arr(r1, L) for L in self.layers}
        per2 = {L: _layer_arr(r2, L) for L in self.layers}
        pre_final1 = r1[("block", pre_final_block)]

        lines = ["verify_read_point diagnostics:"]
        for L in self.layers:
            lines.append(
                f"  layer {L:>2d}: row L2 {np.linalg.norm(per1[L]):.4g}, "
                f"max|pass1-pass2| {np.abs(per1[L] - per2[L]).max():.4g}"
            )
        print("\n".join(lines))

        for L in self.layers:
            assert np.array_equal(per1[L], per2[L]), (
                f"verify_read_point: layer {L} differs across two identical prefill "
                f"passes (max abs diff {np.abs(per1[L] - per2[L]).max():.4g}) — "
                "collection is non-deterministic."
            )
        for a, b in zip(self.layers, self.layers[1:]):
            assert not np.allclose(per1[a], per1[b], atol=1e-3), (
                f"verify_read_point: layer {a} and {b} reads are ~identical — read "
                "points collapsed (off-by-one or duplicate)."
            )

        if want_lnf:
            post1 = per1[self.n_layers]
            assert not np.allclose(post1, pre_final1, atol=1e-3), (
                "verify_read_point: ln_final.output == last block output — the final "
                "read point is the pre-norm residual, not post-final-norm."
            )
            norm_mod = getattr(self.llm.ln_final, "_module", None)
            if norm_mod is None or not hasattr(norm_mod, "weight"):
                raise RuntimeError(
                    "verify_read_point: cannot read the final-norm weight for the "
                    "RMSNorm check."
                )
            w = norm_mod.weight.detach().float().cpu().numpy()
            eps = float(
                getattr(norm_mod, "variance_epsilon", None)
                or getattr(norm_mod, "eps", None)
                or 1e-5
            )
            row_rms = float(np.sqrt(np.mean((post1 / w) ** 2)))
            assert 0.9 < row_rms < 1.1, (
                f"verify_read_point: rms(ln_final.output / weight) = {row_rms:.4g}, "
                "not ≈ 1 — ln_final.output is not the model's RMSNorm output."
            )
            rms = np.sqrt(np.mean(pre_final1 ** 2) + eps)
            recomputed = w * pre_final1 / rms
            rel = float(np.median(np.abs(recomputed - post1) / (np.abs(post1) + 1e-6)))
            assert rel < 5e-2, (
                f"verify_read_point: ln_final.output != numpy RMSNorm(last block) "
                f"(median rel err {rel:.3g}) — off-by-one in the layer→module map?"
            )
            print(
                f"verify_read_point: OK — {len(self.layers)} read points, bitwise "
                f"deterministic; adjacent layers distinct; final layer passes the "
                f"RMSNorm identity (row rms {row_rms:.4f}, rel err {rel:.2g})."
            )
        else:
            print(
                f"verify_read_point: OK — {len(self.layers)} read points, bitwise "
                "deterministic; adjacent layers distinct. (layer n_layers not "
                "requested — final-norm identity check skipped.)"
            )

    # ------------------------------------------------------------------

    def collect(
        self,
        items: list[dict],
        instructions: str,
        dataset: str,
        cache_dir: str | Path,
        verify_read_point: bool,
    ) -> dict:
        """Judge every item and return stacked per-layer representations.

        Args:
            items: One dict per data point, each with ``document_id``,
                ``line_index`` (``None`` for main), ``error_type`` (``None`` for
                main and for valid line rows; ``"inter_document"`` /
                ``"intra_document"`` for invalid line rows -- since 2026-09-15
                the same ``(document_id, line_index, k)`` can carry both
                variants, so ``error_type`` is part of row identity), ``label``
                (bool = the row's ``valid``), ``k``, ``num_invalid_fields``,
                ``invalid_fields``, ``context`` and ``query``.
            instructions: The judge instruction block (fixed for the run).
            dataset: ``"main"`` or ``"line"`` — selects the row-key shape.
            cache_dir: Directory for per-document ``.npz`` shards. A document
                whose shard already holds exactly its expected rows is skipped.
            verify_read_point: Run ``verify_read_point`` on the first item before
                judging anything. Determinism is only in doubt on the actual
                execution path (dtype/device this run uses) — CPU/float32
                evidence from a smoke test does not cover a bf16 GPU run, so
                this is required rather than defaulted, and should be left on
                for any run whose determinism hasn't already been checked on
                this exact (model, device, dtype).

        Returns:
            Dict of parallel arrays (row count ``n``), row order sorted by
            ``(document_id, line_index or -1, k)``:
              - ``representations``: ``{layer: float32 [n, hidden_size]}``
              - ``layers``: ``int64 [len(self.layers)]``
              - ``doc_ids``: ``str [n]``
              - ``line_indices``: ``int64 [n]`` (``-1`` for main)
              - ``error_type``: ``str [n]`` (``""`` for main and for valid line
                rows -- npz string arrays can't hold ``None``)
              - ``k`` / ``num_invalid_fields``: ``int64 [n]``
              - ``labels``: ``bool [n]`` — the row's ``valid``
              - ``p_true`` / ``p_false`` / ``logit_p_true`` / ``logit_p_false``:
                ``float64 [n]``
              - ``verdict_true`` / ``verdict_recognised``: ``bool [n]`` (see ``judge_one``)
              - ``prompt_n_tokens``: ``int64 [n]``
        """
        if not items:
            raise ValueError("items is empty")
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        if verify_read_point:
            first = items[0]
            self.verify_read_point(instructions, first["context"], first["query"])

        by_doc: dict[str, list[dict]] = {}
        for it in items:
            by_doc.setdefault(it["document_id"], []).append(it)

        # Deterministic order and a stable row key per item.
        for doc_id, doc_items in by_doc.items():
            doc_items.sort(key=lambda it: (it["line_index"] if it["line_index"] is not None else -1, it["k"]))
            keys = [
                row_key(
                    {
                        "document_id": it["document_id"],
                        "line_index": it["line_index"],
                        "error_type": it["error_type"],
                        "k": it["k"],
                    },
                    dataset,
                )
                for it in doc_items
            ]
            assert len(set(keys)) == len(keys), f"duplicate row key(s) in document {doc_id!r}"

        for i, doc_id in enumerate(sorted(by_doc)):
            shard = cache_dir / f"{doc_id}.npz"
            expected = [
                json.dumps(
                    row_key(
                        {
                            "document_id": it["document_id"],
                            "line_index": it["line_index"],
                            "error_type": it["error_type"],
                            "k": it["k"],
                        },
                        dataset,
                    )
                )
                for it in by_doc[doc_id]
            ]
            if shard.is_file():
                with np.load(shard, allow_pickle=False) as z:
                    if list(z["row_keys"]) == expected:
                        if self.verbose:
                            print(f"[{i + 1}/{len(by_doc)}] {doc_id}: cached, skipping")
                        continue
                shard.unlink()  # incomplete / stale — recompute

            if self.verbose:
                print(f"[{i + 1}/{len(by_doc)}] {doc_id}: judging {len(by_doc[doc_id])} row(s)")
            self._judge_document(by_doc[doc_id], instructions, dataset, shard, expected)

        return self._aggregate(by_doc, dataset, cache_dir)

    def _judge_document(
        self, doc_items: list[dict], instructions: str, dataset: str, shard: Path, expected: list[str]
    ) -> None:
        """Judge one document's rows and write its shard atomically."""
        per_layer: dict[int, list[np.ndarray]] = {L: [] for L in self.layers}
        scalars: dict[str, list] = {
            k: []
            for k in (
                "line_indices", "error_type", "k", "num_invalid_fields", "labels", "p_true",
                "p_false", "logit_p_true", "logit_p_false", "verdict_true", "verdict_recognised",
                "prompt_n_tokens",
            )
        }
        for it in doc_items:
            res = self.judge_one(instructions, it["context"], it["query"])
            for L in self.layers:
                per_layer[L].append(res["representations"][L])
            scalars["line_indices"].append(it["line_index"] if it["line_index"] is not None else -1)
            scalars["error_type"].append(it["error_type"] if it["error_type"] is not None else "")
            scalars["k"].append(it["k"])
            scalars["num_invalid_fields"].append(it["num_invalid_fields"])
            scalars["labels"].append(bool(it["label"]))
            scalars["p_true"].append(res["p_true"])
            scalars["p_false"].append(res["p_false"])
            scalars["logit_p_true"].append(res["logit_p_true"])
            scalars["logit_p_false"].append(res["logit_p_false"])
            scalars["verdict_true"].append(res["verdict_true"])
            scalars["verdict_recognised"].append(res["verdict_recognised"])
            scalars["prompt_n_tokens"].append(res["prompt_n_tokens"])

        payload = {
            "row_keys": np.asarray(expected, dtype=object).astype("U"),
            "doc_ids": np.asarray([it["document_id"] for it in doc_items], dtype=object).astype("U"),
            "layers": np.asarray(self.layers, dtype=np.int64),
            "line_indices": np.asarray(scalars["line_indices"], dtype=np.int64),
            "error_type": np.asarray(scalars["error_type"], dtype=object).astype("U"),
            "k": np.asarray(scalars["k"], dtype=np.int64),
            "num_invalid_fields": np.asarray(scalars["num_invalid_fields"], dtype=np.int64),
            "labels": np.asarray(scalars["labels"], dtype=bool),
            "p_true": np.asarray(scalars["p_true"], dtype=np.float64),
            "p_false": np.asarray(scalars["p_false"], dtype=np.float64),
            "logit_p_true": np.asarray(scalars["logit_p_true"], dtype=np.float64),
            "logit_p_false": np.asarray(scalars["logit_p_false"], dtype=np.float64),
            "verdict_true": np.asarray(scalars["verdict_true"], dtype=bool),
            "verdict_recognised": np.asarray(scalars["verdict_recognised"], dtype=bool),
            "prompt_n_tokens": np.asarray(scalars["prompt_n_tokens"], dtype=np.int64),
        }
        for L in self.layers:
            payload[f"rep_{L}"] = np.stack(per_layer[L], axis=0).astype(np.float32)

        # np.savez appends ".npz" itself when the path doesn't already end in
        # it, so the temp name must end in ".npz" too or the rename below finds
        # nothing at the path it expects.
        tmp = shard.with_name(shard.stem + ".tmp.npz")
        np.savez(tmp, **payload)
        tmp.rename(shard)

    def _aggregate(self, by_doc: dict, dataset: str, cache_dir: Path) -> dict:
        """Concatenate every document shard in sorted order into one result."""
        rep_blocks: dict[int, list[np.ndarray]] = {L: [] for L in self.layers}
        cols: dict[str, list[np.ndarray]] = {
            k: []
            for k in (
                "doc_ids", "line_indices", "error_type", "k", "num_invalid_fields", "labels",
                "p_true", "p_false", "logit_p_true", "logit_p_false", "verdict_true",
                "verdict_recognised", "prompt_n_tokens",
            )
        }
        for doc_id in sorted(by_doc):
            with np.load(cache_dir / f"{doc_id}.npz", allow_pickle=False) as z:
                assert list(z["layers"]) == self.layers, (doc_id, z["layers"])
                for L in self.layers:
                    rep_blocks[L].append(z[f"rep_{L}"])
                for k in cols:
                    cols[k].append(z[k])

        representations = {
            L: np.concatenate(rep_blocks[L], axis=0).astype(np.float32) for L in self.layers
        }
        out: dict = {"representations": representations, "layers": np.asarray(self.layers, dtype=np.int64)}
        for k, parts in cols.items():
            out[k] = np.concatenate(parts, axis=0)

        n = len(out["doc_ids"])
        expected_n = sum(len(v) for v in by_doc.values())
        assert n == expected_n, f"aggregated {n} rows, expected {expected_n}"
        for L in self.layers:
            assert representations[L].shape == (n, self.hidden_size), (L, representations[L].shape)
            assert np.isfinite(representations[L]).all(), f"non-finite reps at layer {L}"
        for k, v in out.items():
            if k in ("representations", "layers"):
                continue
            assert len(v) == n, f"{k}: len {len(v)} != n {n}"
        # error_type is part of row identity for line rows: since 2026-09-15 the same
        # (doc_id, line_index, k) can carry both an inter- and an intra-document
        # invalid variant, so it must be in the key or those two rows collide here.
        keys = list(
            zip(
                out["doc_ids"].tolist(),
                out["line_indices"].tolist(),
                out["error_type"].tolist(),
                out["k"].tolist(),
            )
        )
        assert len(set(keys)) == n, (
            f"{n - len(set(keys))} duplicate (doc_id, line_index, error_type, k) rows"
        )
        return out
