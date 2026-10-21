"""Local instruct model wrapper that reads hidden states at chosen prompt positions.

``RepresentationLM`` builds one chat prompt per data point
(``## INSTRUCTIONS:`` / ``## CONTEXT:`` / ``## QUERY:`` inside the model's chat
template, generation prompt appended) and runs a single prefill pass through
nnterp. This base class provides prompt assembly, character-to-token offset
mapping, residual-stream read points and a read-point self-check; the task
modules (``entity_detection``, ``relation_detection``) subclass it to select
positions and shard per-document output.

Fail loud: torch / nnterp are imported at module top, and device and dtype are
required arguments with no inferred default.
"""
from __future__ import annotations

import gc

import numpy as np
import torch
from nnterp import StandardizedTransformer

from judge_interp.prompts import resolve_layers

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


def to_npz_array(values: list) -> np.ndarray:
    """Convert a column of metadata values to a typed array for an ``.npz`` shard.

    Args:
        values: values that are all ``bool``, all ``int``, all ``float`` or all ``str``.

    Returns:
        A ``bool``, ``int64``, ``float64`` or unicode array.

    Raises:
        ValueError: mixed or unsupported value types (e.g. ``None``), which would
            otherwise be coerced silently.
    """
    if all(isinstance(v, bool) for v in values):
        return np.asarray(values, dtype=bool)
    if all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in values):
        return np.asarray(values, dtype=np.int64)
    if all(isinstance(v, (float, np.floating)) for v in values):
        return np.asarray(values, dtype=np.float64)
    if all(isinstance(v, str) for v in values):
        return np.asarray(values, dtype=object).astype("U")
    raise ValueError(
        f"unsupported or mixed metadata value types: {sorted({type(v).__name__ for v in values})}"
    )


class RepresentationLM:
    """Instruct model loaded through nnterp, with helpers to read hidden states.

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

        if self.verbose:
            print(
                f"RepresentationLM: {model_name} on {device} ({dtype}), "
                f"{self.n_layers} blocks, hidden {self.hidden_size}, "
                f"collecting layers {self.layers}"
            )

    def _build_prompt(self, instructions: str, context: str, query: str) -> list[int]:
        """Tokenize the chat-templated prompt.

        The template is rendered to a string first (``tokenize=False``) and tokenized
        separately, because ``apply_chat_template(tokenize=True)`` returns a
        ``BatchEncoding`` rather than a list of ids on this transformers version.

        Args:
            instructions: the ``## INSTRUCTIONS:`` text.
            context: the ``## CONTEXT:`` text.
            query: the ``## QUERY:`` text.

        Returns:
            Token ids of the user turn plus the generation prompt.
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
        """Return the nnsight tensor to read for a layer (call inside a trace, in ascending layer order).

        Args:
            layer: ``0`` (token embeddings), ``1..n_layers-1`` (block outputs) or
                ``n_layers`` (post-final-norm); see ``prompts.resolve_layers``.

        Returns:
            The traced tensor, ``[batch, seq, hidden]`` or ``[seq, hidden]``.
        """
        if layer == 0:
            return self.llm.token_embeddings
        if layer == self.n_layers:
            return self.llm.ln_final.output
        return self.llm.model.layers[layer - 1].output[0]

    def verify_read_point(self, instructions: str, context: str, query: str) -> None:
        """Check the per-layer read points and determinism on one data point.

        Args:
            instructions: the ``## INSTRUCTIONS:`` text.
            context: the ``## CONTEXT:`` text.
            query: the ``## QUERY:`` text. Reads are taken at the last prompt token.

        Raises:
            AssertionError: a check below fails.
            RuntimeError: the final-norm weight cannot be read.

        Checks:

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

    def _build_prompt_with_offsets(
        self, instructions: str, context: str, query: str
    ) -> tuple[list[int], list[tuple[int, int]], int]:
        """Tokenize like ``_build_prompt`` and also map tokens to characters.

        Args:
            instructions: the ``## INSTRUCTIONS:`` text.
            context: the ``## CONTEXT:`` text.
            query: the ``## QUERY:`` text; it must occur exactly once in the prompt.

        Returns:
            ``(input_ids, offsets, shift)``: token ids, each token's ``(start, end)``
            character span in the chat-templated string, and the shift to add to a
            ``query``-relative character position to index that string.

        Raises:
            ValueError: the tokenizer is not a fast tokenizer.
            AssertionError: ``query`` or the content string is not unique in its
                container, offset-mapping tokenization disagrees with plain
                tokenization, or offsets are not monotonically non-decreasing.
        """
        if not self.tokenizer.is_fast:
            raise ValueError(
                f"{self.model_name!r}'s tokenizer is not a fast tokenizer; "
                "span-to-token mapping requires return_offsets_mapping support."
            )

        content = _PROMPT_TEMPLATE.format(instructions=instructions, context=context, query=query)
        assert content.count(query) == 1, (
            f"query occurs {content.count(query)} time(s) in the assembled content, not "
            "exactly once -- span offsets would be ambiguous"
        )
        query_shift = content.index(query)

        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        assert isinstance(formatted, str), (
            f"apply_chat_template(tokenize=False) returned {type(formatted)}, not str"
        )
        assert formatted.count(content) == 1, (
            f"content occurs {formatted.count(content)} time(s) in the chat-templated string, "
            "not exactly once -- span offsets would be ambiguous"
        )
        content_shift = formatted.index(content)

        encoded = self.tokenizer(formatted, add_special_tokens=False, return_offsets_mapping=True)
        input_ids = list(encoded["input_ids"])
        offsets = [tuple(o) for o in encoded["offset_mapping"]]

        reference_ids = list(self.tokenizer(formatted, add_special_tokens=False)["input_ids"])
        assert input_ids == reference_ids, (
            "offset-mapping tokenization disagrees with the plain tokenization of the same "
            "string -- return_offsets_mapping must not change input_ids"
        )
        for (s0, e0), (s1, e1) in zip(offsets, offsets[1:]):
            assert s1 >= s0 and e1 >= e0, "token offsets are not monotonically non-decreasing"

        return input_ids, offsets, query_shift + content_shift

    @staticmethod
    def _last_token_covering(offsets: list[tuple[int, int]], char_pos: int) -> int:
        """Find the last token whose character span covers a position.

        "Last" rather than "first" because byte-level BPE can merge adjacent characters
        into one token, so several positions may map to the same token.

        Args:
            offsets: per-token ``(start, end)`` character spans (``end`` exclusive).
            char_pos: character position in the tokenized string.

        Returns:
            Index of the last covering token.

        Raises:
            AssertionError: no token covers ``char_pos``.
        """
        matches = [i for i, (s, e) in enumerate(offsets) if s <= char_pos < e]
        assert matches, f"no token covers character offset {char_pos}"
        return matches[-1]
