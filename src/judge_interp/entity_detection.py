"""Mixed-type entity detection with substituted decoys.

One prompt per ``(document, sample)``. The prompt lists *every* ground-truth
field value of the document, across all ``t`` entity types (main + line), in one
shuffled numbered list; ``k`` of them are replaced by a decoy of the same type
taken from a different document in the same split.

Definitions (fixed here; the numbers recorded per span are ``m`` and ``k``):

- ``F_i`` -- the *set* of distinct non-null ground-truth values of entity type
  ``i`` in document ``D``, read off ``D``'s ``valid: true`` rows. Line types
  therefore contribute each distinct value once, not once per line item.
- ``m = sum_i |F_i|`` -- the number of entries in the prompt's list.
- ``k`` -- drawn uniformly from ``{k_min, ..., m}`` (inclusive) per sample;
  ``k`` of the ``m`` slots are replaced by decoys. The list length stays ``m``.

A decoy for a slot of type ``i`` is a value of type ``i`` from another document
in the *same split*, that is not equal to any value in ``F_i`` (so it is
genuinely not this document's value). Decoy values are drawn uniformly over
*distinct* eligible values (not weighted by how often a value recurs across the
corpus) and never repeat within one type in one prompt. Whether a decoy string
nevertheless occurs somewhere in ``D``'s OCR text is recorded per span
(``decoy_in_ocr``) rather than filtered on: a short value such as a channel
number or a date can coincide with text in ``D``, in which case the "invalid"
label is only invalid as a value *for that field*.

Every span row carries ``m``, ``k``, ``sample_index`` and its own valid/invalid
label, so any stored representation maps back to its prompt's ``(m, k)``.

Only the entity-content span (the value text between the JSON quotes, as it appears in the list)
is read. Layers are pinned to ``[0, "last"]``: the token-embedding output and
the post-final-norm state. No logits are read, and the forward pass runs with
``logits_to_keep=1``, so no ``[T, vocab]`` tensor is ever materialised.
"""
from __future__ import annotations

import gc
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from judge_interp.entity_prompts import _sub_seed
from judge_interp.instruction_prompts import _EXPLICIT_CRITERION
from judge_interp.prompts import LINE_FIELDS, MAIN_FIELDS, load_ocr_context, load_split
from judge_interp.representationlm import RepresentationLM, _to_npz_array

# (entity type, source dataset), canonical order. Main types first.
ENTITY_TYPES: list[tuple[str, str]] = [(f, "main") for f in MAIN_FIELDS] + [
    (f, "line") for f in LINE_FIELDS
]
assert len({t for t, _ in ENTITY_TYPES}) == len(ENTITY_TYPES), "main/line type names collide"
TYPE_NAMES: list[str] = [t for t, _ in ENTITY_TYPES]
_SOURCE_OF: dict[str, str] = dict(ENTITY_TYPES)

_HEADER = """You are validating candidate values extracted from a document by an information-extraction pipeline.

You are given:
1) In ## CONTEXT: the full OCR text of the source document.
2) In ## QUERY: a numbered list of candidate values. Each line gives the field type in square brackets, followed by the candidate value as a JSON string.

The possible field types are:
{types}

For each candidate independently, decide whether it is a VALID value for its stated field type in this document. A candidate is valid only if BOTH criteria below hold:

{criteria}

Judge only against the context. Do not use outside knowledge about the entities involved, and do not guess when the evidence is unclear — unclear evidence is a failed criterion. Judge every candidate independently: one candidate's validity has no bearing on another's.

For each numbered candidate, answer with exactly one line in the form `<n>: true` or `<n>: false`, in order, one line per candidate."""

# Shown to the judge so it can apply the type-correspondence criterion. One line
# per type; keys must equal TYPE_NAMES exactly (checked below).
TYPE_DESCRIPTIONS: dict[str, str] = {
    "advertiser": "the political campaign, candidate or committee that the advertising is bought for",
    "agency": "the advertising agency that placed the buy on the advertiser's behalf",
    "property": "the TV station (typically its call letters) that sells the airtime and receives the order",
    "tv_address": "the mailing address of that TV station (street, city, state, ZIP; may span several lines)",
    "product": "the campaign, candidate or race description that the buy is booked under",
    "contract_num": "the contract, order or estimate identifier of the whole buy",
    "flight_from": "the start date of the whole buy's flight (its overall air period)",
    "flight_to": "the end date of the whole buy's flight (its overall air period)",
    "gross_amount": "the total gross dollar amount of the whole contract",
    "channel": "the station or channel that one individual line item (spot) airs on",
    "program_desc": "the program, time slot or daypart that one individual line item airs in",
    "program_start_date": "the start date of one individual line item's air period",
    "program_end_date": "the end date of one individual line item's air period",
    "sub_amount": "the dollar amount (subtotal) of one individual line item",
}
assert list(TYPE_DESCRIPTIONS) == TYPE_NAMES, "TYPE_DESCRIPTIONS keys must match TYPE_NAMES in order"

# Second criterion, specific to this task: presence in the text is not validity.
_TYPE_CRITERION = """({n}) Type correspondence. The value must be what the stated field type denotes in this document, as described in the list of field types above, not merely text that appears somewhere in the context. A value that is present in the context but is the value of a different field (for example a date or a dollar amount that belongs to another field or to another part of the document) fails this criterion."""

# Span metadata columns, in shard order. All values are str / int / bool.
SPAN_META_KEYS = (
    "sample_index", "m", "k", "entity_type", "source_dataset", "value", "label_valid",
    "list_position", "decoy_source_doc", "decoy_in_ocr",
)


def build_instructions() -> str:
    """The all-types instruction block (fixed for a run)."""
    types = "\n".join(f"- {t}: {TYPE_DESCRIPTIONS[t]}" for t in TYPE_NAMES)
    criteria = "\n".join([_EXPLICIT_CRITERION.format(n=1), _TYPE_CRITERION.format(n=2)])
    return _HEADER.format(types=types, criteria=criteria)


# ----------------------------------------------------------------------
# Ground truth and decoy pools


def load_ground_truth(data_root: str | Path, split: str) -> dict[str, dict[str, list[str]]]:
    """``{document_id: {entity_type: sorted distinct non-null values}}``.

    Every document has all ``t`` keys (an empty list for an all-null type).

    Raises:
        AssertionError: a main document without exactly one valid row, a line
            row for a document with no main row, or a non-``str`` value.
    """
    main_rows = load_split(data_root, "main", split)
    line_rows = load_split(data_root, "line", split)

    truth: dict[str, dict[str, set[str]]] = {}
    for r in main_rows:
        if not r["valid"]:
            continue
        assert r["document_id"] not in truth, f"two valid main rows for {r['document_id']!r}"
        truth[r["document_id"]] = {t: set() for t in TYPE_NAMES}
    main_docs = {r["document_id"] for r in main_rows}
    assert set(truth) == main_docs, "a main document has no valid row"

    for r in main_rows:
        if r["valid"]:
            _add_values(truth[r["document_id"]], r, MAIN_FIELDS)
    for r in line_rows:
        if not r["valid"]:
            continue
        assert r["document_id"] in truth, f"line row for unknown document {r['document_id']!r}"
        _add_values(truth[r["document_id"]], r, LINE_FIELDS)

    return {d: {t: sorted(vals) for t, vals in per.items()} for d, per in truth.items()}


def _add_values(per_type: dict[str, set[str]], row: dict, fields: list[str]) -> None:
    for f in fields:
        v = row[f]
        if v is None:
            continue
        assert isinstance(v, str), f"{row['document_id']!r}/{f!r}: non-str value {v!r}"
        per_type[f].add(v)


def build_decoy_pools(truth: dict[str, dict[str, list[str]]]) -> dict[str, dict[str, list[str]]]:
    """``{entity_type: {value: sorted documents holding it as ground truth}}``."""
    pools: dict[str, dict[str, list[str]]] = {t: {} for t in TYPE_NAMES}
    for doc_id in sorted(truth):
        for t in TYPE_NAMES:
            for v in truth[doc_id][t]:
                pools[t].setdefault(v, []).append(doc_id)
    return pools


# ----------------------------------------------------------------------
# Sampling and rendering


def sample_prompt_entries(
    doc_id: str,
    doc_truth: dict[str, list[str]],
    pools: dict[str, dict[str, list[str]]],
    sample_index: int,
    seed: int,
    k_min: int,
) -> dict:
    """Draw one ``(document, sample)``: ``m``, ``k``, and the shuffled entry list.

    Returns ``{"m", "k", "entries": [{"entity_type", "source_dataset", "value",
    "label_valid", "decoy_source_doc"}, ...]}`` -- ``entries`` in final
    presentation order (a full seeded shuffle, so neither list position nor
    sorted order can reveal which entries were substituted).

    Raises:
        ValueError: ``m < k_min`` or ``k_min < 0``, or a type has fewer eligible
            decoys than slots chosen for replacement.
    """
    slots = [(t, v) for t in TYPE_NAMES for v in doc_truth[t]]
    m = len(slots)
    if k_min < 0 or m < k_min:
        raise ValueError(f"{doc_id!r}: m={m} but k_min={k_min}")

    rng = random.Random(_sub_seed(seed, doc_id, str(sample_index)))
    k = rng.randint(k_min, m)
    replaced = sorted(rng.sample(range(m), k))

    replaced_by_type: dict[str, list[int]] = {}
    for i in replaced:
        replaced_by_type.setdefault(slots[i][0], []).append(i)

    entries: list[dict | None] = [
        {
            "entity_type": t, "source_dataset": _SOURCE_OF[t], "value": v,
            "label_valid": True, "decoy_source_doc": "",
        }
        for t, v in slots
    ]
    for t in TYPE_NAMES:  # fixed type order keeps the RNG stream deterministic
        idxs = replaced_by_type.get(t, [])
        if not idxs:
            continue
        own = set(doc_truth[t])
        eligible = sorted(v for v in pools[t] if v not in own)
        if len(eligible) < len(idxs):
            raise ValueError(
                f"{doc_id!r}/{t!r}: {len(idxs)} decoy(s) needed but only {len(eligible)} "
                "eligible value(s) in the split"
            )
        for i, v in zip(idxs, rng.sample(eligible, len(idxs))):
            sources = pools[t][v]
            assert doc_id not in sources, f"decoy {v!r} for {doc_id!r}/{t!r} is its own value"
            entries[i] = {
                "entity_type": t, "source_dataset": _SOURCE_OF[t], "value": v,
                "label_valid": False, "decoy_source_doc": rng.choice(sources),
            }

    rng.shuffle(entries)
    assert sum(not e["label_valid"] for e in entries) == k
    assert len(entries) == m
    return {"m": m, "k": k, "entries": entries}


def render_entities(entries: list[dict]) -> dict:
    """Render entries as the ``## QUERY:`` block: ``<n>. [<type>] <json value>``.

    Spans are computed during rendering. ``char_start:char_end`` covers the
    JSON-encoded value *between* its quotes (exclusive end, offsets into
    ``text``). The closing quote is deliberately outside the span: byte-level
    BPE merges it with the following newline (``"\\n``), so a span ending on
    the quote would resolve to a token that carries nothing of the entity --
    at layer 0 every entry would read the same embedding.
    """
    if not entries:
        raise ValueError("entries is empty")
    lines: list[str] = []
    spans: list[tuple[int, int]] = []
    offset = 0
    for i, e in enumerate(entries, start=1):
        prefix = f"{i}. [{e['entity_type']}] "
        quoted = json.dumps(e["value"])
        lines.append(prefix + quoted)
        assert len(quoted) > 2, "empty value has no span"
        start = offset + len(prefix)
        spans.append((start + 1, start + len(quoted) - 1))
        offset += len(lines[-1]) + 1
    return {"text": "\n".join(lines), "spans": spans}


def build_items(params: dict, seed: int, repo_root: Path, tokenizer=None) -> list[dict]:
    """One item per ``(document, sample_index)`` for ``EntityDetectionLM.collect_entities``.

    ``params``: ``data_root``, ``split``, ``samples_per_doc`` (``s``),
    ``k_min``, ``context_token_limit`` (``tokenizer`` required iff not null),
    ``doc_subset`` (null | ``{n, seed}`` -- sample of documents).
    """
    data_root = repo_root / params["data_root"]
    split, s, k_min = params["split"], params["samples_per_doc"], params["k_min"]
    if not isinstance(s, int) or isinstance(s, bool) or s < 1:
        raise ValueError(f"samples_per_doc must be an int >= 1, got {s!r}")

    truth = load_ground_truth(data_root, split)
    pools = build_decoy_pools(truth)
    doc_ids = sorted(truth)

    subset = params["doc_subset"]
    if subset is not None:
        n = subset["n"]
        if n > len(doc_ids):
            raise ValueError(f"doc_subset.n {n} > available documents {len(doc_ids)}")
        doc_ids = sorted(random.Random(subset["seed"]).sample(doc_ids, n))

    limit = params["context_token_limit"]
    if (limit is None) != (tokenizer is None):
        raise ValueError("tokenizer must be given iff context_token_limit is not null")

    instructions = build_instructions()
    items: list[dict] = []
    for doc_id in doc_ids:
        context = load_ocr_context(data_root, doc_id)
        if limit is not None:
            n_tokens = len(tokenizer.encode(context, add_special_tokens=False))
            if n_tokens > limit:
                raise ValueError(
                    f"OCR context for {doc_id!r} is {n_tokens} tokens > context_token_limit {limit}"
                )
        for j in range(s):
            drawn = sample_prompt_entries(doc_id, truth[doc_id], pools, j, seed, k_min)
            rendered = render_entities(drawn["entries"])
            spans = []
            for pos, (e, (c0, c1)) in enumerate(zip(drawn["entries"], rendered["spans"]), start=1):
                spans.append({
                    "char_start": c0,
                    "char_end": c1,
                    "sample_index": j,
                    "m": drawn["m"],
                    "k": drawn["k"],
                    "entity_type": e["entity_type"],
                    "source_dataset": e["source_dataset"],
                    "value": e["value"],
                    "label_valid": e["label_valid"],
                    "list_position": pos,
                    "decoy_source_doc": e["decoy_source_doc"],
                    "decoy_in_ocr": (not e["label_valid"]) and (e["value"] in context),
                })
            items.append({
                "document_id": doc_id,
                "sample_index": j,
                "instructions": instructions,
                "context": context,
                "query": rendered["text"],
                "spans": spans,
            })
    return items


# ----------------------------------------------------------------------
# Model side


class EntityDetectionLM(RepresentationLM):
    """``RepresentationLM`` restricted to layers ``[0, n_layers]`` (embedding
    output and post-final-norm), reading entity-content spans only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.layers != [0, self.n_layers]:
            raise ValueError(
                f"entity detection records layers [0, 'last'] only, got {self.layers} "
                f"(n_layers={self.n_layers})"
            )

    def judge_entities(self, instructions: str, context: str, query: str, spans: list[dict]) -> dict:
        """One prefill pass; hidden states at each span's last covering token.

        Returns ``representations`` (``{layer: float32 [n_spans, hidden]}``),
        ``token_index`` / ``exact_end_alignment`` (``[n_spans]``), and
        ``prompt_n_tokens``.
        """
        if not spans:
            raise ValueError("spans is empty")
        input_ids, offsets, shift = self._build_prompt_with_offsets(instructions, context, query)
        prompt_n_tokens = len(input_ids)
        if prompt_n_tokens > self.max_position_embeddings:
            raise ValueError(
                f"prompt is {prompt_n_tokens} tokens > max_position_embeddings "
                f"{self.max_position_embeddings}; truncation would cut into the "
                "query, not spare context -- fix the input rather than truncate."
            )

        token_index: list[int] = []
        exact_end_alignment: list[bool] = []
        for span in spans:
            char_end = shift + span["char_end"]
            idx = self._last_token_covering(offsets, char_end - 1)
            token_index.append(idx)
            exact_end_alignment.append(offsets[idx][1] == char_end)
        assert len(set(token_index)) == len(token_index), (
            "two entities resolved to the same token -- span mapping is broken"
        )
        idx_tensor = torch.tensor(token_index, dtype=torch.long)

        saved: dict[int, "torch.Tensor"] = {}
        with torch.no_grad(), self.llm.trace(input_ids, logits_to_keep=1):
            for layer in self.layers:
                h = self._read_point(layer)
                seq = h[0] if h.ndim == 3 else h
                saved[layer] = seq[idx_tensor, :].detach().to(torch.float32).save()

        representations = {
            layer: np.asarray(t.cpu().numpy(), dtype=np.float32) for layer, t in saved.items()
        }
        n_spans = len(spans)
        for layer, arr in representations.items():
            assert arr.shape == (n_spans, self.hidden_size), (layer, arr.shape)
            assert np.isfinite(arr).all(), f"non-finite representation at layer {layer}"

        del saved
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return {
            "representations": representations,
            "token_index": np.asarray(token_index, dtype=np.int64),
            "exact_end_alignment": np.asarray(exact_end_alignment, dtype=bool),
            "prompt_n_tokens": prompt_n_tokens,
        }

    def _item_fingerprint(self, it: dict) -> str:
        """Identity of everything that determines an item's stored rows."""
        h = hashlib.sha256()
        for part in (self.model_name, it["instructions"], it["context"], it["query"]):
            h.update(part.encode())
            h.update(b"\x00")
        return h.hexdigest()

    def collect_entities(self, items: list[dict], cache_dir: str | Path, verify_read_point: bool) -> dict:
        """Judge every item; one ``.npz`` shard per document (resumable).

        A shard is reused only if its per-item fingerprints (model, instructions,
        context, query) and span count match exactly.

        Returns parallel arrays, one row per span, ordered by ``(sorted
        document_id, sample_index, list_position)``: ``representations``
        (``{layer: float32 [n, hidden]}``), ``layers``, ``doc_ids``,
        ``token_index``, ``exact_end_alignment``, ``prompt_n_tokens``, and every
        key of ``SPAN_META_KEYS`` (``m`` and ``k`` among them).
        """
        if not items:
            raise ValueError("items is empty")
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        by_doc: dict[str, list[dict]] = {}
        for it in items:
            by_doc.setdefault(it["document_id"], []).append(it)
        for doc_id, doc_items in by_doc.items():
            assert [it["sample_index"] for it in doc_items] == list(range(len(doc_items))), (
                f"{doc_id!r}: items are not sample_index 0..s-1 in order"
            )
            for it in doc_items:
                assert it["spans"], f"{doc_id!r}: item with no spans"
                for sp in it["spans"]:
                    assert set(sp) == {"char_start", "char_end", *SPAN_META_KEYS}, sorted(sp)

        if verify_read_point:
            first = items[0]
            self.verify_read_point(first["instructions"], first["context"], first["query"])

        for i, doc_id in enumerate(sorted(by_doc)):
            shard = cache_dir / f"{doc_id}.npz"
            fps = [self._item_fingerprint(it) for it in by_doc[doc_id]]
            n_spans = sum(len(it["spans"]) for it in by_doc[doc_id])
            if shard.is_file():
                with np.load(shard, allow_pickle=False) as z:
                    if list(z["item_fingerprints"]) == fps and int(z["n_spans"]) == n_spans:
                        if self.verbose:
                            print(f"[{i + 1}/{len(by_doc)}] {doc_id}: cached, skipping")
                        continue
                shard.unlink()  # incomplete / stale -- recompute
            if self.verbose:
                print(f"[{i + 1}/{len(by_doc)}] {doc_id}: judging {len(fps)} prompt(s), {n_spans} span(s)")
            self._judge_document_entities(by_doc[doc_id], fps, shard)

        return self._aggregate_entities(by_doc, cache_dir)

    def _judge_document_entities(self, doc_items: list[dict], fps: list[str], shard: Path) -> None:
        per_layer: dict[int, list[np.ndarray]] = {L: [] for L in self.layers}
        token_index: list[int] = []
        exact: list[bool] = []
        n_tokens: list[int] = []
        meta: dict[str, list] = {k: [] for k in SPAN_META_KEYS}

        for it in doc_items:
            res = self.judge_entities(it["instructions"], it["context"], it["query"], it["spans"])
            n = len(it["spans"])
            for L in self.layers:
                per_layer[L].append(res["representations"][L])
            token_index.extend(res["token_index"].tolist())
            exact.extend(res["exact_end_alignment"].tolist())
            n_tokens.extend([res["prompt_n_tokens"]] * n)
            for sp in it["spans"]:
                for k in SPAN_META_KEYS:
                    meta[k].append(sp[k])

        payload = {
            "n_spans": np.asarray(len(token_index), dtype=np.int64),
            "item_fingerprints": np.asarray(fps, dtype=object).astype("U"),
            "layers": np.asarray(self.layers, dtype=np.int64),
            "token_index": np.asarray(token_index, dtype=np.int64),
            "exact_end_alignment": np.asarray(exact, dtype=bool),
            "prompt_n_tokens": np.asarray(n_tokens, dtype=np.int64),
        }
        for L in self.layers:
            payload[f"rep_{L}"] = np.concatenate(per_layer[L], axis=0).astype(np.float32)
        for k in SPAN_META_KEYS:
            payload[f"meta_{k}"] = _to_npz_array(meta[k])

        tmp = shard.with_name(shard.stem + ".tmp.npz")
        np.savez(tmp, **payload)
        tmp.rename(shard)

    def _aggregate_entities(self, by_doc: dict, cache_dir: Path) -> dict:
        rep_blocks: dict[int, list[np.ndarray]] = {L: [] for L in self.layers}
        cols: dict[str, list[np.ndarray]] = {
            k: [] for k in ("token_index", "exact_end_alignment", "prompt_n_tokens")
        }
        meta: dict[str, list[np.ndarray]] = {k: [] for k in SPAN_META_KEYS}
        doc_id_col: list[np.ndarray] = []

        for doc_id in sorted(by_doc):
            with np.load(cache_dir / f"{doc_id}.npz", allow_pickle=False) as z:
                assert list(z["layers"]) == self.layers, (doc_id, z["layers"])
                n = int(z["n_spans"])
                for L in self.layers:
                    rep_blocks[L].append(z[f"rep_{L}"])
                for k in cols:
                    cols[k].append(z[k])
                for k in SPAN_META_KEYS:
                    meta[k].append(z[f"meta_{k}"])
                doc_id_col.append(np.asarray([doc_id] * n, dtype=object).astype("U"))

        out: dict = {
            "representations": {
                L: np.concatenate(rep_blocks[L], axis=0).astype(np.float32) for L in self.layers
            },
            "layers": np.asarray(self.layers, dtype=np.int64),
            "doc_ids": np.concatenate(doc_id_col, axis=0),
        }
        for k, parts in cols.items():
            out[k] = np.concatenate(parts, axis=0)
        for k, parts in meta.items():
            out[k] = np.concatenate(parts, axis=0)

        n = len(out["doc_ids"])
        assert n == sum(len(it["spans"]) for v in by_doc.values() for it in v), "row count mismatch"
        for L in self.layers:
            assert out["representations"][L].shape == (n, self.hidden_size), L
            assert np.isfinite(out["representations"][L]).all(), f"non-finite reps at layer {L}"
        for k, v in out.items():
            if k not in ("representations", "layers"):
                assert len(v) == n, f"{k}: len {len(v)} != n {n}"
        check_prompt_invariants(out)
        return out


def check_prompt_invariants(result: dict) -> None:
    """Every ``(document, sample)`` group has exactly ``m`` rows, ``k`` of them
    invalid, with a single ``(m, k)``, and list positions ``1..m``."""
    key = np.char.add(np.char.add(result["doc_ids"], "|"), result["sample_index"].astype(str))
    uniq, inverse = np.unique(key, return_inverse=True)
    inverse = inverse.reshape(-1)
    counts = np.bincount(inverse)
    invalid = np.bincount(inverse, weights=(~result["label_valid"]).astype(np.float64))
    for name in ("m", "k"):
        lo = np.full(len(uniq), np.iinfo(np.int64).max)
        hi = np.full(len(uniq), np.iinfo(np.int64).min)
        np.minimum.at(lo, inverse, result[name])
        np.maximum.at(hi, inverse, result[name])
        assert (lo == hi).all(), f"{name} is not constant within a prompt"
        if name == "m":
            assert (lo == counts).all(), "a prompt's row count != its recorded m"
        else:
            assert (lo == invalid.astype(np.int64)).all(), "a prompt's invalid count != its recorded k"
    pos_sum = np.bincount(inverse, weights=result["list_position"].astype(np.float64))
    assert (pos_sum == counts * (counts + 1) / 2).all(), "list positions are not 1..m"


def compute_metrics(result: dict) -> dict:
    """Flat scalar counts and composition summaries (no judge verdict is read)."""
    label_valid = result["label_valid"].astype(bool)
    invalid = ~label_valid
    key = np.char.add(np.char.add(result["doc_ids"], "|"), result["sample_index"].astype(str))
    _, first = np.unique(key, return_index=True)
    m, k = result["m"][first], result["k"][first]
    return {
        "n_spans": int(len(label_valid)),
        "n_prompts": int(len(first)),
        "n_valid_spans": int(label_valid.sum()),
        "n_invalid_spans": int(invalid.sum()),
        "mean_m": float(m.mean()),
        "max_m": int(m.max()),
        "mean_k": float(k.mean()),
        "mean_k_over_m": float((k / m).mean()),
        "exact_end_alignment_rate": float(result["exact_end_alignment"].mean()),
        "decoy_in_ocr_rate": float(result["decoy_in_ocr"][invalid].mean()) if invalid.any() else None,
        "min_prompt_n_tokens": int(result["prompt_n_tokens"].min()),
        "max_prompt_n_tokens": int(result["prompt_n_tokens"].max()),
    }
