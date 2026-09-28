"""Relation detection over per-type entity lists, ``line`` dataset.

One prompt per document. ``## QUERY:`` holds one numbered list per line-field
type (``TYPE n ENTITIES (<field>):``), every entry a genuine value of that
document (valid-only; see ``relation_prompts.build_relation_lists``). Entries
carry no verdict cue: each is just ``<n>. <json value>``.

What is read: for every list entry, the hidden state at the last token of the
entity's own value *as it appears in that list*. ``char_start:char_end`` covers
the JSON-encoded value between its quotes (exclusive end); the closing quote is
deliberately outside the span -- byte-level BPE merges it with the following
newline (``"\\n``), so a span ending on the quote would resolve to a token that
carries nothing of the entity. Offsets are computed during rendering, never by
substring search, so a value that recurs across lists is never ambiguous, and
the model side asserts every resolved token lies inside the ``## QUERY:`` block
(never in the OCR context, where the same string may also occur).

Layers are pinned to ``[0, "last"]`` (token-embedding output, post-final-norm);
no logits are read and the forward pass runs with ``logits_to_keep=1``.

This module supersedes the ``relation_detection`` task (``representationlm.
collect_spans`` + ``relation_prompts.render_relation_prompt``), which appended a
``— valid?`` cue to each entry and ended spans on the closing quote. That task
is left untouched so its finished runs stay reproducible.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from judge_interp.entity_detection import TYPE_DESCRIPTIONS, EntityDetectionLM
from judge_interp.prompts import LINE_FIELDS, load_ocr_context
from judge_interp.relation_prompts import build_relation_lists
from judge_interp.representationlm import _to_npz_array

assert set(LINE_FIELDS) <= set(TYPE_DESCRIPTIONS), "a line field has no type description"

_HEADER = """You are validating relationships between entities extracted from a document by an information-extraction pipeline. The entities are the values of the individual line items (spots) of an advertising order, and come in {n_types} types.

You are given:
1) In ## CONTEXT: the full OCR text of the source document.
2) In ## QUERY: one numbered list per entity type, each headed `TYPE <t> ENTITIES (<field>):`. Every entity listed is a genuine value drawn from the document for its type -- you are not judging whether an individual entity is real, only which combinations of them belong together.

The entity types, in the order their lists appear, are:
{types}

A valid {n_types}-tuple selects exactly one entity from each type such that all selected entities are mutually related in the document: they must all describe the same line item (the same broadcast spot). An entity individually present in the context but combined with an entity it does not belong with fails this criterion, even though both entities are themselves genuine.

Some entities may belong to more than one valid tuple (e.g. the same value shared across multiple line items) -- report every valid tuple you find, not just one per entity.

List every valid tuple you find, one per line, in the form `({index_format})`, using the numbered index from each type's list."""

# Span metadata columns, in shard order. All values are str / int.
SPAN_META_KEYS = ("entity_type", "type_num", "value", "list_position")


def build_instructions(entity_types: list[str]) -> str:
    """The instruction block for the types actually rendered, in list order.

    Raises:
        ValueError: fewer than 2 types, a duplicate, or a type outside
            ``LINE_FIELDS``.
    """
    if not isinstance(entity_types, list) or len(entity_types) < 2:
        raise ValueError(f"entity_types must be a list of >= 2 field names, got {entity_types!r}")
    if len(set(entity_types)) != len(entity_types):
        raise ValueError(f"entity_types has duplicate(s): {entity_types!r}")
    unknown = set(entity_types) - set(LINE_FIELDS)
    if unknown:
        raise ValueError(f"entity_types outside LINE_FIELDS: {sorted(unknown)}")
    types = "\n".join(f"- {t}: {TYPE_DESCRIPTIONS[t]}" for t in entity_types)
    index_format = ", ".join(f"<type {i} index>" for i in range(1, len(entity_types) + 1))
    return _HEADER.format(n_types=len(entity_types), types=types, index_format=index_format)


def render_lists(doc_lists: dict[str, list[str]]) -> dict:
    """Render a document's per-type value lists as the ``## QUERY:`` block.

    Args:
        doc_lists: ``{field: [value, ...]}`` (already in presentation order).
            A ``LINE_FIELDS`` type missing from it is omitted from the prompt
            entirely, and the remaining types are renumbered ``1..``.

    Returns:
        ``{"text", "entity_types", "spans"}``: ``entity_types`` is the rendered
        field order; ``spans`` is one dict per entry in text order with
        ``char_start``/``char_end`` (offsets into ``text``, exclusive end,
        covering the JSON value between its quotes) plus ``SPAN_META_KEYS``.

    Raises:
        ValueError: ``doc_lists`` empty, a field outside ``LINE_FIELDS``, or an
            empty list.
        AssertionError: an empty-string value (it has no span).
    """
    if not doc_lists:
        raise ValueError("doc_lists is empty")
    unknown = set(doc_lists) - set(LINE_FIELDS)
    if unknown:
        raise ValueError(f"doc_lists has field(s) outside LINE_FIELDS: {sorted(unknown)}")

    sections: list[str] = []
    spans: list[dict] = []
    entity_types: list[str] = []
    offset = 0
    for field in LINE_FIELDS:  # fixed type order
        if field not in doc_lists:
            continue
        values = doc_lists[field]
        if not values:
            raise ValueError(f"{field!r}: empty list")
        entity_types.append(field)
        type_num = len(entity_types)
        header = f"TYPE {type_num} ENTITIES ({field}):\n"
        lines: list[str] = []
        line_offset = offset + len(header)
        for i, value in enumerate(values, start=1):
            prefix = f"{i}. "
            quoted = json.dumps(value)
            assert len(quoted) > 2, f"{field!r}: empty value has no span"
            start = line_offset + len(prefix)
            spans.append({
                "char_start": start + 1,
                "char_end": start + len(quoted) - 1,
                "entity_type": field,
                "type_num": type_num,
                "value": value,
                "list_position": i,
            })
            lines.append(prefix + quoted)
            line_offset += len(lines[-1]) + 1
        section = header + "\n".join(lines)
        sections.append(section)
        offset += len(section) + 2  # the "\n\n" section separator
    text = "\n\n".join(sections)

    for sp in spans:
        assert text[sp["char_start"]:sp["char_end"]] == json.dumps(sp["value"])[1:-1], (
            f"span for {sp['entity_type']!r} #{sp['list_position']} does not cover its value"
        )
    return {"text": text, "entity_types": entity_types, "spans": spans}


def build_items(params: dict, seed: int, repo_root: Path, tokenizer=None) -> tuple[list[dict], dict, list[str]]:
    """One item per document with >= ``min_types_per_doc`` renderable lists.

    ``params``: ``data_root``, ``split``, ``min_valid_per_type``,
    ``min_types_per_doc`` (>= 2: a relation needs two types), ``context_token_limit``
    (``tokenizer`` required iff not null), ``doc_subset`` (null | ``{n, seed}``,
    sampled from the documents that survive ``min_types_per_doc``).

    Returns ``(items, tuple_membership, skipped_docs)``; ``tuple_membership`` is
    the ground-truth valid / intra-document-invalid tuple bookkeeping (never
    rendered), restricted to the documents in ``items``; ``skipped_docs`` are
    the documents dropped for having too few renderable types.
    """
    min_types = params["min_types_per_doc"]
    if not isinstance(min_types, int) or isinstance(min_types, bool) or min_types < 2:
        raise ValueError(f"min_types_per_doc must be an int >= 2, got {min_types!r}")
    data_root = repo_root / params["data_root"]
    result = build_relation_lists(
        data_root, params["split"], seed=seed, min_valid_per_type=params["min_valid_per_type"]
    )
    limit = params["context_token_limit"]
    if (limit is None) != (tokenizer is None):
        raise ValueError("tokenizer must be given iff context_token_limit is not null")

    by_doc: dict[str, dict[str, list[str]]] = {}
    for (doc_id, field), entities in result["lists"].items():
        by_doc.setdefault(doc_id, {})[field] = [v for v, _ in entities]
    skipped_docs = sorted(d for d, lists in by_doc.items() if len(lists) < min_types)
    doc_ids = sorted(set(by_doc) - set(skipped_docs))
    if not doc_ids:
        raise ValueError("no document has enough renderable types")

    subset = params["doc_subset"]
    if subset is not None:
        n = subset["n"]
        if n > len(doc_ids):
            raise ValueError(f"doc_subset.n {n} > available documents {len(doc_ids)}")
        doc_ids = sorted(random.Random(subset["seed"]).sample(doc_ids, n))

    items: list[dict] = []
    for doc_id in doc_ids:
        context = load_ocr_context(data_root, doc_id)
        if limit is not None:
            n_tokens = len(tokenizer.encode(context, add_special_tokens=False))
            if n_tokens > limit:
                raise ValueError(
                    f"OCR context for {doc_id!r} is {n_tokens} tokens > context_token_limit {limit}"
                )
        rendered = render_lists(by_doc[doc_id])
        items.append({
            "document_id": doc_id,
            "instructions": build_instructions(rendered["entity_types"]),
            "context": context,
            "query": rendered["text"],
            "spans": rendered["spans"],
        })

    kept = {it["document_id"] for it in items}
    tuple_membership = {d: t for d, t in result["tuples"].items() if d in kept}
    return items, tuple_membership, skipped_docs


# ----------------------------------------------------------------------
# Model side


class RelationDetectionLM(EntityDetectionLM):
    """``EntityDetectionLM`` (layers ``[0, n_layers]``, ``logits_to_keep=1``,
    ``judge_entities`` reading each span's last covering token) with relation
    metadata and one-item-per-document shards."""

    def judge_entities(self, instructions: str, context: str, query: str, spans: list[dict]) -> dict:
        res = super().judge_entities(instructions, context, query, spans)
        # Every read token must sit inside the QUERY block: a span that landed
        # in the OCR context (same string, wrong place) is the failure this
        # task exists to rule out.
        _, offsets, shift = self._build_prompt_with_offsets(instructions, context, query)
        for sp, idx in zip(spans, res["token_index"].tolist()):
            assert shift <= offsets[idx][0] < shift + len(query), (
                f"{sp['entity_type']!r} #{sp['list_position']}: token {idx} starts outside the QUERY block"
            )
        return res

    def collect_relations(self, items: list[dict], cache_dir: str | Path, verify_read_point: bool) -> dict:
        """Judge every item; one ``.npz`` shard per document (resumable).

        Returns parallel arrays, one row per span, ordered by ``(sorted
        document_id, list order)``: ``representations`` (``{layer: float32
        [n, hidden]}``), ``layers``, ``doc_ids``, ``token_index``,
        ``exact_end_alignment``, ``prompt_n_tokens`` and every ``SPAN_META_KEYS``.
        """
        if not items:
            raise ValueError("items is empty")
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        doc_ids = [it["document_id"] for it in items]
        assert len(set(doc_ids)) == len(doc_ids), "relation items must be one per document"
        for it in items:
            assert it["spans"], f"{it['document_id']!r}: item with no spans"
            for sp in it["spans"]:
                assert set(sp) == {"char_start", "char_end", *SPAN_META_KEYS}, sorted(sp)

        if verify_read_point:
            first = items[0]
            self.verify_read_point(first["instructions"], first["context"], first["query"])

        by_doc = {it["document_id"]: it for it in items}
        for i, doc_id in enumerate(sorted(by_doc)):
            it = by_doc[doc_id]
            shard = cache_dir / f"{doc_id}.npz"
            fp = self._item_fingerprint(it)
            if shard.is_file():
                with np.load(shard, allow_pickle=False) as z:
                    if list(z["item_fingerprints"]) == [fp] and int(z["n_spans"]) == len(it["spans"]):
                        if self.verbose:
                            print(f"[{i + 1}/{len(by_doc)}] {doc_id}: cached, skipping")
                        continue
                shard.unlink()  # incomplete / stale -- recompute
            if self.verbose:
                print(f"[{i + 1}/{len(by_doc)}] {doc_id}: {len(it['spans'])} span(s)")
            self._judge_document_relations(it, fp, shard)

        return self._aggregate_relations(by_doc, cache_dir)

    def _judge_document_relations(self, it: dict, fp: str, shard: Path) -> None:
        res = self.judge_entities(it["instructions"], it["context"], it["query"], it["spans"])
        n = len(it["spans"])
        payload = {
            "n_spans": np.asarray(n, dtype=np.int64),
            "item_fingerprints": np.asarray([fp], dtype=object).astype("U"),
            "layers": np.asarray(self.layers, dtype=np.int64),
            "token_index": res["token_index"],
            "exact_end_alignment": res["exact_end_alignment"],
            "prompt_n_tokens": np.full(n, res["prompt_n_tokens"], dtype=np.int64),
        }
        for L in self.layers:
            payload[f"rep_{L}"] = res["representations"][L].astype(np.float32)
        for k in SPAN_META_KEYS:
            payload[f"meta_{k}"] = _to_npz_array([sp[k] for sp in it["spans"]])

        tmp = shard.with_name(shard.stem + ".tmp.npz")
        np.savez(tmp, **payload)
        tmp.rename(shard)

    def _aggregate_relations(self, by_doc: dict, cache_dir: Path) -> dict:
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
                assert n == len(by_doc[doc_id]["spans"]), (doc_id, n)
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
        for k, parts in {**cols, **meta}.items():
            out[k] = np.concatenate(parts, axis=0)

        n = len(out["doc_ids"])
        assert n == sum(len(it["spans"]) for it in by_doc.values()), "row count mismatch"
        for L in self.layers:
            assert out["representations"][L].shape == (n, self.hidden_size), L
            assert np.isfinite(out["representations"][L]).all(), f"non-finite reps at layer {L}"
        for k, v in out.items():
            if k not in ("representations", "layers"):
                assert len(v) == n, f"{k}: len {len(v)} != n {n}"
        check_list_invariants(out)
        return out


def check_list_invariants(result: dict) -> None:
    """Within each ``(document, type)``: list positions are exactly ``1..n``
    and ``type_num`` is constant; within each document ``type_num`` values are
    exactly ``1..T`` (each ``entity_type`` on one ``type_num``)."""
    key = np.char.add(np.char.add(result["doc_ids"], "|"), result["entity_type"])
    uniq, inverse = np.unique(key, return_inverse=True)
    inverse = inverse.reshape(-1)
    counts = np.bincount(inverse)
    pos_sum = np.bincount(inverse, weights=result["list_position"].astype(np.float64))
    assert (pos_sum == counts * (counts + 1) / 2).all(), "list positions are not 1..n"
    assert (result["list_position"] <= counts[inverse]).all(), "list positions are not 1..n"
    lo = np.full(len(uniq), np.iinfo(np.int64).max)
    hi = np.full(len(uniq), np.iinfo(np.int64).min)
    np.minimum.at(lo, inverse, result["type_num"])
    np.maximum.at(hi, inverse, result["type_num"])
    assert (lo == hi).all(), "type_num is not constant within a (document, type)"
    for doc in np.unique(result["doc_ids"]):
        nums = sorted(set(result["type_num"][result["doc_ids"] == doc].tolist()))
        assert nums == list(range(1, len(nums) + 1)), f"{doc!r}: type_nums {nums} are not 1..T"


def compute_metrics(result: dict, n_skipped_docs: int) -> dict:
    """Flat scalar counts (no verdict is read)."""
    return {
        "n_documents_skipped_min_types": int(n_skipped_docs),
        "n_spans": int(len(result["doc_ids"])),
        "n_documents": int(len(np.unique(result["doc_ids"]))),
        **{f"n_spans_{t}": int((result["entity_type"] == t).sum()) for t in LINE_FIELDS},
        "exact_end_alignment_rate": float(result["exact_end_alignment"].mean()),
        "min_prompt_n_tokens": int(result["prompt_n_tokens"].min()),
        "max_prompt_n_tokens": int(result["prompt_n_tokens"].max()),
    }
