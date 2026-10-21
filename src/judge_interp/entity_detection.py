"""Entity detection: hidden states read at each entity's closing quote, one prompt per dataset row.

Pipeline:
1. ``build_instructions`` states the judging task: for each non-null field of a row,
   decide whether the value is explicitly supported by the OCR context and is the
   right kind of value for that field.
2. ``build_items`` loads ``data/vrdu/{main,line}/<split>.json``, draws a stratified
   subset of rows (``select_rows``), renders each row (``render_row``) and attaches a
   per-entity label from the row's ``field_errors``.
3. ``EntityDetectionLM.collect_rows`` runs one prefill pass per row and stores, per
   layer, the hidden state at the token covering each closing quote, in resumable
   per-document shards.
4. ``check_row_invariants`` and ``compute_metrics`` check label bookkeeping and
   summarise span counts. No verdict is read: the model is never asked to answer.

Rendering: every non-null value is a JSON string followed by one pad space before its
closing quote, and the query ends with a newline. Under byte-level BPE (Qwen2.5) this
makes the closing quote tokenize the same way on every line and keeps the value's last
character out of the quote token, so the read position carries no entity text.

Labels are per entity: ``field_errors[field]`` is null for a correct entity and
``{"error_type", "source"}`` for a corrupted one. Null fields have no quote and no span.
Layers are whatever the config lists; no logits are materialised (``logits_to_keep=1``).
"""
from __future__ import annotations

import gc
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from judge_interp.prompts import (
    ERROR_TYPES,
    LINE_CARRIED_FIELDS,
    QUERY_FIELDS,
    TYPE_DESCRIPTIONS,
    load_ocr_context,
    load_split,
    render_query,
    sub_seed,
)
from judge_interp.representationlm import RepresentationLM, to_npz_array

DATASETS = ("main", "line")
assert LINE_CARRIED_FIELDS == [], "line rows must not carry main fields in this task"
ALL_ERROR_TYPES = ("inter_document", "intra_document", "cross_type")
_SOURCE_SCOPES = ("other_document", "sentinel", "same_document")
_PAD = " "  # inserted before each closing quote; see the module docstring

assert set(QUERY_FIELDS["main"]).isdisjoint(QUERY_FIELDS["line"]), "main/line field names collide"

_RECORD_NOUN = {"main": "the document's order (one record per document)", "line": "one line item of the document's order"}
_OTHER_PLACE = {
    "main": "to another field of the record or to another part of the document",
    "line": "to another field of this line item, to another line item, or to the order as a whole",
}

_HEADER = """You are validating a record extracted from a document by an information-extraction pipeline.

You are given:
1) In ## CONTEXT: the full OCR text of the source document.
2) In ## QUERY: the extracted record, which is {noun}. It has one line per field, in the form `field: value`. A value is a JSON string in double quotes, or `null` when the field was not extracted. Each string value is followed by one space before its closing quote; that padding space is not part of the value.

The fields are:
{types}

For each field whose value is not null, decide independently whether the value is a VALID value for that field in this document. A value is valid only if BOTH criteria below hold:

{criteria}

Judge only against the context. Do not use outside knowledge about the entities involved, and do not guess when the evidence is unclear — unclear evidence is a failed criterion. Judge every field independently: one field's validity has no bearing on another's.

For each field with a value, answer with exactly one line in the form `<field>: true` or `<field>: false`, in the order shown. Do not answer for fields whose value is null."""

EXPLICIT_CRITERION = """({n}) Explicit support. Every non-null value must appear explicitly in the context. A value may differ from the context only in trivial surface formatting (whitespace, line breaks, letter case, and equivalent number forms such as 10 vs 10.0 or 1,000 vs 1000). A value produced by paraphrasing, summarising, inferring, rounding, unit conversion, or any other transformation of the text is NOT explicit support. A `null` value always satisfies this criterion."""

_TYPE_CRITERION = """({n}) Type correspondence. The field name is a claim about what the value is, and it must hold in this document: the value must be what that field denotes here, as described in the list of fields above. Appearing somewhere in the context is not enough. A value that is present in the context but is actually the value of a different field fails this criterion, even though it is a genuine entity of the document. For example, a value under `advertiser` whose text is really the name of the agency, the TV station, or a product description is invalid, as is a date or a dollar amount that belongs {other_place}. Check each value against the role its field requires, not merely against the text."""

# Span metadata columns, in shard order. All values are str / int / bool.
SPAN_META_KEYS = (
    "dataset", "row_ordinal", "line_index", "row_error_type", "row_k", "field", "field_position",
    "value", "label_valid", "field_error_type", "source_scope", "source_dataset", "source_field",
    "source_line_index", "exchanged_with",
)
# Per-span columns the model side adds.
_MODEL_KEYS = ("token_index", "token_id", "token_n_chars", "exact_end_alignment", "prompt_n_tokens")


def build_instructions(dataset: str) -> str:
    """The instruction block for one dataset (fixed for a run)."""
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    types = "\n".join(f"- {f}: {TYPE_DESCRIPTIONS[f]}" for f in QUERY_FIELDS[dataset])
    criteria = "\n".join([
        EXPLICIT_CRITERION.format(n=1),
        _TYPE_CRITERION.format(n=2, other_place=_OTHER_PLACE[dataset]),
    ])
    return _HEADER.format(noun=_RECORD_NOUN[dataset], types=types, criteria=criteria)


# ----------------------------------------------------------------------
# Rows: validation and labels


def check_row(row: dict, dataset: str) -> None:
    """Assert that a row's label fields agree with each other.

    Args:
        row: a dataset row with ``valid``, ``error_type``, ``k``, ``num_invalid_fields``,
            ``invalid_fields``, ``field_errors`` and one key per field in ``dataset``.
        dataset: ``"main"`` or ``"line"``; selects the expected field list.

    Raises:
        AssertionError: the label fields disagree, a corrupted field has a null value,
            or a ``source`` entry is malformed for its scope.
    """
    fields = QUERY_FIELDS[dataset]
    where = f"{row['document_id']!r}/{row.get('line_index')}"
    fe = row["field_errors"]
    assert list(fe) == fields, f"{where}: field_errors keys {list(fe)} != {fields}"
    bad = [f for f in fields if fe[f] is not None]
    assert row["valid"] == (row["error_type"] is None), f"{where}: valid vs error_type"
    assert row["valid"] == (not bad), f"{where}: valid vs field_errors"
    assert row["k"] == row["num_invalid_fields"] == len(bad) == len(row["invalid_fields"]), where
    assert set(row["invalid_fields"]) == set(bad), f"{where}: invalid_fields vs field_errors"
    if not row["valid"]:
        assert row["error_type"] in ERROR_TYPES[dataset], f"{where}: error_type {row['error_type']!r}"
    for f in fields:
        v = row[f]
        assert v is None or isinstance(v, str), f"{where}/{f}: non-str value {v!r}"
        e = fe[f]
        if e is None:
            continue
        assert v is not None, f"{where}/{f}: corrupted field has a null value"
        assert set(e) == {"error_type", "source"}, f"{where}/{f}: {sorted(e)}"
        assert e["error_type"] == row["error_type"], f"{where}/{f}: field vs row error_type"
        src = e["source"]
        assert src["scope"] in _SOURCE_SCOPES, f"{where}/{f}: scope {src['scope']!r}"
        if src["scope"] == "same_document":
            need = {"scope", "dataset", "field", "line_index"}
            if row["error_type"] == "cross_type":
                need.add("exchanged_with")
            assert set(src) == need, f"{where}/{f}: source keys {sorted(src)}"
            assert src["dataset"] in DATASETS and src["field"] in TYPE_DESCRIPTIONS, f"{where}/{f}"
        else:
            assert set(src) == {"scope"}, f"{where}/{f}: source keys {sorted(src)}"


def _source_columns(entry: dict | None) -> dict:
    """Flatten a ``field_errors`` entry's ``source`` into span metadata columns.

    Args:
        entry: ``field_errors[field]`` (``{"error_type", "source"}``), or ``None`` for a
            correct entity.

    Returns:
        Dict with ``source_scope``, ``source_dataset``, ``source_field``,
        ``source_line_index`` and ``exchanged_with``; ``""`` (``-1`` for the index)
        where the entry has no such value.
    """
    out = {"source_scope": "", "source_dataset": "", "source_field": "", "source_line_index": -1,
           "exchanged_with": ""}
    if entry is None:
        return out
    src = entry["source"]
    out["source_scope"] = src["scope"]
    if src["scope"] == "same_document":
        out["source_dataset"] = src["dataset"]
        out["source_field"] = src["field"]
        out["source_line_index"] = -1 if src["line_index"] is None else src["line_index"]
        if "exchanged_with" in src:
            out["exchanged_with"] = "" if src["exchanged_with"] is None else src["exchanged_with"]
    return out


# ----------------------------------------------------------------------
# Rendering


def render_row(row: dict, dataset: str) -> dict:
    """Render a row as the query text and locate each entity's closing quote.

    The text equals ``prompts.render_query`` with one pad space before each non-null
    value's closing quote and a trailing newline (asserted line by line).

    Args:
        row: a dataset row; every field of ``dataset`` must be a ``str`` or ``None``.
        dataset: ``"main"`` or ``"line"``.

    Returns:
        ``{"text": str, "spans": [(field, char_start, char_end), ...]}``: one span per
        non-null field in canonical order, covering exactly the closing ``"``
        (offsets into ``text``, exclusive end).
    """
    lines: list[str] = []
    spans: list[tuple[str, int, int]] = []
    offset = 0
    for f in QUERY_FIELDS[dataset]:
        v = row[f]
        prefix = f"{f}: "
        encoded = json.dumps(v)
        if v is not None:
            assert isinstance(v, str), f"{f}: non-str value {v!r}"
            encoded = encoded[:-1] + _PAD + '"'
        lines.append(prefix + encoded)
        if v is not None:
            close = offset + len(prefix) + len(encoded) - 1
            spans.append((f, close, close + 1))
        offset += len(lines[-1]) + 1
    text = "\n".join(lines) + "\n"

    base = render_query(row, dataset).split("\n")
    assert len(base) == len(lines), "render_row diverged from prompts.render_query"
    for f, b, mine in zip(QUERY_FIELDS[dataset], base, lines):
        expected = b if row[f] is None else b[:-1] + _PAD + '"'
        assert mine == expected, f"{f}: render_row diverged from prompts.render_query"
    for f, a, b in spans:
        assert text[a:b] == '"', f"{f}: span does not cover a closing quote"
    return {"text": text, "spans": spans}


# ----------------------------------------------------------------------
# Item building


def _stratum(row: dict) -> str:
    """Stratum name of a row: ``"valid"`` or its error type."""
    return "valid" if row["valid"] else row["error_type"]


def select_rows(rows: list[dict], dataset: str, split: str, rows_per_stratum: dict, seed: int) -> list[dict]:
    """Draw a stratified subset of rows.

    Strata are ``"valid"`` plus each error type present in ``rows``.

    Args:
        rows: rows from ``prompts.load_split``.
        dataset: ``"main"`` or ``"line"``; seeds the per-stratum draw.
        split: ``"train"`` or ``"test"``; seeds the per-stratum draw.
        rows_per_stratum: stratum name -> number of rows to draw (``None`` takes all);
            keys must equal the strata present exactly.
        seed: run seed.

    Returns:
        The drawn rows, sorted by ``(document_id, line_index, stratum, k)``.

    Raises:
        ValueError: key mismatch, a count that is not ``None`` or an int >= 1, or a
            stratum smaller than its requested count.
    """
    strata = sorted({_stratum(r) for r in rows})
    if sorted(rows_per_stratum) != strata:
        raise ValueError(f"rows_per_stratum keys {sorted(rows_per_stratum)} != strata present {strata}")
    chosen: list[dict] = []
    for s in strata:
        pool = sorted((r for r in rows if _stratum(r) == s),
                      key=lambda r: (r["document_id"], r.get("line_index", -1), r["k"]))
        n = rows_per_stratum[s]
        if n is None:
            chosen.extend(pool)
            continue
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError(f"rows_per_stratum[{s!r}] must be null or an int >= 1, got {n!r}")
        if n > len(pool):
            raise ValueError(f"stratum {s!r}: {n} rows requested but only {len(pool)} available")
        chosen.extend(random.Random(sub_seed(seed, dataset, split, s)).sample(pool, n))
    return sorted(chosen, key=lambda r: (r["document_id"], r.get("line_index", -1), _stratum(r), r["k"]))


def build_items(params: dict, seed: int, repo_root: Path, tokenizer=None) -> list[dict]:
    """Build one ``EntityDetectionLM.collect_rows`` item per selected row.

    Args:
        params: config ``params`` with ``data_root``, ``dataset``, ``split``,
            ``error_types``, ``rows_per_stratum`` and ``context_token_limit``.
        seed: run seed for the stratified draw.
        repo_root: directory that ``params["data_root"]`` is relative to.
        tokenizer: HF tokenizer used to enforce ``context_token_limit``; must be given
            if and only if the limit is not ``None``.

    Returns:
        Items ordered by ``select_rows``: ``document_id``, ``row_ordinal`` (0-based
        within the document), ``instructions``, ``context``, ``query`` and ``spans``
        (one dict per non-null field with ``char_start``, ``char_end`` and every key
        of ``SPAN_META_KEYS``).

    Raises:
        ValueError: unknown dataset, tokenizer/limit mismatch, or an OCR context
            longer than ``context_token_limit`` tokens.
    """
    data_root = repo_root / params["data_root"]
    dataset, split = params["dataset"], params["split"]
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    limit = params["context_token_limit"]
    if (limit is None) != (tokenizer is None):
        raise ValueError("tokenizer must be given iff context_token_limit is not null")

    rows = load_split(data_root, dataset, split, params["error_types"])
    for r in rows:
        check_row(r, dataset)
    selected = select_rows(rows, dataset, split, params["rows_per_stratum"], seed)

    instructions = build_instructions(dataset)
    contexts: dict[str, str] = {}
    items: list[dict] = []
    ordinal: dict[str, int] = {}
    for r in selected:
        doc_id = r["document_id"]
        if doc_id not in contexts:
            text = load_ocr_context(data_root, doc_id)
            if limit is not None:
                n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
                if n_tokens > limit:
                    raise ValueError(
                        f"OCR context for {doc_id!r} is {n_tokens} tokens > context_token_limit {limit}"
                    )
            contexts[doc_id] = text
        j = ordinal.get(doc_id, 0)
        ordinal[doc_id] = j + 1
        rendered = render_row(r, dataset)
        spans = []
        for pos, (f, c0, c1) in enumerate(rendered["spans"], start=1):
            entry = r["field_errors"][f]
            spans.append({
                "char_start": c0,
                "char_end": c1,
                "dataset": dataset,
                "row_ordinal": j,
                "line_index": r["line_index"] if dataset == "line" else -1,
                "row_error_type": "" if r["valid"] else r["error_type"],
                "row_k": r["k"],
                "field": f,
                "field_position": pos,
                "value": r[f],
                "label_valid": entry is None,
                "field_error_type": "" if entry is None else entry["error_type"],
                **_source_columns(entry),
            })
        assert spans, f"{doc_id!r}: row with no non-null field"
        items.append({
            "document_id": doc_id,
            "row_ordinal": j,
            "instructions": instructions,
            "context": contexts[doc_id],
            "query": rendered["text"],
            "spans": spans,
        })
    assert len(items) == len(selected), "row count changed while building items"
    return items


# ----------------------------------------------------------------------
# Model side


class EntityDetectionLM(RepresentationLM):
    """``RepresentationLM`` that reads hidden states at each entity's closing quote."""

    def judge_row(self, instructions: str, context: str, query: str, spans: list[dict]) -> dict:
        """Run one prefill pass and read the hidden state at each closing quote.

        Args:
            instructions: the ``## INSTRUCTIONS:`` text.
            context: the ``## CONTEXT:`` text (OCR).
            query: the ``## QUERY:`` text; ``spans`` offsets are relative to it.
            spans: dicts with one-character ``char_start`` / ``char_end`` covering a
                closing quote in ``query``.

        Returns:
            Dict with ``representations`` (``{layer: float32 [n_spans, hidden]}``),
            ``token_index``, ``token_id``, ``token_n_chars`` and ``exact_end_alignment``
            (each ``[n_spans]``), and ``prompt_n_tokens``. ``exact_end_alignment`` is
            true when the token is exactly the quote; otherwise BPE merged the quote
            with a neighbouring character.

        Raises:
            ValueError: ``spans`` is empty or the prompt exceeds
                ``max_position_embeddings``.
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
        token_id: list[int] = []
        n_chars: list[int] = []
        exact: list[bool] = []
        for span in spans:
            assert span["char_end"] - span["char_start"] == 1, "a span must be exactly one character"
            pos = shift + span["char_start"]
            idx = self._last_token_covering(offsets, pos)
            s, e = offsets[idx]
            assert s <= pos < e
            tid = int(input_ids[idx])
            assert '"' in self.tokenizer.decode([tid]), (
                f"token {idx} ({self.tokenizer.decode([tid])!r}) covering a closing quote has no quote"
            )
            token_index.append(idx)
            token_id.append(tid)
            n_chars.append(e - s)
            exact.append((s, e) == (pos, pos + 1))
        assert len(set(token_index)) == len(token_index), "two quotes resolved to the same token"
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
            "token_id": np.asarray(token_id, dtype=np.int64),
            "token_n_chars": np.asarray(n_chars, dtype=np.int64),
            "exact_end_alignment": np.asarray(exact, dtype=bool),
            "prompt_n_tokens": prompt_n_tokens,
        }

    def _item_fingerprint(self, it: dict) -> str:
        """Hash of everything that determines an item's stored rows.

        Args:
            it: an item from ``build_items``.

        Returns:
            Hex digest over model name, instructions, context and query.
        """
        h = hashlib.sha256()
        for part in (self.model_name, it["instructions"], it["context"], it["query"]):
            h.update(part.encode())
            h.update(b"\x00")
        return h.hexdigest()

    def collect_rows(self, items: list[dict], cache_dir: str | Path, verify_read_point: bool) -> dict:
        """Run every item and aggregate the spans, with one resumable shard per document.

        A cached shard is reused only if its item fingerprints and span count match.

        Args:
            items: output of ``build_items``; each document's ``row_ordinal`` values
                must be 0..n-1 in order.
            cache_dir: directory for the ``<document_id>.npz`` shards (created if missing).
            verify_read_point: run ``verify_read_point`` on the first item first.

        Returns:
            Parallel arrays with one entry per span, ordered by ``(document_id,
            row_ordinal, field_position)``: ``representations`` (``{layer: float32
            [n, hidden]}``), ``layers``, ``doc_ids``, the model columns, and every key
            of ``SPAN_META_KEYS``.

        Raises:
            ValueError: ``items`` is empty.
        """
        if not items:
            raise ValueError("items is empty")
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        by_doc: dict[str, list[dict]] = {}
        for it in items:
            by_doc.setdefault(it["document_id"], []).append(it)
        for doc_id, doc_items in by_doc.items():
            assert [it["row_ordinal"] for it in doc_items] == list(range(len(doc_items))), (
                f"{doc_id!r}: items are not row_ordinal 0..n-1 in order"
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
                print(f"[{i + 1}/{len(by_doc)}] {doc_id}: judging {len(fps)} row(s), {n_spans} span(s)")
            self._judge_document_rows(by_doc[doc_id], fps, shard)

        return self._aggregate_rows(by_doc, cache_dir)

    def _judge_document_rows(self, doc_items: list[dict], fps: list[str], shard: Path) -> None:
        """Judge one document's items and write its shard.

        Args:
            doc_items: the document's items, in ``row_ordinal`` order.
            fps: ``_item_fingerprint`` of each item.
            shard: destination ``.npz`` path, written atomically via a temporary file.
        """
        per_layer: dict[int, list[np.ndarray]] = {L: [] for L in self.layers}
        cols: dict[str, list] = {k: [] for k in _MODEL_KEYS}
        meta: dict[str, list] = {k: [] for k in SPAN_META_KEYS}

        for it in doc_items:
            res = self.judge_row(it["instructions"], it["context"], it["query"], it["spans"])
            n = len(it["spans"])
            for L in self.layers:
                per_layer[L].append(res["representations"][L])
            for k in ("token_index", "token_id", "token_n_chars", "exact_end_alignment"):
                cols[k].extend(res[k].tolist())
            cols["prompt_n_tokens"].extend([res["prompt_n_tokens"]] * n)
            for sp in it["spans"]:
                for k in SPAN_META_KEYS:
                    meta[k].append(sp[k])

        payload = {
            "n_spans": np.asarray(len(cols["token_index"]), dtype=np.int64),
            "item_fingerprints": np.asarray(fps, dtype=object).astype("U"),
            "layers": np.asarray(self.layers, dtype=np.int64),
        }
        for k, v in cols.items():
            payload[k] = to_npz_array(v)
        for L in self.layers:
            payload[f"rep_{L}"] = np.concatenate(per_layer[L], axis=0).astype(np.float32)
        for k in SPAN_META_KEYS:
            payload[f"meta_{k}"] = to_npz_array(meta[k])

        tmp = shard.with_name(shard.stem + ".tmp.npz")
        np.savez(tmp, **payload)
        tmp.rename(shard)

    def _aggregate_rows(self, by_doc: dict, cache_dir: Path) -> dict:
        """Concatenate the per-document shards in sorted document order.

        Args:
            by_doc: document_id -> its items.
            cache_dir: directory holding one shard per document.

        Returns:
            The dict described in ``collect_rows``, after ``check_row_invariants``.
        """
        rep_blocks: dict[int, list[np.ndarray]] = {L: [] for L in self.layers}
        cols: dict[str, list[np.ndarray]] = {k: [] for k in _MODEL_KEYS}
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
        for parts_by_key in (cols, meta):
            for k, parts in parts_by_key.items():
                out[k] = np.concatenate(parts, axis=0)

        n = len(out["doc_ids"])
        assert n == sum(len(it["spans"]) for v in by_doc.values() for it in v), "row count mismatch"
        for L in self.layers:
            assert out["representations"][L].shape == (n, self.hidden_size), L
            assert np.isfinite(out["representations"][L]).all(), f"non-finite reps at layer {L}"
        for k, v in out.items():
            if k not in ("representations", "layers"):
                assert len(v) == n, f"{k}: len {len(v)} != n {n}"
        check_row_invariants(out)
        return out


# ----------------------------------------------------------------------
# Checks and metrics


def _prompt_groups(result: dict) -> tuple[np.ndarray, np.ndarray]:
    """Group spans by prompt, i.e. by ``(document, row_ordinal)``.

    Args:
        result: aggregated arrays with ``doc_ids`` and ``row_ordinal``.

    Returns:
        ``(keys, inverse)``: the sorted unique prompt keys and, per span, the index of
        its prompt in ``keys``.
    """
    key = np.char.add(
        np.char.add(result["doc_ids"], "|"), result["row_ordinal"].astype(str)
    )
    uniq, inverse = np.unique(key, return_inverse=True)
    return uniq, inverse.reshape(-1)


def check_row_invariants(result: dict) -> None:
    """Assert per-prompt label consistency on the aggregated arrays.

    Within every ``(document, row_ordinal)`` group: ``row_k`` and ``row_error_type``
    are constant, the number of invalid spans equals ``row_k``, a valid row has no
    invalid span and vice versa, each invalid span's ``field_error_type`` equals the
    row's, valid spans have an empty one, and ``field_position`` increases strictly
    from 1.

    Args:
        result: aggregated arrays from ``_aggregate_rows``.

    Raises:
        AssertionError: any of the conditions above fails.
    """
    uniq, inv = _prompt_groups(result)
    label_valid = result["label_valid"].astype(bool)
    n_groups = len(uniq)
    n_invalid = np.bincount(inv, weights=(~label_valid).astype(np.float64), minlength=n_groups)

    row_k = result["row_k"]
    lo = np.full(n_groups, np.iinfo(np.int64).max)
    hi = np.full(n_groups, np.iinfo(np.int64).min)
    np.minimum.at(lo, inv, row_k)
    np.maximum.at(hi, inv, row_k)
    assert (lo == hi).all(), "row_k is not constant within a prompt"
    assert (lo == n_invalid.astype(np.int64)).all(), "a prompt's invalid-span count != its row_k"

    ret, rot = result["row_error_type"], result["field_error_type"]
    first = np.unique(inv, return_index=True)[1]
    assert ((ret[first] == "") == (lo == 0)).all(), "row_error_type empty iff row_k == 0"
    assert (ret == ret[first][inv]).all(), "row_error_type is not constant within a prompt"
    assert (rot[~label_valid] == ret[~label_valid]).all(), "invalid span's error type != its row's"
    assert (rot[label_valid] == "").all(), "valid span with a field_error_type"

    pos = result["field_position"]
    order = np.argsort(inv, kind="stable")
    sp, si = pos[order], inv[order]
    same = si[1:] == si[:-1]
    assert (sp[1:][same] > sp[:-1][same]).all(), "field_position not increasing within a prompt"
    assert (sp[np.r_[True, ~same]] == 1).all(), "field_position does not start at 1"


def compute_metrics(result: dict) -> dict:
    """Summarise span counts and prompt lengths of a run.

    Args:
        result: aggregated arrays from ``EntityDetectionLM.collect_rows``.

    Returns:
        Flat scalar dict: span, prompt, valid/invalid and per-error-type counts,
        ``n_rows_with_mixed_entities``, ``exact_end_alignment_rate``,
        ``n_distinct_quote_token_ids`` and min/max prompt length in tokens.
    """
    label_valid = result["label_valid"].astype(bool)
    uniq, inv = _prompt_groups(result)
    n_invalid = np.bincount(inv, weights=(~label_valid).astype(np.float64), minlength=len(uniq))
    n_spans_per = np.bincount(inv, minlength=len(uniq))
    out = {
        "n_spans": int(len(label_valid)),
        "n_prompts": int(len(uniq)),
        "n_valid_spans": int(label_valid.sum()),
        "n_invalid_spans": int((~label_valid).sum()),
        "n_rows_with_mixed_entities": int(((n_invalid > 0) & (n_invalid < n_spans_per)).sum()),
        "exact_end_alignment_rate": float(result["exact_end_alignment"].mean()),
        "n_distinct_quote_token_ids": int(len(np.unique(result["token_id"]))),
        "min_prompt_n_tokens": int(result["prompt_n_tokens"].min()),
        "max_prompt_n_tokens": int(result["prompt_n_tokens"].max()),
    }
    for et in ALL_ERROR_TYPES:
        out[f"n_spans_{et}"] = int((result["field_error_type"] == et).sum())
    assert sum(out[f"n_spans_{et}"] for et in ALL_ERROR_TYPES) == out["n_invalid_spans"]
    return out
