"""Row-based entity detection: one prompt per ``(document, row)``.

A *row* is one row of ``data/vrdu/{main,line}/<split>.json`` -- valid, or invalid
with one of the dataset's error types. The prompt is the document's OCR text plus
that single row, rendered by ``prompts.render_query`` (one ``field: <json>`` line
per field; line rows carry no main fields). Every non-null string value is
therefore wrapped in double quotes, and the representation of each entity is read
at its **closing quote**, the first position at which a causal model has seen the
whole value delimited.

Labels are per entity, not per row: ``field_errors[field]`` is null for a correct
entity and ``{"error_type", "source"}`` for a corrupted one, so one row can carry
correct and wrong entities side by side. Null fields have no quote and no span.

Nothing here samples or synthesises data; the dataset is the single source of
truth. Subsetting (a full run is far too large) is a stratified draw over the
rows, configured per stratum, with a hard error if a stratum is too small.

Layers are whatever the config lists (no pin). No logits are read, and the
forward pass runs with ``logits_to_keep=1``, so no ``[T, vocab]`` tensor is ever
materialised. The model is never asked for a verdict here; the instructions only
establish the task the representations are read under.

This module is deliberately self-contained: it imports nothing from
``entity_detection.py``.
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
from judge_interp.prompts import (
    ERROR_TYPES,
    LINE_CARRIED_FIELDS,
    QUERY_FIELDS,
    load_ocr_context,
    load_split,
    render_query,
)
from judge_interp.representationlm import RepresentationLM, _to_npz_array

DATASETS = ("main", "line")
assert LINE_CARRIED_FIELDS == [], "line rows must not carry main fields in this task"
ALL_ERROR_TYPES = ("inter_document", "intra_document", "cross_type")
_SOURCE_SCOPES = ("other_document", "sentinel", "same_document")

# Shown to the judge so it can apply the type-correspondence criterion.
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
    "channel": "the station or channel that this line item (spot) airs on",
    "program_desc": "the program, time slot or daypart that this line item airs in",
    "program_start_date": "the start date of this line item's air period",
    "program_end_date": "the end date of this line item's air period",
    "sub_amount": "the dollar amount (subtotal) of this line item",
}
for _ds in DATASETS:
    assert set(QUERY_FIELDS[_ds]) <= set(TYPE_DESCRIPTIONS), f"no description for a {_ds} field"
assert set(QUERY_FIELDS["main"]).isdisjoint(QUERY_FIELDS["line"]), "main/line field names collide"

_RECORD_NOUN = {"main": "the document's order (one record per document)", "line": "one line item of the document's order"}
_OTHER_PLACE = {
    "main": "to another field of the record or to another part of the document",
    "line": "to another field of this line item, to another line item, or to the order as a whole",
}

_HEADER = """You are validating a record extracted from a document by an information-extraction pipeline.

You are given:
1) In ## CONTEXT: the full OCR text of the source document.
2) In ## QUERY: the extracted record, which is {noun}. It has one line per field, in the form `field: value`. A value is a JSON string in double quotes, or `null` when the field was not extracted.

The fields are:
{types}

For each field whose value is not null, decide independently whether the value is a VALID value for that field in this document. A value is valid only if BOTH criteria below hold:

{criteria}

Judge only against the context. Do not use outside knowledge about the entities involved, and do not guess when the evidence is unclear — unclear evidence is a failed criterion. Judge every field independently: one field's validity has no bearing on another's.

For each field with a value, answer with exactly one line in the form `<field>: true` or `<field>: false`, in the order shown. Do not answer for fields whose value is null."""

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
        _EXPLICIT_CRITERION.format(n=1),
        _TYPE_CRITERION.format(n=2, other_place=_OTHER_PLACE[dataset]),
    ])
    return _HEADER.format(noun=_RECORD_NOUN[dataset], types=types, criteria=criteria)


# ----------------------------------------------------------------------
# Rows: validation and labels


def check_row(row: dict, dataset: str) -> None:
    """Assert a row's labels are internally consistent.

    ``valid``, ``error_type``, ``k``, ``num_invalid_fields``, ``invalid_fields``
    and ``field_errors`` must all agree; a corrupted field has a non-null value;
    sources are well formed for their scope.
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
    """Flat source columns; ``""`` / ``-1`` where there is no such source."""
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
    """Render a row and locate each entity's closing quote.

    Returns ``{"text", "spans"}`` where ``text == prompts.render_query(row,
    dataset)`` (asserted) and ``spans`` is a list of ``(field, char_start,
    char_end)`` -- one per non-null field, canonical order -- covering exactly the
    closing ``"`` character (offsets into ``text``, exclusive end).
    """
    lines: list[str] = []
    spans: list[tuple[str, int, int]] = []
    offset = 0
    for f in QUERY_FIELDS[dataset]:
        v = row[f]
        prefix = f"{f}: "
        encoded = json.dumps(v)
        lines.append(prefix + encoded)
        if v is not None:
            assert isinstance(v, str), f"{f}: non-str value {v!r}"
            close = offset + len(prefix) + len(encoded) - 1
            spans.append((f, close, close + 1))
        offset += len(lines[-1]) + 1
    text = "\n".join(lines)
    assert text == render_query(row, dataset), "render_row diverged from prompts.render_query"
    for f, a, b in spans:
        assert text[a:b] == '"', f"{f}: span does not cover a closing quote"
    return {"text": text, "spans": spans}


# ----------------------------------------------------------------------
# Item building


def _stratum(row: dict) -> str:
    return "valid" if row["valid"] else row["error_type"]


def select_rows(rows: list[dict], dataset: str, split: str, rows_per_stratum: dict, seed: int) -> list[dict]:
    """Stratified draw: ``rows_per_stratum[stratum]`` rows (null = all) per stratum.

    Strata are ``"valid"`` plus each error type present in ``rows``; the config
    keys must equal that set exactly. Returned in a stable order
    ``(document_id, line_index, stratum, k)``.

    Raises:
        ValueError: key mismatch, a non-positive count, or a stratum smaller
            than its requested count.
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
        chosen.extend(random.Random(_sub_seed(seed, dataset, split, s)).sample(pool, n))
    return sorted(chosen, key=lambda r: (r["document_id"], r.get("line_index", -1), _stratum(r), r["k"]))


def build_items(params: dict, seed: int, repo_root: Path, tokenizer=None) -> list[dict]:
    """One item per selected row, for ``EntityRowLM.collect_rows``.

    ``params``: ``data_root``, ``dataset``, ``split``, ``error_types``,
    ``rows_per_stratum``, ``context_token_limit`` (``tokenizer`` required iff
    not null).
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


class EntityRowLM(RepresentationLM):
    """``RepresentationLM`` reading the hidden state at each entity's closing quote."""

    def judge_row(self, instructions: str, context: str, query: str, spans: list[dict]) -> dict:
        """One prefill pass; hidden states at the token covering each closing quote.

        Returns ``representations`` (``{layer: float32 [n_spans, hidden]}``),
        ``token_index`` / ``token_id`` / ``token_n_chars`` / ``exact_end_alignment``
        (``[n_spans]``) and ``prompt_n_tokens``. ``exact_end_alignment`` is true
        when the token is the bare quote; otherwise byte-level BPE merged the
        quote with a neighbouring character (``"\\n``, ``",`` ...).
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
        """Identity of everything that determines an item's stored rows."""
        h = hashlib.sha256()
        for part in (self.model_name, it["instructions"], it["context"], it["query"]):
            h.update(part.encode())
            h.update(b"\x00")
        return h.hexdigest()

    def collect_rows(self, items: list[dict], cache_dir: str | Path, verify_read_point: bool) -> dict:
        """Judge every item; one ``.npz`` shard per document (resumable).

        A shard is reused only if its per-item fingerprints (model, instructions,
        context, query) and span count match exactly.

        Returns parallel arrays, one row per span, ordered by ``(sorted
        document_id, row_ordinal, field_position)``: ``representations``
        (``{layer: float32 [n, hidden]}``), ``layers``, ``doc_ids``, the model
        columns and every key of ``SPAN_META_KEYS``.
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
            payload[k] = _to_npz_array(v)
        for L in self.layers:
            payload[f"rep_{L}"] = np.concatenate(per_layer[L], axis=0).astype(np.float32)
        for k in SPAN_META_KEYS:
            payload[f"meta_{k}"] = _to_npz_array(meta[k])

        tmp = shard.with_name(shard.stem + ".tmp.npz")
        np.savez(tmp, **payload)
        tmp.rename(shard)

    def _aggregate_rows(self, by_doc: dict, cache_dir: Path) -> dict:
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
    key = np.char.add(
        np.char.add(result["doc_ids"], "|"), result["row_ordinal"].astype(str)
    )
    uniq, inverse = np.unique(key, return_inverse=True)
    return uniq, inverse.reshape(-1)


def check_row_invariants(result: dict) -> None:
    """Per-prompt label consistency, checked on the aggregated arrays.

    Within every ``(document, row_ordinal)`` group: ``row_k`` and
    ``row_error_type`` are constant; the number of invalid spans equals
    ``row_k``; a valid row has no invalid span and vice versa; each invalid
    span's ``field_error_type`` equals the row's; valid spans have an empty one;
    ``field_position`` is strictly increasing from 1 (rows are contiguous).
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
    """Flat scalar counts and composition summaries (no judge verdict is read)."""
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
