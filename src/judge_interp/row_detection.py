"""Shared machinery of the row-based detection tasks: one prompt per dataset row, hidden
states read at each entity's closing quote.

``entity_detection`` and ``relation_detection`` differ only in their instruction text (and
that relation needs a dataset with ``supports_relation``); everything else lives here:

1. ``build_items`` loads the spec's split file, draws a stratified subset of rows
   (``select_rows``), renders each row (``render_row``) and attaches a per-entity label
   from the row's ``field_errors``. The instruction text is passed in by the task.
2. ``RowDetectionLM.collect_rows`` runs one prefill pass per row and stores, per layer, the
   hidden state at the token covering each closing quote, in resumable per-document shards.
3. ``check_row_invariants`` and ``compute_metrics`` check label bookkeeping and summarise
   span counts. No verdict is read: the model is never asked to answer.

Rendering: only non-null fields are shown (a null field has no line). Every value is a JSON
string followed by one pad space before its closing quote, and the query ends with a newline.
The tokenizers (Qwen2.5, Llama-3.1) pre-tokenize punctuation plus following newlines as one unit,
so the pad, the closing quote and the newline become one token, ``READ_TOKEN`` (' "\\n'). The pad is
what keeps the value's last character out of it (without it, a value ending in ``)`` or ``.``
would be read at ``)"\\n`` etc.), so the read position carries no entity text. Every entity of every
prompt must be read at exactly this token; ``judge_row`` and ``compute_metrics`` assert it.

Labels are per entity: ``field_errors[field]`` is null for a correct entity and
``{"error_type", "source"}`` for a corrupted one. Null fields have no quote and no span.
Layers are whatever the config lists; no logits are materialised (``logits_to_keep=1``).
Field names, descriptions, error types and the file layout come from a ``DatasetSpec``
passed to every function.
"""
from __future__ import annotations

import gc
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from judge_interp.dataset_spec import DatasetSpec
from judge_interp.prompts import load_context, load_split, render_query, sub_seed
from judge_interp.representationlm import RepresentationLM, to_npz_array

_SOURCE_SCOPES = ("other_document", "same_document")
_PAD = " "  # inserted before each closing quote; see the module docstring
READ_TOKEN = _PAD + '"\n'  # the one token every entity is read at: pad, closing quote and newline merged


# Span metadata columns, in shard order. All values are str / int / bool.
SPAN_META_KEYS = (
    "dataset", "row_ordinal", "line_index", "row_error_type", "row_k", "field", "field_position",
    "value", "label_valid", "field_error_type", "source_scope", "source_dataset", "source_field",
    "source_line_index", "exchanged_with",
)
# Per-span columns the model side adds.
_MODEL_KEYS = ("token_index", "token_id", "token_n_chars", "prompt_n_tokens")


def row_index(row: dict, spec: DatasetSpec) -> int | None:
    """The row's index within its document, or ``None`` for a dataset with one row per document."""
    return None if spec.row_index_key is None else row[spec.row_index_key]


# ----------------------------------------------------------------------
# Rows: validation and labels


def check_row(row: dict, spec: DatasetSpec) -> None:
    """Assert that a row's label fields agree with each other.

    Args:
        row: a dataset row with ``valid``, ``error_type``, ``k``, ``num_invalid_fields``,
            ``invalid_fields``, ``field_errors`` and one key per field of ``spec``.
        spec: the dataset; selects the expected field list.

    Raises:
        AssertionError: the label fields disagree, a corrupted field has a null value,
            or a ``source`` entry is malformed for its scope.
    """
    fields = list(spec.fields)
    where = f"{row['document_id']!r}/{row_index(row, spec)}"
    fe = row["field_errors"]
    assert list(fe) == fields, f"{where}: field_errors keys {list(fe)} != {fields}"
    bad = [f for f in fields if fe[f] is not None]
    assert row["valid"] == (row["error_type"] is None), f"{where}: valid vs error_type"
    assert row["valid"] == (not bad), f"{where}: valid vs field_errors"
    assert row["k"] == row["num_invalid_fields"] == len(bad) == len(row["invalid_fields"]), where
    assert set(row["invalid_fields"]) == set(bad), f"{where}: invalid_fields vs field_errors"
    if not row["valid"]:
        assert row["error_type"] in spec.error_types, f"{where}: error_type {row['error_type']!r}"
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
        assert isinstance(src.get("null_fill"), bool), f"{where}/{f}: source null_fill {src.get('null_fill')!r}"
        if src["scope"] == "same_document":
            need = {"scope", "null_fill", "dataset", "field", "line_index"}
            if row["error_type"] == "cross_type":
                need.add("exchanged_with")
            assert set(src) == need, f"{where}/{f}: source keys {sorted(src)}"
            assert isinstance(src["dataset"], str) and isinstance(src["field"], str), f"{where}/{f}"
        else:
            assert set(src) == {"scope", "null_fill"}, f"{where}/{f}: source keys {sorted(src)}"


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


def render_row(row: dict, spec: DatasetSpec) -> dict:
    """Render a row as the query text and locate each entity's closing quote.

    The text equals ``prompts.render_query`` (non-null fields only) with one pad space
    before each value's closing quote and a trailing newline (asserted line by line).

    Args:
        row: a dataset row; every field of ``spec`` must be a ``str`` or ``None``.
        spec: the dataset.

    Returns:
        ``{"text": str, "spans": [(field, char_start, char_end), ...]}``: one span per
        non-null field in canonical order, covering exactly the closing ``"``
        (offsets into ``text``, exclusive end).
    """
    lines: list[str] = []
    spans: list[tuple[str, int, int]] = []
    offset = 0
    shown = [f for f in spec.fields if row[f] is not None]
    for f in shown:
        v = row[f]
        assert isinstance(v, str), f"{f}: non-str value {v!r}"
        prefix = f"{f}: "
        encoded = json.dumps(v)[:-1] + _PAD + '"'
        lines.append(prefix + encoded)
        close = offset + len(prefix) + len(encoded) - 1
        spans.append((f, close, close + 1))
        offset += len(lines[-1]) + 1
    assert lines, "row with no non-null field"
    text = "\n".join(lines) + "\n"

    base = render_query(row, spec).split("\n")
    assert len(base) == len(lines), "render_row diverged from prompts.render_query"
    for f, b, mine in zip(shown, base, lines):
        assert mine == b[:-1] + _PAD + '"', f"{f}: render_row diverged from prompts.render_query"
    for f, a, b in spans:
        assert text[a:b] == '"', f"{f}: span does not cover a closing quote"
    return {"text": text, "spans": spans}


# ----------------------------------------------------------------------
# Item building


def _stratum(row: dict) -> str:
    """Stratum name of a row: ``"valid"`` or its error type."""
    return "valid" if row["valid"] else row["error_type"]


def _sort_index(row: dict, spec: DatasetSpec) -> int:
    """Sort key for a row within its document (constant when a document has one row)."""
    idx = row_index(row, spec)
    return -1 if idx is None else idx


def select_rows(rows: list[dict], spec: DatasetSpec, split: str, rows_per_stratum: dict, seed: int) -> list[dict]:
    """Draw a stratified subset of rows.

    Strata are ``"valid"`` plus each error type present in ``rows``.

    Args:
        rows: rows from ``prompts.load_split``.
        spec: the dataset; its ``name`` seeds the per-stratum draw.
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
                      key=lambda r: (r["document_id"], _sort_index(r, spec), r["k"]))
        n = rows_per_stratum[s]
        if n is None:
            chosen.extend(pool)
            continue
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError(f"rows_per_stratum[{s!r}] must be null or an int >= 1, got {n!r}")
        if n > len(pool):
            raise ValueError(f"stratum {s!r}: {n} rows requested but only {len(pool)} available")
        chosen.extend(random.Random(sub_seed(seed, spec.name, split, s)).sample(pool, n))
    return sorted(chosen, key=lambda r: (r["document_id"], _sort_index(r, spec), _stratum(r), r["k"]))


def build_items(spec: DatasetSpec, params: dict, seed: int, repo_root: Path, tokenizer, instructions: str) -> list[dict]:
    """Build one ``RowDetectionLM.collect_rows`` item per selected row.

    Args:
        spec: the dataset.
        params: config ``params`` with ``data_root``, ``split``, ``error_types``,
            ``rows_per_stratum`` and ``context_token_limit``.
        seed: run seed for the stratified draw.
        repo_root: directory that ``params["data_root"]`` is relative to.
        tokenizer: HF tokenizer used to enforce ``context_token_limit``; must be given
            (not ``None``) if and only if the limit is not ``None``.
        instructions: the ``## INSTRUCTIONS:`` text, identical for every item.

    Returns:
        Items ordered by ``select_rows``: ``document_id``, ``row_ordinal`` (0-based
        within the document), ``instructions``, ``context``, ``query`` and ``spans``
        (one dict per non-null field with ``char_start``, ``char_end`` and every key
        of ``SPAN_META_KEYS``).

    Raises:
        ValueError: tokenizer/limit mismatch, or a context
            longer than ``context_token_limit`` tokens.
    """
    data_root = repo_root / params["data_root"]
    split = params["split"]
    limit = params["context_token_limit"]
    if (limit is None) != (tokenizer is None):
        raise ValueError("tokenizer must be given iff context_token_limit is not null")

    rows = load_split(data_root, spec, split, params["error_types"])
    for r in rows:
        check_row(r, spec)
    selected = select_rows(rows, spec, split, params["rows_per_stratum"], seed)

    contexts: dict[str, str] = {}
    items: list[dict] = []
    ordinal: dict[str, int] = {}
    for r in selected:
        doc_id = r["document_id"]
        if doc_id not in contexts:
            text = load_context(data_root, spec, doc_id)
            if limit is not None:
                n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
                if n_tokens > limit:
                    raise ValueError(
                        f"context for {doc_id!r} is {n_tokens} tokens > context_token_limit {limit}"
                    )
            contexts[doc_id] = text
        j = ordinal.get(doc_id, 0)
        ordinal[doc_id] = j + 1
        rendered = render_row(r, spec)
        spans = []
        for pos, (f, c0, c1) in enumerate(rendered["spans"], start=1):
            entry = r["field_errors"][f]
            spans.append({
                "char_start": c0,
                "char_end": c1,
                "dataset": spec.name,
                "row_ordinal": j,
                "line_index": -1 if row_index(r, spec) is None else row_index(r, spec),
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


class RowDetectionLM(RepresentationLM):
    """``RepresentationLM`` that reads hidden states at each entity's closing quote."""

    def judge_row(self, instructions: str, context: str, query: str, spans: list[dict]) -> dict:
        """Run one prefill pass and read the hidden state at each closing quote.

        Args:
            instructions: the ``## INSTRUCTIONS:`` text.
            context: the ``## CONTEXT:`` text.
            query: the ``## EXTRACTED RECORD:`` text; ``spans`` offsets are relative to it.
            spans: dicts with one-character ``char_start`` / ``char_end`` covering a
                closing quote in ``query``.

        Returns:
            Dict with ``representations`` (``{layer: float32 [n_spans, hidden]}``),
            ``token_index``, ``token_id`` and ``token_n_chars`` (each ``[n_spans]``), and
            ``prompt_n_tokens``. Every ``token_id`` is that of ``READ_TOKEN``.

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
        for span in spans:
            assert span["char_end"] - span["char_start"] == 1, "a span must be exactly one character"
            pos = shift + span["char_start"]
            idx = self._last_token_covering(offsets, pos)
            s, e = offsets[idx]
            assert s <= pos < e
            # The context may contain the same strings; the read must be in the query.
            assert shift <= s < shift + len(query), (
                f"{span['field']!r}: token {idx} starts outside the EXTRACTED RECORD block"
            )
            tid = int(input_ids[idx])
            assert self.tokenizer.decode([tid]) == READ_TOKEN, (
                f"{span['field']!r}: read token {idx} is {self.tokenizer.decode([tid])!r}, expected {READ_TOKEN!r}"
            )
            token_index.append(idx)
            token_id.append(tid)
            n_chars.append(e - s)
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
            for k in ("token_index", "token_id", "token_n_chars"):
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
                assert n == sum(len(it["spans"]) for it in by_doc[doc_id]), (doc_id, n)
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


def compute_metrics(result: dict, spec: DatasetSpec) -> dict:
    """Summarise span counts and prompt lengths of a run.

    Args:
        result: aggregated arrays from ``RowDetectionLM.collect_rows``.
        spec: the dataset; one ``n_spans_<error_type>`` count is emitted per ``spec.error_types``.

    Returns:
        Flat scalar dict: span, prompt, valid/invalid and per-error-type counts,
        ``n_rows_with_mixed_entities``, ``read_token_id`` and min/max prompt length in tokens.

    Raises:
        AssertionError: the entities were not all read at one token id (``READ_TOKEN``).
    """
    label_valid = result["label_valid"].astype(bool)
    read_ids = np.unique(result["token_id"])
    assert len(read_ids) == 1, f"entities were read at {len(read_ids)} different tokens: {read_ids.tolist()}"
    read_token_id = int(read_ids[0])
    uniq, inv = _prompt_groups(result)
    n_invalid = np.bincount(inv, weights=(~label_valid).astype(np.float64), minlength=len(uniq))
    n_spans_per = np.bincount(inv, minlength=len(uniq))
    out = {
        "n_spans": int(len(label_valid)),
        "n_prompts": int(len(uniq)),
        "n_valid_spans": int(label_valid.sum()),
        "n_invalid_spans": int((~label_valid).sum()),
        "n_rows_with_mixed_entities": int(((n_invalid > 0) & (n_invalid < n_spans_per)).sum()),
        "read_token_id": read_token_id,
        "min_prompt_n_tokens": int(result["prompt_n_tokens"].min()),
        "max_prompt_n_tokens": int(result["prompt_n_tokens"].max()),
    }
    for et in spec.error_types:
        out[f"n_spans_{et}"] = int((result["field_error_type"] == et).sum())
    assert sum(out[f"n_spans_{et}"] for et in spec.error_types) == out["n_invalid_spans"]
    return out



# ----------------------------------------------------------------------
# Run

ROW_PARAMS = (
    "data_root", "model", "dataset_config", "split", "layers", "dtype", "device",
    "context_token_limit", "verify_read_point", "error_types", "rows_per_stratum",
)


def write_span_representations(result: dict, path: Path) -> None:
    """Write a ``collect_rows`` result to a compressed ``.npz``.

    Args:
        result: ``collect_rows`` output; ``representations`` is stored as ``rep_<layer>``
            and every other entry under its own key.
        path: destination file.
    """
    payload = {f"rep_{L}": result["representations"][L] for L in result["layers"].tolist()}
    for k, v in result.items():
        if k == "representations":
            continue
        payload[k] = v
    np.savez_compressed(path, **payload)


def run_rows(cfg: dict, spec: DatasetSpec, run_dir: Path, repo_root: Path, build_items_fn, lm_class) -> dict:
    """Collect span representations for a row task and write ``artifacts/span_representations.npz``.

    Args:
        cfg: the config (``id``, ``seed`` and ``params`` as ``ROW_PARAMS``).
        spec: the dataset named by ``params["dataset_config"]``.
        run_dir: run output directory; per-document shards go under ``artifacts/cache``.
        repo_root: directory that ``params["data_root"]`` is relative to.
        build_items_fn: the task's ``build_items``.
        lm_class: the task's ``RowDetectionLM`` subclass.

    Returns:
        ``compute_metrics`` output.
    """
    from transformers import AutoTokenizer

    p, seed = cfg["params"], int(cfg["seed"])
    rlm = lm_class(model_name=p["model"], layers=p["layers"], device=p["device"], dtype=p["dtype"], verbose=True)
    limit = p["context_token_limit"]
    tokenizer = AutoTokenizer.from_pretrained(p["model"]) if limit is not None else None
    items = build_items_fn(spec, p, seed, repo_root, tokenizer)
    print(f"{cfg['id']}: {len(items)} row prompt(s), dataset={spec.name}, split={p['split']}")
    result = rlm.collect_rows(
        items, cache_dir=run_dir / "artifacts" / "cache", verify_read_point=p["verify_read_point"],
    )
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    write_span_representations(result, run_dir / "artifacts" / "span_representations.npz")
    return compute_metrics(result, spec)
