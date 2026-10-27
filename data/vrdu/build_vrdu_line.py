"""Build the VRDU line dataset: full-length line items as minimal pairs, and every subset of them.

Reads ``line/{train,test}_base.json`` and writes, per split, ``line/<split>.json`` (full-length samples),
``line/<split>_subsets.json`` (their subsets), and ``line_build.json`` (provenance).

Full-length sample: one line item shown with ALL its non-null fields (2 to 5 of them; ``m`` is the item's own
count), always in ``LINE_FIELDS`` order. Every invalid sample has one clean **twin**: the same item, the same
fields, the original values. A sample is a dataset row (the usual label keys, one key per line field, null if
not shown) plus ``fields``, ``m``, ``cell``, ``pair_id``, ``sample_id``. An item occurs in one pair per split.

Cells (all of size ``n_per_cell`` invalid samples + their twins; ``m`` varies within a cell):

  intra_document  k, 1 <= k <= max_k: k shown fields each take the same field's value from a distinct
                  other line item of the document.
  cross_item      k = 2: two shown fields swap values.

Subsets: every subset of a full-length sample with ``subset_min_size`` to ``m`` fields (the full set
included), fields in ``LINE_FIELDS`` order. Its label is recomputed, never inherited: ``valid`` is
``matches_item`` of the shown values against every line item of the document. It records its parent
(``parent_id``, ``parent_index``), the parent's corrupted fields it still shows (``invalid_fields``), the
parent's ``error_type`` if any remain, and ``size``. ``error_type`` and ``valid`` can disagree: a kept
corrupted value may still be a real item's (see ``stats["subsets"]``).

Rules: a corruption changes every value it touches (up to ``equivalence_key``); the corrupted sample
is no real line item's (``matches_item``, the label rule); ``from <= to`` holds whenever both shown
dates parse; intra date donors are drawn fragment-neutral (``pick_fragment_neutral``); nulls are
never shown or filled. Items are drawn round-robin over documents, cells most constrained first.

Local data preparation (no cluster job):

    uv run --python 3.12 python data/vrdu/build_vrdu_line.py \\
        --config data/vrdu/build-configs/2026-10-27-vrdu-invalids-07.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import build_vrdu_dataset as base_builder
import yaml
from vrdu_common import (field_kinds, fragment_rate, make_eq, matches_item, pair_ok, parse_date, pick_fragment_neutral,
                         read_base)

LINE_FIELDS = base_builder.LINE_FIELDS
SPLITS = ("train", "test")
INTRA_DOCUMENT, CROSS_ITEM = "intra_document", "cross_item"
CROSS_ITEM_K = 2  # a swap corrupts exactly two fields
BUILD_FILE = "line_build.json"
REQUIRED_LINE_KEYS = {"m_range", "max_k", "n_per_cell", "subset_min_size"}


# --- Cells ----------------------------------------------------------------------


def make_cells(m_range, max_k):
    """The (error_type, k) cells, most constrained first (items are used once, in the first cell that draws them).

    Args:
        m_range: ``[lo, hi]`` allowed full-length sizes (non-null fields of an item), inclusive.
        max_k: largest intra k.

    Returns:
        ``[(error_type, k)]``: intra by descending k, then cross_item.
    """
    lo, hi = m_range
    assert 2 <= lo <= hi <= len(LINE_FIELDS), m_range
    assert 1 <= max_k <= hi, max_k
    return [(INTRA_DOCUMENT, k) for k in range(max_k, 0, -1)] + [(CROSS_ITEM, CROSS_ITEM_K)]


def cell_id(cell):
    """``"<error_type>:k<k>"``."""
    et, k = cell
    return f"{et}:k{k}"


# --- Corruptions ----------------------------------------------------------------


def dates_in_order(values, date_fields, year_pivot):
    """Whether shown values keep ``from <= to`` (true unless both dates are shown and parse)."""
    a, b = date_fields
    if a not in values or b not in values:
        return True
    return pair_ok(parse_date(values[a], year_pivot), parse_date(values[b], year_pivot))


def intra_corruption(item, doc_items, fields, k, rng, eq, date_fields, year_pivot, frag_rate):
    """Corrupt ``k`` of ``fields`` with values of distinct other items of the document.

    Tries the k-subsets of ``fields`` in random order, filling each greedily (one donor per field,
    fields in random order); the first that passes every rule wins.

    Args:
        item: the clean line item.
        doc_items: every valid line item of its document (``item`` included).
        fields: the shown fields (a set; order is irrelevant here).
        k: fields to corrupt.
        rng: ``random.Random``.
        eq: from ``make_eq``.
        date_fields: ``(start, end)``.
        year_pivot: as in ``parse_date``.
        frag_rate: target OCR-fragment rate of donor dates.

    Returns:
        ``(values, sources)``: ``{field: value}`` over ``fields`` and ``{field: donor line_index}`` over
        the corrupted fields; or None if no k-subset works.
    """
    others = [d for d in doc_items if d["line_index"] != item["line_index"]]
    assert len(others) == len(doc_items) - 1, (item["document_id"], item["line_index"])
    subsets = list(itertools.combinations(sorted(fields, key=LINE_FIELDS.index), k))
    rng.shuffle(subsets)
    for chosen in subsets:
        values = {f: item[f] for f in fields}
        sources, used = {}, set()
        for f in rng.sample(chosen, len(chosen)):
            eligible = [d for d in others
                        if d["line_index"] not in used and d[f] is not None and eq(f, d[f]) != eq(f, item[f])
                        and dates_in_order({**values, f: d[f]}, date_fields, year_pivot)]
            if not eligible:
                break
            if f in date_fields:
                donor = pick_fragment_neutral(eligible, frag_rate, lambda d: parse_date(d[f], year_pivot) is None, rng)
            else:
                donor = eligible[rng.randrange(len(eligible))]
            values[f] = donor[f]
            sources[f] = donor["line_index"]
            used.add(donor["line_index"])
        else:
            if not matches_item(values, doc_items, eq):
                return values, sources
    return None


def cross_item_corruption(item, doc_items, fields, rng, eq, date_fields, year_pivot):
    """Swap the values of two of ``fields``.

    Tries the pairs in random order; the first whose swap changes both values, keeps date order and is
    no real item's wins.

    Args:
        item: the clean line item.
        doc_items: every valid line item of its document.
        fields: the shown fields.
        rng: ``random.Random``.
        eq: from ``make_eq``.
        date_fields: ``(start, end)``.
        year_pivot: as in ``parse_date``.

    Returns:
        ``(values, (a, b))``, or None if no pair works.
    """
    pairs = list(itertools.combinations(sorted(fields, key=LINE_FIELDS.index), 2))
    rng.shuffle(pairs)
    for a, b in pairs:
        if eq(a, item[b]) == eq(a, item[a]) or eq(b, item[a]) == eq(b, item[b]):
            continue
        values = {**{f: item[f] for f in fields}, a: item[b], b: item[a]}
        if dates_in_order(values, date_fields, year_pivot) and not matches_item(values, doc_items, eq):
            return values, (a, b)
    return None


def line_source(field, line_index, exchanged_with):
    """A ``field_errors`` source: the value is ``field`` of line item ``line_index`` of the same document.

    ``exchanged_with`` is the swap partner of a cross_item field (None for intra_document, whose source
    has no such key). ``null_fill`` is always False: nulls are never shown or filled.
    """
    src = {"scope": "same_document", "null_fill": False, "dataset": "line", "field": field, "line_index": line_index}
    if exchanged_with is not None:
        src["exchanged_with"] = exchanged_with
    return src


def make_candidate(item, doc_items, cell, m_range, used, rng, eq, date_fields, year_pivot, frag_rate):
    """One invalid sample of ``cell`` from ``item``, shown with all its non-null fields.

    Args:
        item: the clean line item.
        doc_items: every valid line item of its document.
        cell: ``(error_type, k)``.
        m_range: ``[lo, hi]`` allowed full-length sizes.
        used: ``{(document_id, line_index)}`` already in the split.
        rng, eq, date_fields, year_pivot, frag_rate: as in ``intra_corruption``.

    Returns:
        ``{item, fields, values, field_errors}``, or None if the item cannot supply the cell.
    """
    et, k = cell
    fields = [f for f in LINE_FIELDS if item[f] is not None]
    if (item["document_id"], item["line_index"]) in used or not m_range[0] <= len(fields) <= m_range[1]:
        return None
    if et == INTRA_DOCUMENT:
        out = intra_corruption(item, doc_items, fields, k, rng, eq, date_fields, year_pivot, frag_rate)
        if out is None:
            return None
        values, donors = out
        errors = {f: {"error_type": et, "source": line_source(f, li, None)} for f, li in donors.items()}
    else:
        out = cross_item_corruption(item, doc_items, fields, rng, eq, date_fields, year_pivot)
        if out is None:
            return None
        values, (a, b) = out
        errors = {a: {"error_type": et, "source": line_source(b, item["line_index"], b)},
                  b: {"error_type": et, "source": line_source(a, item["line_index"], a)}}
    return {"item": item, "fields": fields, "values": {f: values[f] for f in fields},
            "field_errors": {f: errors.get(f) for f in fields}}


# --- Drawing --------------------------------------------------------------------


def draw_round_robin(by_doc, n, rng):
    """Draw ``n`` candidates, one per document per pass, documents in a seeded order.

    Args:
        by_doc: ``{document_id: [candidates]}`` (non-empty lists).
        n: how many to draw.
        rng: ``random.Random``.

    Returns:
        The drawn candidates.

    Raises:
        ValueError: fewer than ``n`` candidates.
    """
    total = sum(len(v) for v in by_doc.values())
    if total < n:
        raise ValueError(f"only {total} candidates in {len(by_doc)} documents, {n} requested")
    docs = sorted(by_doc)
    rng.shuffle(docs)
    queues = {d: rng.sample(by_doc[d], len(by_doc[d])) for d in docs}
    out = []
    while len(out) < n:
        for d in docs:
            if queues[d] and len(out) < n:
                out.append(queues[d].pop())
    return out


def make_pair(cand, cell):
    """The invalid sample and its clean twin, as dataset rows.

    A row has one key per line field (null when not shown), the standard label keys, and ``fields``:
    the shown fields in the order shown.

    Args:
        cand: from ``make_candidate``.
        cell: ``(error_type, k)``.

    Returns:
        ``[invalid, valid]``.
    """
    item, fields = cand["item"], cand["fields"]
    cid = cell_id(cell)
    pair_id = f"{item['document_id']}|{item['line_index']}|{cid}|{'+'.join(fields)}"
    common = {"pair_id": pair_id, "document_id": item["document_id"], "line_index": item["line_index"],
              "cell": cid, "m": len(fields), "fields": fields}
    invalid_fields = [f for f in fields if cand["field_errors"][f] is not None]
    bad = {"sample_id": f"{pair_id}|invalid", **common, "valid": False, "error_type": cell[0], "k": len(invalid_fields),
           "num_invalid_fields": len(invalid_fields), "invalid_fields": invalid_fields,
           "field_errors": {f: cand["field_errors"].get(f) for f in LINE_FIELDS},
           **{f: cand["values"].get(f) for f in LINE_FIELDS}}
    good = {"sample_id": f"{pair_id}|valid", **common, "valid": True, "error_type": None, "k": 0, "num_invalid_fields": 0,
            "invalid_fields": [], "field_errors": {f: None for f in LINE_FIELDS},
            **{f: item[f] if f in fields else None for f in LINE_FIELDS}}
    return [bad, good]


def shown_values(row):
    """``{field: value}`` of a sample row, in the order shown."""
    return {f: row[f] for f in row["fields"]}


def build_split(base_rows, line_params, split, seed, eq, date_fields, year_pivot):
    """Every sample of one split.

    Args:
        base_rows: the split's valid line items.
        line_params: ``params.line``.
        split: ``"train"`` or ``"test"``.
        seed: the config seed.
        eq, date_fields, year_pivot: as in ``intra_corruption``.

    Returns:
        ``(samples, stats)``: samples cell by cell (invalid then twin), and per-cell counts.
    """
    by_doc = {}
    for r in base_rows:
        by_doc.setdefault(r["document_id"], []).append(r)
    frag_rate = fragment_rate(base_rows, date_fields, year_pivot)
    n = line_params["n_per_cell"][split]
    used, samples, stats = set(), [], {"fragment_target_rate": frag_rate, "cells": {}}
    for cell in make_cells(line_params["m_range"], line_params["max_k"]):
        rng = random.Random(f"{seed}:{split}:{cell_id(cell)}")
        cands = {}
        for doc in sorted(by_doc):
            for item in sorted(by_doc[doc], key=lambda r: r["line_index"]):
                c = make_candidate(item, by_doc[doc], cell, line_params["m_range"], used, rng, eq, date_fields, year_pivot, frag_rate)
                if c is not None:
                    cands.setdefault(doc, []).append(c)
        try:
            drawn = draw_round_robin(cands, n, rng)
        except ValueError as e:
            raise ValueError(f"{split} {cell_id(cell)}: {e}") from None
        for c in drawn:
            used.add((c["item"]["document_id"], c["item"]["line_index"]))
            samples += make_pair(c, cell)
        docs = Counter(c["item"]["document_id"] for c in drawn)
        stats["cells"][cell_id(cell)] = {
            "n_invalid": len(drawn), "n_docs": len(docs), "effective_docs": round(n * n / sum(v * v for v in docs.values()), 1),
            "eligible_items": sum(len(v) for v in cands.values()), "eligible_docs": len(cands),
            "m_counts": dict(sorted(Counter(len(c["fields"]) for c in drawn).items())),
        }
    return samples, stats


# --- Verification ----------------------------------------------------------------


def verify_split(samples, base_rows, cells, m_range, n_per_cell, eq, date_fields, year_pivot):
    """Re-check every sample of a split against its item; raises on any violation.

    Args:
        samples: from ``build_split``.
        base_rows: the split's valid line items.
        cells: from ``make_cells``.
        m_range: ``[lo, hi]`` allowed full-length sizes.
        n_per_cell: invalid samples per cell.
        eq, date_fields, year_pivot: as in ``intra_corruption``.
    """
    items = {(r["document_id"], r["line_index"]): r for r in base_rows}
    by_doc = {}
    for r in base_rows:
        by_doc.setdefault(r["document_id"], []).append(r)
    cell_of = {cell_id(c): c for c in cells}
    assert len({s["sample_id"] for s in samples}) == len(samples), "duplicate sample_id"
    pairs = {}
    for s in samples:
        pairs.setdefault(s["pair_id"], []).append(s)
    shown = Counter()
    for pid, (bad, good) in ((p, sorted(v, key=lambda s: s["valid"])) for p, v in pairs.items()):
        assert not bad["valid"] and good["valid"], (pid, "a pair is not one invalid + one valid sample")
        for key in ("document_id", "line_index", "cell", "m", "fields"):
            assert bad[key] == good[key], (pid, key)
        item = items[(good["document_id"], good["line_index"])]
        fields, (et, k) = good["fields"], cell_of[good["cell"]]
        assert fields == [f for f in LINE_FIELDS if item[f] is not None], (pid, "not the item's non-null fields in LINE_FIELDS order")
        assert len(fields) == good["m"] and m_range[0] <= good["m"] <= m_range[1], (pid, fields)
        shown[(good["document_id"], good["line_index"])] += 1
        doc_items = by_doc[good["document_id"]]
        for s in (bad, good):
            assert list(s["field_errors"]) == LINE_FIELDS, (s["sample_id"], "field_errors keys")
            assert all(isinstance(s[f], str) for f in fields), (s["sample_id"], "a null value is shown")
            assert all(s[f] is None and s["field_errors"][f] is None for f in LINE_FIELDS if f not in fields), (
                s["sample_id"], "a field that is not shown has a value or an error")
            assert dates_in_order(shown_values(s), date_fields, year_pivot), (s["sample_id"], "from > to")
        assert shown_values(good) == {f: item[f] for f in fields} and good["k"] == good["num_invalid_fields"] == 0, pid
        assert good["error_type"] is None and good["invalid_fields"] == [], pid
        assert all(e is None for e in good["field_errors"].values()), pid
        assert matches_item(shown_values(good), doc_items, eq), (pid, "twin is no real item")

        changed = [f for f in fields if bad["field_errors"][f] is not None]
        assert bad["error_type"] == et and bad["k"] == bad["num_invalid_fields"] == k == len(changed), (pid, k, changed)
        assert bad["invalid_fields"] == changed, (pid, "invalid_fields not the corrupted fields in shown order")
        for f in fields:
            same = eq(f, bad[f]) == eq(f, item[f])
            assert same == (f not in changed), (pid, f, "changed iff marked")
        for f in changed:
            e = bad["field_errors"][f]
            assert e["error_type"] == et and e["source"]["scope"] == "same_document" and e["source"]["dataset"] == "line", (pid, f)
            assert e["source"]["null_fill"] is False, (pid, f)
        if et == INTRA_DOCUMENT:
            donors = [bad["field_errors"][f]["source"]["line_index"] for f in changed]
            assert len(set(donors)) == len(donors) and item["line_index"] not in donors, (pid, "donors not distinct others")
            for f in changed:
                src = bad["field_errors"][f]["source"]
                assert "exchanged_with" not in src, (pid, f)
                assert src["field"] == f and items[(item["document_id"], src["line_index"])][f] == bad[f], (pid, f)
        else:
            a, b = changed
            assert bad[a] == item[b] and bad[b] == item[a], (pid, "not a swap")
            for x, y in ((a, b), (b, a)):
                src = bad["field_errors"][x]["source"]
                assert src["field"] == src["exchanged_with"] == y and src["line_index"] == item["line_index"], (pid, x)
        assert not matches_item(shown_values(bad), doc_items, eq), (pid, "invalid sample is a real item")
    assert max(shown.values()) == 1, "an item occurs in two pairs"
    per_cell = Counter(s["cell"] for s in samples if not s["valid"])
    assert per_cell == Counter({cid: n_per_cell for cid in cell_of}), per_cell


def fragment_stats(samples, date_fields, year_pivot):
    """Paired OCR-fragment counts of shown dates, by cell: invalid sample vs its twin at the same positions.

    A date position that received a non-date value (a cross_item swap) is a type error, not a fragment;
    it is skipped on both sides (``non_date_positions``).
    """
    twins = {s["pair_id"]: s for s in samples if s["valid"]}
    out = {}
    for s in samples:
        if s["valid"]:
            continue
        cell, twin = out.setdefault(s["cell"], Counter()), twins[s["pair_id"]]
        for f in date_fields:
            if f not in s["fields"]:
                continue
            err = s["field_errors"][f]
            if err is not None and err["source"]["field"] not in date_fields:
                cell["non_date_positions"] += 1
                continue
            cell["positions"] += 1
            cell["invalid_fragment"] += parse_date(s[f], year_pivot) is None
            cell["twin_fragment"] += parse_date(twin[f], year_pivot) is None
    return {c: dict(sorted(v.items())) for c, v in sorted(out.items())}


def cross_item_stats(samples, kind_of):
    """Swapped field pairs and their kind mix (``same_kind`` iff both fields have the same kind), by cell."""
    out = {}
    for s in samples:
        if s["valid"] or s["error_type"] != CROSS_ITEM:
            continue
        a, b = sorted(s["invalid_fields"], key=LINE_FIELDS.index)
        mix = "same_kind" if kind_of.get(a) is not None and kind_of.get(a) == kind_of.get(b) else "cross_kind"
        cell = out.setdefault(s["cell"], {"pairs": Counter(), "kind_mix": Counter()})
        cell["pairs"][f"{a}~{b}"] += 1
        cell["kind_mix"][mix] += 1
    return {c: {k: dict(sorted(v.items())) for k, v in d.items()} for c, d in sorted(out.items())}


# --- Subsets ----------------------------------------------------------------------


def subset_rows(parent, parent_index, doc_items, min_size, eq):
    """Every subset of one full-length sample, with its recomputed label.

    Args:
        parent: a row of ``line/<split>.json`` (valid or invalid).
        parent_index: its position in that file.
        doc_items: every valid line item of its document.
        min_size: smallest subset size.
        eq: from ``make_eq``.

    Returns:
        One row per subset of ``min_size`` to ``m`` fields (fields in ``LINE_FIELDS`` order), the label
        keys as in ``parent`` (null if not shown), plus ``subset_id``, ``parent_id``, ``parent_index``, ``pair_id``,
        ``parent_valid``, ``parent_error_type``, ``cell``, ``size``, ``valid`` (``matches_item`` of the shown
        values), ``error_type`` (the parent's if a corrupted field is shown, else None), ``invalid_fields``,
        ``num_invalid_fields``.
    """
    fields = parent["fields"]
    assert fields == [f for f in LINE_FIELDS if f in fields], (parent["sample_id"], "parent fields not in LINE_FIELDS order")
    rows = []
    for size in range(min_size, len(fields) + 1):
        for subset in itertools.combinations(fields, size):
            invalid = [f for f in subset if f in parent["invalid_fields"]]
            rows.append({
                "subset_id": f"{parent['sample_id']}|{'+'.join(subset)}", "parent_id": parent["sample_id"],
                "parent_index": parent_index, "pair_id": parent["pair_id"], "document_id": parent["document_id"],
                "line_index": parent["line_index"], "cell": parent["cell"], "parent_valid": parent["valid"],
                "parent_error_type": parent["error_type"], "fields": list(subset), "size": size,
                "valid": matches_item({f: parent[f] for f in subset}, doc_items, eq),
                "error_type": parent["error_type"] if invalid else None, "invalid_fields": invalid,
                "num_invalid_fields": len(invalid), **{f: parent[f] if f in subset else None for f in LINE_FIELDS}})
    return rows


def build_subsets(samples, base_rows, min_size, eq):
    """The subsets of every sample of a split.

    Args:
        samples: from ``build_split``.
        base_rows: the split's valid line items.
        min_size: ``params.line.subset_min_size``.
        eq: from ``make_eq``.

    Returns:
        ``(rows, stats)``: subset rows parent by parent, and label counts (``valid`` vs the parent's error type
        still being shown) by size.
    """
    by_doc = {}
    for r in base_rows:
        by_doc.setdefault(r["document_id"], []).append(r)
    rows = []
    for i, parent in enumerate(samples):
        rows += subset_rows(parent, i, by_doc[parent["document_id"]], min_size, eq)
    tab = {}
    for r in rows:
        key = f"size{r['size']}|parent_{'valid' if r['parent_valid'] else 'invalid'}|corrupted_shown_{bool(r['invalid_fields'])}"
        tab.setdefault(key, Counter())["valid" if r["valid"] else "invalid"] += 1
    return rows, {"n_parents": len(samples), "n_subsets": len(rows),
                  "label_by_size_parent_corrupted_shown": {k: dict(sorted(v.items())) for k, v in sorted(tab.items())}}


def verify_subsets(rows, samples, base_rows, min_size, eq):
    """Re-check every subset against its parent; raises on any violation.

    Args:
        rows: from ``build_subsets``.
        samples: the split's full-length samples.
        base_rows: the split's valid line items.
        min_size: ``params.line.subset_min_size``.
        eq: from ``make_eq``.
    """
    by_doc = {}
    for r in base_rows:
        by_doc.setdefault(r["document_id"], []).append(r)
    assert len({r["subset_id"] for r in rows}) == len(rows), "duplicate subset_id"
    expected = sum(sum(math.comb(s["m"], size) for size in range(min_size, s["m"] + 1)) for s in samples)
    assert len(rows) == expected, (len(rows), expected)
    per_parent = Counter(r["parent_index"] for r in rows)
    assert set(per_parent) == set(range(len(samples))), "a parent has no subsets, or a subset has no parent"
    for r in rows:
        parent = samples[r["parent_index"]]
        sid = r["subset_id"]
        assert r["parent_id"] == parent["sample_id"] and r["pair_id"] == parent["pair_id"], sid
        assert r["document_id"] == parent["document_id"] and r["line_index"] == parent["line_index"], sid
        assert r["parent_valid"] == parent["valid"] and r["parent_error_type"] == parent["error_type"], sid
        fields = r["fields"]
        assert fields == [f for f in LINE_FIELDS if f in fields] and set(fields) <= set(parent["fields"]), (sid, fields)
        assert r["size"] == len(fields) and min_size <= r["size"] <= parent["m"], sid
        assert all(r[f] == parent[f] for f in fields) and all(r[f] is None for f in LINE_FIELDS if f not in fields), sid
        assert r["invalid_fields"] == [f for f in fields if f in parent["invalid_fields"]], sid
        assert r["num_invalid_fields"] == len(r["invalid_fields"]), sid
        assert r["error_type"] == (parent["error_type"] if r["invalid_fields"] else None), sid
        assert r["valid"] == matches_item({f: r[f] for f in fields}, by_doc[r["document_id"]], eq), (sid, "label")
        if not r["invalid_fields"]:
            assert r["valid"], (sid, "a subset with no corrupted field must be valid")
        if r["size"] == parent["m"]:
            assert r["valid"] == parent["valid"], (sid, "the full subset must carry its parent's label")


# --- Config / main ---------------------------------------------------------------


def load_config(path):
    """Read and validate the line part of the build config.

    Args:
        path: the YAML config.

    Returns:
        The parsed config.
    """
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("id", "seed", "params"):
        assert key in config, f"config missing top-level key {key!r}"
    p = config["params"]
    for key in ("data_root", "dates", "cross_type", "line"):
        assert key in p, f"config params need {key!r}"
    line = p["line"]
    assert set(line) == REQUIRED_LINE_KEYS, f"params.line keys {sorted(line)} != {sorted(REQUIRED_LINE_KEYS)}"
    assert set(line["n_per_cell"]) == set(SPLITS), line["n_per_cell"]
    assert all(isinstance(v, int) and v >= 1 for v in line["n_per_cell"].values()), line["n_per_cell"]
    assert len(p["dates"]["line_fields"]) == 2 and set(p["dates"]["line_fields"]) <= set(LINE_FIELDS), p["dates"]
    make_cells(line["m_range"], line["max_k"])
    assert isinstance(line["subset_min_size"], int) and 1 <= line["subset_min_size"] <= line["m_range"][0], line
    return config


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    """Build and write both splits, their subsets and ``line_build.json`` for the config given on the CLI.

    Args:
        argv: CLI arguments (None = ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="data/vrdu/build-configs/<id>.yaml")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    p = config["params"]
    data_root = Path(p["data_root"])
    assert data_root.is_dir(), f"no such directory: {data_root}"
    date_fields, year_pivot = tuple(p["dates"]["line_fields"]), p["dates"]["year_pivot"]
    kind_of = field_kinds(p["cross_type"]["kinds"])
    eq = make_eq(kind_of, year_pivot)
    cells = make_cells(p["line"]["m_range"], p["line"]["max_k"])

    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0],
           "script": "data/vrdu/build_vrdu_line.py", "config_id": config["id"], "config_snapshot": config,
           "cells": [cell_id(c) for c in cells], "splits": {}, "sha256": {}}
    out["git_sha"], out["git_dirty"] = base_builder.git_state()
    for split in SPLITS:
        base = read_base(data_root / "line" / f"{split}_base.json", LINE_FIELDS, ["document_id", "line_index"])
        samples, stats = build_split(base, p["line"], split, config["seed"], eq, date_fields, year_pivot)
        verify_split(samples, base, cells, p["line"]["m_range"], p["line"]["n_per_cell"][split], eq, date_fields, year_pivot)
        stats["date_fragments"] = fragment_stats(samples, date_fields, year_pivot)
        stats["cross_item"] = cross_item_stats(samples, kind_of)
        subsets, stats["subsets"] = build_subsets(samples, base, p["line"]["subset_min_size"], eq)
        verify_subsets(subsets, samples, base, p["line"]["subset_min_size"], eq)
        rel = f"line/{split}.json"
        (data_root / rel).write_text(json.dumps(samples, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        rel_sub = f"line/{split}_subsets.json"   # one row per line
        (data_root / rel_sub).write_text("[\n" + ",\n".join(json.dumps(r, ensure_ascii=False) for r in subsets) + "\n]\n", encoding="utf-8")
        out["splits"][split] = stats
        for r in (rel, rel_sub):
            out["sha256"][r] = sha256_of(data_root / r)
        tight = min(stats["cells"].items(), key=lambda kv: kv[1]["n_docs"])
        print(f"{rel}: {len(samples)} samples, {len(cells)} cells; fewest docs: {tight[0]} ({tight[1]['n_docs']}); "
              f"{rel_sub}: {len(subsets)} subsets")
    (data_root / BUILD_FILE).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"written to {data_root / BUILD_FILE}")


if __name__ == "__main__":
    main()
