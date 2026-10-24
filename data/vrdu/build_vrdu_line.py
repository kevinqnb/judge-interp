"""Build the VRDU line dataset: random ordered field subsets of line items, as minimal pairs.

Reads ``line/{train,test}_base.json`` and writes ``line/{train,test}.json`` and ``line_build.json``
(provenance). A sample shows ``m`` of one line item's non-null fields, in a random order. Every
invalid sample has one clean **twin**: the same item, the same ordered fields, the original values.
So every cell, and every union of cells, has the same items, documents, field sets and orders in
both classes.

Cells (all of size ``n_per_cell`` invalid samples + their twins):

  intra_document  (m, k), 1 <= k <= min(m, max_k): k shown fields each take the same field's value
                  from a distinct other line item of the document.
  cross_item      (m, 2): two shown fields swap values.

Rules: a corruption changes every value it touches (up to ``equivalence_key``); the corrupted sample
is no real line item's (``matches_item``, the label rule); ``from <= to`` holds whenever both shown
dates parse; intra date donors are drawn fragment-neutral (``pick_fragment_neutral``); nulls are
never shown or filled. Items are drawn round-robin over documents; an (item, ordered fields) pair
occurs once per split.

Local data preparation (no cluster job):

    uv run --python 3.12 python data/vrdu/build_vrdu_line.py \\
        --config data/vrdu/build-configs/2026-10-24-vrdu-invalids-06.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
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
REQUIRED_LINE_KEYS = {"m_range", "max_k", "n_per_cell"}


# --- Cells ----------------------------------------------------------------------


def make_cells(m_range, max_k):
    """The (error_type, m, k) cells.

    Args:
        m_range: ``[lo, hi]`` subset sizes, inclusive.
        max_k: largest intra k.

    Returns:
        ``[(error_type, m, k)]``, intra first, each by m then k.
    """
    lo, hi = m_range
    assert 2 <= lo <= hi <= len(LINE_FIELDS), m_range
    assert 1 <= max_k <= hi, max_k
    intra = [(INTRA_DOCUMENT, m, k) for m in range(lo, hi + 1) for k in range(1, min(m, max_k) + 1)]
    return intra + [(CROSS_ITEM, m, CROSS_ITEM_K) for m in range(lo, hi + 1)]


def cell_id(cell):
    """``"<error_type>:m<m>:k<k>"``."""
    et, m, k = cell
    return f"{et}:m{m}:k{k}"


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


def make_candidate(item, doc_items, cell, used, rng, eq, date_fields, year_pivot, frag_rate):
    """One invalid sample of ``cell`` from ``item``, in an order not yet used for the item.

    Args:
        item: the clean line item.
        doc_items: every valid line item of its document.
        cell: ``(error_type, m, k)``.
        used: ``{(document_id, line_index, fields tuple)}`` already in the split.
        rng, eq, date_fields, year_pivot, frag_rate: as in ``intra_corruption``.

    Returns:
        ``{item, fields, values, field_errors}``, or None if the item cannot supply the cell.
    """
    et, m, k = cell
    present = [f for f in LINE_FIELDS if item[f] is not None]
    subsets = list(itertools.combinations(present, m))
    rng.shuffle(subsets)
    key = (item["document_id"], item["line_index"])
    for subset in subsets:
        orders = [o for o in itertools.permutations(subset) if (*key, o) not in used]
        if not orders:
            continue
        if et == INTRA_DOCUMENT:
            out = intra_corruption(item, doc_items, subset, k, rng, eq, date_fields, year_pivot, frag_rate)
            if out is None:
                continue
            values, donors = out
            errors = {f: {"error_type": et, "source": {"line_index": li, "field": f}} for f, li in donors.items()}
        else:
            out = cross_item_corruption(item, doc_items, subset, rng, eq, date_fields, year_pivot)
            if out is None:
                continue
            values, (a, b) = out
            errors = {a: {"error_type": et, "source": {"line_index": item["line_index"], "field": b}},
                      b: {"error_type": et, "source": {"line_index": item["line_index"], "field": a}}}
        order = list(orders[rng.randrange(len(orders))])
        return {"item": item, "fields": order, "values": {f: values[f] for f in order},
                "field_errors": {f: errors.get(f) for f in order}}
    return None


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
    """The invalid sample and its clean twin.

    Args:
        cand: from ``make_candidate``.
        cell: ``(error_type, m, k)``.

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
           "invalid_fields": invalid_fields, "values": cand["values"], "field_errors": cand["field_errors"]}
    good = {"sample_id": f"{pair_id}|valid", **common, "valid": True, "error_type": None, "k": 0, "invalid_fields": [],
            "values": {f: item[f] for f in fields}, "field_errors": {f: None for f in fields}}
    return [bad, good]


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
                c = make_candidate(item, by_doc[doc], cell, used, rng, eq, date_fields, year_pivot, frag_rate)
                if c is not None:
                    cands.setdefault(doc, []).append(c)
        try:
            drawn = draw_round_robin(cands, n, rng)
        except ValueError as e:
            raise ValueError(f"{split} {cell_id(cell)}: {e}") from None
        for c in drawn:
            used.add((c["item"]["document_id"], c["item"]["line_index"], tuple(c["fields"])))
            samples += make_pair(c, cell)
        docs = Counter(c["item"]["document_id"] for c in drawn)
        stats["cells"][cell_id(cell)] = {
            "n_invalid": len(drawn), "n_docs": len(docs), "effective_docs": round(n * n / sum(v * v for v in docs.values()), 1),
            "eligible_items": sum(len(v) for v in cands.values()), "eligible_docs": len(cands),
        }
    return samples, stats


# --- Verification ----------------------------------------------------------------


def verify_split(samples, base_rows, cells, n_per_cell, eq, date_fields, year_pivot):
    """Re-check every sample of a split against its item; raises on any violation.

    Args:
        samples: from ``build_split``.
        base_rows: the split's valid line items.
        cells: from ``make_cells``.
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
        fields, (et, m, k) = good["fields"], cell_of[good["cell"]]
        assert len(fields) == len(set(fields)) == m == good["m"] and set(fields) <= set(LINE_FIELDS), (pid, fields)
        shown[(good["document_id"], good["line_index"], tuple(fields))] += 1
        doc_items = by_doc[good["document_id"]]
        for s in (bad, good):
            assert list(s["values"]) == fields and list(s["field_errors"]) == fields, (s["sample_id"], "not in field order")
            assert all(isinstance(v, str) for v in s["values"].values()), (s["sample_id"], "a null value is shown")
            assert dates_in_order(s["values"], date_fields, year_pivot), (s["sample_id"], "from > to")
        assert good["values"] == {f: item[f] for f in fields} and good["k"] == 0 and good["error_type"] is None, pid
        assert all(e is None for e in good["field_errors"].values()) and good["invalid_fields"] == [], pid
        assert matches_item(good["values"], doc_items, eq), (pid, "twin is no real item")

        changed = [f for f in fields if bad["field_errors"][f] is not None]
        assert bad["error_type"] == et and bad["k"] == k == len(changed) and bad["invalid_fields"] == changed, (pid, k, changed)
        for f in fields:
            same = eq(f, bad["values"][f]) == eq(f, item[f])
            assert same == (f not in changed), (pid, f, "changed iff marked")
        if et == INTRA_DOCUMENT:
            donors = [bad["field_errors"][f]["source"]["line_index"] for f in changed]
            assert len(set(donors)) == len(donors) and item["line_index"] not in donors, (pid, "donors not distinct others")
            for f in changed:
                src = bad["field_errors"][f]["source"]
                assert src["field"] == f and items[(item["document_id"], src["line_index"])][f] == bad["values"][f], (pid, f)
        else:
            a, b = changed
            assert bad["values"][a] == item[b] and bad["values"][b] == item[a], (pid, "not a swap")
            assert bad["field_errors"][a]["source"]["field"] == b and bad["field_errors"][b]["source"]["field"] == a, pid
        assert not matches_item(bad["values"], doc_items, eq), (pid, "invalid sample is a real item")
    assert max(shown.values()) == 1, "an (item, ordered fields) pair occurs twice"
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
            if f not in s["values"]:
                continue
            err = s["field_errors"][f]
            if err is not None and err["source"]["field"] not in date_fields:
                cell["non_date_positions"] += 1
                continue
            cell["positions"] += 1
            cell["invalid_fragment"] += parse_date(s["values"][f], year_pivot) is None
            cell["twin_fragment"] += parse_date(twin["values"][f], year_pivot) is None
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
    return config


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    """Build and write both splits and ``line_build.json`` for the config given on the CLI.

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
        verify_split(samples, base, cells, p["line"]["n_per_cell"][split], eq, date_fields, year_pivot)
        stats["date_fragments"] = fragment_stats(samples, date_fields, year_pivot)
        stats["cross_item"] = cross_item_stats(samples, kind_of)
        rel = f"line/{split}.json"
        (data_root / rel).write_text(json.dumps(samples, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        out["splits"][split] = stats
        out["sha256"][rel] = sha256_of(data_root / rel)
        tight = min(stats["cells"].items(), key=lambda kv: kv[1]["n_docs"])
        print(f"{rel}: {len(samples)} samples, {len(cells)} cells; fewest docs: {tight[0]} ({tight[1]['n_docs']})")
    (data_root / BUILD_FILE).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"written to {data_root / BUILD_FILE}")


if __name__ == "__main__":
    main()
