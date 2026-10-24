"""Build the VRDU main dataset: one row per document, valid or with ``k`` corrupted fields.

Reads ``main/{train,test}_base.json`` (and ``line/*_base.json``, a source of cross_type values) and writes
``main/{train,test}.json`` and ``main_build.json`` (provenance). An invalid row differs from its document's
valid row in exactly ``k`` fields; its identity is ``(document_id, error_type, k)``. One row per achievable
``k`` in each type's range (never a requested ``k`` the row cannot reach). The valid row is the clean twin
of every invalid row of its document (``main_samples.main_entity_pairs``).

Error types:

  inter_document  the value comes from another document of the split.
  cross_type      a value of another same-kind field of the document (name / date / amount), main or line.

Rules: a corruption must change the value up to ``equivalence_key`` (a reformat is not an error); a null
may be filled (``params.main.corrupt_from_null``; recorded as ``source["null_fill"]``) but a value is never
nulled; ``from <= to`` holds whenever both dates parse; OCR-fragment dates are drawn at their natural rate.
Generation rules are those of ``build_vrdu_invalids.py`` (-05), moved here unchanged except that inter
rows are emitted only at achievable ``k`` (``k`` always equals the number of corrupted fields) and each
(split, error type) has its own RNG stream.

Local data preparation (no cluster job):

    uv run --python 3.12 python data/vrdu/build_vrdu_main.py \\
        --config data/vrdu/build-configs/2026-10-24-vrdu-invalids-06.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import build_vrdu_dataset as base_builder
import yaml
from vrdu_common import (LINE_FIELDS, MAIN_FIELDS, field_kinds, fragment_rate, make_eq, pair_ok, parse_date,
                         pick_fragment_neutral, read_base)

SPLITS = ("train", "test")
INTER_DOCUMENT, CROSS_TYPE = "inter_document", "cross_type"
ERROR_TYPES = {"main": (INTER_DOCUMENT, CROSS_TYPE)}
ID_KEYS = ["document_id"]
BUILD_FILE = "main_build.json"
REQUIRED_MAIN_KEYS = {"k_range", "pool_scope", "corrupt_from_null"}


# --- Pools ---------------------------------------------------------------------


def main_pools(base_rows):
    """Value pools for main inter_document corruption.

    Args:
        base_rows: the split's valid main rows.

    Returns:
        ``{field: [non-null values]}`` (OCR fragments included, repeats kept).
    """
    return {f: [r[f] for r in base_rows if r[f] is not None] for f in MAIN_FIELDS}


# --- Feasibility -----------------------------------------------------------


def _nondate_invalidatable(row, target_fields, date_fields, pools, eq):
    """Non-date fields of a row that some pool value can change.

    Args:
        row: the valid row.
        target_fields: fields that may be corrupted.
        date_fields: excluded here.
        pools: ``{field: [values]}``.
        eq: from ``make_eq``.

    Returns:
        The list of such fields.
    """
    return [
        f for f in target_fields
        if f not in date_fields and any(eq(f, v) != eq(f, row[f]) for v in pools[f])
    ]


def _date_candidates(row, field, pools, year_pivot, eq):
    """Candidate replacements for one date field.

    Args:
        row: the valid row.
        field: a date field.
        pools: ``{field: [values]}``.
        year_pivot: as in ``parse_date``.
        eq: from ``make_eq``.

    Returns:
        ``parse_date`` of every pool value not equivalent to ``row[field]`` (None = fragment).
    """
    original = eq(field, row[field])
    return [parse_date(v, year_pivot) for v in pools[field] if eq(field, v) != original]


def _date_feasibility(row, date_fields, pools, year_pivot, eq):
    """Which date corruptions keep ``from <= to``.

    Args:
        row: the valid row.
        date_fields: ``(start, end)``.
        pools: ``{field: [values]}``.
        year_pivot: as in ``parse_date``.
        eq: from ``make_eq``.

    Returns:
        ``(start_alone, end_alone, both)``: whether the start alone, the end alone, or both can change.
    """
    from_field, to_field = date_fields
    parsed_s = parse_date(row[from_field], year_pivot)
    parsed_e = parse_date(row[to_field], year_pivot)
    from_c = _date_candidates(row, from_field, pools, year_pivot, eq)
    to_c = _date_candidates(row, to_field, pools, year_pivot, eq)

    start_alone = any(pair_ok(p, parsed_e) for p in from_c)
    end_alone = any(pair_ok(parsed_s, q) for q in to_c)
    both = bool(from_c) and bool(to_c) and (
        None in from_c or None in to_c
        or min(p for p in from_c if p is not None) <= max(q for q in to_c if q is not None)
    )
    return start_alone, end_alone, both


def achievable_counts(n_nondate, date_feasibility):
    """Numbers of corrupted fields a row can realize.

    Args:
        n_nondate: how many non-date fields can change.
        date_feasibility: from ``_date_feasibility``.

    Returns:
        The set of achievable counts.
    """
    start_alone, end_alone, both = date_feasibility
    date_counts = {0}
    if start_alone or end_alone:
        date_counts.add(1)
    if both:
        date_counts.add(2)
    return {a + d for a in range(n_nondate + 1) for d in date_counts}


def achievable_targets(base_row, target_fields, date_fields, pools, year_pivot, k_range, eq):
    """The ``k`` values in ``k_range`` a row can realize.

    Args:
        base_row: the valid row.
        target_fields: fields that may be corrupted.
        date_fields: ``(start, end)``.
        pools: ``{field: [values]}``.
        year_pivot: as in ``parse_date``.
        k_range: inclusive ``(k_lo, k_hi)``.
        eq: from ``make_eq``.

    Returns:
        Sorted achievable ``k`` values.
    """
    k_lo, k_hi = k_range
    feasible_nondate = _nondate_invalidatable(base_row, target_fields, date_fields, pools, eq)
    date_feasibility = _date_feasibility(base_row, date_fields, pools, year_pivot, eq)
    achievable = achievable_counts(len(feasible_nondate), date_feasibility)
    return sorted(x for x in achievable if k_lo <= x <= k_hi)


# --- Corruption ------------------------------------------------------------


def sample_new(pool, original, rng, eq, field):
    """Draw a replacement value that differs from the original.

    Args:
        pool: candidate values.
        original: the current value.
        rng: ``random.Random``.
        eq: from ``make_eq``.
        field: the field (for ``eq``).

    Returns:
        A random pool value not equivalent to ``original``; asserts one exists.
    """
    original_key = eq(field, original)
    for value in rng.sample(pool, len(pool)):
        if eq(field, value) != original_key:
            return value
    raise AssertionError(f"pool of {len(pool)} values has none differing from {original!r}")


def _choose_fields(feasible_nondate, date_fields, date_feasibility, k, rng):
    """Choose which fields to corrupt for one (row, k).

    Args:
        feasible_nondate: changeable non-date fields.
        date_fields: ``(start, end)``.
        date_feasibility: from ``_date_feasibility``.
        k: requested count (capped to the largest achievable count <= k).
        rng: ``random.Random``.

    Returns:
        Sorted list of fields (empty if none can change).
    """
    from_field, to_field = date_fields
    start_alone, end_alone, both = date_feasibility
    achievable = achievable_counts(len(feasible_nondate), date_feasibility)
    j = max((x for x in achievable if x <= k), default=0)
    if j == 0:
        return []
    k = j

    date_options = [frozenset()]
    if start_alone:
        date_options.append(frozenset([from_field]))
    if end_alone:
        date_options.append(frozenset([to_field]))
    if both:
        date_options.append(frozenset([from_field, to_field]))
    usable = sorted(
        (option for option in date_options if 0 <= k - len(option) <= len(feasible_nondate)),
        key=sorted,
    )
    chosen_dates = usable[rng.randrange(len(usable))]
    chosen_nondate = rng.sample(sorted(feasible_nondate), k - len(chosen_dates))
    return sorted([*chosen_nondate, *chosen_dates])


def _corrupt_dates(new_row, chosen_dates, row, date_fields, pools, year_pivot, rng, eq, frag_rate):
    """Write new values into the chosen date field(s), keeping ``from <= to``.

    Args:
        new_row: the row being corrupted (modified in place).
        chosen_dates: the date field(s) to change.
        row: the valid row.
        date_fields: ``(start, end)``.
        pools: ``{field: [values]}``.
        year_pivot: as in ``parse_date``.
        rng: ``random.Random``.
        eq: from ``make_eq``.
        frag_rate: target fragment rate.
    """
    from_field, to_field = date_fields
    parsed_s = parse_date(row[from_field], year_pivot)
    parsed_e = parse_date(row[to_field], year_pivot)
    orig_s, orig_e = eq(from_field, row[from_field]), eq(to_field, row[to_field])

    def parsed(v):
        return parse_date(v, year_pivot)

    def is_fragment(v):
        return parsed(v) is None

    def pick(eligible):
        return pick_fragment_neutral(eligible, frag_rate, is_fragment, rng)

    if chosen_dates == frozenset([from_field]):
        new_row[from_field] = pick([
            v for v in pools[from_field]
            if eq(from_field, v) != orig_s and pair_ok(parsed(v), parsed_e)
        ])
    elif chosen_dates == frozenset([to_field]):
        new_row[to_field] = pick([
            w for w in pools[to_field]
            if eq(to_field, w) != orig_e and pair_ok(parsed_s, parsed(w))
        ])
    else:
        to_c = _date_candidates(row, to_field, pools, year_pivot, eq)
        to_free = None in to_c
        latest_to = max((q for q in to_c if q is not None), default=None)
        new_from = pick([
            v for v in pools[from_field]
            if eq(from_field, v) != orig_s
            and (parsed(v) is None or to_free or parsed(v) <= latest_to)
        ])
        parsed_from = parsed(new_from)
        new_to = pick([
            w for w in pools[to_field]
            if eq(to_field, w) != orig_e and pair_ok(parsed_from, parsed(w))
        ])
        new_row[from_field], new_row[to_field] = new_from, new_to


def make_invalid_row(base_row, k, target_fields, date_fields, pools, year_pivot, rng, eq, frag_rate):
    """Corrupt one valid row in (up to) ``k`` fields.

    Args:
        base_row: the valid row.
        k: requested number of corrupted fields.
        target_fields: fields that may be corrupted.
        date_fields: ``(start, end)``.
        pools: ``{field: [values]}``.
        year_pivot: as in ``parse_date``.
        rng: ``random.Random``.
        eq: from ``make_eq``.
        frag_rate: target fragment rate.

    Returns:
        ``(new_row, invalid_fields)``, or None if no field can change.
    """
    feasible_nondate = _nondate_invalidatable(base_row, target_fields, date_fields, pools, eq)
    date_feasibility = _date_feasibility(base_row, date_fields, pools, year_pivot, eq)
    chosen = _choose_fields(feasible_nondate, date_fields, date_feasibility, k, rng)
    if not chosen:
        return None

    new_row = dict(base_row)
    for field in chosen:
        if field not in date_fields:
            new_row[field] = sample_new(pools[field], base_row[field], rng, eq, field)
    chosen_dates = frozenset(f for f in chosen if f in date_fields)
    if chosen_dates:
        _corrupt_dates(new_row, chosen_dates, base_row, date_fields, pools, year_pivot, rng, eq, frag_rate)

    changed = sorted(f for f in base_row if new_row[f] != base_row[f])
    assert changed == chosen, f"changed {changed} != chosen {chosen} ({base_row.get('document_id')})"
    for f in chosen:
        assert eq(f, new_row[f]) != eq(f, base_row[f]), (
            f"{base_row.get('document_id')}: {f} {base_row[f]!r} -> {new_row[f]!r} is a reformat, not an error"
        )
    return new_row, chosen


# --- Row emission: main --------------------------------------------------


def build_field_errors(fields, error_type, sources):
    """The ``field_errors`` label of one row.

    Args:
        fields: the row's data fields.
        error_type: the row's error type (None if valid).
        sources: ``{corrupted field: source dict}``.

    Returns:
        ``{field: None | {"error_type", "source"}}`` for every field.
    """
    assert set(sources) <= set(fields), sorted(set(sources) - set(fields))
    return {
        f: (None if f not in sources else {"error_type": error_type, "source": sources[f]})
        for f in fields
    }


def tag_row(row, fields, id_keys, k, valid, invalid_fields, error_type, sources):
    """Assemble an output row: identity keys, label fields, then data fields.

    Args:
        row: the row's values.
        fields: data fields.
        id_keys: identity keys (``document_id`` [, ``line_index``]).
        k: number of corrupted fields.
        valid: whether the row is valid.
        invalid_fields: the corrupted fields.
        error_type: None if valid.
        sources: ``{corrupted field: source dict}``.

    Returns:
        The output row.
    """
    assert set(sources) == set(invalid_fields), (sorted(sources), sorted(invalid_fields))
    assert (error_type is None) == valid
    tagged = {key: row[key] for key in id_keys}
    tagged["valid"] = valid
    tagged["error_type"] = error_type
    tagged["k"] = k
    tagged["num_invalid_fields"] = len(invalid_fields)
    tagged["invalid_fields"] = list(invalid_fields)
    tagged["field_errors"] = build_field_errors(fields, error_type, sources)
    for field in fields:
        tagged[field] = row[field]
    return tagged


def generate_inter_rows(base_rows, k_range, pools, date_fields, year_pivot, rng, eq, frag_rate):
    """The valid rows and the inter_document rows of one split, one per achievable k.

    Args:
        base_rows: valid main rows.
        k_range: inclusive ``(k_lo, k_hi)``.
        pools: from ``main_pools``.
        date_fields: ``(start, end)``.
        year_pivot: as in ``parse_date``.
        rng: this split's inter_document ``random.Random``.
        eq: from ``make_eq``.
        frag_rate: target fragment rate.

    Returns:
        ``(rows, skipped, capped)``: valid + invalid output rows, rows with no achievable k, rows whose
        largest achievable k is below ``k_hi``.
    """
    k_lo, k_hi = k_range
    assert 1 <= k_lo <= k_hi <= len(MAIN_FIELDS), f"k_range {k_range} outside 1..{len(MAIN_FIELDS)}"

    out = [tag_row(row, MAIN_FIELDS, ID_KEYS, 0, True, [], None, {}) for row in base_rows]
    skipped, capped = [], []
    for base_row in base_rows:
        row_id = {key: base_row[key] for key in ID_KEYS}
        ks = achievable_targets(base_row, MAIN_FIELDS, date_fields, pools, year_pivot, k_range, eq)
        if not ks:
            skipped.append(row_id)
            continue
        if ks[-1] < k_hi:
            capped.append({**row_id, "reached": ks[-1]})
        for k in ks:
            result = make_invalid_row(base_row, k, MAIN_FIELDS, date_fields, pools, year_pivot, rng, eq, frag_rate)
            assert result is not None and len(result[1]) == k, "k came from achievable_targets"
            new_row, invalid_fields = result
            out.append(tag_row(new_row, MAIN_FIELDS, ID_KEYS, k, False, invalid_fields, INTER_DOCUMENT,
                               {f: {"scope": "other_document"} for f in invalid_fields}))
    return out, skipped, capped


def verify_output(out_rows, base_by_id, fields, target_fields, id_keys, k_hi, date_fields, year_pivot, eq):
    """Check one generated main split; raises AssertionError on any inconsistency.

    Args:
        out_rows: the split's output rows.
        base_by_id: ``{id tuple: valid row}``.
        fields, target_fields: all fields / fields that may be corrupted.
        id_keys: identity keys.
        k_hi: the largest allowed k.
        date_fields: ``(start, end)``.
        year_pivot: as in ``parse_date``.
        eq: from ``make_eq``.
    """
    valid_rows = [r for r in out_rows if r["valid"]]
    invalid_rows = [r for r in out_rows if not r["valid"]]
    carried = [f for f in fields if f not in target_fields]

    assert len(valid_rows) == len(base_by_id), (len(valid_rows), len(base_by_id))
    for row in valid_rows:
        key = tuple(row[k] for k in id_keys)
        assert row["k"] == 0 and row["num_invalid_fields"] == 0 and row["invalid_fields"] == []
        assert row["error_type"] is None
        assert {f: row[f] for f in fields} == {f: base_by_id[key][f] for f in fields}

    identity = [(*[row[k] for k in id_keys], row["error_type"], row["k"]) for row in out_rows]
    assert len(identity) == len(set(identity)), "output rows not uniquely keyed by (*id, error_type, k)"
    verify_field_errors(out_rows, fields, ERROR_TYPES["main"])

    for row in invalid_rows:
        key = tuple(row[k] for k in id_keys)
        assert key in base_by_id, f"invalid row for unknown id {key}"
        base = base_by_id[key]
        changed = sorted(f for f in fields if row[f] != base[f])
        assert changed == sorted(row["invalid_fields"]), (key, changed, row["invalid_fields"])
        assert set(changed) <= set(target_fields), (key, changed)
        assert 1 <= row["num_invalid_fields"] == len(changed) <= row["k"] <= k_hi
        for field in changed:
            assert row[field] is not None, f"{key}: field {field} corrupted to null"
            assert eq(field, row[field]) != eq(field, base[field]), f"{key}: {field} only reformatted"
        for field in carried:
            assert row[field] == base[field], f"{key}: carried field {field} changed"

    from_field, to_field = date_fields
    for row in invalid_rows:
        if from_field not in row["invalid_fields"] and to_field not in row["invalid_fields"]:
            continue
        start, end = row[from_field], row[to_field]
        if start is None or end is None:
            continue
        key = tuple(row[k] for k in id_keys)
        assert pair_ok(parse_date(start, year_pivot), parse_date(end, year_pivot)), (key, start, end)


def annotate_null_fill(rows, base_by_id, id_keys):
    """Record on every corrupted entity whether it filled a null (``source["null_fill"]``).

    Args:
        rows: output rows (modified in place).
        base_by_id: ``{id tuple: valid row}``.
        id_keys: identity keys.
    """
    for row in rows:
        if row["valid"]:
            continue
        base = base_by_id[tuple(row[k] for k in id_keys)]
        for f in row["invalid_fields"]:
            row["field_errors"][f]["source"]["null_fill"] = base[f] is None


def verify_field_errors(rows, fields, allowed_error_types):
    """Check that ``error_type``, ``field_errors`` and ``invalid_fields`` agree; raises AssertionError.

    Args:
        rows: output rows.
        fields: data fields.
        allowed_error_types: the dataset's error types.
    """
    for row in rows:
        assert list(row["field_errors"]) == list(fields), row["field_errors"].keys()
        wrong = [f for f in fields if row["field_errors"][f] is not None]
        assert sorted(wrong) == sorted(row["invalid_fields"]), (wrong, row["invalid_fields"])
        if row["valid"]:
            assert row["error_type"] is None and not wrong
            continue
        assert row["error_type"] in allowed_error_types, row["error_type"]
        for f in wrong:
            assert row["field_errors"][f]["error_type"] == row["error_type"], (f, row["field_errors"][f])
            assert isinstance(row["field_errors"][f]["source"], dict)
            assert isinstance(row["field_errors"][f]["source"].get("null_fill"), bool), (f, row["field_errors"][f])


# --- cross_type -----------------------------------------------------------------


def document_truth(main_base, line_base, eq):
    """The valid values of every field of every document.

    Args:
        main_base: valid main rows.
        line_base: valid line rows.
        eq: from ``make_eq``.

    Returns:
        ``{document_id: {field: set of equivalence keys}}`` over main and line fields.
    """
    truth = {r["document_id"]: {f: set() for f in [*MAIN_FIELDS, *LINE_FIELDS]} for r in main_base}
    for r in main_base:
        for f in MAIN_FIELDS:
            if r[f] is not None:
                truth[r["document_id"]][f].add(eq(f, r[f]))
    for r in line_base:
        for f in LINE_FIELDS:
            if r[f] is not None:
                truth[r["document_id"]][f].add(eq(f, r[f]))
    return truth


def cross_type_candidates(target, kinds, kind_of, main_row, line_rows, truth_doc, counts, eq, own_line_index):
    """Values that could be filed under ``target`` as a cross_type error.

    Values of the document's other same-kind fields, excluding values already valid for ``target`` and
    values from the row's own line item.

    Args:
        target: the field to corrupt.
        kinds: ``{kind: [fields]}``.
        kind_of: ``{field: kind}``.
        main_row: the document's valid main row.
        line_rows: the document's valid line rows.
        truth_doc: ``document_truth`` of the document.
        counts: Counter of considered / excluded candidates (updated).
        eq: from ``make_eq``.
        own_line_index: the corrupted line row's index (None for a main row).

    Returns:
        ``[(value, source dict)]``.
    """
    kind = kind_of[target]
    out = []
    for other in kinds[kind]:
        if other == target:
            continue
        if other in MAIN_FIELDS:
            supplied = [(main_row[other], "main", None)]
        else:
            supplied = [(r[other], "line", r["line_index"]) for r in line_rows if r["line_index"] != own_line_index]
        for value, dataset, line_index in supplied:
            if value is None:
                continue
            counts["considered"] += 1
            if eq(target, value) in truth_doc[target]:
                counts["excluded_valid_for_target"] += 1
                continue
            out.append((value, {"scope": "same_document", "dataset": dataset,
                                "field": other, "line_index": line_index}))
    return out


def _same_record(source, dataset, row):
    """Whether a cross_type source is the corrupted record itself.

    Args:
        source: a source dict.
        dataset: ``"main"`` or ``"line"``.
        row: the corrupted row.

    Returns:
        True if the source is the same main row or the same line item.
    """
    return source["dataset"] == dataset and (dataset == "main" or source["line_index"] == row["line_index"])


def mark_exchanges(sources, dataset, row):
    """Mark direct exchanges: two fields of one record that took each other's values.

    Args:
        sources: ``{field: source dict}`` (``exchanged_with`` set in place).
        dataset: ``"main"`` or ``"line"``.
        row: the corrupted row.
    """
    for f, src in sources.items():
        partner = src["field"]
        back = sources.get(partner)
        src["exchanged_with"] = (
            partner
            if _same_record(src, dataset, row) and back is not None
            and _same_record(back, dataset, row) and back["field"] == f
            else None
        )


def generate_cross_type_rows(dataset, base_rows, fields, id_keys, main_by_doc, line_by_doc, truth,
                             kinds, date_fields, year_pivot, k_range, rng, eq, frag_rate):
    """cross_type rows for one dataset/split, one per achievable k.

    Args:
        dataset: ``"main"`` or ``"line"``.
        base_rows: valid rows of that dataset.
        fields: data fields.
        id_keys: identity keys.
        main_by_doc: ``{document_id: valid main row}``.
        line_by_doc: ``{document_id: [valid line rows]}``.
        truth: ``document_truth``.
        kinds: ``{kind: [fields]}``.
        date_fields: ``(start, end)``.
        year_pivot: as in ``parse_date``.
        k_range: inclusive ``(k_lo, k_hi)``.
        rng: this dataset/split's own ``random.Random``.
        eq: from ``make_eq``.
        frag_rate: target fragment rate.

    Returns:
        ``(rows, skipped, capped, stats)``.
    """
    kind_of = field_kinds(kinds)
    targets = [f for f in fields if f in kind_of]
    assert set(date_fields) <= set(targets), "date fields must be in the date kind"
    k_lo, k_hi = k_range
    assert 1 <= k_lo <= k_hi <= len(targets), (k_range, len(targets))

    rows, skipped, capped = [], [], []
    stats = {"candidate_counts": Counter(), "pair_counts": Counter(),
             "n_exchanged_fields": 0, "n_rows_with_exchange": 0}
    for base in base_rows:
        doc = base["document_id"]
        row_id = {key: base[key] for key in id_keys}
        cands = {
            t: cross_type_candidates(t, kinds, kind_of, main_by_doc[doc], line_by_doc.get(doc, []),
                                     truth[doc], stats["candidate_counts"], eq,
                                     base["line_index"] if dataset == "line" else None)
            for t in targets
        }
        pools = {t: [v for v, _ in cands[t]] for t in targets}
        sources_by_value = {t: {} for t in targets}
        for t in targets:
            for v, src in cands[t]:
                sources_by_value[t].setdefault(v, []).append(src)

        ks = achievable_targets(base, targets, date_fields, pools, year_pivot, (k_lo, k_hi), eq)
        if not ks:
            skipped.append(row_id)
            continue
        if ks[-1] < k_hi:
            capped.append({**row_id, "reached": ks[-1]})
        for k in ks:
            result = make_invalid_row(base, k, targets, date_fields, pools, year_pivot, rng, eq, frag_rate)
            assert result is not None and len(result[1]) == k, "k came from achievable_targets"
            new_row, invalid_fields = result
            sources = {f: dict(rng.choice(sources_by_value[f][new_row[f]])) for f in invalid_fields}
            mark_exchanges(sources, dataset, base)
            n_ex = sum(src["exchanged_with"] is not None for src in sources.values())
            stats["n_exchanged_fields"] += n_ex
            stats["n_rows_with_exchange"] += n_ex > 0
            for f, src in sources.items():
                stats["pair_counts"][f"{f}<-{src['field']}"] += 1
            rows.append(tag_row(new_row, fields, id_keys, k, False, invalid_fields, CROSS_TYPE, sources))
    return rows, skipped, capped, stats


def verify_cross_type_rows(rows, dataset, base_by_id, id_keys, main_by_doc, line_by_doc, truth, kinds,
                           date_fields, year_pivot, eq):
    """Check cross_type rows: each value is a real same-kind value of the document, not valid for its
    field, and not from the row's own line item; raises AssertionError.

    Args:
        rows: cross_type rows.
        dataset: ``"main"`` or ``"line"``.
        base_by_id: ``{id tuple: valid row}``.
        id_keys: identity keys.
        main_by_doc: ``{document_id: valid main row}``.
        line_by_doc: ``{document_id: [valid line rows]}``.
        truth: ``document_truth``.
        kinds: ``{kind: [fields]}``.
        date_fields: ``(start, end)``.
        year_pivot: as in ``parse_date``.
        eq: from ``make_eq``.
    """
    kind_of = field_kinds(kinds)
    for row in rows:
        assert row["error_type"] == CROSS_TYPE
        key = tuple(row[k] for k in id_keys)
        doc = row["document_id"]
        base = base_by_id[key]
        for f in row["invalid_fields"]:
            src = row["field_errors"][f]["source"]
            assert src["scope"] == "same_document" and src["field"] != f, (key, f, src)
            assert kind_of[f] == kind_of[src["field"]], (key, f, src)
            if src["dataset"] == "main":
                assert src["line_index"] is None and src["field"] in MAIN_FIELDS
                taken = main_by_doc[doc][src["field"]]
            else:
                assert src["field"] in LINE_FIELDS
                assert dataset != "line" or src["line_index"] != row["line_index"], (
                    f"{key}: {f} took a value from its own line item (that is cross_item, not cross_type)")
                donor = next(r for r in line_by_doc[doc] if r["line_index"] == src["line_index"])
                taken = donor[src["field"]]
            assert taken is not None and row[f] == taken, (key, f, src)
            assert eq(f, row[f]) not in truth[doc][f], f"{key}: {f} filed a value that is valid for it"
            ex = src["exchanged_with"]
            if ex is not None:
                back = row["field_errors"][ex]["source"]
                assert back["field"] == f and back["exchanged_with"] == f, (key, f, ex)
                assert row[f] == base[ex] and row[ex] == base[f], (key, f, ex)


# --- Diagnostics ------------------------------------------------------------


def date_fragment_stats(rows, date_fields, year_pivot):
    """Fragment rates per row type, to check that fragments do not predict corruption.

    Args:
        rows: output rows.
        date_fields: the date fields.
        year_pivot: as in ``parse_date``.

    Returns:
        ``{error type or "valid": {corrupted, corrupted_fragment, kept, kept_fragment}}``.
    """
    stats = {}
    for row in rows:
        cell = stats.setdefault(row["error_type"] or "valid", Counter())
        for f in date_fields:
            if row[f] is None:
                continue
            side = "corrupted" if row["field_errors"][f] is not None else "kept"
            cell[side] += 1
            cell[side + "_fragment"] += parse_date(row[f], year_pivot) is None
    return {et: {key: cell[key] for key in ("corrupted", "corrupted_fragment", "kept", "kept_fragment")}
            for et, cell in sorted(stats.items())}


def stratum_stats(rows):
    """Rows and null-fill entities per (error type, k).

    Args:
        rows: one split's output rows.

    Returns:
        ``{error_type: {k: {n_rows, n_entities, n_null_fill}}}`` (one row per document, so ``n_rows`` is also
        the document count).
    """
    out = {}
    for r in rows:
        if r["valid"]:
            continue
        cell = out.setdefault(r["error_type"], {}).setdefault(r["k"], Counter())
        cell["n_rows"] += 1
        cell["n_entities"] += len(r["invalid_fields"])
        cell["n_null_fill"] += sum(r["field_errors"][f]["source"]["null_fill"] for f in r["invalid_fields"])
    return {et: {k: dict(v) for k, v in sorted(ks.items())} for et, ks in sorted(out.items())}


# --- Config / main ---------------------------------------------------------------


def load_config(path):
    """Read and validate the main part of the build config.

    Args:
        path: the YAML config.

    Returns:
        The parsed config.
    """
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("id", "seed", "params"):
        assert key in config, f"config missing top-level key {key!r}"
    p = config["params"]
    for key in ("data_root", "dates", "cross_type", "main"):
        assert key in p, f"config params need {key!r}"
    m = p["main"]
    assert set(m) == REQUIRED_MAIN_KEYS, f"params.main keys {sorted(m)} != {sorted(REQUIRED_MAIN_KEYS)}"
    assert set(m["k_range"]) == set(ERROR_TYPES["main"]), m["k_range"]
    assert m["pool_scope"] == "per_split", m["pool_scope"]
    assert m["corrupt_from_null"] is True, "this builder implements corrupt_from_null: true only"
    assert {"main_fields", "year_pivot"} <= set(p["dates"]), p["dates"]
    assert len(p["dates"]["main_fields"]) == 2 and set(p["dates"]["main_fields"]) <= set(MAIN_FIELDS), p["dates"]
    return config


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    """Build and write both main splits and ``main_build.json`` for the config given on the CLI.

    Args:
        argv: CLI arguments (None = ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="data/vrdu/build-configs/<id>.yaml")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    p, seed = config["params"], config["seed"]
    data_root = Path(p["data_root"])
    assert data_root.is_dir(), f"no such directory: {data_root}"
    date_fields, year_pivot = list(p["dates"]["main_fields"]), p["dates"]["year_pivot"]
    kinds = {kind: list(members) for kind, members in p["cross_type"]["kinds"].items()}
    eq = make_eq(field_kinds(kinds), year_pivot)
    k_range = {et: tuple(p["main"]["k_range"][et]) for et in ERROR_TYPES["main"]}

    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0],
           "script": "data/vrdu/build_vrdu_main.py", "config_id": config["id"], "config_snapshot": config,
           "splits": {}, "sha256": {}}
    out["git_sha"], out["git_dirty"] = base_builder.git_state()
    for split in SPLITS:
        main_base = read_base(data_root / "main" / f"{split}_base.json", MAIN_FIELDS, ID_KEYS)
        line_base = read_base(data_root / "line" / f"{split}_base.json", LINE_FIELDS, ["document_id", "line_index"])
        main_by_doc = {r["document_id"]: r for r in main_base}
        line_by_doc = {}
        for r in sorted(line_base, key=lambda r: (r["document_id"], r["line_index"])):
            line_by_doc.setdefault(r["document_id"], []).append(r)
        assert set(line_by_doc) <= set(main_by_doc), "line documents missing from the main base"
        frag_rate = fragment_rate(main_base, date_fields, year_pivot)
        rngs = {et: random.Random(f"{seed}:{split}:main:{et}") for et in ERROR_TYPES["main"]}

        rows, skipped, capped = generate_inter_rows(main_base, k_range[INTER_DOCUMENT], main_pools(main_base), date_fields,
                                                    year_pivot, rngs[INTER_DOCUMENT], eq, frag_rate)
        truth = document_truth(main_base, line_base, eq)
        cross_rows, cross_skipped, cross_capped, cross_stats = generate_cross_type_rows(
            "main", main_base, MAIN_FIELDS, ID_KEYS, main_by_doc, line_by_doc, truth, kinds, date_fields, year_pivot,
            k_range[CROSS_TYPE], rngs[CROSS_TYPE], eq, frag_rate)
        rows += cross_rows
        base_by_id = {(r["document_id"],): r for r in main_base}
        annotate_null_fill(rows, base_by_id, ID_KEYS)
        verify_output(rows, base_by_id, MAIN_FIELDS, MAIN_FIELDS, ID_KEYS, max(hi for _, hi in k_range.values()),
                      date_fields, year_pivot, eq)
        verify_cross_type_rows(cross_rows, "main", base_by_id, ID_KEYS, main_by_doc, line_by_doc, truth, kinds,
                               date_fields, year_pivot, eq)
        for r in rows:
            assert r["k"] == r["num_invalid_fields"], (r["document_id"], r["error_type"], r["k"], "k != corrupted fields")
            if not r["valid"]:
                lo, hi = k_range[r["error_type"]]
                assert lo <= r["k"] <= hi, (r["document_id"], r["error_type"], r["k"])

        rel = f"main/{split}.json"
        (data_root / rel).write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        out["sha256"][rel] = sha256_of(data_root / rel)
        out["splits"][split] = {
            "n_valid": sum(r["valid"] for r in rows), "strata": stratum_stats(rows),
            "skipped_rows": {INTER_DOCUMENT: skipped, CROSS_TYPE: cross_skipped},
            "capped_rows": {INTER_DOCUMENT: capped, CROSS_TYPE: cross_capped},
            "cross_type": {"candidate_counts": dict(cross_stats["candidate_counts"]),
                           "pair_counts": dict(sorted(cross_stats["pair_counts"].items())),
                           "n_exchanged_fields": cross_stats["n_exchanged_fields"],
                           "n_rows_with_exchange": cross_stats["n_rows_with_exchange"]},
            "fragment_target_rate": frag_rate,
            "date_fragment_rates": date_fragment_stats(rows, date_fields, year_pivot),
        }
        n_by_type = Counter(r["error_type"] for r in rows if not r["valid"])
        print(f"{rel}: {out['splits'][split]['n_valid']} valid + {n_by_type[INTER_DOCUMENT]} inter-document + "
              f"{n_by_type[CROSS_TYPE]} cross-type = {len(rows)}")
    (data_root / BUILD_FILE).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"written to {data_root / BUILD_FILE}")


if __name__ == "__main__":
    main()
