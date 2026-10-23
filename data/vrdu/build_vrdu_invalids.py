"""Add deliberately invalid rows to the VRDU main/line datasets.

Reads ``{main,line}/{train,test}_base.json`` (valid rows) and writes ``{main,line}/{train,test}.json``
(valid + invalid rows) and ``invalids.json`` (provenance). An invalid row differs from its valid row
in exactly ``k`` fields; its identity is ``(document_id, [line_index], error_type, k)``. Every row has
``error_type`` (null if valid) and per-field ``field_errors`` (null, or the source of the wrong value).

Error types (one per row; one row per achievable k):

  inter_document  the value comes from another document of the split.
  intra_document  line only: the same field of another line item of the document (a chain, one donor
                  item per corrupted field).
  cross_type      a value of another same-kind field of the document (name / date / amount), never from
                  the row's own line item.

Rules for every type: a corruption must change the value up to ``equivalence_key`` (a reformat is not
an error); a null may be filled but a value is never nulled; ``from <= to`` holds whenever both dates
parse; OCR-fragment dates are drawn at their natural rate (``pick_fragment_neutral``); and an invalid
line row may not match a valid line item on its non-null fields (``matches_item``, the label rule).

Local data preparation (no cluster job):

    uv run --python 3.12 python data/vrdu/build_vrdu_invalids.py \
        --config data/vrdu/build-configs/2026-10-22-vrdu-invalids-05.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from functools import cache
from pathlib import Path

import build_vrdu_dataset as base_builder
import yaml

# Field lists: single source of truth in the base-file builder.
MAIN_FIELDS = base_builder.MAIN_FIELDS
LINE_FIELDS = base_builder.LINE_FIELDS
LINE_CARRIED_FIELDS = base_builder.LINE_CARRIED_FIELDS  # always [] since 2026-09-15

REQUIRED_PARAM_KEYS = {"data_root", "main", "line", "corrupt_from_null", "corrupt_to_null", "dates", "cross_type"}
INTER_DOCUMENT, INTRA_DOCUMENT, CROSS_TYPE = "inter_document", "intra_document", "cross_type"
ERROR_TYPES = {"main": (INTER_DOCUMENT, CROSS_TYPE), "line": (INTER_DOCUMENT, INTRA_DOCUMENT, CROSS_TYPE)}
assert not set(MAIN_FIELDS) & set(LINE_FIELDS), "main/line field names collide"
REQUIRED_DATE_KEYS = {"main_fields", "line_fields", "year_pivot"}


# --- Dates --------------------------------------------------------------------


@cache
def parse_date(text, year_pivot):
    """Parse an ``m/d/y`` date (2- or 4-digit year).

    Args:
        text: the date string (or None).
        year_pivot: 2-digit years below it are 20YY, others 19YY.

    Returns:
        ``(year, month, day)``, or None if ``text`` is None or not a full date (e.g. an OCR fragment).
    """
    if text is None:
        return None
    match = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})", text.strip())
    if match is None:
        return None
    month, day, year = (int(group) for group in match.groups())
    if year < 100:
        year = 2000 + year if year < year_pivot else 1900 + year
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return (year, month, day)


def pair_ok(parsed_from, parsed_to):
    """Check the date-order rule ``from <= to`` (always true if either date did not parse).

    Args:
        parsed_from: ``parse_date`` of the start date.
        parsed_to: ``parse_date`` of the end date.

    Returns:
        True if the pair is in order.
    """
    return parsed_from is None or parsed_to is None or parsed_from <= parsed_to


@cache
def equivalence_key(kind, text, year_pivot):
    """Canonical form of a value, so a reformat (``2/18/20`` vs ``02/18/20``) counts as the same value.

    Args:
        kind: the field's kind (``"date"``, ``"amount"``, or other / None).
        text: the value.
        year_pivot: as in ``parse_date``.

    Returns:
        A parsed date, a number, or the case- and whitespace-folded text, tagged by type.
    """
    if kind == "date":
        parsed = parse_date(text, year_pivot)
        if parsed is not None:
            return ("date", parsed)
    elif kind == "amount":
        digits = re.sub(r"[\s$,]", "", text)
        if re.fullmatch(r"\d+(\.\d+)?", digits):
            return ("amount", Decimal(digits))
    return ("text", " ".join(text.split()).casefold())


def make_eq(kind_of, year_pivot):
    """Build the per-field equivalence function.

    Args:
        kind_of: ``{field: kind}``.
        year_pivot: as in ``parse_date``.

    Returns:
        ``eq(field, value)``: the ``equivalence_key`` of ``value`` as a ``field`` value (None for None).
    """
    def eq(field, value):
        return None if value is None else equivalence_key(kind_of.get(field), value, year_pivot)
    return eq


# --- Label rule ---------------------------------------------------------------


def nonnull_values(row, fields):
    """The values a judge is shown for a row (nulls are not shown).

    Args:
        row: a data row.
        fields: the fields to consider.

    Returns:
        ``{field: value}`` for the non-null ``fields``.
    """
    return {f: row[f] for f in fields if row[f] is not None}


def item_keys(items, fields, eq):
    """Precompute the equivalence keys of line items, for ``matches_keys``.

    Args:
        items: valid line rows.
        fields: the fields to key.
        eq: from ``make_eq``.

    Returns:
        One ``{field: key}`` per item.
    """
    return [{f: eq(f, item[f]) for f in fields} for item in items]


def matches_keys(keys, items_keys):
    """``matches_item`` on precomputed keys.

    Args:
        keys: ``{field: key}`` of the shown values (non-null).
        items_keys: from ``item_keys``.

    Returns:
        True iff some item has every one of these keys.
    """
    assert keys and all(k is not None for k in keys.values()), keys
    return any(all(ik[f] == k for f, k in keys.items()) for ik in items_keys)


def matches_item(values, items, eq):
    """The label rule: are the shown values one real line item's?

    Args:
        values: ``{field: non-null value}`` as shown.
        items: the document's valid line rows.
        eq: from ``make_eq``.

    Returns:
        True iff some item equals ``values`` (by ``equivalence_key``) on every shown field; an item's null
        matches nothing.
    """
    return matches_keys({f: eq(f, v) for f, v in values.items()}, item_keys(items, list(values), eq))


# --- Pools (main dataset + line inter-document) --------------------------------


def main_pools(base_rows):
    """Value pools for main inter_document corruption.

    Args:
        base_rows: the split's valid main rows.

    Returns:
        ``{field: [non-null values]}`` (OCR fragments included, repeats kept).
    """
    return {f: [r[f] for r in base_rows if r[f] is not None] for f in MAIN_FIELDS}


def line_inter_pools_by_doc(line_base, date_fields):
    """Value pools for line inter_document corruption, per document.

    Args:
        line_base: the split's valid line rows.
        date_fields: ``(start, end)``; the two share one pool.

    Returns:
        ``{document_id: {field: [non-null values from every other document]}}``.
    """
    docs = sorted({r["document_id"] for r in line_base})
    pools = {}
    for doc in docs:
        others = [r for r in line_base if r["document_id"] != doc]
        date_pool = [v for f in date_fields for r in others for v in [r[f]] if v is not None]
        field_pools = {}
        for field in LINE_FIELDS:
            if field in date_fields:
                field_pools[field] = date_pool
            else:
                field_pools[field] = [r[field] for r in others if r[field] is not None]
        pools[doc] = field_pools
    return pools


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


# --- Corruption (shared: main + line inter-document) ------------------------


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


def fragment_rate(rows, date_fields, year_pivot):
    """Share of OCR-fragment dates among the non-null dates of ``rows``.

    Args:
        rows: data rows.
        date_fields: the date fields.
        year_pivot: as in ``parse_date``.

    Returns:
        The share, in [0, 1].
    """
    values = [r[f] for r in rows for f in date_fields if r[f] is not None]
    assert values, "no date values to take a fragment rate from"
    return sum(1 for v in values if parse_date(v, year_pivot) is None) / len(values)


def pick_fragment_neutral(eligible, rate, is_fragment, rng):
    """Pick a candidate so fragments are drawn at ``rate`` (a fragment should not predict a corruption).

    Args:
        eligible: candidates.
        rate: target fragment rate (from ``fragment_rate``).
        is_fragment: ``candidate -> bool``.
        rng: ``random.Random``.

    Returns:
        One candidate.
    """
    assert 0.0 <= rate <= 1.0, rate
    fragments = [x for x in eligible if is_fragment(x)]
    reals = [x for x in eligible if not is_fragment(x)]
    if fragments and reals:
        chosen = fragments if rng.random() < rate else reals
    else:
        chosen = fragments or reals
    assert chosen, "no date candidate despite feasibility"
    return chosen[rng.randrange(len(chosen))]


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


def generate_split(base_rows, fields, target_fields, id_keys, k_range,
                   pool_for_row, date_fields, year_pivot, rng, source_for, eq, frag_rate):
    """Main-dataset inter_document rows for one split (one per k, capped to what each row allows).

    Args:
        base_rows: valid main rows.
        fields, target_fields: all fields / fields that may be corrupted.
        id_keys: identity keys.
        k_range: inclusive ``(k_lo, k_hi)``.
        pool_for_row: ``row -> pools``.
        date_fields: ``(start, end)``.
        year_pivot: as in ``parse_date``.
        rng: ``random.Random``.
        source_for: ``(field, value) -> source dict``.
        eq: from ``make_eq``.
        frag_rate: target fragment rate.

    Returns:
        ``(rows, skipped, capped)``: valid + invalid output rows, rows with no corruption, rows capped below k.
    """
    k_lo, k_hi = k_range
    assert 1 <= k_lo <= k_hi <= len(target_fields), f"k_range {k_range} outside 1..{len(target_fields)}"

    out = [tag_row(row, fields, id_keys, 0, True, [], None, {}) for row in base_rows]
    skipped, capped = [], []
    for base_row in base_rows:
        pools = pool_for_row(base_row)
        row_id = {key: base_row[key] for key in id_keys}
        row_caps, produced = [], 0
        for k in range(k_lo, k_hi + 1):
            result = make_invalid_row(base_row, k, target_fields, date_fields, pools, year_pivot, rng, eq, frag_rate)
            if result is None:
                row_caps.append({**row_id, "k": k, "realized": 0})
                continue
            new_row, invalid_fields = result
            produced += 1
            if len(invalid_fields) < k:
                row_caps.append({**row_id, "k": k, "realized": len(invalid_fields)})
            out.append(tag_row(new_row, fields, id_keys, k, False, invalid_fields, INTER_DOCUMENT,
                               {f: source_for(f, new_row[f]) for f in invalid_fields}))
        if produced:
            capped.extend(row_caps)  # a row that produced nothing at all is skipped, not capped
        else:
            skipped.append(row_id)
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


# --- Row emission: line ---------------------------------------------------


def _intra_feasible(base_row, current_row, other_rows, other_keys, target_fields, date_fields,
                     corrupted, used_donors, year_pivot, eq):
    """The (field, donor) pairs open at the next step of an intra chain.

    A donor value is eligible if it changes the original value, keeps ``from <= to``, and does not make
    the row match another item (``matches_keys``).

    Args:
        base_row: the valid row.
        current_row: the row as corrupted so far.
        other_rows: the document's other line items (donors).
        other_keys: ``item_keys(other_rows)``.
        target_fields: fields that may be corrupted.
        date_fields: ``(start, end)``.
        corrupted: fields already corrupted.
        used_donors: donor indices already used.
        year_pivot: as in ``parse_date``.
        eq: from ``make_eq``.

    Returns:
        ``{field: [eligible donor index]}``.
    """
    from_field, to_field = date_fields
    feasible = {}
    for field in target_fields:
        if field in corrupted:
            continue
        eligible = []
        for idx, donor in enumerate(other_rows):
            if idx in used_donors:
                continue
            value = donor[field]
            if value is None or eq(field, value) == eq(field, base_row[field]):
                continue
            if field in date_fields:
                partner = to_field if field == from_field else from_field
                parsed_value = parse_date(value, year_pivot)
                parsed_partner = parse_date(current_row[partner], year_pivot)
                if field == from_field and not pair_ok(parsed_value, parsed_partner):
                    continue
                if field == to_field and not pair_ok(parsed_partner, parsed_value):
                    continue
            hypothetical = {f: eq(f, value if f == field else current_row[f]) for f in target_fields}
            if matches_keys({f: k for f, k in hypothetical.items() if k is not None}, other_keys):
                continue
            eligible.append(idx)
        if eligible:
            feasible[field] = eligible
    return feasible


def build_intra_chain(base_row, other_rows, target_fields, date_fields, year_pivot, k_hi, rng, eq, frag_rate):
    """Intra-document variants of one line row with 1, 2, ... corrupted fields.

    Each step copies one field from a distinct donor item; stops at ``k_hi`` or when nothing is eligible.

    Args:
        base_row: the valid row.
        other_rows: the document's other line items.
        target_fields: fields that may be corrupted.
        date_fields: ``(start, end)``.
        year_pivot: as in ``parse_date``.
        k_hi: the largest k.
        rng: ``random.Random``.
        eq: from ``make_eq``.
        frag_rate: target fragment rate.

    Returns:
        ``[(row, invalid_fields, {field: donor line_index})]``, one per achieved k.
    """
    other_keys = item_keys(other_rows, target_fields, eq)
    current = dict(base_row)
    corrupted = []
    used_donors = set()
    donors = {}
    results = []
    while len(corrupted) < k_hi:
        feasible = _intra_feasible(base_row, current, other_rows, other_keys, target_fields,
                                    date_fields, corrupted, used_donors, year_pivot, eq)
        if not feasible:
            break
        feasible_fields = [f for f in target_fields if f in feasible]
        field = feasible_fields[rng.randrange(len(feasible_fields))]
        if field in date_fields:
            donor_idx = pick_fragment_neutral(
                feasible[field], frag_rate,
                lambda i: parse_date(other_rows[i][field], year_pivot) is None, rng)
        else:
            donor_idx = feasible[field][rng.randrange(len(feasible[field]))]

        current[field] = other_rows[donor_idx][field]
        corrupted.append(field)
        used_donors.add(donor_idx)
        donors[field] = other_rows[donor_idx]["line_index"]

        snapshot = dict(current)
        assert not matches_item(nonnull_values(snapshot, target_fields), other_rows, eq), (
            f"intra-document corruption at {base_row.get('document_id')}/"
            f"{base_row.get('line_index')} reproduced another valid row on its non-null fields "
            "(should have been excluded by _intra_feasible -- logic error)"
        )
        results.append((snapshot, sorted(corrupted), dict(donors)))
    return results


def line_inter_document(row, pool, target_fields, date_fields, year_pivot, k_range, rng, eq, frag_rate):
    """Line inter_document rows for one valid row, one per achievable k.

    Args:
        row: the valid row.
        pool: the document's inter pools (``line_inter_pools_by_doc``).
        target_fields: fields that may be corrupted.
        date_fields: ``(start, end)``.
        year_pivot: as in ``parse_date``.
        k_range: inclusive ``(k_lo, k_hi)``.
        rng: ``random.Random``.
        eq: from ``make_eq``.
        frag_rate: target fragment rate.

    Returns:
        ``[(new_row, invalid_fields)]``.
    """
    targets = achievable_targets(row, target_fields, date_fields, pool, year_pivot, k_range, eq)
    rows = []
    for k in targets:
        result = make_invalid_row(row, k, target_fields, date_fields, pool, year_pivot, rng, eq, frag_rate)
        assert result is not None and len(result[1]) == k, (
            f"k={k} came from achievable_targets so it must be exactly realizable"
        )
        rows.append(result)
    return rows


def generate_line_split(line_base, date_fields, year_pivot, k_range, rngs, eq, frag_rate):
    """Line inter_document and intra_document rows for one split.

    Args:
        line_base: valid line rows.
        date_fields: ``(start, end)``.
        year_pivot: as in ``parse_date``.
        k_range: inclusive ``(k_lo, k_hi)``.
        rngs: ``{error type: random.Random}``, one stream per type.
        eq: from ``make_eq``.
        frag_rate: target fragment rate.

    Returns:
        ``(rows, skipped, capped, dropped)``: valid + invalid output rows; skipped / capped rows by type;
        inter rows dropped because they matched a valid item.
    """
    assert set(rngs) == {INTER_DOCUMENT, INTRA_DOCUMENT}, sorted(rngs)
    k_lo, k_hi = k_range
    id_keys = ["document_id", "line_index"]
    out = [tag_row(row, LINE_FIELDS, id_keys, 0, True, [], None, {}) for row in line_base]

    by_doc = {}
    for row in line_base:
        by_doc.setdefault(row["document_id"], []).append(row)
    for rows in by_doc.values():
        rows.sort(key=lambda r: r["line_index"])

    inter_pools = line_inter_pools_by_doc(line_base, date_fields)

    skipped = {INTER_DOCUMENT: [], INTRA_DOCUMENT: []}
    capped = {INTER_DOCUMENT: [], INTRA_DOCUMENT: []}
    dropped = []

    for doc_id, rows in by_doc.items():
        keys = item_keys(rows, LINE_FIELDS, eq)
        for row in rows:
            row_id = {"document_id": row["document_id"], "line_index": row["line_index"]}

            inter_rows = []
            for new_row, invalid_fields in line_inter_document(row, inter_pools[doc_id], LINE_FIELDS, date_fields, year_pivot,
                                                              k_range, rngs[INTER_DOCUMENT], eq, frag_rate):
                shown = {f: eq(f, v) for f, v in nonnull_values(new_row, LINE_FIELDS).items()}
                if matches_keys(shown, keys):
                    dropped.append({**row_id, "k": len(invalid_fields)})
                else:
                    inter_rows.append((new_row, invalid_fields))
            if not inter_rows:
                skipped["inter_document"].append(row_id)
            else:
                reached = inter_rows[-1][1]
                if len(reached) < k_hi:
                    capped["inter_document"].append({**row_id, "reached": len(reached)})
                for new_row, invalid_fields in inter_rows:
                    out.append(tag_row(new_row, LINE_FIELDS, id_keys, len(invalid_fields), False,
                                       invalid_fields, INTER_DOCUMENT,
                                       {f: {"scope": "other_document"} for f in invalid_fields}))

            other_rows = [r for r in rows if r is not row]
            chain = build_intra_chain(row, other_rows, LINE_FIELDS, date_fields, year_pivot, k_hi, rngs[INTRA_DOCUMENT],
                                      eq, frag_rate)
            chain = [(r, fields, donors) for r, fields, donors in chain if len(fields) >= k_lo]
            if not chain:
                skipped["intra_document"].append(row_id)
            else:
                if len(chain[-1][1]) < k_hi:
                    capped["intra_document"].append({**row_id, "reached": len(chain[-1][1])})
                for new_row, invalid_fields, donors in chain:
                    out.append(tag_row(
                        new_row, LINE_FIELDS, id_keys, len(invalid_fields), False, invalid_fields, INTRA_DOCUMENT,
                        {f: {"scope": "same_document", "dataset": "line", "field": f, "line_index": donors[f]}
                         for f in invalid_fields}))

    return out, skipped, capped, dropped


def verify_line_output(out_rows, base_by_id, date_fields, year_pivot, line_by_doc, eq):
    """Check one generated line split, including the label rule; raises AssertionError.

    Args:
        out_rows: the split's output rows.
        base_by_id: ``{(document_id, line_index): valid row}``.
        date_fields: ``(start, end)``.
        year_pivot: as in ``parse_date``.
        line_by_doc: ``{document_id: [valid line rows]}``.
        eq: from ``make_eq``.
    """
    valid_rows = [r for r in out_rows if r["valid"]]
    invalid_rows = [r for r in out_rows if not r["valid"]]

    assert len(valid_rows) == len(base_by_id), (len(valid_rows), len(base_by_id))
    for row in valid_rows:
        key = (row["document_id"], row["line_index"])
        assert row["k"] == 0 and row["num_invalid_fields"] == 0 and row["invalid_fields"] == []
        assert row["error_type"] is None
        assert {f: row[f] for f in LINE_FIELDS} == {f: base_by_id[key][f] for f in LINE_FIELDS}
    verify_field_errors(out_rows, LINE_FIELDS, ERROR_TYPES["line"])

    identity = [(row["document_id"], row["line_index"], row["error_type"], row["k"]) for row in out_rows]
    assert len(identity) == len(set(identity)), (
        "output rows not uniquely keyed by (document_id, line_index, error_type, k)"
    )

    from_field, to_field = date_fields
    for row in invalid_rows:
        key = (row["document_id"], row["line_index"])
        assert key in base_by_id, f"invalid row for unknown id {key}"
        base = base_by_id[key]
        changed = sorted(f for f in LINE_FIELDS if row[f] != base[f])
        assert row["error_type"] in ERROR_TYPES["line"], row["error_type"]
        assert changed == sorted(row["invalid_fields"]), (key, row["error_type"], changed, row["invalid_fields"])
        assert row["k"] == row["num_invalid_fields"] == len(changed) >= 1, (key, row["error_type"], row)
        for field in changed:
            assert row[field] is not None, f"{key}: field {field} corrupted to null"
            assert eq(field, row[field]) != eq(field, base[field]), f"{key}: {field} only reformatted"

        # Intra values must come from the named donor (inter values may recur by chance).
        if row["error_type"] == INTRA_DOCUMENT:
            for field in changed:
                src = row["field_errors"][field]["source"]
                donor = next(r for r in line_by_doc[row["document_id"]] if r["line_index"] == src["line_index"])
                assert src["line_index"] != row["line_index"] and row[field] == donor[field], (key, field, src)
            doc_values = {
                f: {r[f] for r in line_by_doc[row["document_id"]] if r[f] is not None}
                for f in LINE_FIELDS
            }
            for field in changed:
                assert row[field] in doc_values[field], (
                    f"{key}: intra-document value for {field} does not appear in its document"
                )

        if from_field in changed or to_field in changed:
            start, end = row[from_field], row[to_field]
            if start is not None and end is not None:
                assert pair_ok(parse_date(start, year_pivot), parse_date(end, year_pivot)), (key, start, end)

    # Label rule: no invalid row may match a valid item.
    for row in invalid_rows:
        key = (row["document_id"], row["line_index"])
        assert not matches_item(nonnull_values(row, LINE_FIELDS), line_by_doc[row["document_id"]], eq), (
            f"{key}: {row['error_type']} k={row['k']} row has the non-null values of a valid line item"
        )


# --- cross_type ----------------------------------------------------------


def field_kinds(kinds):
    """Invert the kinds config.

    Args:
        kinds: ``{kind: [fields]}``.

    Returns:
        ``{field: kind}``; asserts no field is in two kinds.
    """
    out = {}
    for kind, members in kinds.items():
        for f in members:
            assert f not in out, f"field {f!r} is in kinds {out[f]!r} and {kind!r}"
            assert f in MAIN_FIELDS or f in LINE_FIELDS, f"kinds: unknown field {f!r}"
            out[f] = kind
    return out


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


# --- Config / IO --------------------------------------------------------------


def load_config(path):
    """Read and validate the invalids config.

    Args:
        path: the YAML config.

    Returns:
        The parsed config; asserts every required key is present.
    """
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("id", "project", "seed", "params"):
        assert key in config, f"config missing top-level key {key!r}"
    params = config["params"]
    assert not REQUIRED_PARAM_KEYS - set(params), f"config params missing {sorted(REQUIRED_PARAM_KEYS - set(params))}"
    assert set(params["dates"]) == REQUIRED_DATE_KEYS, (
        f"config params.dates keys {sorted(params['dates'])} != {sorted(REQUIRED_DATE_KEYS)}"
    )
    assert set(params["main"]) == {"k_range", "pool_scope"}, "params.main keys"
    assert set(params["line"]) == {"k_range"}, "params.line keys"
    cross = params["cross_type"]
    assert set(cross) == {"kinds", "main", "line"}, "params.cross_type keys"
    assert set(cross["main"]) == {"k_range"} and set(cross["line"]) == {"k_range"}, "params.cross_type.main/line keys"
    return config


def read_base(path, fields, id_keys):
    """Read and validate one ``*_base.json`` file.

    Args:
        path: the base file.
        fields: expected data fields.
        id_keys: identity keys.

    Returns:
        Its rows; asserts the keys, ``valid: true`` and unique ids.
    """
    rows = json.loads(path.read_text(encoding="utf-8"))
    expected = {*id_keys, "valid", *fields}
    for row in rows:
        assert set(row) == expected, f"{path.name}: row keys {set(row)} != {expected}"
        assert row["valid"] is True, f"{path.name}: base row not valid: {row}"
    ids = [tuple(row[k] for k in id_keys) for row in rows]
    assert len(ids) == len(set(ids)), f"{path.name}: duplicate identity keys"
    return rows


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


# --- Main --------------------------------------------------------------------


def main(argv=None):
    """Generate and write the invalid rows and ``invalids.json`` for the config given on the CLI.

    Args:
        argv: CLI arguments (None = ``sys.argv``).
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, type=Path, help="configs/<id>.yaml")
    args = parser.parse_args(argv)

    assert args.config.is_file(), f"no such file: {args.config}"
    config = load_config(args.config)
    params = config["params"]
    seed = config["seed"]
    dates = params["dates"]
    year_pivot = dates["year_pivot"]
    main_date_fields = list(dates["main_fields"])
    line_date_fields = list(dates["line_fields"])
    assert set(main_date_fields) <= set(MAIN_FIELDS) and len(main_date_fields) == 2
    assert set(line_date_fields) <= set(LINE_FIELDS) and len(line_date_fields) == 2
    assert params["corrupt_from_null"] is True and params["corrupt_to_null"] is False, (
        "this build assumes corrupt_from_null=true, corrupt_to_null=false"
    )
    assert params["main"]["pool_scope"] == "per_split"
    main_k_range = tuple(params["main"]["k_range"])
    line_k_range = tuple(params["line"]["k_range"])
    assert 1 <= main_k_range[0] <= main_k_range[1] <= len(MAIN_FIELDS), main_k_range
    assert 1 <= line_k_range[0] <= line_k_range[1] <= len(LINE_FIELDS), line_k_range

    cross = params["cross_type"]
    kinds = {kind: list(members) for kind, members in cross["kinds"].items()}
    kind_of = field_kinds(kinds)
    eq = make_eq(kind_of, year_pivot)
    assert set(kinds.get("date", [])) == {*main_date_fields, *line_date_fields}, (
        "cross_type.kinds.date must be exactly the main + line date fields"
    )
    cross_k = {"main": tuple(cross["main"]["k_range"]), "line": tuple(cross["line"]["k_range"])}
    for ds, fields in (("main", MAIN_FIELDS), ("line", LINE_FIELDS)):
        n_kinded = sum(f in kind_of for f in fields)
        assert 1 <= cross_k[ds][0] <= cross_k[ds][1] <= n_kinded, (ds, cross_k[ds], n_kinded)
    print("cross_type kinds:", {k: v for k, v in kinds.items()})
    print("fields with no kind (never cross_type):",
          [f for f in [*MAIN_FIELDS, *LINE_FIELDS] if f not in kind_of])

    data_root = Path(params["data_root"])
    assert data_root.is_dir(), f"no such directory: {data_root}"

    rng = random.Random(seed)
    provenance = {
        "counts": {"main": {}, "line": {}},
        "num_invalid_fields_dist": {"main": {}, "line": {}},
        "invalid_field_freq": {"main": {}, "line": {}},
        "skipped_rows": {"main": {}, "line": {}},
        "capped_rows": {"main": {}, "line": {}},
        "cross_type": {"kinds": kinds, "fields_without_kind": [f for f in [*MAIN_FIELDS, *LINE_FIELDS] if f not in kind_of],
                       "main": {}, "line": {}},
        "main_pool_sizes": {},
        "line_inter_date_pool_sizes": {},
        "line_intra_donor_counts": {},
        "line_inter_coincidental_overlap": {},
        "line_inter_dropped_matches_item": {},
        "date_fragment_rates": {"main": {}, "line": {}},
        "fragment_target_rate": {},
    }

    for split in ("train", "test"):
        main_base = read_base(data_root / "main" / f"{split}_base.json", MAIN_FIELDS, ["document_id"])
        main_by_id = {r["document_id"]: r for r in main_base}
        line_base = read_base(data_root / "line" / f"{split}_base.json", LINE_FIELDS,
                              ["document_id", "line_index"])

        line_docs = sorted({r["document_id"] for r in line_base})
        assert set(line_docs) <= set(main_by_id), "line documents missing from the main base"

        # --- main dataset ---
        pools_main = main_pools(main_base)
        main_frag_rate = fragment_rate(main_base, main_date_fields, year_pivot)
        line_frag_rate = fragment_rate(line_base, line_date_fields, year_pivot)
        provenance["fragment_target_rate"][split] = {"main": main_frag_rate, "line": line_frag_rate}
        provenance["main_pool_sizes"][split] = {f: len(p) for f, p in pools_main.items()}
        out_rows, skipped, capped = generate_split(
            main_base, MAIN_FIELDS, MAIN_FIELDS, ["document_id"], main_k_range,
            lambda _r, _p=pools_main: _p, main_date_fields, year_pivot, rng,
            lambda _field, _value: {"scope": "other_document"}, eq, main_frag_rate,
        )

        line_by_doc = {}
        for row in line_base:
            line_by_doc.setdefault(row["document_id"], []).append(row)
        for rows_ in line_by_doc.values():
            rows_.sort(key=lambda r: r["line_index"])
        truth = document_truth(main_base, line_base, eq)

        cross_rng = random.Random(f"{seed}:{split}:main:{CROSS_TYPE}")
        cross_rows, cross_skipped, cross_capped, cross_stats = generate_cross_type_rows(
            "main", main_base, MAIN_FIELDS, ["document_id"], main_by_id, line_by_doc, truth, kinds,
            main_date_fields, year_pivot, cross_k["main"], cross_rng, eq, main_frag_rate,
        )
        out_rows = out_rows + cross_rows
        base_by_id = {(r["document_id"],): r for r in main_base}
        annotate_null_fill(out_rows, base_by_id, ["document_id"])
        verify_output(out_rows, base_by_id, MAIN_FIELDS, MAIN_FIELDS, ["document_id"],
                      max(main_k_range[1], cross_k["main"][1]), main_date_fields, year_pivot, eq)
        verify_cross_type_rows(cross_rows, "main", base_by_id, ["document_id"], main_by_id, line_by_doc,
                               truth, kinds, main_date_fields, year_pivot, eq)
        (data_root / "main" / f"{split}.json").write_text(
            json.dumps(out_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        provenance["date_fragment_rates"]["main"][split] = date_fragment_stats(out_rows, main_date_fields, year_pivot)
        n_valid = sum(1 for r in out_rows if r["valid"])
        n_by_type = {et: sum(1 for r in out_rows if r["error_type"] == et) for et in ERROR_TYPES["main"]}
        provenance["counts"]["main"][split] = {
            "valid": n_valid, "invalid": len(out_rows) - n_valid, **n_by_type, "total": len(out_rows)
        }
        provenance["num_invalid_fields_dist"]["main"][split] = {
            et: dict(sorted(Counter(r["num_invalid_fields"] for r in out_rows if r["error_type"] == et).items()))
            for et in ERROR_TYPES["main"]
        }
        provenance["invalid_field_freq"]["main"][split] = {
            et: {f: sum(1 for r in out_rows if r["error_type"] == et and f in r["invalid_fields"]) for f in MAIN_FIELDS}
            for et in ERROR_TYPES["main"]
        }
        provenance["skipped_rows"]["main"][split] = {INTER_DOCUMENT: skipped, CROSS_TYPE: cross_skipped}
        provenance["capped_rows"]["main"][split] = {INTER_DOCUMENT: capped, CROSS_TYPE: cross_capped}
        provenance["cross_type"]["main"][split] = {
            "candidate_counts": dict(cross_stats["candidate_counts"]),
            "pair_counts": dict(sorted(cross_stats["pair_counts"].items())),
            "n_exchanged_fields": cross_stats["n_exchanged_fields"],
            "n_rows_with_exchange": cross_stats["n_rows_with_exchange"],
        }
        print(
            f"main/{split}: {n_valid} valid + {n_by_type[INTER_DOCUMENT]} inter-document + "
            f"{n_by_type[CROSS_TYPE]} cross-type = {len(out_rows)}  (skipped inter {len(skipped)}/cross "
            f"{len(cross_skipped)}, capped inter {len(capped)}/cross {len(cross_capped)})"
        )

        # --- line dataset ---
        inter_pools = line_inter_pools_by_doc(line_base, line_date_fields)
        inter_sizes = [len(p[line_date_fields[0]]) for p in inter_pools.values()]
        provenance["line_inter_date_pool_sizes"][split] = {
            "min": min(inter_sizes), "median": statistics.median(inter_sizes), "max": max(inter_sizes),
        }
        donor_counts = [len(rows) - 1 for rows in line_by_doc.values()]
        provenance["line_intra_donor_counts"][split] = {
            "min": min(donor_counts), "median": statistics.median(donor_counts), "max": max(donor_counts),
            "docs_with_no_donor": sum(1 for c in donor_counts if c == 0),
        }

        line_rngs = {t: random.Random(f"{seed}:{split}:line:{t}") for t in (INTER_DOCUMENT, INTRA_DOCUMENT)}
        out_rows, skipped, capped, inter_dropped = generate_line_split(
            line_base, line_date_fields, year_pivot, line_k_range, line_rngs, eq, line_frag_rate)
        provenance["line_inter_dropped_matches_item"][split] = inter_dropped
        cross_rng = random.Random(f"{seed}:{split}:line:{CROSS_TYPE}")
        cross_rows, cross_skipped, cross_capped, cross_stats = generate_cross_type_rows(
            "line", line_base, LINE_FIELDS, ["document_id", "line_index"], main_by_id, line_by_doc, truth,
            kinds, line_date_fields, year_pivot, cross_k["line"], cross_rng, eq, line_frag_rate,
        )
        out_rows = out_rows + cross_rows
        base_by_id_line = {(r["document_id"], r["line_index"]): r for r in line_base}
        annotate_null_fill(out_rows, base_by_id_line, ["document_id", "line_index"])
        verify_line_output(out_rows, base_by_id_line, line_date_fields, year_pivot, line_by_doc, eq)
        verify_cross_type_rows(cross_rows, "line", base_by_id_line, ["document_id", "line_index"], main_by_id,
                               line_by_doc, truth, kinds, line_date_fields, year_pivot, eq)
        (data_root / "line" / f"{split}.json").write_text(
            json.dumps(out_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        provenance["date_fragment_rates"]["line"][split] = date_fragment_stats(out_rows, line_date_fields, year_pivot)
        n_valid = sum(1 for r in out_rows if r["valid"])
        n_inter = sum(1 for r in out_rows if r["error_type"] == INTER_DOCUMENT)
        n_intra = sum(1 for r in out_rows if r["error_type"] == INTRA_DOCUMENT)
        n_cross = sum(1 for r in out_rows if r["error_type"] == CROSS_TYPE)
        provenance["counts"]["line"][split] = {
            "valid": n_valid, INTER_DOCUMENT: n_inter, INTRA_DOCUMENT: n_intra, CROSS_TYPE: n_cross,
            "total": len(out_rows),
        }
        provenance["num_invalid_fields_dist"]["line"][split] = {
            error_type: dict(sorted(Counter(
                r["num_invalid_fields"] for r in out_rows if r["error_type"] == error_type
            ).items()))
            for error_type in ERROR_TYPES["line"]
        }
        provenance["invalid_field_freq"]["line"][split] = {
            error_type: {
                f: sum(1 for r in out_rows if r["error_type"] == error_type and f in r["invalid_fields"])
                for f in LINE_FIELDS
            }
            for error_type in ERROR_TYPES["line"]
        }
        provenance["skipped_rows"]["line"][split] = {**skipped, CROSS_TYPE: cross_skipped}
        provenance["capped_rows"]["line"][split] = {**capped, CROSS_TYPE: cross_capped}
        provenance["cross_type"]["line"][split] = {
            "candidate_counts": dict(cross_stats["candidate_counts"]),
            "pair_counts": dict(sorted(cross_stats["pair_counts"].items())),
            "n_exchanged_fields": cross_stats["n_exchanged_fields"],
            "n_rows_with_exchange": cross_stats["n_rows_with_exchange"],
        }

        # Informational: inter values that also occur in the row's document (near 100% = crossed pool).
        inter_rows = [r for r in out_rows if r["error_type"] == "inter_document"]
        overlap_count = 0
        for row in inter_rows:
            doc_values = {
                f: {r[f] for r in line_by_doc[row["document_id"]] if r[f] is not None}
                for f in LINE_FIELDS
            }
            if any(row[f] in doc_values[f] for f in row["invalid_fields"]):
                overlap_count += 1
        provenance["line_inter_coincidental_overlap"][split] = {
            "rows_with_overlap": overlap_count, "total_inter_rows": len(inter_rows),
        }
        print(
            f"line/{split}: {n_valid} valid + {n_inter} inter-document + {n_intra} intra-document + "
            f"{n_cross} cross-type = {len(out_rows)}  (skipped inter {len(skipped['inter_document'])}/intra "
            f"{len(skipped['intra_document'])}/cross {len(cross_skipped)}, capped inter "
            f"{len(capped['inter_document'])}/intra {len(capped['intra_document'])}/cross {len(cross_capped)})"
        )

    sha, dirty = base_builder.git_state()
    (data_root / "invalids.json").write_text(
        json.dumps(
            {"build": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git_sha": sha,
                "git_dirty": dirty,
                "python": sys.version.split()[0],  # random.sample order is not guaranteed stable across CPython
                "script": "data/vrdu/build_vrdu_invalids.py",
                "config_id": config["id"],
                "config_snapshot": config,
                "seed": seed,
                **provenance,
            }},
            indent=2, ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"written to {data_root}")


if __name__ == "__main__":
    main()
