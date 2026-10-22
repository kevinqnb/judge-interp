"""Populate the VRDU main/line datasets with deliberately-invalid extractions.

Reads the four ``*_base.json`` files (every row a correct extraction, ``valid:
true``) and writes, under ``<data_root>/``:

    main/{train,test}.json   valid base rows + generated invalid rows
    line/{train,test}.json   valid base rows + generated invalid rows
    invalids.json            provenance: config snapshot, seed, pool sizes, counts,
                             and every capped / skipped row

An invalid row keeps the ``document_id`` (and ``line_index``) of the valid row it
was derived from, and differs from it in exactly ``num_invalid_fields`` fields,
named in ``invalid_fields``. ``k`` always equals ``num_invalid_fields`` -- there is
no "requested but uncapped" value that a row falls short of; a row's identity is
``(document_id, error_type, k)`` for main, ``(document_id, line_index, error_type,
k)`` for line.

Every row (valid or not, both datasets) carries two explicit label fields:

  error_type    ``null`` on a valid row; otherwise the one error type the whole row
                was built with (rows are single-error-type, so each type can be
                studied on its own).
  field_errors  one entry per data field. ``null`` = that entity is correct (it may
                be correct on an invalid row); otherwise ``{"error_type": ...,
                "source": {...}}`` saying how the wrong value got there. Sources:
                  {"scope": "other_document"}       drawn from another document
                  {"scope": "same_document", "dataset", "field", "line_index",
                   ["exchanged_with"]}              a value taken from this document
                                                    (``line_index`` null for main;
                                                    ``exchanged_with`` is cross_type
                                                    only -- see below)
                Every source also carries ``null_fill``: True when the valid row had NO value
                in that field (the corruption adds an entity that was absent), False when
                it replaces one.
                ``invalid_fields`` is exactly the set of non-null ``field_errors``.

Error types (main: inter_document, cross_type; line: all three):

  inter_document  values come from other documents (below).
  intra_document  line only: values come from the row's other line items.
  cross_type      the value is a genuine value of THIS document, but of a different
                  field of the same *kind* (``params.cross_type.kinds``: name, date,
                  amount), so the entity is real but filed under the wrong type.
                  Each corrupted field independently pulls a value from a same-kind
                  field of the same document -- in either dataset (a main
                  ``gross_amount`` can take a line ``sub_amount`` and vice versa).
                  Two fields that happen to pull each other's values are a direct
                  exchange, recorded as ``exchanged_with`` in both sources. A value
                  is eligible for field ``t`` only if it is NOT already a valid value
                  of ``t`` anywhere in the document (otherwise the "wrong type" row
                  would actually be correct); "the same value" is judged by
                  ``equivalence_key``, so a reformat counts as the same. Fields with
                  no kind are never touched. Uses its own RNG stream, so adding it does not
                  perturb inter_document / intra_document rows.

  main   pool = every non-null value for the field within the split being written
         (OCR-fragment dates included; nothing synthetic). For each row, one
         invalid row is emitted per k in ``main.k_range``; a k the row cannot
         fully reach is realized at the largest achievable count below it and
         recorded under ``capped_rows`` (it has not happened so far: every row
         reaches every k).

  line   Two independent error types, each producing (up to) one invalid row per
         achievable count in ``line.k_range`` (now 1..5, all five LINE_FIELDS):

           inter_document  pool = every non-null value for the field from every
                            OTHER document in the split, so a line date is only
                            ever corrupted to a date that genuinely appears in
                            some other document.
           intra_document  values come exclusively from the row's OWN document's
                            OTHER line items, one distinct donor item per
                            corrupted field -- a donor used for one field cannot
                            be reused for another. Built as a chain: field/donor
                            pairs are drawn one at a time (field uniform over
                            those still open, donor uniform over those eligible
                            for it) until either k_hi fields are corrupted or no
                            further (field, donor) pair is eligible, at which
                            point the row's realized count -- never assumed to be
                            k_hi -- is what gets recorded. Guarded against
                            reproducing another valid row in the document
                            (up to ``equivalence_key``).

A corruption must differ from the original by ``equivalence_key`` (parsed date,
numeric amount, case- and whitespace-folded text): a reformat such as ``2/18/20``
-> ``02/18/20`` or ``ISSUE`` -> ``Issue`` is never an error.

OCR-fragment dates (``/08/20``, ``02/21/``) are ordinary values in every pool and
every error type. They have no position in time, so ``from <= to`` is enforced only
when BOTH dates parse (``pair_ok``). Order-filtering alone would over-select
fragments (they always pass), so ``pick_fragment_neutral`` draws the class
(fragment / real date) at the pool's own rate first; ``date_fragment_rates`` in
``invalids.json`` records the resulting corrupted-vs-kept rates per error type.

Field selection is uniform over the fields invalidatable for that row. The two
date fields are coupled: a value must keep ``from <= to`` (for every error type,
cross_type included; see above), so whether a date field can be corrupted depends on whether
its partner is corrupted too (and, for the intra-document chain, on the partner's
*current* -- possibly already-corrupted -- value). A consequence for cross_type:
two date fields of one record cannot be directly exchanged (that would reverse
their order) unless they hold equal dates, which the eligibility rule excludes.

No cluster job: this is local data preparation. Run it directly:

    uv run --python 3.12 python data/vrdu/build_vrdu_invalids.py \
        --config configs/2026-10-21-vrdu-invalids-04.yaml
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

# Canonical field lists come from the base-file builder (a sibling script; its
# directory is on sys.path) -- one source of truth, and the base-file row keys
# are asserted against them below.
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
    """Parse ``m/d/y`` (2- or 4-digit year) into a ``(year, month, day)`` tuple.

    Returns None for anything not exactly that shape -- OCR fragments such as
    ``"02/21/"`` or ``"/12/20"`` do not parse.
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
    """Date-order rule: ``from <= to``, enforced only when BOTH sides parse.

    A null or an OCR fragment (``parse_date`` -> None) has no position in time,
    so it neither violates nor blocks the order. Every error type, feasibility
    check and verifier uses this one rule.
    """
    return parsed_from is None or parsed_to is None or parsed_from <= parsed_to


@cache
def equivalence_key(kind, text, year_pivot):
    """Canonical form of a value: two values with the same key are the same entity.

    A reformat of the original (``2/18/20`` vs ``02/18/20``, ``$1,234.00`` vs
    ``1234``, ``ISSUE`` vs ``Issue``) is not an error, so it must never count as
    a corruption. Dates compare as parsed dates and amounts as numbers (by
    ``kind``); everything else, fragments included, as whitespace-collapsed
    case-folded text.
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
    """``eq(field, value)``: the equivalence key of ``value`` as a ``field`` value (None for None)."""
    def eq(field, value):
        return None if value is None else equivalence_key(kind_of.get(field), value, year_pivot)
    return eq


# --- Pools (main dataset + line inter-document) --------------------------------


def main_pools(base_rows):
    """One value list per main field: its non-null values, frequency-weighted.

    Date fields keep OCR fragments like every other value: the corrupted-date
    distribution must look like the valid-date distribution, or "is a fragment"
    would predict "is correct".
    """
    return {f: [r[f] for r in base_rows if r[f] is not None] for f in MAIN_FIELDS}


def line_inter_pools_by_doc(line_base, date_fields):
    """One value pool per document: every LINE_FIELD's non-null values, drawn
    exclusively from every OTHER document's line rows in the split.

    The two date fields share one pool -- every date value (fragments
    included) on any other document's line rows.
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
    return [
        f for f in target_fields
        if f not in date_fields and any(eq(f, v) != eq(f, row[f]) for v in pools[f])
    ]


def _date_candidates(row, field, pools, year_pivot, eq):
    """Parsed value (None = fragment) of every pool value that is not equivalent to ``row[field]``."""
    original = eq(field, row[field])
    return [parse_date(v, year_pivot) for v in pools[field] if eq(field, v) != original]


def _date_feasibility(row, date_fields, pools, year_pivot, eq):
    """Which date corruptions keep the pair ordered (see ``pair_ok``).

    Returns ``(start_alone, end_alone, both)``: whether the from-date can be
    changed with the to-date retained, vice versa, and whether both can be
    changed together. A fragment candidate (or a null / fragment partner) is
    never blocked by order.
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
    """Set of realizable ``num_invalid_fields`` values for a row."""
    start_alone, end_alone, both = date_feasibility
    date_counts = {0}
    if start_alone or end_alone:
        date_counts.add(1)
    if both:
        date_counts.add(2)
    return {a + d for a in range(n_nondate + 1) for d in date_counts}


def achievable_targets(base_row, target_fields, date_fields, pools, year_pivot, k_range, eq):
    """Sorted, distinct ``num_invalid_fields`` values this row can realize in ``k_range``.

    Line-side only (inter-document + the ceiling check for intra-document).
    Every value returned is genuinely achievable with this row's pools, so a
    caller looping over it (using the ordinary ``_choose_fields``/``make_invalid_row``,
    which self-cap to the achievable max <= the requested k) never actually
    triggers that capping -- the requested k is already the achieved one.
    """
    k_lo, k_hi = k_range
    feasible_nondate = _nondate_invalidatable(base_row, target_fields, date_fields, pools, eq)
    date_feasibility = _date_feasibility(base_row, date_fields, pools, year_pivot, eq)
    achievable = achievable_counts(len(feasible_nondate), date_feasibility)
    return sorted(x for x in achievable if k_lo <= x <= k_hi)


# --- Corruption (shared: main + line inter-document) ------------------------


def sample_new(pool, original, rng, eq, field):
    """First value of a random permutation of ``pool`` not equivalent to ``original``."""
    original_key = eq(field, original)
    for value in rng.sample(pool, len(pool)):
        if eq(field, value) != original_key:
            return value
    raise AssertionError(f"pool of {len(pool)} values has none differing from {original!r}")


def _choose_fields(feasible_nondate, date_fields, date_feasibility, k, rng):
    """Pick the fields to invalidate for one (row, k). Returns a sorted list.

    Realizes the largest achievable field count not exceeding k, then splits it
    between a uniformly chosen date-corruption option and a uniform sample of the
    non-date fields. (On the line side, callers only ever pass a k already in the
    achievable set -- see ``achievable_targets`` -- so this never actually caps
    there; main-dataset callers rely on the capping.)
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
    """Share of the non-null date values of ``rows`` that are OCR fragments (do not parse)."""
    values = [r[f] for r in rows for f in date_fields if r[f] is not None]
    assert values, "no date values to take a fragment rate from"
    return sum(1 for v in values if parse_date(v, year_pivot) is None) / len(values)


def pick_fragment_neutral(eligible, rate, is_fragment, rng):
    """Pick one of ``eligible`` so OCR fragments are drawn at ``rate``, not at their eligible share.

    A fragment passes every order filter and a real date only sometimes does, and the
    cross_type / intra_document pools (a document's own dates, minus values already valid for
    the target) are skewed toward fragments -- either would make "is a fragment" predict "was
    corrupted". So the class (fragment / real) is drawn at ``rate`` (the split's fragment rate
    among its valid date values, from ``fragment_rate``), then a value uniformly within it.
    Only when one class has no feasible value is the other forced.
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
    """Assign new value(s) to the chosen date field(s), keeping ``from <= to`` where both parse.

    Feasibility has already guaranteed a valid choice exists; the assertions here
    fire only on a logic error.
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
    """Build one invalid row from ``base_row``. Returns ``(new_row, invalid_fields)`` or None.

    On the main-dataset path ``k`` may exceed what the row supports, in which
    case ``_choose_fields`` caps to the achievable max (or returns ``[]``,
    signalled here as None) -- see ``generate_split``. Line-side callers only
    ever pass an already-achievable ``k`` (from ``achievable_targets``), so this
    never returns None or caps there.
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
    """``field_errors`` for one row: ``{field: None | {"error_type", "source"}}`` for every field.

    ``sources`` maps each corrupted field to its source dict; ``{}`` for a valid row.
    """
    assert set(sources) <= set(fields), sorted(set(sources) - set(fields))
    return {
        f: (None if f not in sources else {"error_type": error_type, "source": sources[f]})
        for f in fields
    }


def tag_row(row, fields, id_keys, k, valid, invalid_fields, error_type, sources):
    """Ordered output row: identity keys, label tags, then the data fields.

    ``error_type`` is None on a valid row; ``sources`` maps each of
    ``invalid_fields`` to its source dict (see the module docstring).
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
    """Return ``(output_rows, skipped, capped)`` for one main-dataset split.

    All emitted invalid rows are ``inter_document``; ``source_for(field, value)``
    gives each corrupted field's source dict.

    Unchanged main-dataset generation: for each k in ``k_range``, one invalid
    row per valid base row, capped (recorded) when the row does not support k
    invalidatable fields, skipped (recorded) when it supports none at all.
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
    """Boundary checks on one emitted main-dataset split. Raises on any inconsistency."""
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
    """Set ``source["null_fill"]`` on every corrupted entity: True when the valid row had no value there.

    A null field is rendered as absent, so filling one in is a spurious entity rather than a
    wrong value for a real one -- a different kind of error, tagged so it can be stratified.
    Written as an explicit True / False on every corrupted entity, never only when True.
    """
    for row in rows:
        if row["valid"]:
            continue
        base = base_by_id[tuple(row[k] for k in id_keys)]
        for f in row["invalid_fields"]:
            row["field_errors"][f]["source"]["null_fill"] = base[f] is None


def verify_field_errors(rows, fields, allowed_error_types):
    """``error_type`` / ``field_errors`` / ``invalid_fields`` agree on every row."""
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


def _intra_feasible(base_row, current_row, other_rows, other_tuples, target_fields, date_fields,
                     corrupted, used_donors, year_pivot, eq):
    """Fields still open to intra-document corruption at this step of the chain.

    Returns ``{field: [eligible donor index, ...]}`` for fields not yet
    corrupted with at least one not-yet-used donor supplying a non-null value
    that (a) is not equivalent to the row's ORIGINAL value (``eq``: a reformat is
    not an error), (b) for a date field, keeps ``from <= to`` against the partner
    date field's *current* (possibly already-corrupted) value whenever both
    parse (``pair_ok``), and (c) would not make the row equivalent to another
    valid row in the document -- a (field, donor) pair that WOULD reproduce one
    is simply excluded here, not fatal: two line items
    differing in only one field is plausible real data, and the chain should
    just decline that specific pair (or stop, if nothing else is left) rather
    than abort the whole build over it.
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
            hypothetical = tuple(
                eq(f, value if f == field else current_row[f]) for f in target_fields
            )
            if hypothetical in other_tuples:
                continue
            eligible.append(idx)
        if eligible:
            feasible[field] = eligible
    return feasible


def build_intra_chain(base_row, other_rows, target_fields, date_fields, year_pivot, k_hi, rng, eq, frag_rate):
    """Grow ``base_row`` into up to ``k_hi`` intra-document invalid variants.

    Each successive field corruption draws its value from a distinct,
    not-yet-used row in ``other_rows`` (other line items in the same document):
    field choice is uniform over the fields still open, donor choice uniform
    over the donors eligible for that field. Stops the moment no (field, donor)
    pair is eligible -- the returned list's length is the row's actual
    achievable count, never assumed to be ``k_hi``.

    Returns ``[(row_dict, invalid_fields, donors), ...]`` for num_invalid_fields
    1, 2, ..., achieved (empty list if the row has no usable donor at all);
    ``donors`` maps each corrupted field to its donor's ``line_index``.
    """
    other_tuples = {tuple(eq(f, r[f]) for f in target_fields) for r in other_rows}
    current = dict(base_row)
    corrupted = []
    used_donors = set()
    donors = {}
    results = []
    while len(corrupted) < k_hi:
        feasible = _intra_feasible(base_row, current, other_rows, other_tuples, target_fields,
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
        full_tuple = tuple(eq(f, snapshot[f]) for f in target_fields)
        assert full_tuple not in other_tuples, (
            f"intra-document corruption at {base_row.get('document_id')}/"
            f"{base_row.get('line_index')} reproduced another valid row verbatim "
            "(should have been excluded by _intra_feasible -- logic error)"
        )
        results.append((snapshot, sorted(corrupted), dict(donors)))
    return results


def line_inter_document(row, pool, target_fields, date_fields, year_pivot, k_range, rng, eq, frag_rate):
    """One invalid row per achievable num_invalid_fields for the inter-document type."""
    targets = achievable_targets(row, target_fields, date_fields, pool, year_pivot, k_range, eq)
    rows = []
    for k in targets:
        result = make_invalid_row(row, k, target_fields, date_fields, pool, year_pivot, rng, eq, frag_rate)
        assert result is not None and len(result[1]) == k, (
            f"k={k} came from achievable_targets so it must be exactly realizable"
        )
        rows.append(result)
    return rows


def generate_line_split(line_base, date_fields, year_pivot, k_range, rng, eq, frag_rate):
    """Build inter- and intra-document invalid line rows for one split.

    Returns ``(output_rows, skipped, capped)`` where ``skipped``/``capped`` are
    dicts keyed by ``"inter_document"``/``"intra_document"`` (cross_type is built
    separately, see ``generate_cross_type_rows``).
    """
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

    for doc_id, rows in by_doc.items():
        for row in rows:
            row_id = {"document_id": row["document_id"], "line_index": row["line_index"]}

            inter_rows = line_inter_document(row, inter_pools[doc_id], LINE_FIELDS, date_fields, year_pivot, k_range, rng, eq, frag_rate)
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
            chain = build_intra_chain(row, other_rows, LINE_FIELDS, date_fields, year_pivot, k_hi, rng, eq, frag_rate)
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

    return out, skipped, capped


def verify_line_output(out_rows, base_by_id, date_fields, year_pivot, line_by_doc, eq):
    """Boundary checks on one emitted line split. Raises on any inconsistency."""
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

        # Only checked for intra-document: its values are drawn directly from
        # another row in this document, so this must always hold by
        # construction -- a failure here means the wrong pool got wired in.
        # Not checked for inter-document: field values (channel numbers, dates)
        # legitimately recur across documents in this corpus, so "the drawn
        # value also happens to occur in this document" is not a bug signal
        # (see provenance's inter_document_coincidental_overlap for the count).
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

    for row in invalid_rows:
        if row["error_type"] != "intra_document":
            continue
        key = (row["document_id"], row["line_index"])
        full = tuple(eq(f, row[f]) for f in LINE_FIELDS)
        others = {
            tuple(eq(f, r[f]) for f in LINE_FIELDS) for r in line_by_doc[row["document_id"]]
            if (r["document_id"], r["line_index"]) != key
        }
        assert full not in others, f"{key}: intra-document row duplicates another valid row"


# --- cross_type ----------------------------------------------------------


def field_kinds(kinds):
    """``{field: kind}`` from ``{kind: [fields]}``. A field in two kinds is an error."""
    out = {}
    for kind, members in kinds.items():
        for f in members:
            assert f not in out, f"field {f!r} is in kinds {out[f]!r} and {kind!r}"
            assert f in MAIN_FIELDS or f in LINE_FIELDS, f"kinds: unknown field {f!r}"
            out[f] = kind
    return out


def document_truth(main_base, line_base, eq):
    """``{document_id: {field: set of equivalence keys of the non-null valid values}}`` over main + line fields."""
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


def cross_type_candidates(target, kinds, kind_of, main_row, line_rows, truth_doc, counts, eq):
    """Eligible ``[(value, source)]`` for filing a value under ``target``.

    Every non-null value of every other same-kind field of this document, in
    either dataset, except values equivalent to a valid value of ``target`` in
    this document (OCR-fragment dates are eligible, like any other value).
    ``counts`` accumulates how many candidates the rule removed.
    """
    kind = kind_of[target]
    out = []
    for other in kinds[kind]:
        if other == target:
            continue
        if other in MAIN_FIELDS:
            supplied = [(main_row[other], "main", None)]
        else:
            supplied = [(r[other], "line", r["line_index"]) for r in line_rows]
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
    return source["dataset"] == dataset and (dataset == "main" or source["line_index"] == row["line_index"])


def mark_exchanges(sources, dataset, row):
    """Set ``exchanged_with`` on each source: the field it was taken from, if that
    field (in this very record) was in turn filled from this field; else None."""
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
    """cross_type invalid rows for one dataset/split: one per achievable ``k`` in ``k_range``.

    Returns ``(rows, skipped, capped, stats)``. ``rng`` must be this
    dataset/split's own stream. Reuses ``achievable_targets`` /
    ``make_invalid_row`` with per-row pools of eligible same-kind values, so the
    field choice, ``k`` bookkeeping and date-order handling are those of the
    other error types; the source of each value is then drawn uniformly among the
    fields in this document holding it.
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
                                     truth[doc], stats["candidate_counts"], eq)
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
    """Every cross_type field is a real, same-kind value of its own document that
    is not (equivalent to) a value already valid for the field it was filed under."""
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
    """Per row type, how many date entities are OCR fragments, split by corrupted / kept.

    ``{"valid" | error_type: {"corrupted": n, "corrupted_fragment": n, "kept": n,
    "kept_fragment": n}}`` over the non-null date entities. If fragments were a
    label signal (a fragment predicting "correct", or "corrupted"), the
    corrupted fragment rate would differ from the kept / valid one.
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

        # --- main dataset (inter_document generation unchanged; cross_type appended) ---
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

        # --- line dataset (inter- + intra-document, then cross_type) ---
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

        out_rows, skipped, capped = generate_line_split(line_base, line_date_fields, year_pivot, line_k_range, rng, eq, line_frag_rate)
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

        # Informational, not asserted: how often an inter-document draw happens
        # to coincide with a value already present in the row's own document
        # (channel numbers and dates recur naturally across this corpus, so
        # some overlap is expected). A count near 100% would flag a crossed pool.
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
