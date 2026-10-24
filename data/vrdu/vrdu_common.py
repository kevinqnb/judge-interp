"""Helpers shared by the VRDU builders: dates, the equivalence key, the label rule, fragments, IO.

Moved unchanged from ``build_vrdu_invalids.py`` (-05).
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from functools import cache

import build_vrdu_dataset as base_builder

# Field lists: single source of truth in the base-file builder.
MAIN_FIELDS = base_builder.MAIN_FIELDS
LINE_FIELDS = base_builder.LINE_FIELDS
assert not set(MAIN_FIELDS) & set(LINE_FIELDS), "main/line field names collide"


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



# --- OCR fragments ------------------------------------------------------------


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



# --- IO -----------------------------------------------------------------------


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
