"""Populate the VRDU datasets with deliberately-invalid extractions.

Reads the four ``*_base.json`` files (every row a correct extraction, ``valid:
true``) and writes, under ``<data_root>/``:

    main/{train,test}.json   valid base rows + generated invalid rows
    line/{train,test}.json   valid base rows + generated invalid rows
    invalids.json            provenance: config snapshot, seed, pool sizes, counts,
                             and every capped / skipped row

An invalid row keeps the ``document_id`` (and ``line_index``) of the valid row it
was derived from, and differs from it in exactly ``num_invalid_fields`` fields,
named in ``invalid_fields``. ``k`` is the requested field count for the generating
loop iteration (0 on a valid row); ``num_invalid_fields`` is the realized count,
smaller than ``k`` when the row does not have k invalidatable fields.
``(*id_keys, k)`` is a unique key over each split.

  main   pool = every non-null value for the field within the split being written.
         Date fields also carry two synthetic sentinels so an ordered pair always
         exists. k ranges over ``main.k_range``.
  line   pool = every non-null value for the field within the same document; only
         the line-item fields are invalidated, carried context stays fixed. The
         two date fields share one pool: every parseable date appearing anywhere
         on the page -- both program-date columns plus the document's
         ``flight_from`` / ``flight_to``. No sentinels: a line date is only
         corrupted to another date that really appears on that page, or left
         alone. k ranges over ``line.k_range``.

Field selection is uniform over the fields invalidatable for that row. The two
date fields are coupled: a value must keep ``from <= to``, so whether a date field
can be corrupted depends on whether its partner is corrupted too. When k exceeds
what the row supports, the realized count is capped (recorded); a row that
supports nothing is skipped (recorded).

No cluster job: this is local data preparation. Run it directly:

    uv run --python 3.12 python scripts/build_vrdu_invalids.py \
        --config configs/2026-09-10-vrdu-invalids-01.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

import build_vrdu_dataset as base_builder
import yaml

# Canonical field lists come from the base-file builder (a sibling script; its
# directory is on sys.path) -- one source of truth, and the base-file row keys
# are asserted against them below.
MAIN_FIELDS = base_builder.MAIN_FIELDS
LINE_FIELDS = base_builder.LINE_FIELDS
LINE_CARRIED_FIELDS = base_builder.LINE_CARRIED_FIELDS

REQUIRED_PARAM_KEYS = {"data_root", "main", "line", "corrupt_from_null", "corrupt_to_null", "dates"}
REQUIRED_DATE_KEYS = {"main_fields", "line_fields", "line_context_fields", "year_pivot", "main_sentinels"}


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


# --- Pools -------------------------------------------------------------------


def main_pools(base_rows, sentinels, date_fields, year_pivot):
    """One value list per main field: its non-null values, frequency-weighted.

    Date fields keep only parseable values, plus the sentinels, so an OCR
    fragment is never emitted as an invalid date and an ordered pair always
    exists.
    """
    pools = {}
    for field in MAIN_FIELDS:
        non_null = [r[field] for r in base_rows if r[field] is not None]
        if field in date_fields:
            parseable = [v for v in non_null if parse_date(v, year_pivot) is not None]
            pools[field] = parseable + list(sentinels)
        else:
            pools[field] = non_null
    return pools


def line_pools(doc_rows, main_row, context_fields, date_fields, year_pivot):
    """Value lists for one document's line fields.

    Non-date fields: the field's non-null in-document values. The two date fields
    share one pool -- every parseable date that appears anywhere on the page:
    both program-date columns across all line items plus the document's
    ``context_fields`` (``flight_from`` / ``flight_to``). Frequency-weighted; no
    sentinels.
    """
    page_dates = []
    for field in date_fields:
        page_dates += [r[field] for r in doc_rows if r[field] is not None]
    page_dates += [main_row[f] for f in context_fields if main_row[f] is not None]
    page_dates = [v for v in page_dates if parse_date(v, year_pivot) is not None]

    pools = {}
    for field in LINE_FIELDS:
        if field in date_fields:
            pools[field] = page_dates
        else:
            pools[field] = [r[field] for r in doc_rows if r[field] is not None]
    return pools


# --- Feasibility -----------------------------------------------------------


def _nondate_invalidatable(row, target_fields, date_fields, pools):
    return [
        f for f in target_fields
        if f not in date_fields and any(v != row[f] for v in pools[f])
    ]


def _date_feasibility(row, date_fields, pools, year_pivot):
    """Which date corruptions keep the pair ordered.

    Returns ``(start_alone, end_alone, both)``: whether the from-date can be
    changed with the to-date retained, vice versa, and whether both can be
    changed together. Date pools contain only parseable values, so every
    candidate parses.
    """
    from_field, to_field = date_fields
    s0, e0 = row[from_field], row[to_field]
    parsed_s, parsed_e = parse_date(s0, year_pivot), parse_date(e0, year_pivot)
    fragment_s = s0 is not None and parsed_s is None
    fragment_e = e0 is not None and parsed_e is None

    from_candidates = [parse_date(v, year_pivot) for v in pools[from_field] if v != s0]
    to_candidates = [parse_date(v, year_pivot) for v in pools[to_field] if v != e0]
    earliest_from = min(from_candidates) if from_candidates else None
    latest_to = max(to_candidates) if to_candidates else None

    start_alone = (
        not fragment_e and earliest_from is not None
        and (parsed_e is None or earliest_from <= parsed_e)
    )
    end_alone = (
        not fragment_s and latest_to is not None
        and (parsed_s is None or latest_to >= parsed_s)
    )
    both = earliest_from is not None and latest_to is not None and earliest_from <= latest_to
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


# --- Corruption ----------------------------------------------------------


def sample_new(pool, original, rng):
    """First value of a random permutation of ``pool`` that differs from ``original``."""
    for value in rng.sample(pool, len(pool)):
        if value != original:
            return value
    raise AssertionError(f"pool of {len(pool)} values has none differing from {original!r}")


def _choose_fields(feasible_nondate, date_fields, date_feasibility, k, rng):
    """Pick the fields to invalidate for one (row, k). Returns a sorted list.

    Realizes the largest achievable field count not exceeding k, then splits it
    between a uniformly chosen date-corruption option and a uniform sample of the
    non-date fields.
    """
    from_field, to_field = date_fields
    start_alone, end_alone, both = date_feasibility
    achievable = achievable_counts(len(feasible_nondate), date_feasibility)
    j = max((x for x in achievable if x <= k), default=0)
    if j == 0:
        return []

    date_options = [frozenset()]
    if start_alone:
        date_options.append(frozenset([from_field]))
    if end_alone:
        date_options.append(frozenset([to_field]))
    if both:
        date_options.append(frozenset([from_field, to_field]))
    usable = sorted(
        (option for option in date_options if 0 <= j - len(option) <= len(feasible_nondate)),
        key=sorted,
    )
    chosen_dates = usable[rng.randrange(len(usable))]
    chosen_nondate = rng.sample(sorted(feasible_nondate), j - len(chosen_dates))
    return sorted([*chosen_nondate, *chosen_dates])


def _corrupt_dates(new_row, chosen_dates, row, date_fields, pools, year_pivot, rng):
    """Assign new value(s) to the chosen date field(s), keeping ``from <= to``.

    Feasibility has already guaranteed a valid choice exists; the assertions here
    fire only on a logic error.
    """
    from_field, to_field = date_fields
    s0, e0 = row[from_field], row[to_field]

    def pick(candidates):
        assert candidates, "no date candidate despite feasibility"
        return candidates[rng.randrange(len(candidates))]

    if chosen_dates == frozenset([from_field]):
        bound = parse_date(e0, year_pivot)  # None -> to-date is null, unconstrained
        new_row[from_field] = pick([
            v for v in pools[from_field]
            if v != s0 and (bound is None or parse_date(v, year_pivot) <= bound)
        ])
    elif chosen_dates == frozenset([to_field]):
        bound = parse_date(s0, year_pivot)
        new_row[to_field] = pick([
            w for w in pools[to_field]
            if w != e0 and (bound is None or parse_date(w, year_pivot) >= bound)
        ])
    else:
        latest_to = max(parse_date(w, year_pivot) for w in pools[to_field] if w != e0)
        new_from = pick([
            v for v in pools[from_field] if v != s0 and parse_date(v, year_pivot) <= latest_to
        ])
        parsed_from = parse_date(new_from, year_pivot)
        new_to = pick([
            w for w in pools[to_field] if w != e0 and parse_date(w, year_pivot) >= parsed_from
        ])
        new_row[from_field], new_row[to_field] = new_from, new_to


def make_invalid_row(base_row, k, target_fields, date_fields, pools, year_pivot, rng):
    """Build one invalid row from ``base_row``. Returns ``(new_row, invalid_fields)`` or None."""
    feasible_nondate = _nondate_invalidatable(base_row, target_fields, date_fields, pools)
    date_feasibility = _date_feasibility(base_row, date_fields, pools, year_pivot)
    chosen = _choose_fields(feasible_nondate, date_fields, date_feasibility, k, rng)
    if not chosen:
        return None

    new_row = dict(base_row)
    for field in chosen:
        if field not in date_fields:
            new_row[field] = sample_new(pools[field], base_row[field], rng)
    chosen_dates = frozenset(f for f in chosen if f in date_fields)
    if chosen_dates:
        _corrupt_dates(new_row, chosen_dates, base_row, date_fields, pools, year_pivot, rng)

    changed = sorted(f for f in base_row if new_row[f] != base_row[f])
    assert changed == chosen, f"changed {changed} != chosen {chosen} ({base_row.get('document_id')})"
    return new_row, chosen


# --- Row emission -------------------------------------------------------


def tag_row(row, fields, id_keys, k, valid, invalid_fields):
    """Ordered output row: identity keys, validity tags, then the data fields."""
    tagged = {key: row[key] for key in id_keys}
    tagged["valid"] = valid
    tagged["k"] = k
    tagged["num_invalid_fields"] = len(invalid_fields)
    tagged["invalid_fields"] = list(invalid_fields)
    for field in fields:
        tagged[field] = row[field]
    return tagged


def generate_split(base_rows, fields, target_fields, id_keys, k_range,
                   pool_for_row, date_fields, year_pivot, rng):
    """Return ``(output_rows, skipped, capped)`` for one split."""
    k_lo, k_hi = k_range
    assert 1 <= k_lo <= k_hi <= len(target_fields), f"k_range {k_range} outside 1..{len(target_fields)}"

    out = [tag_row(row, fields, id_keys, 0, True, []) for row in base_rows]
    skipped, capped = [], []
    for base_row in base_rows:
        pools = pool_for_row(base_row)
        row_id = {key: base_row[key] for key in id_keys}
        row_caps, produced = [], 0
        for k in range(k_lo, k_hi + 1):
            result = make_invalid_row(base_row, k, target_fields, date_fields, pools, year_pivot, rng)
            if result is None:
                row_caps.append({**row_id, "k": k, "realized": 0})
                continue
            new_row, invalid_fields = result
            produced += 1
            if len(invalid_fields) < k:
                row_caps.append({**row_id, "k": k, "realized": len(invalid_fields)})
            out.append(tag_row(new_row, fields, id_keys, k, False, invalid_fields))
        if produced:
            capped.extend(row_caps)  # a row that produced nothing at all is skipped, not capped
        else:
            skipped.append(row_id)
    return out, skipped, capped


# --- Verification --------------------------------------------------------


def verify_output(out_rows, base_by_id, fields, target_fields, id_keys, k_hi, date_fields, year_pivot):
    """Boundary checks on one emitted split. Raises on any inconsistency."""
    valid_rows = [r for r in out_rows if r["valid"]]
    invalid_rows = [r for r in out_rows if not r["valid"]]
    carried = [f for f in fields if f not in target_fields]

    assert len(valid_rows) == len(base_by_id), (len(valid_rows), len(base_by_id))
    for row in valid_rows:
        key = tuple(row[k] for k in id_keys)
        assert row["k"] == 0 and row["num_invalid_fields"] == 0 and row["invalid_fields"] == []
        assert {f: row[f] for f in fields} == {f: base_by_id[key][f] for f in fields}

    identity = [(*[row[k] for k in id_keys], row["k"]) for row in out_rows]
    assert len(identity) == len(set(identity)), "output rows not uniquely keyed by (*id, k)"

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
        for field in carried:
            assert row[field] == base[field], f"{key}: carried field {field} changed"

    from_field, to_field = date_fields
    for row in invalid_rows:
        if from_field not in row["invalid_fields"] and to_field not in row["invalid_fields"]:
            continue
        start, end = row[from_field], row[to_field]
        if start is None or end is None:
            continue
        parsed_start, parsed_end = parse_date(start, year_pivot), parse_date(end, year_pivot)
        key = tuple(row[k] for k in id_keys)
        assert parsed_start is not None and parsed_end is not None, (key, start, end)
        assert parsed_start <= parsed_end, (key, start, end)


# --- Config / IO --------------------------------------------------------------


def _git(*args):
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=Path(__file__).resolve().parent
    )
    assert result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.strip()}"
    return result.stdout


def git_state():
    return _git("rev-parse", "HEAD").strip(), bool(_git("status", "--porcelain").strip())


def load_config(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("id", "project", "seed", "params"):
        assert key in config, f"config missing top-level key {key!r}"
    params = config["params"]
    assert not REQUIRED_PARAM_KEYS - set(params), f"config params missing {sorted(REQUIRED_PARAM_KEYS - set(params))}"
    assert not REQUIRED_DATE_KEYS - set(params["dates"]), (
        f"config params.dates missing {sorted(REQUIRED_DATE_KEYS - set(params['dates']))}"
    )
    for scope_key in ("main", "line"):
        assert set(params[scope_key]) == {"k_range", "pool_scope"}, f"params.{scope_key} keys"
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
    sentinels = list(dates["main_sentinels"])
    main_date_fields = list(dates["main_fields"])
    line_date_fields = list(dates["line_fields"])
    context_fields = list(dates["line_context_fields"])
    assert set(main_date_fields) <= set(MAIN_FIELDS) and len(main_date_fields) == 2
    assert set(line_date_fields) <= set(LINE_FIELDS) and len(line_date_fields) == 2
    assert set(context_fields) <= set(MAIN_FIELDS)
    assert all(parse_date(s, year_pivot) is not None for s in sentinels), "unparseable sentinel"
    assert params["corrupt_from_null"] is True and params["corrupt_to_null"] is False, (
        "this build assumes corrupt_from_null=true, corrupt_to_null=false"
    )
    assert params["main"]["pool_scope"] == "per_split" and params["line"]["pool_scope"] == "per_document"

    data_root = Path(params["data_root"])
    assert data_root.is_dir(), f"no such directory: {data_root}"

    rng = random.Random(seed)
    provenance = {"counts": {"main": {}, "line": {}},
                  "num_invalid_fields_dist": {"main": {}, "line": {}},
                  "invalid_field_freq": {"main": {}, "line": {}},
                  "skipped_rows": {"main": {}, "line": {}},
                  "capped_rows": {"main": {}, "line": {}},
                  "main_pool_sizes": {}, "line_date_pool_sizes": {}}

    for split in ("train", "test"):
        main_base = read_base(data_root / "main" / f"{split}_base.json", MAIN_FIELDS, ["document_id"])
        main_by_id = {r["document_id"]: r for r in main_base}
        line_fields = [*LINE_CARRIED_FIELDS, *LINE_FIELDS]
        line_base = read_base(data_root / "line" / f"{split}_base.json", line_fields, ["document_id", "line_index"])

        line_docs = sorted({r["document_id"] for r in line_base})
        assert set(line_docs) <= set(main_by_id), "line documents missing from the main base"

        pools_main = main_pools(main_base, sentinels, main_date_fields, year_pivot)
        line_by_doc = {}
        for row in line_base:
            line_by_doc.setdefault(row["document_id"], []).append(row)
        pools_line = {
            doc: line_pools(rows, main_by_id[doc], context_fields, line_date_fields, year_pivot)
            for doc, rows in line_by_doc.items()
        }

        provenance["main_pool_sizes"][split] = {f: len(p) for f, p in pools_main.items()}
        date_sizes = [len(set(p[line_date_fields[0]])) for p in pools_line.values()]
        empty_docs = [doc for doc, p in pools_line.items() if not p[line_date_fields[0]]]
        provenance["line_date_pool_sizes"][split] = {
            "min": min(date_sizes), "median": statistics.median(date_sizes), "max": max(date_sizes),
            "docs_with_no_page_date": len(empty_docs),
            "line_rows_with_no_page_date": sum(len(line_by_doc[doc]) for doc in empty_docs),
        }

        for kind, base_rows, fields, target_fields, id_keys, date_fields, k_range, pool_for_row in (
            ("main", main_base, MAIN_FIELDS, MAIN_FIELDS, ["document_id"], main_date_fields,
             params["main"]["k_range"], lambda _r, _p=pools_main: _p),
            ("line", line_base, line_fields, LINE_FIELDS, ["document_id", "line_index"], line_date_fields,
             params["line"]["k_range"], lambda r, _p=pools_line: _p[r["document_id"]]),
        ):
            out_rows, skipped, capped = generate_split(
                base_rows, fields, target_fields, id_keys, k_range, pool_for_row, date_fields, year_pivot, rng
            )
            base_by_id = {tuple(r[k] for k in id_keys): r for r in base_rows}
            verify_output(out_rows, base_by_id, fields, target_fields, id_keys,
                          k_range[1], date_fields, year_pivot)

            (data_root / kind / f"{split}.json").write_text(
                json.dumps(out_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            n_valid = sum(1 for r in out_rows if r["valid"])
            provenance["counts"][kind][split] = {
                "valid": n_valid, "invalid": len(out_rows) - n_valid, "total": len(out_rows)
            }
            provenance["num_invalid_fields_dist"][kind][split] = dict(
                sorted(Counter(r["num_invalid_fields"] for r in out_rows).items())
            )
            field_freq = Counter(f for r in out_rows for f in r["invalid_fields"])
            provenance["invalid_field_freq"][kind][split] = {f: field_freq[f] for f in target_fields}
            provenance["skipped_rows"][kind][split] = skipped
            provenance["capped_rows"][kind][split] = capped
            print(
                f"{kind}/{split}: {n_valid} valid + {len(out_rows) - n_valid} invalid "
                f"= {len(out_rows)}  (skipped {len(skipped)}, capped {len(capped)})"
            )

    sha, dirty = git_state()
    (data_root / "invalids.json").write_text(
        json.dumps(
            {"build": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git_sha": sha,
                "git_dirty": dirty,
                "python": sys.version.split()[0],  # random.sample order is not guaranteed stable across CPython
                "script": "scripts/build_vrdu_invalids.py",
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
