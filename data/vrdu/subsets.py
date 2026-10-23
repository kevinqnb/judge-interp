"""Load the shared random field subsets of VRDU line rows (``subsets.json``).

A sample is a randomly ordered subset of one line row's non-null fields. It is valid iff its
shown values are one real line item's (``build_vrdu_invalids.matches_item``).

Error types: a sample carries its row's ``error_type``; ``cross_item`` (subsets only) swaps the
values of two shown fields of a clean row (k=2).

    sample_id = <doc>|<line_index>|<error_type or "valid">|<k>|<f1>+<f2>+...[|<a>~<b>]

``load_subsets`` is the only reader. It refuses stale inputs and re-derives every sample from the rows.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from build_vrdu_dataset import LINE_FIELDS
from build_vrdu_invalids import field_kinds, make_eq, matches_item
from build_vrdu_manifest import MANIFEST_FILE, SPLITS, row_key, sha256_of

SUBSETS_FILE = "subsets.json"
INVALIDS_FILE = "invalids.json"
VALID_TOKEN = "valid"
CROSS_ITEM = "cross_item"
CROSS_ITEM_K = 2
SOURCES = ("manifest", "all_rows")
BALANCE_MODES = ("field_set_error_type",)
REQUIRED_PARAMS = {"seed", "source", "n_replicates", "samples_per_row", "min_subset_size", "n_per_error_type", "balance_by",
                   "max_error_type_share_deviation"}


def make_sample_id(row: dict, fields: list[str], swap: tuple[str, str] | None = None) -> str:
    """The id of a sample.

    Args:
        row: the sample's row (the clean row for a cross_item sample).
        fields: the sample's fields, in order.
        swap: the swapped pair ``(a, b)`` of a cross_item sample, else None.

    Returns:
        The ``sample_id`` string.
    """
    if swap is None:
        parts = [row["document_id"], str(row["line_index"]), row["error_type"] or VALID_TOKEN, str(row["k"]), "+".join(fields)]
    else:
        a, b = swap
        assert row["error_type"] is None, (row["document_id"], row["line_index"], "cross_item from a corrupted row")
        assert a in fields and b in fields and LINE_FIELDS.index(a) < LINE_FIELDS.index(b), (swap, fields)
        parts = [row["document_id"], str(row["line_index"]), CROSS_ITEM, str(CROSS_ITEM_K), "+".join(fields), f"{a}~{b}"]
    assert not any("|" in p for p in parts) and all("+" not in f and "~" not in f for f in fields), parts
    return "|".join(parts)


def parse_sample_id(sample_id: str) -> dict:
    """Inverse of ``make_sample_id``.

    Args:
        sample_id: a sample id.

    Returns:
        ``{document_id, line_index, error_type (None if valid), k, fields, swap (None unless cross_item)}``.
    """
    parts = sample_id.split("|")
    assert len(parts) in (5, 6), sample_id
    doc, line_index, et, k, fields = parts[:5]
    sample = {"document_id": doc, "line_index": int(line_index), "error_type": None if et == VALID_TOKEN else et,
              "k": int(k), "fields": fields.split("+"), "swap": None}
    assert (len(parts) == 6) == (et == CROSS_ITEM), sample_id
    if len(parts) == 6:
        a, b = parts[5].split("~")
        sample["swap"] = (a, b)
        assert sample["k"] == CROSS_ITEM_K, sample_id
        valid_row = {"document_id": doc, "line_index": sample["line_index"], "error_type": None}
        assert make_sample_id(valid_row, sample["fields"], sample["swap"]) == sample_id, sample_id
    else:
        assert make_sample_id(sample, sample["fields"]) == sample_id, sample_id
    return sample


def sample_values(row: dict, fields: list[str], swap: tuple[str, str] | None = None) -> dict[str, str]:
    """The values a sample shows.

    Args:
        row: the sample's row.
        fields: the sample's fields, in order.
        swap: the cross_item pair, or None.

    Returns:
        ``{field: value}`` in sample order, with ``swap``'s two values exchanged.
    """
    values = {f: row[f] for f in fields}
    if swap is not None:
        a, b = swap
        values[a], values[b] = row[b], row[a]
    assert all(v is not None for v in values.values()), (row["document_id"], row["line_index"], fields, "null shown")
    return values


def sample_invalid_fields(row: dict, fields: list[str], swap: tuple[str, str] | None = None) -> list[str]:
    """The corrupted fields of a sample.

    Args:
        row: the sample's row.
        fields: the sample's fields, in order.
        swap: the cross_item pair, or None.

    Returns:
        The swapped pair, or the shown fields corrupted in ``row``, in sample order.
    """
    if swap is not None:
        return [f for f in fields if f in swap]
    return [f for f in fields if f in row["invalid_fields"]]


def sample_label_valid(row: dict, fields: list[str], swap: tuple[str, str] | None = None) -> bool:
    """Whether a sample is valid (no shown field corrupted).

    Args:
        row, fields, swap: as in ``sample_invalid_fields``.

    Returns:
        True if valid.
    """
    return not sample_invalid_fields(row, fields, swap)


def swap_is_noop(row: dict, a: str, b: str, eq) -> bool:
    """Whether swapping two fields would leave a value unchanged.

    Args:
        row: the clean row.
        a, b: the two fields.
        eq: from ``build_vrdu_invalids.make_eq``.

    Returns:
        True if either field would keep an equivalent value.
    """
    return eq(a, row[b]) == eq(a, row[a]) or eq(b, row[a]) == eq(b, row[b])


def items_by_document(rows: list[dict]) -> dict[str, list[dict]]:
    """The real line items of each document.

    Args:
        rows: a split's line rows.

    Returns:
        ``{document_id: [valid line rows]}``.
    """
    out: dict[str, list[dict]] = {}
    for r in rows:
        if r["error_type"] is None:
            out.setdefault(r["document_id"], []).append(r)
    return out


def equivalence_from_invalids(data_root) -> dict:
    """The equivalence settings the rows were built with.

    Args:
        data_root: the ``data/vrdu`` directory.

    Returns:
        ``{kinds, year_pivot}`` from ``invalids.json``'s config snapshot.
    """
    params = json.loads((Path(data_root) / INVALIDS_FILE).read_text(encoding="utf-8"))["build"]["config_snapshot"]["params"]
    return {"kinds": params["cross_type"]["kinds"], "year_pivot": params["dates"]["year_pivot"]}


def make_sample_eq(equivalence: dict):
    """The equivalence function for recorded settings.

    Args:
        equivalence: ``{kinds, year_pivot}``.

    Returns:
        ``eq(field, value)`` (``build_vrdu_invalids.make_eq``).
    """
    assert set(equivalence) == {"kinds", "year_pivot"}, sorted(equivalence)
    return make_eq(field_kinds(equivalence["kinds"]), equivalence["year_pivot"])


def invalid_error_type_shares(error_types: list[str]) -> dict[str, float]:
    """Share of each error type.

    Args:
        error_types: the invalid samples' error types.

    Returns:
        ``{error_type: share}``, sorted by type.
    """
    assert error_types and all(e is not None for e in error_types)
    c = Counter(error_types)
    return {e: c[e] / len(error_types) for e in sorted(c)}


def check_error_type_shares(error_types: list[str], expected: list[str], tolerance: float) -> dict[str, float]:
    """Check that every error type has about an equal share of the invalid class.

    Args:
        error_types: the invalid samples' error types.
        expected: every error type that should appear.
        tolerance: allowed distance from ``1 / len(expected)``.

    Returns:
        The shares; raises AssertionError if one is off.
    """
    shares = invalid_error_type_shares(error_types)
    assert set(shares) <= set(expected), (sorted(shares), expected)
    for e in expected:
        dev = abs(shares.get(e, 0.0) - 1 / len(expected))
        assert dev <= tolerance, f"invalid-class share of {e} is {shares.get(e, 0.0):.3f}, {dev:.3f} from equal (tolerance {tolerance})"
    return shares


def _check_sample(sample: dict, row: dict, min_subset_size: int) -> None:
    """Check a sample's fields against its row; raises AssertionError.

    Args:
        sample: a parsed sample.
        row: its row.
        min_subset_size: the smallest allowed subset.
    """
    fields = sample["fields"]
    assert len(fields) >= min_subset_size, (sample["sample_id"], "subset smaller than min_subset_size")
    assert len(set(fields)) == len(fields) and set(fields) <= set(LINE_FIELDS), (sample["sample_id"], "bad fields")
    assert all(row[f] is not None for f in fields), (sample["sample_id"], "a field is null in its row")


def load_subsets(data_root, split: str, replicate: int) -> list[dict]:
    """Load one replicate of samples, each joined to its row and checked.

    Args:
        data_root: the ``data/vrdu`` directory.
        split: ``"train"`` or ``"test"``.
        replicate: index in ``range(n_replicates)``.

    Returns:
        Samples in file order: ``{sample_id, document_id, line_index, error_type, k, fields, swap, values,
        invalid_fields, label_valid, row}``.

    Raises:
        AssertionError: stale inputs, a sample that disagrees with its row or the label rule, or wrong balance.
    """
    data_root = Path(data_root)
    assert split in SPLITS, split
    blob = json.loads((data_root / SUBSETS_FILE).read_text(encoding="utf-8"))
    assert set(blob) == {"build", "line"}, sorted(blob)
    build = blob["build"]
    params = build["subsets_params"]
    assert set(params) == REQUIRED_PARAMS and params["source"] in SOURCES and params["balance_by"] in BALANCE_MODES, params
    assert isinstance(replicate, int) and 0 <= replicate < params["n_replicates"], (replicate, params["n_replicates"])

    rel = f"line/{split}.json"
    assert build["sha256"][rel] == sha256_of(data_root / rel), f"{SUBSETS_FILE} is stale for {rel}"
    if params["source"] == "manifest":
        assert build["sha256"][MANIFEST_FILE] == sha256_of(data_root / MANIFEST_FILE), f"{SUBSETS_FILE} is stale for {MANIFEST_FILE}"
    else:
        assert MANIFEST_FILE not in build["sha256"]
    assert build["equivalence"] == equivalence_from_invalids(data_root), (
        f"{SUBSETS_FILE} was built with a different equivalence than {INVALIDS_FILE} records")
    eq = make_sample_eq(build["equivalence"])
    assert CROSS_ITEM in build["error_types"], build["error_types"]

    entry = blob["line"][split]["replicates"][replicate]
    assert entry["replicate"] == replicate
    rows = json.loads((data_root / rel).read_text(encoding="utf-8"))
    assert not any(r["error_type"] == CROSS_ITEM for r in rows), f"{rel} has {CROSS_ITEM} rows: the id spaces collide"
    by_key = {tuple(row_key(r, "line")): r for r in rows}
    assert len(by_key) == len(rows), "line row keys are not unique"
    items = items_by_document(rows)

    samples = []
    for sid in entry["samples"]:
        s = parse_sample_id(sid)
        row_et, row_k = (None, 0) if s["swap"] is not None else (s["error_type"], s["k"])
        row = by_key[(s["document_id"], s["line_index"], row_et, row_k)]
        s["sample_id"] = sid
        _check_sample(s, row, params["min_subset_size"])
        if s["swap"] is not None:
            assert not swap_is_noop(row, *s["swap"], eq), (sid, "swap leaves a value unchanged")
        s["values"] = sample_values(row, s["fields"], s["swap"])
        s["invalid_fields"] = sample_invalid_fields(row, s["fields"], s["swap"])
        s["label_valid"] = not s["invalid_fields"]
        assert s["label_valid"] == matches_item(s["values"], items[s["document_id"]], eq), (
            sid, "label disagrees with the match rule")
        s["row"] = row
        samples.append(s)
    assert len({s["sample_id"] for s in samples}) == len(samples), "duplicate sample_id"
    per_type = params["n_per_error_type"][split]
    n = per_type * len(build["error_types"])
    assert sum(s["label_valid"] for s in samples) == n and sum(not s["label_valid"] for s in samples) == n, "class balance"
    assert not any(s["error_type"] is None and not s["label_valid"] for s in samples), "invalid sample from a valid row"
    assert sum(s["error_type"] == CROSS_ITEM for s in samples) == per_type, f"{CROSS_ITEM} count != n_per_error_type"
    # both classes have the same number of samples for every field set
    hist = {lab: Counter(frozenset(s["fields"]) for s in samples if s["label_valid"] == lab) for lab in (True, False)}
    assert hist[True] == hist[False], "field-set histograms of the two classes differ"
    check_error_type_shares([s["error_type"] for s in samples if not s["label_valid"]], build["error_types"],
                            params["max_error_type_share_deviation"])
    # one unordered subset per (row, error type) at most once: orders of the same set must not both appear
    keys = [(s["document_id"], s["line_index"], s["error_type"], s["k"], frozenset(s["fields"])) for s in samples]
    assert len(set(keys)) == len(keys), "the same unordered subset of a row appears twice"
    return samples
