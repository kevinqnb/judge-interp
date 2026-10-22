"""Load the shared random field subsets of VRDU line rows (``subsets.json``).

Line dataset only. A *sample* is an ordered subset of one line row's non-null fields: ``fields``
is f1..fn in a random order drawn at build time (for the chain-rule factorisation
P(f1..fn) = P(fn | f1..fn-1) ... P(f2 | f1) P(f1)). It is valid iff none of its fields is in the row's
``invalid_fields``. Samples are identified by ``sample_id``:

    <document_id>|<line_index>|<error_type or "valid">|<k>|<f1>+<f2>+...

``build_vrdu_subsets.py`` writes the file; ``load_subsets`` is its only reader. It refuses a file
whose recorded sha256 of ``line/<split>.json`` (or of ``manifest.json`` when the source was the
manifest) differs from the one on disk, and re-derives every sample's row, fields and label from the
rows.
"""

from __future__ import annotations

import json
from pathlib import Path

from build_vrdu_dataset import LINE_FIELDS
from build_vrdu_manifest import MANIFEST_FILE, SPLITS, row_key, sha256_of

SUBSETS_FILE = "subsets.json"
VALID_TOKEN = "valid"
SOURCES = ("manifest", "all_rows")
REQUIRED_PARAMS = {"seed", "source", "n_replicates", "samples_per_row", "min_subset_size", "n_per_class"}


def make_sample_id(row: dict, fields: list[str]) -> str:
    """The ``sample_id`` of ``fields`` (in sample order) drawn from the line ``row``."""
    parts = [row["document_id"], str(row["line_index"]), row["error_type"] or VALID_TOKEN, str(row["k"]), "+".join(fields)]
    assert not any("|" in p for p in parts) and all("+" not in f for f in fields), parts
    return "|".join(parts)


def parse_sample_id(sample_id: str) -> dict:
    """Inverse of ``make_sample_id``: ``document_id``, ``line_index``, ``error_type`` (None if valid), ``k``, ``fields``."""
    doc, line_index, et, k, fields = sample_id.split("|")
    sample = {"document_id": doc, "line_index": int(line_index), "error_type": None if et == VALID_TOKEN else et,
              "k": int(k), "fields": fields.split("+")}
    assert make_sample_id(sample, sample["fields"]) == sample_id, sample_id
    return sample


def sample_label_valid(row: dict, fields: list[str]) -> bool:
    """True iff no field of the sample is corrupted in ``row``."""
    return not (set(fields) & set(row["invalid_fields"]))


def _check_sample(sample: dict, row: dict, min_subset_size: int) -> None:
    fields = sample["fields"]
    assert len(fields) >= min_subset_size, (sample["sample_id"], "subset smaller than min_subset_size")
    assert len(set(fields)) == len(fields) and set(fields) <= set(LINE_FIELDS), (sample["sample_id"], "bad fields")
    assert all(row[f] is not None for f in fields), (sample["sample_id"], "a field is null in its row")


def load_subsets(data_root, split: str, replicate: int) -> list[dict]:
    """The samples of one replicate, each joined to its row.

    Args:
        data_root: the ``data/vrdu`` directory.
        split: ``"train"`` or ``"test"``.
        replicate: replicate index in ``range(n_replicates)``.

    Returns:
        Samples in file order. Each is ``{sample_id, document_id, line_index, error_type, k, fields,
        label_valid, row}``, where ``fields`` is the sample's field order and ``row`` the line row.

    Raises:
        AssertionError: the file is stale, a sample does not match its row, or the class balance is wrong.
    """
    data_root = Path(data_root)
    assert split in SPLITS, split
    blob = json.loads((data_root / SUBSETS_FILE).read_text(encoding="utf-8"))
    assert set(blob) == {"build", "line"}, sorted(blob)
    build = blob["build"]
    params = build["subsets_params"]
    assert set(params) == REQUIRED_PARAMS and params["source"] in SOURCES, params
    assert isinstance(replicate, int) and 0 <= replicate < params["n_replicates"], (replicate, params["n_replicates"])

    rel = f"line/{split}.json"
    assert build["sha256"][rel] == sha256_of(data_root / rel), f"{SUBSETS_FILE} is stale for {rel}"
    if params["source"] == "manifest":
        assert build["sha256"][MANIFEST_FILE] == sha256_of(data_root / MANIFEST_FILE), f"{SUBSETS_FILE} is stale for {MANIFEST_FILE}"
    else:
        assert MANIFEST_FILE not in build["sha256"]

    entry = blob["line"][split]["replicates"][replicate]
    assert entry["replicate"] == replicate
    rows = json.loads((data_root / rel).read_text(encoding="utf-8"))
    by_key = {tuple(row_key(r, "line")): r for r in rows}
    assert len(by_key) == len(rows), "line row keys are not unique"

    samples = []
    for sid in entry["samples"]:
        s = parse_sample_id(sid)
        row = by_key[(s["document_id"], s["line_index"], s["error_type"], s["k"])]
        s["sample_id"] = sid
        _check_sample(s, row, params["min_subset_size"])
        s["label_valid"] = sample_label_valid(row, s["fields"])
        s["row"] = row
        samples.append(s)
    assert len({s["sample_id"] for s in samples}) == len(samples), "duplicate sample_id"
    n = params["n_per_class"][split]
    assert sum(s["label_valid"] for s in samples) == n and sum(not s["label_valid"] for s in samples) == n, "class balance"
    assert not any(s["error_type"] is None and not s["label_valid"] for s in samples), "invalid sample from a valid row"
    # one unordered subset per row at most once: orders of the same set must not both appear
    keys = [(s["document_id"], s["line_index"], s["error_type"], s["k"], frozenset(s["fields"])) for s in samples]
    assert len(set(keys)) == len(keys), "the same unordered subset of a row appears twice"
    return samples
