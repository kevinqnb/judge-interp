"""Select a matched evaluation subset of the VRDU invalid rows and write ``manifest.json``.

The invalid rows are not comparable across error types as generated: the types have
different sizes, and the rows of each (error type, k) cell come from different base rows
(intra_document reaches k=5 for 3% of base rows, inter_document for all of them), so a
difference between types is confounded with document length. The manifest removes that:

  common support  a base row (a valid main/line row) qualifies when EVERY error type of
                  its dataset has a row at EVERY k in ``k_values``.
  pairing         each selected base row contributes its valid row plus exactly one row per
                  (error type, k in ``k_values``). Every cell therefore has the same size
                  and is built from the same base rows, with the same k (not a bucket).

``params.manifest`` in the invalids config holds ``seed`` and, per dataset, ``k_values`` and
``n_base_rows`` (per split; ``null`` = every common-support row, otherwise a seeded sample).
The manifest records the invalids config id and the sha256 of every ``{train,test}.json`` it
was drawn from; ``load_manifest_split`` refuses a manifest whose data has since changed.

No cluster job: this is local data preparation. Run it directly, after
``build_vrdu_invalids.py``:

    uv run --python 3.12 python data/vrdu/build_vrdu_manifest.py \\
        --config configs/2026-10-21-vrdu-invalids-04.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import build_vrdu_dataset as base_builder
import build_vrdu_invalids as invalids
import yaml

ID_KEYS = {"main": ["document_id"], "line": ["document_id", "line_index"]}
SPLITS = ("train", "test")
MANIFEST_FILE = "manifest.json"
REQUIRED_MANIFEST_KEYS = {"seed", "main", "line"}
REQUIRED_DATASET_KEYS = {"k_values", "n_base_rows"}


def row_key(row, dataset):
    """``[document_id, line_index | None, error_type | None, k]`` -- a row's identity within a split."""
    line_index = row["line_index"] if dataset == "line" else None
    return [row["document_id"], line_index, row["error_type"], row["k"]]


def base_id(row, dataset):
    return tuple(row[k] for k in ID_KEYS[dataset])


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def common_support(rows, dataset, k_values):
    """Sorted base ids having a row for every (error type, k) in ``k_values``."""
    needed = {(et, k) for et in invalids.ERROR_TYPES[dataset] for k in k_values}
    have = {}
    for row in rows:
        if not row["valid"]:
            have.setdefault(base_id(row, dataset), set()).add((row["error_type"], row["k"]))
    return sorted(i for i, cells in have.items() if needed <= cells)


def select_split(rows, dataset, k_values, n_base_rows, rng):
    """Matched row keys for one dataset/split.

    Returns ``(keys, info)``: ``keys`` are ``row_key`` lists (valid rows first per base
    row), ``info`` the support / selection sizes and the per-cell counts.
    """
    assert k_values and all(isinstance(k, int) and k >= 1 for k in k_values), k_values
    assert list(k_values) == sorted(set(k_values)), f"k_values must be sorted and distinct: {k_values}"
    support = common_support(rows, dataset, k_values)
    assert support, f"{dataset}: no base row has every error type at every k in {k_values}"
    if n_base_rows is None:
        selected = support
    else:
        assert isinstance(n_base_rows, int) and 1 <= n_base_rows <= len(support), (n_base_rows, len(support))
        selected = sorted(rng.sample(support, n_base_rows))

    by_key = {tuple(row_key(r, dataset)): r for r in rows}
    assert len(by_key) == len(rows), f"{dataset}: row keys are not unique"
    keys = []
    for bid in selected:
        doc, line_index = bid[0], (bid[1] if dataset == "line" else None)
        keys.append([doc, line_index, None, 0])
        for et in invalids.ERROR_TYPES[dataset]:
            keys.extend([doc, line_index, et, k] for k in k_values)
    for key in keys:
        assert tuple(key) in by_key, f"manifest row {key} is not in the split"
    cells = Counter("valid" if key[2] is None else f"{key[2]}|{key[3]}" for key in keys)
    return keys, {"n_common_support": len(support), "n_base_rows": len(selected), "cells": dict(sorted(cells.items()))}


def verify_split_entry(entry, dataset):
    """Cell sizes equal, rows unique, one valid row per base row. Raises on any inconsistency."""
    keys = [tuple(k) for k in entry["rows"]]
    assert len(keys) == len(set(keys)), "duplicate manifest rows"
    n = entry["n_base_rows"]
    expected_cells = {"valid": n, **{f"{et}|{k}": n for et in invalids.ERROR_TYPES[dataset] for k in entry["k_values"]}}
    assert entry["cells"] == dict(sorted(expected_cells.items())), (entry["cells"], expected_cells)
    assert len(keys) == sum(expected_cells.values())
    recount = Counter("valid" if k[2] is None else f"{k[2]}|{k[3]}" for k in keys)
    assert dict(recount) == expected_cells, "rows disagree with the recorded cells"


def load_manifest_split(data_root, dataset, split):
    """The manifest's row keys for ``dataset``/``split`` as a set of tuples.

    Raises:
        AssertionError: the manifest's recorded sha256 of ``<dataset>/<split>.json`` differs
            from the file now on disk (the invalids were regenerated after the manifest).
    """
    data_root = Path(data_root)
    manifest = json.loads((data_root / MANIFEST_FILE).read_text(encoding="utf-8"))
    rel = f"{dataset}/{split}.json"
    recorded = manifest["build"]["invalids_sha256"][rel]
    actual = sha256_of(data_root / rel)
    assert recorded == actual, f"{MANIFEST_FILE} is stale for {rel}: sha256 {recorded[:12]} != {actual[:12]}"
    entry = manifest["datasets"][dataset][split]
    verify_split_entry(entry, dataset)
    return {tuple(k) for k in entry["rows"]}


def apply_manifest(rows, dataset, keys):
    """The rows of ``rows`` named by ``keys`` (a ``load_manifest_split`` result), in input order.

    Raises:
        AssertionError: a manifest key has no row, or a row is matched twice.
    """
    chosen = [r for r in rows if tuple(row_key(r, dataset)) in keys]
    assert len(chosen) == len(keys), f"{len(keys)} manifest keys but {len(chosen)} rows matched"
    return chosen


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("id", "params"):
        assert key in config, f"config missing top-level key {key!r}"
    params = config["params"]
    assert "data_root" in params and "manifest" in params, "config params need data_root and manifest"
    manifest = params["manifest"]
    assert set(manifest) == REQUIRED_MANIFEST_KEYS, f"params.manifest keys {sorted(manifest)}"
    for dataset in ID_KEYS:
        assert set(manifest[dataset]) == REQUIRED_DATASET_KEYS, f"params.manifest.{dataset} keys"
        assert set(manifest[dataset]["n_base_rows"]) == set(SPLITS), f"params.manifest.{dataset}.n_base_rows keys"
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="configs/<invalids id>.yaml")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    params = config["params"]
    mparams = params["manifest"]
    data_root = Path(params["data_root"])
    assert data_root.is_dir(), f"no such directory: {data_root}"

    sha256 = {}
    datasets = {}
    for dataset in ID_KEYS:
        datasets[dataset] = {}
        k_values = list(mparams[dataset]["k_values"])
        for split in SPLITS:
            rel = f"{dataset}/{split}.json"
            rows = json.loads((data_root / rel).read_text(encoding="utf-8"))
            sha256[rel] = sha256_of(data_root / rel)
            rng = random.Random(f"{mparams['seed']}:{dataset}:{split}:manifest")
            keys, info = select_split(rows, dataset, k_values, mparams[dataset]["n_base_rows"][split], rng)
            entry = {"k_values": k_values, **info, "rows": keys}
            verify_split_entry(entry, dataset)
            datasets[dataset][split] = entry
            print(f"{dataset}/{split}: common support {info['n_common_support']} base rows, "
                  f"selected {info['n_base_rows']}, {len(keys)} rows, every cell {info['n_base_rows']}")

    sha, dirty = base_builder.git_state()
    out = {
        "build": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_sha": sha,
            "git_dirty": dirty,
            "script": "data/vrdu/build_vrdu_manifest.py",
            "config_id": config["id"],
            "manifest_params": mparams,
            "invalids_sha256": sha256,
        },
        "datasets": datasets,
    }
    (data_root / MANIFEST_FILE).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for dataset in ID_KEYS:
        for split in SPLITS:
            load_manifest_split(data_root, dataset, split)   # round-trip: hashes match, entries verify
    print(f"written to {data_root / MANIFEST_FILE}")


if __name__ == "__main__":
    main()
