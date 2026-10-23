"""Select a matched evaluation subset of the VRDU invalid rows and write ``manifest.json``.

A base row qualifies when every error type of its dataset has a row at every k in ``k_values``;
each selected base row contributes its valid row plus one row per (error type, k). Every cell
then has the same size, base rows and k. The manifest records the sha256 of the row files and
``load_manifest_split`` refuses a stale one. Values are in ``params.manifest`` of the invalids config.

Run after ``build_vrdu_invalids.py``:

    uv run --python 3.12 python data/vrdu/build_vrdu_manifest.py \\
        --config data/vrdu/build-configs/2026-10-22-vrdu-invalids-05.yaml
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
    """A row's identity within a split.

    Args:
        row: a data row.
        dataset: ``"main"`` or ``"line"``.

    Returns:
        ``[document_id, line_index or None, error_type or None, k]``.
    """
    line_index = row["line_index"] if dataset == "line" else None
    return [row["document_id"], line_index, row["error_type"], row["k"]]


def base_id(row, dataset):
    """The id of a row's valid base row.

    Args:
        row: a data row.
        dataset: ``"main"`` or ``"line"``.

    Returns:
        ``(document_id,)`` or ``(document_id, line_index)``.
    """
    return tuple(row[k] for k in ID_KEYS[dataset])


def sha256_of(path):
    """Hex sha256 of a file.

    Args:
        path: the file.

    Returns:
        The digest.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def common_support(rows, dataset, k_values):
    """Base rows that have a row for every (error type, k).

    Args:
        rows: a split's rows.
        dataset: ``"main"`` or ``"line"``.
        k_values: the k values required.

    Returns:
        Sorted base ids.
    """
    needed = {(et, k) for et in invalids.ERROR_TYPES[dataset] for k in k_values}
    have = {}
    for row in rows:
        if not row["valid"]:
            have.setdefault(base_id(row, dataset), set()).add((row["error_type"], row["k"]))
    return sorted(i for i, cells in have.items() if needed <= cells)


def select_split(rows, dataset, k_values, n_base_rows, rng):
    """Select the matched rows of one dataset/split.

    Args:
        rows: the split's rows.
        dataset: ``"main"`` or ``"line"``.
        k_values: sorted, distinct k values.
        n_base_rows: number of base rows to sample (None = all of the common support).
        rng: ``random.Random`` for the sample.

    Returns:
        ``(keys, info)``: ``row_key`` lists (each base row's valid row first), and the support size,
        selected size and per-cell counts.
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
    """Check a manifest entry: equal cell sizes, unique rows; raises AssertionError.

    Args:
        entry: one dataset/split entry of the manifest.
        dataset: ``"main"`` or ``"line"``.
    """
    keys = [tuple(k) for k in entry["rows"]]
    assert len(keys) == len(set(keys)), "duplicate manifest rows"
    n = entry["n_base_rows"]
    expected_cells = {"valid": n, **{f"{et}|{k}": n for et in invalids.ERROR_TYPES[dataset] for k in entry["k_values"]}}
    assert entry["cells"] == dict(sorted(expected_cells.items())), (entry["cells"], expected_cells)
    assert len(keys) == sum(expected_cells.values())
    recount = Counter("valid" if k[2] is None else f"{k[2]}|{k[3]}" for k in keys)
    assert dict(recount) == expected_cells, "rows disagree with the recorded cells"


def load_manifest_split(data_root, dataset, split):
    """Read the manifest's rows for one dataset/split.

    Args:
        data_root: the ``data/vrdu`` directory.
        dataset: ``"main"`` or ``"line"``.
        split: ``"train"`` or ``"test"``.

    Returns:
        The set of ``row_key`` tuples.

    Raises:
        AssertionError: the row file changed since the manifest was built.
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
    """Keep the rows named by the manifest.

    Args:
        rows: a split's rows.
        dataset: ``"main"`` or ``"line"``.
        keys: from ``load_manifest_split``.

    Returns:
        The matching rows, in input order; asserts every key matched exactly once.
    """
    chosen = [r for r in rows if tuple(row_key(r, dataset)) in keys]
    assert len(chosen) == len(keys), f"{len(keys)} manifest keys but {len(chosen)} rows matched"
    return chosen


def load_config(path):
    """Read and validate the manifest part of the invalids config.

    Args:
        path: the YAML config.

    Returns:
        The parsed config.
    """
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
    """Build and write ``manifest.json`` for the config given on the CLI.

    Args:
        argv: CLI arguments (None = ``sys.argv``).
    """
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
