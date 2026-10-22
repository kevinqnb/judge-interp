"""Draw the shared random field subsets of VRDU line rows and write ``subsets.json``.

Clef and the probe are evaluated on the same samples, and the average over random subsets is
kept by drawing several independent replicates. Line dataset only.

  sample      an ordered subset of one line row's non-null fields. ``fields`` is f1..fn in a
              random order (for the chain-rule factorisation P(f1..fn) = P(fn|f1..fn-1)...P(f1)).
              Valid iff none of its fields is in the row's ``invalid_fields``, so a subset of an
              invalid row that avoids every corrupted field is a valid sample.
  draw        per source row, ``samples_per_row`` distinct *unordered* subsets of size
              >= ``min_subset_size`` (uniform, without replacement; all of them if fewer exist),
              then a uniformly random permutation of each. Ordered tuples are not enumerated,
              which would weight a subset by its n!.
  balance     each replicate is cut to exactly ``n_per_class`` valid and ``n_per_class`` invalid
              samples, at random. ``balance_by: none`` balances the classes only. A larger subset is
              more likely to contain a corrupted field, so subset size and field presence then predict the
              label (test AUROC ~0.63 / ~0.65). ``balance_by: field_set`` gives both classes exactly the
              same number of samples for every set of fields (hence the same sizes): the target share of a
              set is its share among all subsets (size >= ``min_subset_size``, equal weight) of the source
              *valid* rows, so it depends only on which fields real line items have, not on the corruption,
              the error types or ``samples_per_row``; counts use largest-remainder rounding and a set
              whose valid or invalid pool is too small raises. The invalid class keeps the error types in
              (nearly) equal shares, as the manifest's cells are; ``max_error_type_share_deviation`` bounds
              the drift and is asserted here and by the loader. (``cross_type`` never touches ``channel`` /
              ``program_desc``, so a set made of those two has no cross_type invalid samples.)
  replicates  replicate r of split s uses ``default_rng([seed, split index, r])``: independent of
              the others, and raising ``n_replicates`` leaves replicates 0..R-1 unchanged.

``params.subsets`` in the invalids config holds every value (see there). The file stores only
``sample_id`` strings (see ``subsets.py``) and the sha256 of every input it was drawn from;
``subsets.load_subsets`` refuses a file whose inputs have since changed.

No cluster job: this is local data preparation. Run it directly, after
``build_vrdu_manifest.py``:

    uv run --python 3.12 python data/vrdu/build_vrdu_subsets.py \\
        --config data/vrdu/build-configs/2026-10-21-vrdu-invalids-04.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import build_vrdu_dataset as base_builder
import numpy as np
import yaml
from build_vrdu_manifest import MANIFEST_FILE, SPLITS, apply_manifest, load_manifest_split, row_key, sha256_of
from subsets import (BALANCE_MODES, LINE_FIELDS, REQUIRED_PARAMS, SOURCES, SUBSETS_FILE, check_error_type_shares,
                     load_subsets, make_sample_id, sample_label_valid)


def source_rows(rows: list[dict], source: str, data_root: Path, split: str) -> list[dict]:
    """The rows samples are drawn from, sorted by ``row_key`` (so the draw does not depend on file order)."""
    assert source in SOURCES, source
    if source == "manifest":
        rows = apply_manifest(rows, "line", load_manifest_split(data_root, "line", split))
    key = lambda r: (r["document_id"], r["line_index"], r["error_type"] or "", r["k"])
    return sorted(rows, key=key)


def draw_samples(rows: list[dict], samples_per_row: int, min_subset_size: int, rng: np.random.Generator) -> tuple[list[dict], dict]:
    """Random ordered subsets of each row's non-null fields.

    Returns:
        ``(samples, stats)``: samples ``{sample_id, row, fields, label_valid}`` in row order, and the counts of
        rows too small for ``min_subset_size`` and rows with fewer distinct subsets than ``samples_per_row``.
    """
    assert min_subset_size >= 1 and samples_per_row >= 1
    samples, stats = [], {"n_source_rows": len(rows), "n_rows_too_small": 0, "n_rows_fewer_subsets_than_requested": 0}
    for row in rows:
        present = [f for f in LINE_FIELDS if row[f] is not None]
        if len(present) < min_subset_size:
            stats["n_rows_too_small"] += 1
            continue
        subsets = [c for r in range(min_subset_size, len(present) + 1) for c in itertools.combinations(present, r)]
        take = min(samples_per_row, len(subsets))
        stats["n_rows_fewer_subsets_than_requested"] += take < samples_per_row
        for j in rng.choice(len(subsets), size=take, replace=False):
            subset = subsets[int(j)]
            fields = [subset[i] for i in rng.permutation(len(subset))]
            assert sorted(fields) == sorted(subset)
            samples.append({"sample_id": make_sample_id(row, fields), "row": row, "fields": fields,
                            "label_valid": sample_label_valid(row, fields)})
    assert samples, "no samples drawn"
    return samples, stats


def balance(samples: list[dict], n_per_class: int, rng: np.random.Generator) -> list[dict]:
    """Exactly ``n_per_class`` valid and ``n_per_class`` invalid samples, random; draw order kept."""
    assert isinstance(n_per_class, int) and n_per_class >= 1
    keep = []
    for lab, name in ((True, "valid"), (False, "invalid")):
        pool = [i for i, s in enumerate(samples) if s["label_valid"] == lab]
        if len(pool) < n_per_class:
            raise ValueError(f"only {len(pool)} {name} samples available, {n_per_class} requested")
        keep.extend(rng.choice(pool, size=n_per_class, replace=False).tolist())
    out = [samples[i] for i in sorted(keep)]
    assert sum(s["label_valid"] for s in out) == n_per_class and len(out) == 2 * n_per_class
    return out


def target_field_set_counts(rows: list[dict], min_subset_size: int) -> Counter:
    """How often each field set occurs among all subsets (size >= ``min_subset_size``) of the valid rows' non-null fields."""
    counts: Counter = Counter()
    for row in rows:
        if row["error_type"] is not None:
            continue
        present = [f for f in LINE_FIELDS if row[f] is not None]
        for r in range(min_subset_size, len(present) + 1):
            counts.update(frozenset(c) for c in itertools.combinations(present, r))
    assert counts, "no valid source row has enough non-null fields"
    return counts


def allocate(target: Counter, n: int) -> dict[frozenset, int]:
    """Counts per field set summing to ``n``, proportional to ``target`` (largest remainder, ties by set order)."""
    sets = sorted(target, key=lambda f: (len(f), sorted(f)))
    total = sum(target.values())
    raw = [target[f] * n / total for f in sets]
    alloc = {f: int(r) for f, r in zip(sets, raw)}
    by_remainder = sorted(range(len(sets)), key=lambda i: (-(raw[i] - int(raw[i])), i))
    for i in by_remainder[: n - sum(alloc.values())]:
        alloc[sets[i]] += 1
    assert sum(alloc.values()) == n and all(a >= 0 for a in alloc.values())
    return alloc


def balance_field_set(samples: list[dict], n_per_class: int, target: Counter, rng: np.random.Generator) -> list[dict]:
    """``n_per_class`` valid and invalid samples with identical per-field-set counts (``allocate`` of ``target``); draw order kept."""
    alloc = allocate(target, n_per_class)
    pools: dict[tuple, list[int]] = {}
    for i, s in enumerate(samples):
        pools.setdefault((frozenset(s["fields"]), s["label_valid"]), []).append(i)
    keep = []
    for fs, a in alloc.items():
        for lab in (True, False):
            pool = pools.get((fs, lab), [])
            if len(pool) < a:
                raise ValueError(f"field set {'+'.join(sorted(fs))}: only {len(pool)} {'valid' if lab else 'invalid'} samples, {a} requested")
            keep.extend(pool[j] for j in rng.choice(len(pool), size=a, replace=False))
    out = [samples[i] for i in sorted(keep)]
    assert len(out) == 2 * n_per_class and sum(s["label_valid"] for s in out) == n_per_class
    return out


def draw_replicate(rows: list[dict], p: dict, split: str, replicate: int) -> dict:
    """One replicate: ``{replicate, stats, samples}`` (``samples`` = sample_id strings)."""
    rng = np.random.default_rng([p["seed"], SPLITS.index(split), replicate])
    drawn, stats = draw_samples(rows, p["samples_per_row"], p["min_subset_size"], rng)
    stats.update(n_drawn=len(drawn), n_drawn_valid=sum(s["label_valid"] for s in drawn))
    n = p["n_per_class"][split]
    if p["balance_by"] == "none":
        kept = balance(drawn, n, rng)
    else:
        kept = balance_field_set(drawn, n, target_field_set_counts(rows, p["min_subset_size"]), rng)
        assert Counter(frozenset(s["fields"]) for s in kept if s["label_valid"]) == Counter(frozenset(s["fields"]) for s in kept if not s["label_valid"])
    error_types = sorted({r["error_type"] for r in rows if r["error_type"] is not None})
    shares = check_error_type_shares([s["row"]["error_type"] for s in kept if not s["label_valid"]], error_types,
                                     p["max_error_type_share_deviation"])
    stats["invalid_error_type_shares"] = {e: round(v, 4) for e, v in shares.items()}
    stats["valid_from_clean_rows"] = sum(s["label_valid"] and s["row"]["error_type"] is None for s in kept)
    stats["kept_by_error_type"] = dict(sorted(Counter(s["row"]["error_type"] or "valid" for s in kept).items()))
    for name, lab in (("valid", True), ("invalid", False)):   # subset size can leak the label: bigger subsets hit a corrupted field more often
        stats[f"kept_{name}_by_subset_size"] = dict(sorted(Counter(len(s["fields"]) for s in kept if s["label_valid"] == lab).items()))
    return {"replicate": replicate, "stats": stats, "samples": [s["sample_id"] for s in kept]}


def load_config(path: Path) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("id", "params"):
        assert key in config, f"config missing top-level key {key!r}"
    assert "data_root" in config["params"] and "subsets" in config["params"], "config params need data_root and subsets"
    p = config["params"]["subsets"]
    assert set(p) == REQUIRED_PARAMS, f"params.subsets keys {sorted(p)} (line only: no per-dataset keys)"
    assert p["source"] in SOURCES and set(p["n_per_class"]) == set(SPLITS) and p["balance_by"] in BALANCE_MODES, p
    assert 0 < p["max_error_type_share_deviation"] < 1, p
    assert all(isinstance(p[k], int) and p[k] >= 1 for k in ("n_replicates", "samples_per_row", "min_subset_size"))
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="data/vrdu/build-configs/<invalids id>.yaml")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    p = config["params"]["subsets"]
    data_root = Path(config["params"]["data_root"])
    assert data_root.is_dir(), f"no such directory: {data_root}"

    sha256, line, error_types = {}, {}, {}
    for split in SPLITS:
        rel = f"line/{split}.json"
        all_rows = json.loads((data_root / rel).read_text(encoding="utf-8"))
        sha256[rel] = sha256_of(data_root / rel)
        rows = source_rows(all_rows, p["source"], data_root, split)
        error_types[split] = sorted({r["error_type"] for r in rows if r["error_type"] is not None})
        reps = [draw_replicate(rows, p, split, r) for r in range(p["n_replicates"])]
        line[split] = {"replicates": reps}
        sets = [set(r["samples"]) for r in reps]
        overlap = [len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)]
        print(f"line/{split}: {len(rows)} source rows, {p['n_replicates']} replicates x {2 * p['n_per_class'][split]} samples, "
              f"mean pairwise Jaccard between replicates {np.mean(overlap):.3f}" if overlap else f"line/{split}: single replicate")
    assert error_types["train"] == error_types["test"], error_types
    if p["source"] == "manifest":
        sha256[MANIFEST_FILE] = sha256_of(data_root / MANIFEST_FILE)

    sha, dirty = base_builder.git_state()
    out = {
        "build": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_sha": sha,
            "git_dirty": dirty,
            "script": "data/vrdu/build_vrdu_subsets.py",
            "config_id": config["id"],
            "subsets_params": p,
            "error_types": error_types["train"],
            "sha256": sha256,
        },
        "line": line,
    }
    (data_root / SUBSETS_FILE).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for split in SPLITS:
        for r in range(p["n_replicates"]):
            load_subsets(data_root, split, r)   # round-trip: hashes match, every sample verifies against its row
    print(f"written to {data_root / SUBSETS_FILE}")


if __name__ == "__main__":
    main()
