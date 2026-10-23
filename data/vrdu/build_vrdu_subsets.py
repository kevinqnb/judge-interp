"""Draw the shared random field subsets of VRDU line rows and write ``subsets.json``.

Line only; several independent replicates per split. Per replicate:

  draw        per source row, ``samples_per_row`` distinct field subsets (size >= ``min_subset_size``),
              each in a random order.
  match rule  invalid samples whose shown values are a real line item's are dropped (they are valid).
  cross_item  from each subset of a clean row, one shown pair whose swap changes both values is swapped.
  balance     both classes have the same count for every field set; each error type contributes
              ``n_per_error_type`` invalid samples (cross_item exactly, the row types within
              ``max_error_type_share_deviation``: cross_type cannot fill channel/program_desc-only sets).
              A short cell raises.

All values are in ``params.subsets`` of the invalids config. Run after ``build_vrdu_manifest.py``:

    uv run --python 3.12 python data/vrdu/build_vrdu_subsets.py \\
        --config data/vrdu/build-configs/2026-10-22-vrdu-invalids-05.yaml
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
from build_vrdu_invalids import field_kinds, matches_item
from build_vrdu_manifest import MANIFEST_FILE, SPLITS, apply_manifest, load_manifest_split, sha256_of
from subsets import (BALANCE_MODES, CROSS_ITEM, LINE_FIELDS, REQUIRED_PARAMS, SOURCES, SUBSETS_FILE,
                     check_error_type_shares, equivalence_from_invalids, items_by_document, load_subsets,
                     make_sample_eq, make_sample_id, sample_invalid_fields, sample_values, swap_is_noop)


def source_rows(rows: list[dict], source: str, data_root: Path, split: str) -> list[dict]:
    """The rows samples are drawn from.

    Args:
        rows: a split's line rows.
        source: ``"manifest"`` (the manifest's rows) or ``"all_rows"``.
        data_root: the ``data/vrdu`` directory.
        split: ``"train"`` or ``"test"``.

    Returns:
        The rows, sorted by identity (so the draw does not depend on file order).
    """
    assert source in SOURCES, source
    if source == "manifest":
        rows = apply_manifest(rows, "line", load_manifest_split(data_root, "line", split))
    key = lambda r: (r["document_id"], r["line_index"], r["error_type"] or "", r["k"])
    return sorted(rows, key=key)


def make_sample(row: dict, fields: list[str], swap: tuple[str, str] | None = None) -> dict:
    """Build one sample.

    Args:
        row: the sample's row.
        fields: its fields, in order.
        swap: the cross_item pair, or None.

    Returns:
        ``{sample_id, row, fields, swap, values, invalid_fields, label_valid, error_type}``.
    """
    invalid = sample_invalid_fields(row, fields, swap)
    return {"sample_id": make_sample_id(row, fields, swap), "row": row, "fields": fields, "swap": swap,
            "values": sample_values(row, fields, swap), "invalid_fields": invalid, "label_valid": not invalid,
            "error_type": CROSS_ITEM if swap is not None else row["error_type"]}


def draw_samples(rows: list[dict], samples_per_row: int, min_subset_size: int, rng: np.random.Generator) -> tuple[list[dict], dict]:
    """Draw random ordered field subsets of each row.

    Args:
        rows: source rows.
        samples_per_row: distinct subsets per row (all of them if fewer exist).
        min_subset_size: smallest subset size.
        rng: ``numpy.random.Generator``.

    Returns:
        ``(samples, stats)``: samples in row order, and counts of rows too small or with too few subsets.
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
            samples.append(make_sample(row, fields))
    assert samples, "no samples drawn"
    return samples, stats


def drop_matching(samples: list[dict], items: dict[str, list[dict]], eq) -> tuple[list[dict], Counter]:
    """Drop invalid samples whose shown values are a real line item's (they are valid).

    Args:
        samples: drawn samples.
        items: ``{document_id: [valid line rows]}``.
        eq: from ``build_vrdu_invalids.make_eq``.

    Returns:
        ``(kept, dropped)``: the remaining samples, and the dropped count by error type.
    """
    kept, dropped = [], Counter()
    for s in samples:
        if not s["label_valid"] and matches_item(s["values"], items[s["row"]["document_id"]], eq):
            dropped[s["error_type"]] += 1
        else:
            kept.append(s)
    return kept, dropped


def cross_item_candidates(samples: list[dict], items: dict[str, list[dict]], eq,
                          rng: np.random.Generator) -> tuple[list[dict], int]:
    """Make cross_item samples: swap one shown pair of each clean-row sample.

    Args:
        samples: drawn samples (only clean-row ones are used).
        items: ``{document_id: [valid line rows]}``.
        eq: from ``build_vrdu_invalids.make_eq``.
        rng: ``numpy.random.Generator``.

    Returns:
        ``(candidates, n_noop)``: one sample per source with an eligible pair, and the number of pairs
        skipped because the swap changed nothing.
    """
    out, n_noop = [], 0
    for s in samples:
        row = s["row"]
        if row["error_type"] is not None:
            continue
        assert s["label_valid"], s["sample_id"]
        shown = sorted(s["fields"], key=LINE_FIELDS.index)
        pairs = []
        for a, b in itertools.combinations(shown, 2):
            if swap_is_noop(row, a, b, eq):
                n_noop += 1
            else:
                pairs.append((a, b))
        if not pairs:
            continue
        c = make_sample(row, s["fields"], pairs[int(rng.integers(len(pairs)))])
        assert not matches_item(c["values"], items[row["document_id"]], eq), (
            c["sample_id"], "a swap reproduced a real line item")
        out.append(c)
    return out, n_noop


def target_field_set_counts(rows: list[dict], min_subset_size: int) -> Counter:
    """How often each field set occurs among the subsets of the valid rows (the balance target).

    Args:
        rows: source rows (only valid ones count).
        min_subset_size: smallest subset size.

    Returns:
        ``Counter({frozenset(fields): count})``.
    """
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
    """Split ``n`` across field sets in proportion to ``target`` (largest remainder).

    Args:
        target: ``Counter`` of field sets.
        n: total to allocate.

    Returns:
        ``{field set: count}``, summing to ``n``.
    """
    sets = sorted(target, key=lambda f: (len(f), sorted(f)))
    total = sum(target.values())
    raw = [target[f] * n / total for f in sets]
    alloc = {f: int(r) for f, r in zip(sets, raw)}
    by_remainder = sorted(range(len(sets)), key=lambda i: (-(raw[i] - int(raw[i])), i))
    for i in by_remainder[: n - sum(alloc.values())]:
        alloc[sets[i]] += 1
    assert sum(alloc.values()) == n and all(a >= 0 for a in alloc.values())
    return alloc


def split_equally(n: int, types: list[str], running: Counter) -> dict[str, int]:
    """Split ``n`` equally across error types.

    Args:
        n: count to split.
        types: the error types.
        running: running totals per type (updated); the remainder goes to the types lowest in it.

    Returns:
        ``{type: count}``.
    """
    assert types and n >= 0
    q, r = divmod(n, len(types))
    first = sorted(types, key=lambda t: (running[t], t))[:r]
    out = {t: q + (t in first) for t in sorted(types)}
    running.update(out)
    return out


def balance_field_set_error_type(drawn: list[dict], candidates: list[dict], per_type: int, row_types: list[str],
                                 target: Counter, rng: np.random.Generator) -> list[dict]:
    """Cut one replicate to the balanced sample set.

    Both classes get the same count per field set; each error type gets ``per_type`` invalid samples
    (cross_item exactly, the row types split equally within each field set).

    Args:
        drawn: drawn samples (after ``drop_matching``).
        candidates: cross_item samples.
        per_type: invalid samples per error type.
        row_types: the row error types.
        target: from ``target_field_set_counts``.
        rng: ``numpy.random.Generator``.

    Returns:
        The kept samples (drawn first, then cross_item, each in draw order).

    Raises:
        ValueError: a (field set, class or type) cell has too few samples.
    """
    assert isinstance(per_type, int) and per_type >= 1 and row_types and CROSS_ITEM not in row_types
    alloc_rows = allocate(target, per_type * len(row_types))
    alloc_ci = allocate(target, per_type)
    pool: dict[tuple, list[int]] = {}
    for i, s in enumerate(drawn):
        cell = "valid" if s["label_valid"] else s["error_type"]
        pool.setdefault((frozenset(s["fields"]), cell), []).append(i)
    for i, s in enumerate(candidates):
        pool.setdefault((frozenset(s["fields"]), CROSS_ITEM), []).append(len(drawn) + i)

    def take(fs, cell, n):
        cands = pool.get((fs, cell), [])
        if len(cands) < n:
            raise ValueError(f"field set {'+'.join(sorted(fs))}: only {len(cands)} {cell} samples, {n} requested")
        return [cands[j] for j in rng.choice(len(cands), size=n, replace=False)] if n else []

    keep, running = [], Counter()
    for fs in alloc_rows:   # allocate's set order
        keep += take(fs, "valid", alloc_rows[fs] + alloc_ci[fs])
        if alloc_rows[fs]:
            present = [t for t in row_types if pool.get((fs, t))]
            if not present:
                raise ValueError(f"field set {'+'.join(sorted(fs))}: no invalid row-type sample, {alloc_rows[fs]} requested")
            for t, n in split_equally(alloc_rows[fs], present, running).items():
                keep += take(fs, t, n)
        keep += take(fs, CROSS_ITEM, alloc_ci[fs])
    combined = drawn + candidates
    out = [combined[i] for i in sorted(keep)]
    n_class = per_type * (len(row_types) + 1)
    assert len(out) == 2 * n_class and sum(s["label_valid"] for s in out) == n_class
    assert sum(s["error_type"] == CROSS_ITEM for s in out) == per_type
    return out


def draw_replicate(rows: list[dict], items: dict[str, list[dict]], p: dict, split: str, replicate: int, eq,
                   kind_of: dict[str, str]) -> dict:
    """Draw one replicate.

    Args:
        rows: source rows.
        items: ``{document_id: [valid line rows]}`` over the whole split.
        p: ``params.subsets``.
        split: ``"train"`` or ``"test"``.
        replicate: replicate index (seeds the rng).
        eq: from ``build_vrdu_invalids.make_eq``.
        kind_of: ``{field: kind}``.

    Returns:
        ``{replicate, stats, samples}`` (``samples`` = sample ids).
    """
    rng = np.random.default_rng([p["seed"], SPLITS.index(split), replicate])
    drawn, stats = draw_samples(rows, p["samples_per_row"], p["min_subset_size"], rng)
    stats.update(n_drawn=len(drawn), n_drawn_valid=sum(s["label_valid"] for s in drawn))
    drawn, dropped = drop_matching(drawn, items, eq)
    stats["n_dropped_matches_item"] = dict(sorted(dropped.items()))
    candidates, n_noop = cross_item_candidates(drawn, items, eq, rng)
    stats.update(n_cross_item_candidates=len(candidates), n_cross_item_noop_pairs=n_noop)

    row_types = sorted({r["error_type"] for r in rows if r["error_type"] is not None})
    kept = balance_field_set_error_type(drawn, candidates, p["n_per_error_type"][split], row_types,
                                        target_field_set_counts(rows, p["min_subset_size"]), rng)
    assert Counter(frozenset(s["fields"]) for s in kept if s["label_valid"]) == Counter(frozenset(s["fields"]) for s in kept if not s["label_valid"])
    shares = check_error_type_shares([s["error_type"] for s in kept if not s["label_valid"]], sorted([*row_types, CROSS_ITEM]),
                                     p["max_error_type_share_deviation"])
    stats["invalid_error_type_shares"] = {e: round(v, 4) for e, v in shares.items()}
    stats["kept_by_error_type"] = dict(sorted(Counter(s["error_type"] or "valid" for s in kept).items()))
    stats["valid_from_clean_rows"] = sum(s["label_valid"] and s["row"]["error_type"] is None for s in kept)
    ci = [s for s in kept if s["error_type"] == CROSS_ITEM]
    stats["cross_item_pairs"] = dict(sorted(Counter("~".join(s["swap"]) for s in ci).items()))
    same_kind = lambda a, b: kind_of.get(a) is not None and kind_of.get(a) == kind_of.get(b)
    stats["cross_item_kind_mix"] = dict(sorted(Counter("same_kind" if same_kind(*s["swap"]) else "cross_kind" for s in ci).items()))
    valid_sets = {(s["row"]["document_id"], s["row"]["line_index"], frozenset(s["fields"])) for s in kept if s["label_valid"]}
    stats["n_cross_item_with_valid_twin"] = sum((s["row"]["document_id"], s["row"]["line_index"], frozenset(s["fields"])) in valid_sets for s in ci)
    for name, lab in (("valid", True), ("invalid", False)):   # subset size can leak the label: bigger subsets hit a corrupted field more often
        stats[f"kept_{name}_by_subset_size"] = dict(sorted(Counter(len(s["fields"]) for s in kept if s["label_valid"] == lab).items()))
    return {"replicate": replicate, "stats": stats, "samples": [s["sample_id"] for s in kept]}


def load_config(path: Path) -> dict:
    """Read and validate the subsets part of the invalids config.

    Args:
        path: the YAML config.

    Returns:
        The parsed config.
    """
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("id", "params"):
        assert key in config, f"config missing top-level key {key!r}"
    for key in ("data_root", "subsets", "cross_type", "dates"):
        assert key in config["params"], f"config params need {key!r}"
    p = config["params"]["subsets"]
    assert set(p) == REQUIRED_PARAMS, f"params.subsets keys {sorted(p)} != {sorted(REQUIRED_PARAMS)} (line only: no per-dataset keys)"
    assert p["source"] in SOURCES and p["balance_by"] in BALANCE_MODES, p
    assert set(p["n_per_error_type"]) == set(SPLITS) and all(isinstance(v, int) and v >= 1 for v in p["n_per_error_type"].values()), p
    assert 0 < p["max_error_type_share_deviation"] < 1, p
    assert all(isinstance(p[k], int) and p[k] >= 1 for k in ("n_replicates", "samples_per_row", "min_subset_size"))
    return config


def main(argv=None):
    """Draw every replicate and write ``subsets.json`` for the config given on the CLI.

    Args:
        argv: CLI arguments (None = ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="data/vrdu/build-configs/<invalids id>.yaml")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    p = config["params"]["subsets"]
    data_root = Path(config["params"]["data_root"])
    assert data_root.is_dir(), f"no such directory: {data_root}"
    equivalence = {"kinds": config["params"]["cross_type"]["kinds"], "year_pivot": config["params"]["dates"]["year_pivot"]}
    assert equivalence == equivalence_from_invalids(data_root), "config's cross_type.kinds / dates.year_pivot differ from invalids.json"
    eq = make_sample_eq(equivalence)
    kind_of = field_kinds(equivalence["kinds"])

    sha256, line, error_types = {}, {}, {}
    for split in SPLITS:
        rel = f"line/{split}.json"
        all_rows = json.loads((data_root / rel).read_text(encoding="utf-8"))
        assert not any(r["error_type"] == CROSS_ITEM for r in all_rows), f"{rel} has {CROSS_ITEM} rows"
        sha256[rel] = sha256_of(data_root / rel)
        rows = source_rows(all_rows, p["source"], data_root, split)
        items = items_by_document(all_rows)
        error_types[split] = sorted({*(r["error_type"] for r in rows if r["error_type"] is not None), CROSS_ITEM})
        reps = [draw_replicate(rows, items, p, split, r, eq, kind_of) for r in range(p["n_replicates"])]
        line[split] = {"replicates": reps}
        sets = [set(r["samples"]) for r in reps]
        overlap = [len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)]
        n_class = p["n_per_error_type"][split] * len(error_types[split])
        print(f"line/{split}: {len(rows)} source rows, {p['n_replicates']} replicates x {2 * n_class} samples, "
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
            "equivalence": equivalence,
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
