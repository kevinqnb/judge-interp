"""Turn the line subsets (``data/vrdu/subsets.json``) into Clef samples files.

- ``clef_relation_conditional``: the first field is the primary entity, the rest are conditions.
- ``clef_record_oneshot``: all fields, in order, are the entities of one record.

An entity is ``{field, value, valid}`` (the sample's shown value; ``valid`` iff the field is not corrupted).

Usage (repo root)::

    uv run --python 3.12 python experiments/clef_samples.py --data-root data/vrdu --split test \\
        --replicates 0 1 2 3 4 --out-dir experiments/clef-samples

Writes ``<out-dir>/{conditional,oneshot}-<split>-rep<r>.json`` and ``provenance.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "vrdu"))

from subsets import SUBSETS_FILE, load_subsets  # noqa: E402

TASK_FILE_PREFIX = {"clef_relation_conditional": "conditional", "clef_record_oneshot": "oneshot"}


def entities(sample: dict) -> list[dict]:
    """The entities of one ``load_subsets`` sample, in field order."""
    assert list(sample["values"]) == sample["fields"], (sample["sample_id"], "values not in field order")
    ents = [{"field": f, "value": v, "valid": f not in sample["invalid_fields"]} for f, v in sample["values"].items()]
    assert all(isinstance(e["value"], str) for e in ents), (sample["sample_id"], "a null value in a sample")
    assert all(e["valid"] for e in ents) == sample["label_valid"], (sample["sample_id"], "entity validity != subset label")
    return ents


def to_sample(sample: dict, task: str) -> dict:
    """One ``load_subsets`` sample as a samples-file entry for ``task``."""
    assert task in TASK_FILE_PREFIX, task
    ents = entities(sample)
    assert len(ents) >= 2, (sample["sample_id"], "fewer than two fields")
    out = {"sample_id": sample["sample_id"], "document_id": sample["document_id"], "error_type": sample["error_type"] or ""}
    if task == "clef_relation_conditional":
        out.update(primary=ents[0], conditions=ents[1:])
    else:
        out.update(entities=ents)
    return out


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(data_root: Path, split: str, replicates: list[int], out_dir: Path) -> dict:
    """Write one samples file per (task, replicate) and ``provenance.json``; returns the provenance."""
    assert replicates and len(set(replicates)) == len(replicates), replicates
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for r in replicates:
        subsets = load_subsets(data_root, split, r)  # refuses a stale subsets.json and re-checks every sample
        for task, prefix in TASK_FILE_PREFIX.items():
            samples = [to_sample(s, task) for s in subsets]
            assert len(samples) == len(subsets) and len({s["sample_id"] for s in samples}) == len(samples)
            path = out_dir / f"{prefix}-{split}-rep{r}.json"
            path.write_text(json.dumps(samples, indent=1) + "\n", encoding="utf-8")
            files[path.name] = {"task": task, "replicate": r, "n_samples": len(samples),
                                "n_valid": sum(s["label_valid"] for s in subsets), "sha256": sha256_of(path)}
    provenance = {"split": split, "subsets_sha256": sha256_of(data_root / SUBSETS_FILE),
                  "split_sha256": sha256_of(data_root / "line" / f"{split}.json"), "files": files}
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n", encoding="utf-8")
    return provenance


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True, type=Path, help="the data/vrdu directory")
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--replicates", required=True, type=int, nargs="+")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args(argv)
    prov = build(args.data_root, args.split, args.replicates, args.out_dir)
    for name, f in prov["files"].items():
        print(f"{name}: {f['n_samples']} samples ({f['n_valid']} valid)")


if __name__ == "__main__":
    main()
