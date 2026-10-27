"""Turn the line dataset (``data/vrdu/line/<split>.json`` or ``<split>_subsets.json``) into Clef samples files.

- ``clef_relation_conditional``: the first shown field is the primary entity, the rest are conditions.
- ``clef_record_oneshot``: all shown fields, in order, are the entities of one record.

An entity is ``{field, value, valid}`` (the shown value; ``valid`` iff the field is not corrupted).
A sample's own label is ``label_valid``: the row's ``valid``. Every row becomes one sample, so the minimal
pairs (an invalid sample and its clean twin) carry over; ``pair_id`` and ``cell`` are carried through for
stratification.

``--source pairs`` reads the full-length rows; every entity of a
valid row is valid and an invalid row has at least one corrupted entity. ``--source subsets`` reads the
subset rows (``sample_id`` = the subset's id, which embeds its parent's sample id): a subset keeps the
parent's corrupted entities, but its label is its own ``valid`` (``matches_item``), so a subset that shows a
corrupted entity can still be valid. Its ``error_type`` is the parent's if a corrupted entity is shown.

Usage (repo root)::

    uv run --python 3.12 python experiments/clef_samples.py --data-root data/vrdu --split test \\
        --source pairs --out-dir experiments/clef-samples

Writes ``<out-dir>/{conditional,oneshot}-<split>[-subsets].json`` and ``provenance-<split>[-subsets].json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "vrdu"))
from vrdu_common import LINE_FIELDS, check_build  # noqa: E402

TASK_FILE_PREFIX = {"clef_relation_conditional": "conditional", "clef_record_oneshot": "oneshot"}


SOURCES = ("pairs", "subsets")


def entities(row: dict, source: str) -> list[dict]:
    """The entities of one line row, in the order shown."""
    assert source in SOURCES, source
    sid = row["sample_id"] if source == "pairs" else row["subset_id"]
    assert all(row[f] is None for f in LINE_FIELDS if f not in row["fields"]), (sid, "a hidden field has a value")
    assert set(row["invalid_fields"]) <= set(row["fields"]), (sid, "a corrupted field is not shown")
    ents = [{"field": f, "value": row[f], "valid": f not in row["invalid_fields"]} for f in row["fields"]]
    assert all(isinstance(e["value"], str) for e in ents), (sid, "a null value is shown")
    if source == "pairs":
        assert all(row["field_errors"][f] is None for f in LINE_FIELDS if f not in row["invalid_fields"]), (sid, "field_errors != invalid_fields")
        assert all(e["valid"] for e in ents) == row["valid"], (sid, "entity validity != row label")
    else:
        assert "field_errors" not in row, (sid, "a subset row has field_errors")
        assert row["valid"] or row["invalid_fields"], (sid, "an invalid subset without a corrupted field")
    return ents


def to_sample(row: dict, task: str, source: str) -> dict:
    """One line row (``source`` ``pairs`` or ``subsets``) as a samples-file entry for ``task``."""
    assert task in TASK_FILE_PREFIX, task
    ents = entities(row, source)
    assert len(ents) >= 2, (row.get("sample_id", row.get("subset_id")), "fewer than two fields")
    sid = row["sample_id"] if source == "pairs" else row["subset_id"]
    out = {"sample_id": sid, "document_id": row["document_id"], "error_type": row["error_type"] or "",
           "pair_id": row["pair_id"], "cell": row["cell"], "label_valid": row["valid"]}
    if task == "clef_relation_conditional":
        out.update(primary=ents[0], conditions=ents[1:])
    else:
        out.update(entities=ents)
    return out


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(data_root: Path, split: str, source: str, out_dir: Path) -> dict:
    """Write one samples file per task and ``provenance-<split>[-subsets].json``; returns the provenance."""
    assert source in SOURCES, source
    suffix = "" if source == "pairs" else "-subsets"
    rel = f"line/{split}.json" if source == "pairs" else f"line/{split}_subsets.json"
    line_build = check_build(data_root, rel)   # refuses a stale split file
    rows = json.loads((data_root / rel).read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for task, prefix in TASK_FILE_PREFIX.items():
        samples = [to_sample(r, task, source) for r in rows]
        assert len({s["sample_id"] for s in samples}) == len(samples) == len(rows)
        path = out_dir / f"{prefix}-{split}{suffix}.json"
        path.write_text(json.dumps(samples, indent=1) + "\n", encoding="utf-8")
        files[path.name] = {"task": task, "n_samples": len(samples), "n_valid": sum(r["valid"] for r in rows),
                            "sha256": sha256_of(path)}
    provenance = {"split": split, "source": source, "line_config_id": line_build["config_id"],
                  "split_sha256": sha256_of(data_root / rel), "files": files}
    (out_dir / f"provenance-{split}{suffix}.json").write_text(json.dumps(provenance, indent=1) + "\n", encoding="utf-8")
    return provenance


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True, type=Path, help="the data/vrdu directory")
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--source", required=True, choices=SOURCES, help="pairs: line/<split>.json; subsets: line/<split>_subsets.json")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args(argv)
    prov = build(args.data_root, args.split, args.source, args.out_dir)
    for name, f in prov["files"].items():
        print(f"{name}: {f['n_samples']} samples ({f['n_valid']} valid)")


if __name__ == "__main__":
    main()
