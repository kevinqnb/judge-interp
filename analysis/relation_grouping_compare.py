"""Relation grouping vs relation detection: the same clustering analysis on a rerun with grouping instructions, on fixed examples.

Runs ``relation_detection_clustering.run`` on ``relation_grouping`` runs (every row of a few documents; ``params.models`` names those
runs), with the example prompts FIXED to those of an earlier clustering run (``params.examples_from``: its ``artifacts/examples.json``)
instead of drawn at random, and writes a per-prompt comparison with the earlier run's numbers (``params.baseline_prompt_metrics``: label ->
path, relative to the runs root, of that run's ``<label>_prompt_metrics.npz``). The clustering code is used unchanged (only its example
choice is replaced, in this process); outputs go under this config's own id, so the earlier run's results and figures are not touched.

Outputs (runs root / <id>): ``metrics.json`` (clustering metrics on the grouping runs), ``artifacts/comparison.json``
(per example prompt and model: baseline and grouping ``win_frac``, ``twin_win_frac``, ``contrast``), figures in ``analysis/figures/<id>/``.
Config: ``analysis/configs/relation_grouping_compare/<id>.yaml`` (the clustering params plus the three keys above).
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

_here = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("analysis_relation_detection_clustering", _here / "relation_detection_clustering.py")
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)

COMPARE_KEYS = ("win_frac", "twin_win_frac", "contrast")
CLUSTERING_ONLY = ("examples_from", "baseline_prompt_metrics", "original_runs")
ROW_KEYS = ("doc_ids", "field", "field_position", "line_index", "value", "pair_id", "cell", "source_field", "source_line_index", "label_valid")


def fixed_examples(examples: list[dict]):
    """A drop-in for ``choose_examples`` returning the given ``[{"doc", "pair_ids"}]`` (every pair id must be among the kept prompts)."""
    def choose(prompts, cells, m, n_docs, rng):
        by_pid = {q["pair_id"]: q for q in prompts}
        out = []
        for ex in examples:
            missing = [pid for pid in ex["pair_ids"] if pid not in by_pid]
            assert not missing, f"fixed example prompts not among the kept prompts (excluded or absent): {missing}"
            chosen = [by_pid[pid] for pid in ex["pair_ids"]]
            assert {q["doc"] for q in chosen} == {ex["doc"]}
            out.append({"doc": ex["doc"], "prompts": chosen})
        return out
    return choose


def compare(examples: list[dict], base: dict[str, Path], new: dict[str, Path]) -> dict:
    """Per model and example prompt: baseline and new values of ``COMPARE_KEYS``. Both files hold eligible prompts only, so a prompt must be in both
    exactly once, or in neither and be a ``cross_item`` prompt (recorded as ``eligible: False``)."""
    out: dict[str, list] = {}
    for label in base:
        b, n = np.load(base[label], allow_pickle=True), np.load(new[label], allow_pickle=True)
        rows = []
        for ex in examples:
            for pid in ex["pair_ids"]:
                ib, inew = np.flatnonzero(b["pair_id"] == pid), np.flatnonzero(n["pair_id"] == pid)
                assert len(ib) == len(inew) <= 1, f"{label}: {pid!r} appears {len(ib)} times in the baseline, {len(inew)} in the new run"
                if not len(ib):   # the metric files hold eligible prompts only; a cross_item prompt never is (its swapped values are the row's own item's)
                    assert pid.split("|")[2].startswith("cross_item"), f"{label}: {pid!r} is missing from both files but is not a cross_item prompt"
                    rows.append({"pair_id": pid, "cell": pid.split("|")[2], "eligible": False})
                    continue
                rows.append({"pair_id": pid, "cell": str(n["cell"][inew[0]]), "eligible": True,
                             **{f"baseline_{k}": float(b[k][ib[0]]) for k in COMPARE_KEYS},
                             **{f"grouping_{k}": float(n[k][inew[0]]) for k in COMPARE_KEYS}})
        out[label] = rows
    return out


def check_same_rows(new: dict, orig: dict) -> int:
    """Assert the grouping run's entities are, in order, exactly the original run's entities of the same documents; returns their number."""
    docs = set(new["doc_ids"].tolist())
    keep = np.array([d in docs for d in orig["doc_ids"].tolist()])
    assert keep.sum() == len(new["doc_ids"]), (int(keep.sum()), len(new["doc_ids"]))
    for k in ROW_KEYS:
        assert np.array_equal(np.asarray(new[k]).astype(str), np.asarray(orig[k])[keep].astype(str)), f"row metadata {k!r} differs from the original run"
    return int(keep.sum())


def accept_grouping_task():
    """Let the clustering code's ``load_split`` read ``relation_grouping`` runs (it asks for ``relation_detection``; the read-out and
    files are the same, only the instructions differ). Any other task is still refused."""
    load = rc.er.load_split

    def load_split(run_id, runs_root, layer, max_docs, expected_task, extra_meta):
        task = json.loads((runs_root / run_id / "run.json").read_text())["task"]
        assert task in ("relation_detection", "relation_grouping"), f"{run_id}: task {task!r}"
        return load(run_id, runs_root, layer, max_docs, task, extra_meta)
    rc.er.load_split = load_split


def main() -> None:
    cfg_path = Path(sys.argv[1])
    cfg = yaml.safe_load(cfg_path.read_text())
    runs_root = Path(os.environ["RUNS_ROOT"])
    out = runs_root / cfg["id"]
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cfg_path, out / "config.snapshot.yaml")
    p = cfg["params"]
    examples_all = json.loads((runs_root / p["examples_from"] / "artifacts" / "examples.json").read_text())
    labels = [m["label"] for m in p["models"]]
    assert set(labels) == set(examples_all) == set(p["baseline_prompt_metrics"]), (labels, sorted(examples_all))
    first = examples_all[labels[0]]
    assert all(examples_all[l] == first for l in labels), "the earlier run drew different examples per model"
    accept_grouping_task()
    assert set(p["original_runs"]) == set(labels)
    for m in p["models"]:   # known answer: the same entities, in the same order, as the original run (only the instructions differ)
        kw = (runs_root, m["layer"], None, "relation_detection", rc.EXTRA_META)
        n = check_same_rows(rc.er.load_split(m["run_id"], *kw), rc.er.load_split(p["original_runs"][m["label"]], *kw))
        print(m["label"], "entities identical to the original run's:", n)
    rc.choose_examples = fixed_examples(first)
    clustering_cfg = {**cfg, "params": {k: v for k, v in p.items() if k not in CLUSTERING_ONLY}}
    rc.run(clustering_cfg, runs_root)
    cmp = compare(first, {l: runs_root / p["baseline_prompt_metrics"][l] for l in labels},
                  {l: out / "artifacts" / f"{l}_prompt_metrics.npz" for l in labels})
    (out / "artifacts" / "comparison.json").write_text(json.dumps(cmp, indent=2) + "\n")
    for l, rows in cmp.items():
        print(l)
        for r in (r for r in rows if r["eligible"]):
            print(f"  {r['cell']:<20} win {r['baseline_win_frac']:.2f} -> {r['grouping_win_frac']:.2f}   twin {r['baseline_twin_win_frac']:.2f} -> {r['grouping_twin_win_frac']:.2f}")


if __name__ == "__main__":
    main()
