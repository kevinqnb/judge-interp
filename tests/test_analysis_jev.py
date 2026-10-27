"""Unit tests for analysis/jev.py on hand-built Jev run directories (no API involved).

The metric code is analysis/clef.py's (tested in test_analysis_clef.py); these tests cover what is Jev's own:
the loader and its gates, the resolution metrics, and that the reused path gives the hand-worked numbers
through a Jev run. Fixtures as in test_analysis_clef.py: the main rows of test_analysis_entity_detection.py
(3 minimal pairs; invalid spans P(true) 0.1, valid 0.9 -> AUROC 1.0, Brier 0.01) and the 8 rows (4 minimal
pairs) of the relation_detection_mini line fixture.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import test_analysis_clef as tac
import test_analysis_entity_detection as ted

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analysis_jev", ROOT / "analysis" / "jev.py")
aj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aj)


def _write_run(root: Path, run_id: str, kind: str, cols: dict, metrics_over=None, drop_metric=None) -> None:
    d = root / run_id
    (d / "artifacts").mkdir(parents=True)
    np.savez_compressed(d / "artifacts" / "jev_predictions.npz", **cols)
    lab = np.asarray(cols["label_valid"], dtype=bool)
    metrics = {"n_predictions": len(lab), "n_valid": int(lab.sum()), "determinism_max_abs_diff": 0.03, **(metrics_over or {})}
    metrics.pop(drop_metric, None)
    (d / "metrics.json").write_text(json.dumps(metrics))
    (d / "run.json").write_text(json.dumps({"status": "success", "task": aj.KINDS[kind]}))
    (d / "config.snapshot.yaml").write_text(yaml.safe_dump({"params": {"context_control": "none"}}))


def test_entity_run_through_the_reused_metrics(tmp_path):
    cols = tac._entity_cols()
    _write_run(tmp_path, "r", "entity", cols)
    run = aj.load_run("r", tmp_path, "entity")
    assert run["n_excluded"] == 0 and run["determinism_max_abs_diff"] == 0.03 and run["kind"] == "entity"
    ev = aj.ac.entity_eval_set(run, ted.MAIN_ROWS, 0)
    m = aj.ac.analyse_run("e", run, ev, {"bss_ref_rate": aj.ac.REF_SET_RATE}, tac.P, tmp_path / "fig", 0)
    assert m["n"] == 6 and m["test_auroc"] == 1.0 and m["test_brier"] == pytest.approx(0.01)
    assert m["test_bss"] == pytest.approx(1 - 0.01 / 0.25) and m["n_excluded_overlong"] == 0


def test_samples_run_through_the_reused_metrics(tmp_path):
    _write_run(tmp_path, "r", "oneshot", tac._sample_cols())
    run = aj.load_run("r", tmp_path, "oneshot")
    ev = aj.ac.sample_eval_set(run)
    m = aj.ac.analyse_run("o", run, ev, {"bss_ref_rate": 0.5}, tac.P, tmp_path / "fig", 0)
    assert m["test_auroc"] == 1.0 and m["test_baseline_size_auroc"] == 0.5 and m["test_n_cross_item"] == 4


def test_resolution_metrics_hand_worked(tmp_path):
    cols = tac._sample_cols()
    cols["p_true"] = np.array([0.99, 0.5, 0.01, 0.0, 0.8, 0.8, 0.3, 1.0])
    _write_run(tmp_path, "r", "conditional", cols)
    m = aj.resolution_metrics(aj.load_run("r", tmp_path, "conditional"))
    # distinct: {0.99, 0.5, 0.01, 0.0, 0.8, 0.3, 1.0}; saturated (<= 0.01 or >= 0.99): 0.99, 0.01, 0.0, 1.0
    assert m == {"n_distinct_p_true": 7, "frac_p_saturated": 4 / 8, "determinism_max_abs_diff": 0.03}


def test_loader_gates(tmp_path):
    cols = tac._entity_cols()
    _write_run(tmp_path, "bad_count", "entity", cols, metrics_over={"n_predictions": 7})
    with pytest.raises(AssertionError, match="metrics n_predictions"):
        aj.load_run("bad_count", tmp_path, "entity")
    _write_run(tmp_path, "wrong_task", "entity", cols)
    with pytest.raises(AssertionError, match="task"):
        aj.load_run("wrong_task", tmp_path, "oneshot")
    _write_run(tmp_path, "off_grid", "entity", {**cols, "p_true": cols["p_true"] + 0.003})
    with pytest.raises(AssertionError, match="grid"):
        aj.load_run("off_grid", tmp_path, "entity")
    _write_run(tmp_path, "out_of_range", "entity", {**cols, "p_true": cols["p_true"] + 0.5})
    with pytest.raises(AssertionError, match=r"outside \[0, 1\]"):
        aj.load_run("out_of_range", tmp_path, "entity")
    _write_run(tmp_path, "has_logits", "entity", {**cols, "logit_true": np.zeros(len(cols["p_true"]))})
    with pytest.raises(AssertionError, match="no logits"):
        aj.load_run("has_logits", tmp_path, "entity")
    _write_run(tmp_path, "clef_like", "entity", cols, metrics_over={"n_excluded_overlong": 0})
    with pytest.raises(AssertionError, match="excludes nothing"):
        aj.load_run("clef_like", tmp_path, "entity")
    flipped = {**cols, "label_valid": ~cols["label_valid"]}
    _write_run(tmp_path, "bad_label", "entity", flipped)
    with pytest.raises(AssertionError, match="error type empty iff valid"):
        aj.load_run("bad_label", tmp_path, "entity")
