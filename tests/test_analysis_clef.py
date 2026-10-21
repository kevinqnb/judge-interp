"""Unit tests for analysis/clef.py on hand-built Clef run directories (no model involved).

Entity run, 8 items (valid=V): 6 valid with P(true) 0.9, 2 invalid with P(true) 0.1, of which one is
inter_document and one cross_type -> AUROC 1.0, accuracy 1.0, Brier = mean((0.1)^2) = 0.01.
With bss_ref_rate = set rate 0.75, brier_ref = 0.75 * 0.25 = 0.1875 -> BSS = 1 - 0.01/0.1875.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

spec = importlib.util.spec_from_file_location("analysis_clef", Path(__file__).resolve().parents[1] / "analysis" / "clef.py")
ac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac)

P = dict(decision_threshold=0.5, ece_n_boot=20, n_shuffles=50, permutation_auroc_tolerance=0.2,
         hist_n_bins=10)


def _write_run(root: Path, run_id: str, kind: str, cols: dict, **over) -> None:
    d = root / run_id
    (d / "artifacts").mkdir(parents=True)
    p_true = np.asarray(cols["p_true"], dtype=np.float64)
    cols = {**cols, "logit_true": np.log(p_true), "logit_false": np.log(1 - p_true)}
    np.savez_compressed(d / "artifacts" / "clef_predictions.npz", **cols)
    lab = np.asarray(cols["label_valid"], dtype=bool)
    (d / "metrics.json").write_text(json.dumps({
        "n_predictions": len(lab), "n_valid": int(lab.sum()), "n_excluded_overlong": 0, **over}))
    (d / "run.json").write_text(json.dumps({"status": "success", "task": ac.KINDS[kind]}))
    (d / "config.snapshot.yaml").write_text(yaml.safe_dump({"params": {"context_control": "none"}}))


def _entity_cols(p_true):
    n = 8
    return dict(
        doc_ids=np.array(["a"] * 4 + ["b"] * 4), row_ordinal=np.zeros(n, dtype=np.int64), row_k=np.array([0] * 6 + [1, 1]),
        field=np.array(["advertiser", "agency", "property", "product", "advertiser", "agency", "property", "product"]),
        label_valid=np.array([True] * 6 + [False, False]),
        field_error_type=np.array([""] * 6 + ["inter_document", "cross_type"]),
        row_error_type=np.array([""] * 4 + ["inter_document", "inter_document", "cross_type", "cross_type"]),
        p_true=np.asarray(p_true), n_tokens=np.full(n, 100), context_doc=np.array(["a"] * 4 + ["b"] * 4),
    )


def test_entity_run_hand_worked_metrics(tmp_path):
    p_true = [0.9] * 6 + [0.1, 0.1]
    _write_run(tmp_path, "r", "entity", _entity_cols(p_true))
    run = ac.load_run("r", tmp_path, "entity")
    m = ac.analyse_run("e", run, {"bss_ref_rate": ac.REF_SET_RATE}, P, tmp_path / "fig", 0)
    assert m["test_auroc"] == 1.0 and m["test_accuracy"] == 1.0
    assert m["test_brier"] == pytest.approx(0.01)
    assert m["bss_ref_rate"] == 0.75 and m["test_bss"] == pytest.approx(1 - 0.01 / 0.1875)
    assert m["test_detect_inter_document"] == 1.0 and m["test_detect_cross_type"] == 1.0
    assert m["test_valid_acc_clean_rows"] == 1.0 and m["test_n_valid_clean_rows"] == 4
    assert (tmp_path / "fig" / "e" / "calibration.png").is_file()


def test_entity_run_with_a_missed_error_type(tmp_path):
    p_true = [0.9] * 6 + [0.1, 0.8]  # the cross_type item is called valid
    _write_run(tmp_path, "r", "entity", _entity_cols(p_true))
    m = ac.analyse_run("e", ac.load_run("r", tmp_path, "entity"), {"bss_ref_rate": 0.5}, P, tmp_path / "fig", 0)
    assert m["test_detect_inter_document"] == 1.0 and m["test_detect_cross_type"] == 0.0
    assert m["test_accuracy"] == pytest.approx(7 / 8)
    # ranking is still perfect (0.8 < every valid 0.9) although the 0.5 threshold misses that item
    assert m["test_auroc_cross_type_vs_valid"] == 1.0 and m["test_auroc"] == 1.0


def test_loader_gates(tmp_path):
    cols = _entity_cols([0.9] * 6 + [0.1, 0.1])
    _write_run(tmp_path, "bad_count", "entity", cols, n_predictions=7)
    with pytest.raises(AssertionError, match="metrics n_predictions"):
        ac.load_run("bad_count", tmp_path, "entity")
    _write_run(tmp_path, "wrong_task", "entity", cols)
    with pytest.raises(AssertionError, match="task"):
        ac.load_run("wrong_task", tmp_path, "relation")
    _write_run(tmp_path, "bad_logit", "entity", cols)
    d = tmp_path / "bad_logit" / "artifacts" / "clef_predictions.npz"
    z = dict(np.load(d))
    z["logit_true"] = z["logit_true"] + 1.0
    np.savez_compressed(d, **z)
    with pytest.raises(AssertionError, match="softmax"):
        ac.load_run("bad_logit", tmp_path, "entity")


def test_relation_run_size_baselines(tmp_path):
    n = 12
    size = np.array([2, 2, 3, 3, 4, 4] * 2)
    label = np.array([True, True, True, False, False, False] * 2)
    cols = dict(
        doc_ids=np.array(["a"] * 6 + ["b"] * 6), label_valid=label,
        row_error_type=np.where(label, "", "inter_document"), size=size, fields=np.array(["x|y"] * n),
        n_invalid_fields=(~label).astype(np.int64), rows=np.zeros((n, 5), dtype=np.int64),
        p_true=np.where(label, 0.8, 0.3), n_tokens=np.full(n, 50), context_doc=np.array(["a"] * 6 + ["b"] * 6),
    )
    _write_run(tmp_path, "r", "relation", cols)
    m = ac.analyse_run("rel", ac.load_run("r", tmp_path, "relation"), {"bss_ref_rate": 0.5}, P, tmp_path / "fig", 0)
    assert m["test_auroc"] == 1.0
    # score -size; valid sizes {2 x4, 3 x2}, invalid {3 x2, 4 x4}. Of the 36 (valid, invalid) pairs:
    # valid-2 beats all 6 invalid (4*6 = 24), valid-3 ties invalid-3 (2*2 pairs * 0.5 = 2) and beats invalid-4 (2*4 = 8)
    assert m["test_baseline_size_auroc"] == pytest.approx(34 / 36)
    assert m["test_auroc_size3"] == 1.0 and m["test_n_size4"] == 4 and "test_auroc_size2" not in m  # size 2: no invalid
    assert m["test_detect_inter_document"] == 1.0
