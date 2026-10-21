"""Unit tests for analysis/entity_detection.py on a tiny hand-built fixture.

Hand-worked fixture (7 test spans, fields are made up, error types are the real names):
    idx  valid  row_type          field_error_type  prob
     0   T      ""                ""                0.9
     1   T      ""                ""                0.2   (wrongly called invalid)
     2   T      "cross_type"      ""                0.8
     3   F      "cross_type"      "cross_type"      0.1
     4   F      "cross_type"      "cross_type"      0.7   (missed)
     5   F      "inter_document"  "inter_document"  0.3
     6   F      "inter_document"  "inter_document"  0.4
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "analysis_entity_detection", Path(__file__).resolve().parents[1] / "analysis" / "entity_detection.py"
)
er = importlib.util.module_from_spec(spec)
spec.loader.exec_module(er)

Y = np.array([True, True, True, False, False, False, False])
P = np.array([0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 0.4])
ROW = np.array(["", "", "cross_type", "cross_type", "cross_type", "inter_document", "inter_document"])
INV = np.array(["", "", "", "cross_type", "cross_type", "inter_document", "inter_document"])
TYPES = ["cross_type", "inter_document"]


def test_brier_known_answers():
    y = np.array([True, False, True, False])
    b, ref, bss = er.brier_skill_score(y, y.astype(float), 0.5)
    assert (b, ref, bss) == (0.0, 0.25, 1.0)  # perfect forecast
    b, ref, bss = er.brier_skill_score(y, np.full(4, 0.5), 0.5)
    assert (b, ref, bss) == (0.25, 0.25, 0.0)  # the reference forecast has no skill
    b, _, bss = er.brier_skill_score(y, 1.0 - y.astype(float), 0.5)
    assert b == 1.0 and bss == -3.0  # perfectly wrong


def test_brier_hand_value():
    # probs (0.9, 0.2, 0.8, ...) vs Y: squared errors .01 .64 .04 .01 .49 .09 .16 -> mean 1.44/7
    b, ref, bss = er.brier_skill_score(Y, P, 3 / 7)
    assert b == pytest.approx(1.44 / 7)
    # reference at rate 3/7: 3 * (4/7)^2 + 4 * (3/7)^2 over 7 = (48 + 36) / 343
    assert ref == pytest.approx(84 / 343)
    assert bss == pytest.approx(1 - (1.44 / 7) / (84 / 343))


def test_brier_rejects_degenerate_reference():
    with pytest.raises(AssertionError):
        er.brier_skill_score(Y, P, 1.0)


def test_error_type_report_known_answer():
    r = er.error_type_report(Y, P, 0.5, INV, ROW, TYPES)
    assert r["valid_acc_clean_rows"] == 0.5 and r["n_valid_clean_rows"] == 2  # 0.9 ok, 0.2 wrong
    assert r["detect_cross_type"] == 0.5 and r["n_invalid_cross_type"] == 2  # 0.1 caught, 0.7 missed
    assert r["detect_inter_document"] == 1.0 and r["n_invalid_inter_document"] == 2
    assert r["valid_acc_in_cross_type_rows"] == 1.0 and r["n_valid_in_cross_type_rows"] == 1
    assert "valid_acc_in_inter_document_rows" not in r  # no valid span in an inter_document row
    # cross_type-invalid {0.1, 0.7} vs valid {0.9, 0.2, 0.8}: pairs where valid > invalid: 0.1: 3/3, 0.7: 2/3 -> 5/6
    assert r["auroc_cross_type_vs_valid"] == pytest.approx(5 / 6)


def test_error_type_report_inter_document_auroc_by_hand():
    # inter_document-invalid {0.3, 0.4} vs valid {0.9, 0.2, 0.8}; AUROC = P(valid > invalid):
    # 0.9 beats both, 0.8 beats both, 0.2 beats neither -> 4/6
    r = er.error_type_report(Y, P, 0.5, INV, ROW, TYPES)
    assert r["auroc_inter_document_vs_valid"] == pytest.approx(4 / 6)


def test_error_type_report_rejects_unlisted_type():
    with pytest.raises(AssertionError):
        er.error_type_report(Y, P, 0.5, INV, ROW, ["cross_type"])


def test_error_type_report_rejects_type_with_no_items():
    with pytest.raises(AssertionError):
        er.error_type_report(Y, P, 0.5, INV, ROW, TYPES + ["intra_document"])


P_LR = {"lr_C": 1.0, "lr_max_iter": 1000, "lr_tol": 1e-4, "n_shuffles": 30}


def test_shuffled_label_auroc_is_chance_when_x_is_independent_of_y_and_deterministic():
    rng = np.random.default_rng(0)
    x_tr, x_te = rng.standard_normal((300, 5)), rng.standard_normal((300, 5))
    y_tr, y_te = rng.random(300) < 0.4, rng.random(300) < 0.4  # labels independent of x
    m, s = er.shuffled_label_auroc(x_tr, y_tr, x_te, y_te, P_LR, 0)
    assert abs(m - 0.5) < 0.05 and s > 0
    assert (m, s) == er.shuffled_label_auroc(x_tr, y_tr, x_te, y_te, P_LR, 0)


def test_shuffled_label_auroc_requires_n_shuffles():
    x, y = np.zeros((4, 2)), np.array([True, False, True, False])
    with pytest.raises(KeyError):
        er.shuffled_label_auroc(x, y, x, y, {k: v for k, v in P_LR.items() if k != "n_shuffles"}, 0)


def test_onehot():
    m = er.onehot(np.array(["b", "a", "b"]), ["a", "b"])
    np.testing.assert_array_equal(m, [[0, 1], [1, 0], [0, 1]])
    with pytest.raises(AssertionError):
        er.onehot(np.array(["c"]), ["a", "b"])


# ---------------------------------------------------------------- loader on a synthetic run
def write_run(root: Path, run_id: str, task: str = "entity_detection", rep0_constant: bool = True) -> None:
    run_dir = root / run_id
    (run_dir / "artifacts").mkdir(parents=True)
    docs = np.array(["a", "a", "a", "b", "b"])
    valid = np.array([True, False, True, True, False])
    fet = np.where(valid, "", "cross_type")
    rep0 = np.ones((5, 4), dtype=np.float32)
    if not rep0_constant:
        rep0[2, 0] = 2.0
    np.savez_compressed(
        run_dir / "artifacts" / "span_representations.npz",
        layers=np.array([0, 8]), rep_0=rep0, rep_8=np.arange(20, dtype=np.float32).reshape(5, 4),
        doc_ids=docs, row_ordinal=np.zeros(5, dtype=np.int64), field=np.array(["f"] * 5),
        field_position=np.array([1, 2, 3, 1, 2]), label_valid=valid, field_error_type=fet,
        row_error_type=fet, row_k=np.array([1, 1, 1, 1, 1]),
    )
    (run_dir / "run.json").write_text(json.dumps({"status": "success", "task": task}))
    (run_dir / "metrics.json").write_text(json.dumps({
        "n_spans": 5, "n_valid_spans": 3, "n_invalid_spans": 2,
        "n_spans_inter_document": 0, "n_spans_intra_document": 0, "n_spans_cross_type": 2,
    }))


def test_load_split_roundtrip(tmp_path):
    write_run(tmp_path, "r")
    s = er.load_split("r", tmp_path, 8, None, "entity_detection")
    np.testing.assert_array_equal(s["X"], np.arange(20, dtype=np.float32).reshape(5, 4))  # raw rep_8, no rep_0 subtraction
    assert s["label_valid"].tolist() == [True, False, True, True, False]


def test_load_split_max_docs(tmp_path):
    write_run(tmp_path, "r")
    s = er.load_split("r", tmp_path, 8, 1, "entity_detection")
    assert s["doc_ids"].tolist() == ["a"] * 3 and s["X"].shape == (3, 4)


def test_load_split_rejects_wrong_task_layer_and_nonconstant_layer0(tmp_path):
    write_run(tmp_path, "r")
    with pytest.raises(AssertionError):
        er.load_split("r", tmp_path, 8, None, "relation_detection")
    with pytest.raises(AssertionError):
        er.load_split("r", tmp_path, 14, None, "entity_detection")
    write_run(tmp_path, "bad", rep0_constant=False)
    with pytest.raises(AssertionError, match="layer-0"):
        er.load_split("bad", tmp_path, 8, None, "entity_detection")


def test_load_split_rejects_count_mismatch(tmp_path):
    write_run(tmp_path, "r")
    m = json.loads((tmp_path / "r" / "metrics.json").read_text())
    m["n_valid_spans"] = 4
    (tmp_path / "r" / "metrics.json").write_text(json.dumps(m))
    with pytest.raises(AssertionError):
        er.load_split("r", tmp_path, 8, None, "entity_detection")


# --- shared preprocessing and probe fit ---------------------------------------------------

DOCS = np.array(["a"] * 5 + ["b"] * 2)


def test_center_per_document_hand_values():
    x = np.array([[1, 10], [3, 10], [5, 10], [0, 0], [4, 8], [7, 7], [9, 9]], dtype=np.float32)
    er.center_per_document(x, DOCS)
    # doc-a mean = ([1,3,5,0,4], [10,10,10,0,8]) = (2.6, 7.6)
    np.testing.assert_allclose(x[0], [1 - 2.6, 10 - 7.6], atol=1e-5)
    np.testing.assert_allclose(x[6], [1.0, 1.0], atol=1e-5)  # doc-b mean (8,8)
    np.testing.assert_allclose(x[5], [-1.0, -1.0], atol=1e-5)


def test_global_mean_is_zero_after_per_doc_centering():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(7, 4)).astype(np.float32) + 5
    er.center_per_document(x, DOCS)
    assert np.abs(x.mean(axis=0)).max() < 1e-5


def test_fit_logreg_separable_and_shuffle_control():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = rng.random(400) < 0.5
    x = rng.normal(size=(400, 5)).astype(np.float32)
    x[:, 0] += 4 * y
    p = {"lr_C": 1.0, "lr_max_iter": 200, "lr_tol": 1e-4}
    clf = er.fit_logreg(x, y, p, 0)
    assert roc_auc_score(y, clf.predict_proba(x)[:, 1]) > 0.99
    again = er.fit_logreg(x, y, p, 0)
    assert np.array_equal(clf.predict_proba(x), again.predict_proba(x))
