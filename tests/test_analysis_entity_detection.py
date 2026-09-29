"""Unit tests for analysis/entity_detection.py on a tiny hand-built fixture.

Two documents, two prompts each (hand-worked):
    doc-a s0: m=3 k=1  -> rows valid,valid,invalid   k/m = 1/3
    doc-a s1: m=2 k=2  -> rows invalid,invalid       k/m = 1
    doc-b s0: m=2 k=0  -> rows valid,valid           k/m = 0
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "analysis_entity_detection", Path(__file__).resolve().parents[1] / "analysis" / "entity_detection.py"
)
ed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ed)

DOCS = np.array(["a"] * 5 + ["b"] * 2)
SAMPLE = np.array([0, 0, 0, 1, 1, 0, 0])
M = np.array([3, 3, 3, 2, 2, 2, 2])
K = np.array([1, 1, 1, 2, 2, 0, 0])
VALID = np.array([True, True, False, False, False, True, True])


def test_error_rate_known_answer():
    r = ed.error_rate_from_counts(DOCS, SAMPLE, M, K, VALID)
    np.testing.assert_allclose(r, [1 / 3] * 3 + [1.0, 1.0, 0.0, 0.0])


def test_error_rate_rejects_wrong_k():
    bad_k = K.copy()
    bad_k[:3] = 2  # claims 2 decoys but only 1 invalid row
    with pytest.raises(AssertionError):
        ed.error_rate_from_counts(DOCS, SAMPLE, M, bad_k, VALID)


def test_error_rate_rejects_row_count_mismatch():
    with pytest.raises(AssertionError):
        ed.error_rate_from_counts(DOCS[:-1], SAMPLE[:-1], M[:-1], K[:-1], VALID[:-1])  # doc-b has 1 row, m=2


def test_center_per_document_hand_values():
    x = np.array([[1, 10], [3, 10], [5, 10], [0, 0], [4, 8], [7, 7], [9, 9]], dtype=np.float32)
    ed.center_per_document(x, DOCS)
    # doc-a mean = ([1,3,5,0,4], [10,10,10,0,8]) = (2.6, 7.6)
    np.testing.assert_allclose(x[0], [1 - 2.6, 10 - 7.6], atol=1e-5)
    np.testing.assert_allclose(x[6], [1.0, 1.0], atol=1e-5)  # doc-b mean (8,8)
    np.testing.assert_allclose(x[5], [-1.0, -1.0], atol=1e-5)


def test_global_mean_is_zero_after_per_doc_centering():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(7, 4)).astype(np.float32) + 5
    ed.center_per_document(x, DOCS)
    assert np.abs(x.mean(axis=0)).max() < 1e-5


def test_classification_metrics_positive_class_is_valid():
    y = np.array([True, True, False, False])
    p = np.array([0.9, 0.4, 0.6, 0.1])  # preds T,F,T,F -> tp1 fn1 fp1 tn1
    m = ed.classification_metrics(y, p, 0.5)
    assert m["accuracy"] == 0.5 and m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["auroc"] == 0.75


def test_fit_logreg_separable_and_shuffle_control():
    rng = np.random.default_rng(0)
    y = rng.random(400) < 0.5
    x = rng.normal(size=(400, 5)).astype(np.float32)
    x[:, 0] += 4 * y
    p = {"lr_C": 1.0, "lr_max_iter": 200, "lr_tol": 1e-4}
    clf = ed.fit_logreg(x, y, p, 0)
    assert roc_auc(y, clf.predict_proba(x)[:, 1]) > 0.99
    again = ed.fit_logreg(x, y, p, 0)
    assert np.array_equal(clf.predict_proba(x), again.predict_proba(x))


def roc_auc(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p)


def test_type_means_hand_values_and_train_means_applied_to_test():
    types = np.array(["p", "p", "q", "q"])
    x = np.array([[1, 5], [3, 7], [10, 0], [20, 4]], dtype=np.float32)
    means = ed.fit_type_means(x, types)
    np.testing.assert_allclose(means["p"], [2, 6])
    np.testing.assert_allclose(means["q"], [15, 2])
    xt = np.array([[4, 6], [15, 2]], dtype=np.float32)
    ed.subtract_type_means(xt, np.array(["p", "q"]), means)  # train means, not test's own
    np.testing.assert_allclose(xt, [[2, 0], [0, 0]])
    with pytest.raises(AssertionError):
        ed.subtract_type_means(xt, np.array(["p", "zzz"]), means)


def test_dim_std_hand_values_and_constant_dim_rejected():
    x = np.array([[0, 1], [2, 1 + 4]], dtype=np.float32)
    np.testing.assert_allclose(ed.fit_dim_std(x), [1.0, 2.0])
    with pytest.raises(AssertionError):
        ed.fit_dim_std(np.array([[1, 3], [1, 5]], dtype=np.float32))


def test_project_out_hand_values():
    x = np.array([[3.0, 4.0], [1.0, -2.0]], dtype=np.float32)
    u = np.array([1.0, 0.0], dtype=np.float32)
    out = ed.project_out(x, u)
    np.testing.assert_allclose(out, [[0, 4], [0, -2]])
    np.testing.assert_allclose(x, [[3, 4], [1, -2]])  # input untouched
    with pytest.raises(AssertionError):
        ed.project_out(x, np.array([2.0, 0.0], dtype=np.float32))
