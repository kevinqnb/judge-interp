"""Unit tests for analysis/relation_detection_leace.py (model-free; toy data with known answers)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analysis_relation_detection_leace", ROOT / "analysis" / "relation_detection_leace.py")
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)
TOL = dict(eig_rel_tol=1e-10, rank_tol=1e-8)


def _toy(n=4000, d=6, seed=0, shift=3.0):
    """Isotropic noise plus a class shift of ``shift`` along coordinate 0 (two classes): the eraser must remove exactly coordinate 0's class signal."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    x = rng.normal(size=(n, d))
    x[:, 0] += shift * y
    return x, y


def test_one_hot_and_unknown_class():
    z = lc.one_hot(np.array(["a", "b", "a"]), ["a", "b", "c"])
    assert z.tolist() == [[1, 0, 0], [0, 1, 0], [1, 0, 0]]
    with pytest.raises(AssertionError):
        lc.one_hot(np.array(["a", "z"]), ["a", "b"])


def test_known_answer_two_classes_isotropic():
    x, y = _toy()
    e = lc.fit_leace(x, lc.one_hot(y, [0, 1]), **TOL)
    assert e["rank"] == 1
    xe = lc.apply_leace(e, x)
    # the class means coincide, coordinates 1.. are untouched (up to the tiny estimation tilt of the fitted direction)
    assert abs(xe[y == 0, 0].mean() - xe[y == 1, 0].mean()) < 1e-8
    assert np.allclose(xe[:, 1:], x[:, 1:], atol=0.1)
    assert lc.cross_cov_ratio(e, x, lc.one_hot(y, [0, 1])) < 1e-8


def test_cross_covariance_vanishes_with_correlated_features():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(5000, 8))
    x = a @ rng.normal(size=(8, 8))   # strongly correlated coordinates
    y = rng.integers(0, 3, size=5000)
    x = x + np.outer(y == 1, rng.normal(size=8)) * 2 + np.outer(y == 2, rng.normal(size=8)) * 3
    z = lc.one_hot(y, [0, 1, 2])
    e = lc.fit_leace(x, z, **TOL)
    assert e["rank"] == 2   # three classes -> at most two directions
    assert lc.cross_cov_ratio(e, x, z) < 1e-8
    xe = lc.apply_leace(e, x)
    for c in (0, 1, 2):   # class means all equal
        assert np.allclose(xe[y == c].mean(axis=0), xe.mean(axis=0), atol=1e-8)


def test_idempotent_and_low_rank_change():
    x, y = _toy(d=10)
    z = lc.one_hot(y, [0, 1])
    e = lc.fit_leace(x, z, **TOL)
    xe = lc.apply_leace(e, x)
    assert np.allclose(lc.apply_leace(e, xe), xe, atol=1e-8)   # erasing twice = once
    assert np.linalg.matrix_rank(x - xe, tol=1e-8) <= e["rank"]


def test_irrelevant_concept_changes_little():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(6000, 200))
    z = lc.one_hot(rng.integers(0, 4, size=6000), [0, 1, 2, 3])   # independent of x: the estimated cross-covariance is noise
    e = lc.fit_leace(x, z, **TOL)
    assert e["rank"] == 3
    # three random directions of 200 isotropic ones are removed: relative change about sqrt(3/200) = 0.12
    rel = np.linalg.norm(lc.apply_leace(e, x) - x) / np.linalg.norm(x - x.mean(axis=0))
    assert rel == pytest.approx(np.sqrt(3 / 200), abs=0.04)


def test_constant_concept_is_refused():
    x, _ = _toy()
    with pytest.raises(AssertionError, match="no linear information"):
        lc.fit_leace(x, np.ones((len(x), 1)), **TOL)


def test_probe_drops_to_the_majority_rate_after_erasure():
    x, y = _toy(n=3000, d=6)
    z = lc.one_hot(y, [0, 1])
    acc0, maj = lc.probe_accuracy(x[:2000], y[:2000], x[2000:], y[2000:], 1.0, 200, 0)
    e = lc.fit_leace(x[:2000], z[:2000], **TOL)
    xe = lc.apply_leace(e, x)
    acc1, maj1 = lc.probe_accuracy(xe[:2000], y[:2000], xe[2000:], y[2000:], 1.0, 200, 0)
    assert acc0 > 0.9 and acc1 <= maj1 + 0.05 and maj == maj1


def test_linear_r2_is_positive_before_and_near_zero_after_erasure():
    x, y = _toy(n=4000, d=6)
    z = lc.one_hot(y, [0, 1])
    e = lc.fit_leace(x[:2500], z[:2500], **TOL)
    before = lc.linear_r2(x[:2500], z[:2500], x[2500:], z[2500:], 1.0)
    xe = lc.apply_leace(e, x)
    after = lc.linear_r2(xe[:2500], z[:2500], xe[2500:], z[2500:], 1.0)
    assert before > 0.4 and abs(after) < 0.01
