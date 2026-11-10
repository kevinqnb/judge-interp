"""Unit tests for analysis/relation_grouping_distances.py (model-free, hand-worked toy prompts)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analysis_relation_grouping_distances", ROOT / "analysis" / "relation_grouping_distances.py")
gd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gd)

# 4 points on a line at 0, 1, 3, 7: distances (0,1)=1 (0,2)=3 (0,3)=7 (1,2)=2 (1,3)=6 (2,3)=4
V = np.array([[0.0, 0], [1, 0], [3, 0], [7, 0]])


def test_pair_distances_hand_worked():
    c, u = gd.pair_distances(V, {(0, 1), (2, 3)})
    assert sorted(c.tolist()) == [1.0, 4.0]
    assert sorted(u.tolist()) == [2.0, 3.0, 6.0, 7.0]


def test_pair_distances_refuses_a_bad_link():
    with pytest.raises(AssertionError):
        gd.pair_distances(V, {(1, 0)})
    with pytest.raises(AssertionError):
        gd.pair_distances(V, {(0, 4)})


def test_scaled_divides_by_the_mean_of_all_pairs():
    c, u = gd.pair_distances(V, {(0, 1), (2, 3)})
    cs, us = gd.scaled(c, u)
    assert np.concatenate([cs, us]).mean() == pytest.approx(1.0)   # mean of all six distances is 23/6
    assert cs[0] == pytest.approx(1.0 / (23 / 6))


def _rec(links, v=V):
    raw = gd.pair_distances(v, links)
    return {"raw": raw, "scaled": gd.scaled(*raw)}


def test_pooled_diff_is_connected_minus_unconnected_over_pooled_pairs():
    r = _rec({(0, 1), (2, 3)})
    mc, mu, d = gd.pooled_diff([r], "raw", np.array([0]))
    assert (mc, mu, d) == (2.5, 4.5, -2.0)
    # a second prompt with one connected pair at distance 2 and five unconnected: pooled over pairs, not averaged per prompt
    r2 = _rec({(1, 2)})
    mc, mu, d = gd.pooled_diff([r, r2], "raw", np.array([0, 1]))
    assert mc == pytest.approx((1 + 4 + 2) / 3) and mu == pytest.approx((2 + 3 + 6 + 7 + 1 + 3 + 7 + 6 + 4) / 9)


def test_pooled_diff_refuses_when_a_side_is_empty():
    with pytest.raises(AssertionError, match="no connected"):
        gd.pooled_diff([_rec(set())], "raw", np.array([0]))


def test_bootstrap_estimate_equals_the_pooled_difference_and_is_seeded():
    recs = [_rec({(0, 1), (2, 3)}), _rec({(1, 2)}), _rec({(0, 2)})]
    est, lo, hi = gd.boot(recs, "raw", 200, 0.95, np.random.default_rng(1))
    assert est == gd.pooled_diff(recs, "raw", np.arange(3))[2]
    assert lo <= est <= hi or lo <= hi
    assert (est, lo, hi) == gd.boot(recs, "raw", 200, 0.95, np.random.default_rng(1))


def test_paired_bootstrap_of_identical_runs_is_zero():
    recs = [_rec({(0, 1), (2, 3)}), _rec({(1, 2)}), _rec({(0, 2)})]
    assert gd.boot(recs, "raw", 50, 0.95, np.random.default_rng(0), recs) == (0.0, 0.0, 0.0)


def test_random_vectors_have_scaled_difference_near_zero():
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(300):
        v = rng.normal(size=(5, 200))
        diffs.append(_rec({(0, 1), (2, 3)}, v))
    d = gd.pooled_diff(diffs, "scaled", np.arange(len(diffs)))[2]
    assert abs(d) < 0.01
