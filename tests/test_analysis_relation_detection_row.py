"""Unit tests for analysis/relation_detection_row.py on a hand-built fixture.

Doc "a", two line-item rows (LINE_FIELDS = channel, program_desc, program_start_date,
program_end_date, sub_amount):
    row 0 (clean):               channel, program_desc, sub_amount        all valid     -> spans 0,1,2
    row 1 (inter_document):      channel, program_desc, program_start_date
                                 program_desc is the invalid entry                      -> spans 3,4,5
Doc "b", one row with a single field (too small to form a pair)                         -> span 6

Activations are 2-D so distances are checked against 3-4-5 triangles.
"""
from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
spec = importlib.util.spec_from_file_location(
    "analysis_relation_detection_row", Path(__file__).resolve().parents[1] / "analysis" / "relation_detection_row.py"
)
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)

CH, DESC, SD, ED, AMT = rr.LINE_FIELDS


def make_split():
    return {
        "doc_ids": np.array(["a"] * 6 + ["b"]),
        "row_ordinal": np.array([0, 0, 0, 1, 1, 1, 0]),
        "field": np.array([CH, DESC, AMT, CH, DESC, SD, CH]),
        "field_position": np.array([1, 2, 3, 1, 2, 3, 1]),
        "row_error_type": np.array(["", "", "", "inter_document", "inter_document", "inter_document", ""]),
        "label_valid": np.array([True, True, True, True, False, True, True]),
        "X": np.array([[0, 0], [3, 4], [6, 8], [1, 1], [4, 5], [1, 2], [9, 9]], dtype=np.float32),
    }


def test_pairs_are_the_ten_unordered_field_pairs():
    assert len(rr.PAIRS) == 10 and all(a != b for a, b in rr.PAIRS)
    assert rr.PAIRS[0] == (CH, DESC) and rr.PAIRS[-1] == (ED, AMT)


def test_group_rows():
    gs = rr.group_rows(make_split())
    by = {(g["doc"], g["ordinal"]): g for g in gs}
    assert set(by) == {("a", 0), ("a", 1), ("b", 0)}
    assert by[("a", 1)]["rows"] == [3, 4, 5] and by[("a", 1)]["error_type"] == "inter_document"
    assert by[("a", 0)]["error_type"] == ""


def test_group_rows_rejects_varying_error_type_and_duplicate_field():
    s = make_split()
    s["row_error_type"][4] = "cross_type"
    with pytest.raises(AssertionError):
        rr.group_rows(s)
    s = make_split()
    s["field"][4] = CH
    with pytest.raises(AssertionError):
        rr.group_rows(s)


def test_pair_features_known_distances():
    s = make_split()
    # sample 0: row 0 spans (0, 1, 2): ch=(0,0), desc=(3,4), amt=(6,8) -> d(ch,desc)=5, d(ch,amt)=10, d(desc,amt)=5
    # sample 1: row 0 spans (0, 1) only -> 5, rest absent
    rows = np.array([[0, 1, 2, -1, -1], [0, 1, -1, -1, -1]])
    f, present = rr.pair_features(s["X"], s["field"], rows, absent_fill=0.0)
    j = rr.PAIR_INDEX
    assert f[0, j[(CH, DESC)]] == 5 and f[0, j[(CH, AMT)]] == 10 and f[0, j[(DESC, AMT)]] == 5
    assert present[0].sum() == 3 and (f[0][~present[0]] == 0).all()
    assert f[1, j[(CH, DESC)]] == 5 and present[1].sum() == 1 and (f[1][~present[1]] == 0).all()


def test_pair_features_absent_fill_and_order_invariance():
    s = make_split()
    f, present = rr.pair_features(s["X"], s["field"], np.array([[2, 0, -1, -1, -1]]), absent_fill=-7.0)  # row order flipped
    assert f[0, rr.PAIR_INDEX[(CH, AMT)]] == 10 and (f[0][~present[0]] == -7.0).all()


def test_sample_label_is_invalid_iff_any_span_invalid():
    s = make_split()
    gs = [g for g in rr.group_rows(s) if g["doc"] == "a"]
    z = rr.draw_samples(gs, s["label_valid"], samples_per_row=10, min_subset_size=2, rng=np.random.default_rng(0))
    # row 0: 4 subsets of size >= 2 of 3 spans; row 1: the same 4. All requested (10) > available (4): all 4 each.
    assert len(z["label"]) == 8 and z["stats"]["n_rows_fewer_subsets_than_requested"] == 2
    for rows, lab, et in zip(z["rows"], z["label"], z["error_type"]):
        used = [int(r) for r in rows if r >= 0]
        assert lab == all(s["label_valid"][used])
        if et == "":
            assert lab
    # row 1's subsets containing span 4 (the invalid one): {3,4}, {4,5}, {3,4,5} -> 3 invalid, {3,5} valid
    r1 = z["ordinal"] == 1
    assert int((~z["label"][r1]).sum()) == 3 and int(z["label"][r1].sum()) == 1
    assert int(z["label"][~r1].sum()) == 4  # clean row: every subset valid


def test_draw_samples_drops_rows_below_min_size_and_counts_them():
    s = make_split()
    z = rr.draw_samples(rr.group_rows(s), s["label_valid"], 1, 2, np.random.default_rng(0))
    assert z["stats"]["n_rows_too_small"] == 1 and "b" not in z["doc"].tolist()
    assert len(z["label"]) == 2  # one sample for each of the two usable rows
    assert (z["size"] >= 2).all()


def test_draw_samples_is_seed_deterministic():
    s = make_split()
    gs = rr.group_rows(s)
    a = rr.draw_samples(gs, s["label_valid"], 2, 2, np.random.default_rng([0, 1]))
    b = rr.draw_samples(gs, s["label_valid"], 2, 2, np.random.default_rng([0, 1]))
    assert np.array_equal(a["rows"], b["rows"]) and np.array_equal(a["label"], b["label"])


def test_draw_samples_rejects_min_size_below_two():
    s = make_split()
    with pytest.raises(AssertionError):
        rr.draw_samples(rr.group_rows(s), s["label_valid"], 1, 1, np.random.default_rng(0))


def test_balance_exact_counts_and_hard_error():
    s = make_split()
    z = rr.draw_samples([g for g in rr.group_rows(s) if g["doc"] == "a"], s["label_valid"], 10, 2, np.random.default_rng(0))
    assert int(z["label"].sum()) == 5 and int((~z["label"]).sum()) == 3  # pool: 5 valid, 3 invalid
    b = rr.balance(z, 2, np.random.default_rng(0))
    assert int(b["label"].sum()) == 2 and int((~b["label"]).sum()) == 2 and len(b["rows"]) == 4
    with pytest.raises(ValueError, match="invalid samples available"):
        rr.balance(z, 4, np.random.default_rng(0))


def test_per_document_centering_leaves_within_sample_distances_unchanged():
    """Every span of a sample is from one document, so a per-document shift cancels in the distance."""
    ed = rr.ed
    s = make_split()
    rows = np.array([[0, 1, 2, -1, -1], [3, 4, 5, -1, -1]])
    f0, _ = rr.pair_features(s["X"], s["field"], rows, 0.0)
    x = s["X"].copy()
    ed.center_per_document(x, s["doc_ids"])
    f1, _ = rr.pair_features(x, s["field"], rows, 0.0)
    np.testing.assert_allclose(f0, f1, atol=1e-5)


def test_all_pairs_filled_when_all_fields_present():
    x = np.arange(10, dtype=np.float32).reshape(5, 2)
    fields = np.array(rr.LINE_FIELDS)
    f, present = rr.pair_features(x, fields, np.array([[0, 1, 2, 3, 4]]), 0.0)
    assert present.all() and len(list(itertools.combinations(range(5), 2))) == f.shape[1]


# ------------------------------------------------------------ per-pair standardization
def _pair_fixture():
    # 2 pairs of interest (slots 0 and 1); present entries: slot 0 rows 0,1,2 = 2,4,6 ; slot 1 rows 1,3 = 10,30
    f = np.zeros((4, len(rr.PAIRS)))
    present = np.zeros(f.shape, dtype=bool)
    present[[0, 1, 2], 0], f[[0, 1, 2], 0] = True, [2.0, 4.0, 6.0]
    present[[1, 3], 1], f[[1, 3], 1] = True, [10.0, 30.0]
    for j in range(2, len(rr.PAIRS)):  # fill the remaining slots so every pair has >= 2 present entries
        present[[0, 3], j], f[[0, 3], j] = True, [1.0, 3.0]
    return f, present


def test_fit_pair_stats_uses_present_entries_only():
    f, present = _pair_fixture()
    mean, std = rr.fit_pair_stats(f, present)
    assert mean[0] == 4.0 and std[0] == pytest.approx(np.sqrt(8 / 3))  # absent zeros are ignored
    assert mean[1] == 20.0 and std[1] == 10.0


def test_apply_pair_stats_standardizes_present_and_fills_absent():
    f, present = _pair_fixture()
    mean, std = rr.fit_pair_stats(f, present)
    g = rr.apply_pair_stats(f, present, mean, std, fill=0.0)
    assert g[3, 0] == 0.0 and g[0, 1] == 0.0  # absent -> fill, whatever the raw value
    assert g[1, 1] == -1.0 and g[3, 1] == 1.0
    assert g[present[:, 0], 0].mean() == pytest.approx(0) and g[present[:, 0], 0].std() == pytest.approx(1)
    assert (rr.apply_pair_stats(f, present, mean, std, fill=-9.0)[~present] == -9.0).all()


def test_fit_pair_stats_rejects_too_few_or_constant():
    f, present = _pair_fixture()
    present2 = present.copy()
    present2[[1, 3], 1] = [True, False]  # slot 1 now has one present entry
    with pytest.raises(AssertionError, match="< 2 present"):
        rr.fit_pair_stats(f, present2)
    f2 = f.copy()
    f2[[1, 3], 1] = 5.0
    with pytest.raises(AssertionError, match="constant"):
        rr.fit_pair_stats(f2, present)


def test_make_features_flag_off_is_identity_and_on_uses_train_stats_for_test():
    f, present = _pair_fixture()
    a, b = rr.make_features(f, present, f, present, {"standardize_pair_distances": False, "absent_fill": 0.0})
    assert a is f and b is f
    f_te = f.copy()
    f_te[1, 1] = 20.0  # == train mean of slot 1
    g_tr, g_te = rr.make_features(f, present, f_te, present, {"standardize_pair_distances": True, "absent_fill": 0.0})
    assert g_te[1, 1] == 0.0 and g_tr[1, 1] == -1.0  # test is scaled by TRAIN stats
    with pytest.raises(KeyError):
        rr.make_features(f, present, f, present, {"absent_fill": 0.0})
