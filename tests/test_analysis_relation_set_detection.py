"""Unit tests for analysis/relation_set_detection.py on the relation_detection_mini line fixture (see
tests/test_analysis_relation_detection.py for its items and samples).

``make_split`` gives span ``i`` the activation ``X[i] = [2i, 2i + 1]``; spans follow the row file:
    rows 0/1  doc-a intra m5 (invalid/twin)  spans 0-4 / 5-9    [CH, DESC, SD, ED, AMT]
    rows 2/3  doc-a cross m3                 spans 10-12 / 13-15 [AMT, CH, ED]
    rows 4/5  doc-a intra m2                 spans 16-17 / 18-19 [ED, CH]
    rows 6/7  doc-b cross m2                 spans 20-21 / 22-23 [DESC, CH]
Samples are ordered by pair_id (cross m3, intra m5, intra m2, doc-b), invalid members first.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from test_analysis_relation_detection import CH, DESC, ED, ITEMS, ROWS, SD, AMT, make_data, make_split, rr

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analysis_relation_set_detection", ROOT / "analysis" / "relation_set_detection.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)


def samples_of(data=None):
    s = make_split()
    return s, rs.build_samples(s, rr.group_rows(s), data or make_data())


def test_samples_hand_worked():
    s, z = samples_of()
    assert z["n_samples"] == 8 and len(z["rows"]) == 8 and z["y"].tolist() == [False] * 4 + [True] * 4
    assert [r.tolist() for r in z["rows"]] == [[10, 11, 12], [0, 1, 2, 3, 4], [16, 17], [20, 21],
                                               [13, 14, 15], [5, 6, 7, 8, 9], [18, 19], [22, 23]]
    assert z["fields"][0] == f"{AMT},{CH},{ED}" and z["fields"][3] == z["fields"][7] == f"{DESC},{CH}"
    assert z["cell"][:4].tolist() == ["cross_item:m3:k2", "intra_document:m5:k2", "intra_document:m2:k1", "cross_item:m2:k2"]
    assert z["error_type"][1] == "intra_document" and z["doc"][3] == "doc-b"
    assert not z["no_invalid_pair"].any()  # every invalid sample here has an invalid field pair
    # last shown: cross m3 ED (clean), intra m5 AMT (<- li2), intra m2 CH (<- li0), doc-b CH (swapped)
    assert z["last_corrupted"].tolist() == [False, True, True, True] * 2


def test_features_hand_worked():
    s, z = samples_of()
    x = s["X"]
    # sum over spans 0-4 = [2*(0+..+4), 2*(0+..+4) + 5]
    assert rs.sample_features(x, z["rows"], "sum")[1].tolist() == [20, 25]
    # doc-b invalid [DESC, CH]: f_k = CH (span 21), context = DESC (span 20)
    assert rs.sample_features(x, z["rows"], "cond")[3].tolist() == [42, 43, 40, 41]
    # cross m3 [AMT, CH, ED] spans 10-12: f_k = ED (span 12), context = spans 10 + 11
    assert rs.sample_features(x, z["rows"], "cond")[0].tolist() == [24, 25, 42, 44]
    assert rs.sample_features(x, z["rows"], "last")[0].tolist() == [24, 25]
    assert rs.sample_features(x, z["rows"], "context")[0].tolist() == [42, 44]
    f = rs.sample_features(x, z["rows"], "cond")
    assert (f == np.concatenate([rs.sample_features(x, z["rows"], "last"), rs.sample_features(x, z["rows"], "context")], axis=1)).all()
    with pytest.raises(AssertionError, match="unknown feature"):
        rs.sample_features(x, z["rows"], "mean")


def test_an_invalid_sample_with_only_valid_field_pairs_is_flagged():
    # an item on which the doc-a cross m3 invalid sample's every field pair, but not the whole sample, lands:
    # (AMT=7, CH=$100.00), (AMT=7, ED=01/10/20), (CH=$100.00, ED=01/10/20) each matched by a different item
    extra = [{CH: "$100.00", DESC: None, SD: None, ED: "01/10/20", AMT: "$1.00"},
             {CH: "1", DESC: None, SD: None, ED: "01/10/20", AMT: "7"},
             {CH: "$100.00", DESC: None, SD: None, ED: "01/01/99", AMT: "7"}]
    s, z = samples_of(make_data({**ITEMS, "doc-a": ITEMS["doc-a"] + extra}))
    assert z["no_invalid_pair"].tolist() == [True] + [False] * 7


def test_a_twin_that_is_no_items_is_refused():
    items = {**ITEMS, "doc-b": [{**ITEMS["doc-b"][0], CH: "10"}]}
    with pytest.raises(AssertionError, match="no item's"):
        samples_of(make_data(items))


def test_an_invalid_sample_that_is_an_items_is_refused():
    items = {**ITEMS, "doc-a": ITEMS["doc-a"] + [{CH: "7", DESC: None, SD: None, ED: "01/20/20", AMT: None}]}
    with pytest.raises(AssertionError, match="an item's"):
        samples_of(make_data(items))


def test_a_pair_not_judged_whole_is_refused():
    s = make_split(ROWS[:-1])
    with pytest.raises(AssertionError, match="not judged whole"):
        rs.build_samples(s, rr.group_rows(s), make_data())
