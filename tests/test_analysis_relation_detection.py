"""Unit tests for analysis/relation_detection.py on a hand-built fixture.

LINE_FIELDS = channel (CH), program_desc (DESC), program_start_date (SD), program_end_date (ED), sub_amount (AMT).

Row file of doc "a" (valid line items):  li  CH   DESC  SD        ED        AMT
                                          0   c0   d0    -         -         $1
                                          1   c1   d1    01/01/20  01/09/20  $2
                                          2   c2   d2    -         02/09/20  $1
                                          3   c3   d3    -         03/09/20  $3
Spans (one prompt per row; only non-null fields are shown):
    row 0 (clean, li 0)          CH c0, DESC d0, AMT $1                          -> spans 0,1,2
    row 1 (inter_document, li 1) CH c1, DESC "Xd" (corrupted), SD 01/01/20      -> spans 3,4,5
    row 2 (intra_document, li 2) DESC d0 (from li 0), ED 01/09/20 (from li 1), AMT $1 -> spans 6,7,8
    row 3 (intra_document, li 3) DESC d1, ED 01/09/20 (both from li 1)          -> spans 9,10
Doc "b": one row with a single field (no pair)                                   -> span 11

Labels by the match rule: row 2's (DESC d0, AMT $1) is item 0's -> valid although DESC is corrupted;
row 2's (DESC, ED) is no item's -> invalid; row 3's (DESC, ED) is item 1's -> valid.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
spec = importlib.util.spec_from_file_location(
    "analysis_relation_detection", Path(__file__).resolve().parents[1] / "analysis" / "relation_detection.py"
)
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)

CH, DESC, SD, ED, AMT = rr.LINE_FIELDS
IN, INTRA = "inter_document", "intra_document"
KINDS = {"name": ["advertiser", "agency", "property", "tv_address", "product"],
         "date": ["flight_from", "flight_to", "program_start_date", "program_end_date"],
         "amount": ["gross_amount", "sub_amount"]}
ITEMS_A = [
    {CH: "c0", DESC: "d0", SD: None, ED: None, AMT: "$1"},
    {CH: "c1", DESC: "d1", SD: "01/01/20", ED: "01/09/20", AMT: "$2"},
    {CH: "c2", DESC: "d2", SD: None, ED: "02/09/20", AMT: "$1"},
    {CH: "c3", DESC: "d3", SD: None, ED: "03/09/20", AMT: "$3"},
]


def make_split():
    # (doc, ordinal, line_index, row_k, field, value, valid, field_error_type, row_error_type, src_scope, src_field, src_line)
    spans = [
        ("a", 0, 0, 0, CH, "c0", True, "", "", "", "", -1),
        ("a", 0, 0, 0, DESC, "d0", True, "", "", "", "", -1),
        ("a", 0, 0, 0, AMT, "$1", True, "", "", "", "", -1),
        ("a", 1, 1, 1, CH, "c1", True, "", IN, "", "", -1),
        ("a", 1, 1, 1, DESC, "Xd", False, IN, IN, "other_document", "", -1),
        ("a", 1, 1, 1, SD, "01/01/20", True, "", IN, "", "", -1),
        ("a", 2, 2, 2, DESC, "d0", False, INTRA, INTRA, "same_document", DESC, 0),
        ("a", 2, 2, 2, ED, "01/09/20", False, INTRA, INTRA, "same_document", ED, 1),
        ("a", 2, 2, 2, AMT, "$1", True, "", INTRA, "", "", -1),
        ("a", 3, 3, 2, DESC, "d1", False, INTRA, INTRA, "same_document", DESC, 1),
        ("a", 3, 3, 2, ED, "01/09/20", False, INTRA, INTRA, "same_document", ED, 1),
        ("b", 0, 0, 0, CH, "cb", True, "", "", "", "", -1),
    ]
    c = list(zip(*spans))
    n = len(spans)
    return {
        "doc_ids": np.array(c[0]), "row_ordinal": np.array(c[1]), "line_index": np.array(c[2]), "row_k": np.array(c[3]),
        "field": np.array(c[4]), "value": np.array(c[5]), "label_valid": np.array(c[6]), "field_error_type": np.array(c[7]),
        "row_error_type": np.array(c[8]), "source_scope": np.array(c[9]), "source_field": np.array(c[10]),
        "source_line_index": np.array(c[11]), "source_dataset": np.array(["line" if s else "" for s in c[9]]),
        "X": np.arange(2 * n, dtype=np.float32).reshape(n, 2),
    }


def make_data():
    """``load_line_data``'s output for the fixture's row file."""
    def r(doc, li, et, k, values):
        return {"document_id": doc, "line_index": li, "error_type": et, "k": k, **values}
    rows = [r("a", li, None, 0, it) for li, it in enumerate(ITEMS_A)]
    rows.append(r("b", 0, None, 0, {CH: "cb", DESC: None, SD: None, ED: None, AMT: None}))
    rows.append(r("a", 1, IN, 1, {**ITEMS_A[1], DESC: "Xd"}))
    rows.append(r("a", 2, INTRA, 2, {**ITEMS_A[2], DESC: "d0", ED: "01/09/20"}))
    rows.append(r("a", 3, INTRA, 2, {**ITEMS_A[3], DESC: "d1", ED: "01/09/20"}))
    return {"items": rr.items_by_document(rows),
            "rows": {(x["document_id"], x["line_index"], x["error_type"] or "", x["k"]): x for x in rows},
            "eq": rr.make_sample_eq({"kinds": KINDS, "year_pivot": 50})}


def test_pairs_are_the_ten_unordered_field_pairs():
    assert len(rr.PAIRS) == 10 and all(a != b for a, b in rr.PAIRS)
    assert rr.PAIRS[0] == (CH, DESC) and rr.PAIRS[-1] == (ED, AMT)


def test_group_rows():
    by = {(g["doc"], g["ordinal"]): g for g in rr.group_rows(make_split())}
    assert set(by) == {("a", 0), ("a", 1), ("a", 2), ("a", 3), ("b", 0)}
    assert by[("a", 1)]["rows"] == [3, 4, 5] and by[("a", 1)]["error_type"] == IN
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


def test_pair_labels_follow_the_match_rule():
    s = make_split()
    pairs = rr.build_pairs(s, rr.group_rows(s), make_data())
    got = {(int(i), int(j)): bool(l) for i, j, l in zip(pairs["i"], pairs["j"], pairs["label"])}
    # clean row: all valid
    assert [got[k] for k in ((0, 1), (0, 2), (1, 2))] == [True] * 3
    # inter_document: the corrupted DESC makes both of its pairs invalid; CH + SD are item 1's
    assert got[(3, 4)] is False and got[(4, 5)] is False and got[(3, 5)] is True
    # intra row 2: DESC d0 next to AMT $1 is item 0's -> valid; the other two are no item's
    assert got[(6, 8)] is True and got[(7, 8)] is False and got[(6, 7)] is False
    # intra row 3: both values from item 1 -> valid; the singleton row has no pair
    assert got[(9, 10)] is True
    assert len(got) == 3 + 3 + 3 + 1 == len(pairs["i"])
    assert pairs["relabelled"] == {"intra_document_dropped_to_invalid": 1, "intra_document_dropped_to_valid": 1,
                                   "intra_document_invalid_to_valid": 1}


def test_a_span_that_disagrees_with_the_row_file_is_rejected():
    s = make_split()
    s["value"][3] = "c9"
    with pytest.raises(AssertionError, match="row file"):
        rr.build_pairs(s, rr.group_rows(s), make_data())
    s = make_split()
    s["row_k"][3] = 2   # no inter_document k=2 row for line 1
    with pytest.raises(AssertionError, match="no row"):
        rr.build_pairs(s, rr.group_rows(s), make_data())


def test_two_correct_entities_that_are_no_items_are_rejected():
    s = make_split()
    data = make_data()
    data["items"]["a"] = [x for x in data["items"]["a"] if x["line_index"] != 0]   # row 0's item is gone
    with pytest.raises(AssertionError, match="no item"):
        rr.build_pairs(s, rr.group_rows(s), data)


def test_clean_row_with_invalid_pair_is_rejected():
    s = make_split()
    data = make_data()
    data["rows"][("a", 0, "", 0)][CH] = s["value"][0] = "zz"   # the row file agrees, but no valid item has CH zz
    data["items"]["a"][0] = {**data["items"]["a"][0], CH: "c0"}
    s["label_valid"][0] = False
    with pytest.raises(AssertionError):
        rr.build_pairs(s, rr.group_rows(s), data)


def test_pair_features_concatenate_in_field_order():
    s = make_split()
    pairs = rr.build_pairs(s, rr.group_rows(s), make_data())
    f = rr.pair_features(s["X"], pairs)
    assert f.shape == (len(pairs["i"]), 4)
    k = next(n for n, (i, j) in enumerate(zip(pairs["i"], pairs["j"])) if (i, j) == (0, 1))
    assert f[k].tolist() == [0.0, 1.0, 2.0, 3.0]  # X[0] then X[1]
    k = next(n for n, (i, j) in enumerate(zip(pairs["i"], pairs["j"])) if (i, j) == (4, 5))
    assert f[k].tolist() == [8.0, 9.0, 10.0, 11.0]  # DESC before SD
