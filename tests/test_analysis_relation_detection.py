"""Unit tests for analysis/relation_detection.py on a hand-built fixture.

LINE_FIELDS = channel (CH), program_desc (DESC), program_start_date (SD), program_end_date (ED), sub_amount (AMT).

Doc "a":
    row 0 (clean)                 CH, DESC, AMT all valid                          -> spans 0,1,2
    row 1 (inter_document)        CH valid, DESC invalid, SD valid                 -> spans 3,4,5
    row 2 (intra_document)        DESC and ED invalid, AMT valid                   -> spans 6,7,8
    row 3 (intra_document)        DESC and ED invalid (only two fields)            -> spans 9,10
Doc "b": one row with a single field (no pair)                                     -> span 11
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


def make_split():
    # (doc, ordinal, line_index, field, valid, field_error_type, row_error_type, src_scope, src_field, src_line)
    spans = [
        ("a", 0, 0, CH, True, "", "", "", "", -1),
        ("a", 0, 0, DESC, True, "", "", "", "", -1),
        ("a", 0, 0, AMT, True, "", "", "", "", -1),
        ("a", 1, 1, CH, True, "", IN, "", "", -1),
        ("a", 1, 1, DESC, False, IN, IN, "other_document", "", -1),
        ("a", 1, 1, SD, True, "", IN, "", "", -1),
        ("a", 2, 2, DESC, False, INTRA, INTRA, "same_document", DESC, 0),
        ("a", 2, 2, ED, False, INTRA, INTRA, "same_document", ED, 0),
        ("a", 2, 2, AMT, True, "", INTRA, "", "", -1),
        ("a", 3, 3, DESC, False, INTRA, INTRA, "same_document", DESC, 0),
        ("a", 3, 3, ED, False, INTRA, INTRA, "same_document", ED, 1),
        ("b", 0, 0, CH, True, "", "", "", "", -1),
    ]
    c = list(zip(*spans))
    n = len(spans)
    return {
        "doc_ids": np.array(c[0]), "row_ordinal": np.array(c[1]), "line_index": np.array(c[2]), "field": np.array(c[3]),
        "label_valid": np.array(c[4]), "field_error_type": np.array(c[5]), "row_error_type": np.array(c[6]),
        "source_scope": np.array(c[7]), "source_field": np.array(c[8]), "source_line_index": np.array(c[9]),
        "source_dataset": np.array(["line" if s else "" for s in c[7]]),
        "X": np.arange(2 * n, dtype=np.float32).reshape(n, 2),
    }


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


def test_pair_labels_drop_both_invalid():
    s = make_split()
    pairs = rr.build_pairs(s, rr.group_rows(s))
    got = {(int(i), int(j)): bool(l) for i, j, l in zip(pairs["i"], pairs["j"], pairs["label"])}
    # clean row: all valid
    assert [got[k] for k in ((0, 1), (0, 2), (1, 2))] == [True] * 3
    # inter_document: the invalid DESC poisons both pairs it is in; the valid-valid pair stays valid
    assert got[(3, 4)] is False and got[(4, 5)] is False and got[(3, 5)] is True
    # intra row 2: DESC/ED both invalid -> dropped; each with the valid AMT -> invalid
    assert (6, 7) not in got and got[(6, 8)] is False and got[(7, 8)] is False
    # row 3: its only pair is both-invalid -> dropped; singleton row has no pair
    assert (9, 10) not in got
    assert pairs["n_dropped"] == 2
    assert len(got) == 3 + 3 + 2 == len(pairs["i"])


def test_clean_row_with_invalid_pair_is_rejected():
    s = make_split()
    s["label_valid"][0] = False
    with pytest.raises(AssertionError):
        rr.build_pairs(s, rr.group_rows(s))


def test_pair_features_concatenate_in_field_order():
    s = make_split()
    pairs = rr.build_pairs(s, rr.group_rows(s))
    f = rr.pair_features(s["X"], pairs)
    assert f.shape == (len(pairs["i"]), 4)
    k = next(n for n, (i, j) in enumerate(zip(pairs["i"], pairs["j"])) if (i, j) == (0, 1))
    assert f[k].tolist() == [0.0, 1.0, 2.0, 3.0]  # X[0] then X[1]
    k = next(n for n, (i, j) in enumerate(zip(pairs["i"], pairs["j"])) if (i, j) == (4, 5))
    assert f[k].tolist() == [8.0, 9.0, 10.0, 11.0]  # DESC before SD
