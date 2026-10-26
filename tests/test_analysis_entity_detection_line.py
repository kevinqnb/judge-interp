"""Unit tests for the line-dataset path of analysis/entity_detection.py (``line_span_pairs``), on the
hand-built relation_detection_mini line fixture (see tests/test_analysis_relation_detection.py for its items).

Samples (an invalid sample and its twin each; one span per shown field):
    intra_document:m5:k2  doc-a  end date and sub_amount corrupted; channel, desc, start date kept
    cross_item:m3:k2      doc-a  sub_amount and channel swapped; end date kept
    intra_document:m2:k1  doc-a  channel corrupted; end date kept
    cross_item:m2:k2      doc-b  desc and channel swapped
-> 7 matched entity pairs (intra_document 3, cross_item 4) and 5 kept/clean (still-valid) pairs.
"""
from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analysis_entity_detection", ROOT / "analysis" / "entity_detection.py")
er = importlib.util.module_from_spec(spec)
spec.loader.exec_module(er)

ROWS = json.loads((ROOT / "tests/fixtures/relation_detection_mini/vrdu/line/train.json").read_text())


def make_split(rows=ROWS):
    """Spans of a run that judged ``rows``: one prompt per row, one span per shown field."""
    cols = defaultdict(list)
    for r in rows:
        for pos, f in enumerate(r["fields"], start=1):
            err = r["field_errors"][f]
            for c, v in (("doc_ids", r["document_id"]), ("field", f), ("field_position", pos), ("value", r[f]),
                         ("label_valid", err is None), ("field_error_type", "" if err is None else err["error_type"]),
                         ("row_error_type", r["error_type"] or ""), ("row_k", r["k"]), ("pair_id", r["pair_id"]),
                         ("cell", r["cell"])):
                cols[c].append(v)
    return {k: np.array(v) for k, v in cols.items()}


def pairs_of(rows=ROWS, split_rows=None):
    s = make_split(rows if split_rows is None else split_rows)
    return s, er.line_span_pairs(s, ROWS)


def test_matched_pairs_hand_worked():
    s, z = pairs_of()
    assert len(z["inv"]) == 7 and len(z["kept"]) == 5
    assert sorted(z["error_type"].tolist()) == ["cross_item"] * 4 + ["intra_document"] * 3
    assert not z["null_fill"].any()
    for a, b in zip(z["inv"], z["val"]):
        assert s["field"][a] == s["field"][b] and s["pair_id"][a] == s["pair_id"][b]
        assert not s["label_valid"][a] and s["label_valid"][b] and s["value"][a] != s["value"][b]
        assert s["row_error_type"][a] != "" and s["row_error_type"][b] == ""


def test_kept_spans_are_the_uncorrupted_ones_with_identical_twin_values():
    s, z = pairs_of()
    for a, b in zip(z["kept"], z["clean"]):
        assert s["label_valid"][a] and s["label_valid"][b] and s["value"][a] == s["value"][b]
        assert s["row_error_type"][a] != "" and s["row_error_type"][b] == ""
    assert sorted(z["kept_error_type"].tolist()) == ["cross_item"] + ["intra_document"] * 4


def test_each_pair_carries_its_cell_and_k():
    s, z = pairs_of()
    cells = z["cell"].tolist()
    assert [cells.count(c) for c in ("intra_document:m5:k2", "cross_item:m3:k2", "intra_document:m2:k1", "cross_item:m2:k2")] == [2, 2, 1, 2]
    assert sorted(z["k"].tolist()) == [1, 2, 2, 2, 2, 2, 2]


def test_stacked_set_is_exactly_balanced_in_every_cell_and_field():
    s, z = pairs_of()
    st = er.stack_pairs(z["inv"], z["val"], cell=z["cell"], field=z["field"])
    assert st["y"].sum() == len(z["inv"]) and len(st["y"]) == 14
    for col in ("cell", "field"):
        for v in set(st[col].tolist()):
            assert st["y"][st[col] == v].mean() == 0.5


def test_a_twin_not_judged_is_refused():
    with pytest.raises(AssertionError, match="twin"):
        pairs_of(split_rows=ROWS[:1])


def test_a_span_that_disagrees_with_the_row_file_is_refused():
    s = make_split()
    s["value"][0] = "zz"
    with pytest.raises(AssertionError, match="row file"):
        er.line_span_pairs(s, ROWS)


def test_a_span_label_that_disagrees_with_field_errors_is_refused():
    s = make_split()
    s["label_valid"][0] = not s["label_valid"][0]
    with pytest.raises(AssertionError, match="label"):
        er.line_span_pairs(s, ROWS)


def test_a_corruption_that_left_the_value_unchanged_is_refused():
    rows = json.loads(json.dumps(ROWS))
    bad = rows[0]
    bad[bad["invalid_fields"][0]] = rows[1][bad["invalid_fields"][0]]   # the corrupted value is the original again
    s = make_split(rows)
    with pytest.raises(AssertionError, match="unchanged"):
        er.line_span_pairs(s, rows)
