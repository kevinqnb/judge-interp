"""Unit tests for analysis/relation_detection.py on the hand-built relation_detection_mini line fixture
(tests/fixtures/make_line_fixtures.py).

Line items:  doc-a li0 = {7, Movie, 01/05/20, 01/10/20, $100.00}; li1 = {8, null, 01/11/20, 01/20/20, $250.00};
                   li2 = {8, News, 01/21/20, 01/31/20, $400.00};   doc-b li0 = {9, Weather, 02/02/20, 02/09/20, $300.00}
Samples (each an invalid sample and its twin; one prompt per sample, spans in shown order):
    intra_document:m5:k2  doc-a li0, all five; end date <- li1 (01/20/20), sub_amount <- li2 ($400.00)
        10 field pairs: (CH, DESC), (CH, SD), (DESC, SD) are still li0's -> context; the other 7 are invalid
    cross_item:m3:k2      doc-a li0 [AMT, CH, ED], CH <-> AMT: all 3 field pairs invalid
    intra_document:m2:k1  doc-a li1 [ED, CH], CH <- li0's "7": (CH, ED) = (7, 01/20/20) is no item's -> invalid
    cross_item:m2:k2      doc-b [DESC, CH] swapped: (CH, DESC) invalid
-> 12 matched field pairs (each with the twin's same pair), 3 context pairs.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analysis_relation_detection", ROOT / "analysis" / "relation_detection.py")
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)
sys.path.insert(0, str(ROOT / "data" / "vrdu"))
from vrdu_common import field_kinds, item_keys, make_eq  # noqa: E402

CH, DESC, SD, ED, AMT = rr.LINE_FIELDS
KINDS = {"name": ["advertiser", "agency", "property", "tv_address", "product"],
         "date": ["flight_from", "flight_to", "program_start_date", "program_end_date"],
         "amount": ["gross_amount", "sub_amount"]}
EQ = make_eq(field_kinds(KINDS), 50)
ROWS = json.loads((ROOT / "tests/fixtures/relation_detection_mini/vrdu/line/train.json").read_text())
ITEMS = {"doc-a": [{CH: "7", DESC: "Movie", SD: "01/05/20", ED: "01/10/20", AMT: "$100.00"},
                   {CH: "8", DESC: None, SD: "01/11/20", ED: "01/20/20", AMT: "$250.00"},
                   {CH: "8", DESC: "News", SD: "01/21/20", ED: "01/31/20", AMT: "$400.00"}],
         "doc-b": [{CH: "9", DESC: "Weather", SD: "02/02/20", ED: "02/09/20", AMT: "$300.00"}]}


def make_split(rows=ROWS):
    """Spans of a run that judged ``rows``: one prompt per row, spans in shown order."""
    cols, ordinal = defaultdict(list), defaultdict(int)
    for r in rows:
        o = ordinal[r["document_id"]]
        ordinal[r["document_id"]] += 1
        for pos, f in enumerate(r["fields"], start=1):
            for c, v in (("doc_ids", r["document_id"]), ("row_ordinal", o), ("field", f), ("field_position", pos),
                         ("value", r[f]), ("label_valid", r["field_errors"][f] is None),
                         ("row_error_type", r["error_type"] or ""), ("pair_id", r["pair_id"]), ("cell", r["cell"])):
                cols[c].append(v)
    out = {k: np.array(v) for k, v in cols.items()}
    out["X"] = np.arange(2 * len(out["field"]), dtype=np.float32).reshape(-1, 2)
    return out


def make_data(items=ITEMS):
    return {"rows": {(r["pair_id"], r["valid"]): r for r in ROWS}, "eq": EQ,
            "item_keys": {d: item_keys(its, rr.LINE_FIELDS, EQ) for d, its in items.items()}}


def pairs_of(rows=ROWS, data=None):
    s = make_split(rows)
    return s, rr.build_pairs(s, rr.group_rows(s), data or make_data())


def test_pairs_are_the_ten_unordered_field_pairs():
    assert len(rr.PAIRS) == 10 and rr.PAIRS[0] == (CH, DESC) and rr.PAIRS[-1] == (ED, AMT)


def test_group_rows_keeps_shown_order_and_rejects_a_mixed_sample():
    s = make_split()
    g = rr.group_rows(s)
    assert len(g) == len(ROWS) and [x["fields"] for x in g] == [r["fields"] for r in ROWS]
    assert [x["valid"] for x in g] == [r["valid"] for r in ROWS]
    s["pair_id"][1] = "other"
    with pytest.raises(AssertionError, match="varies"):
        rr.group_rows(s)


def test_matched_pairs_hand_worked():
    s, z = pairs_of()
    assert len(z["inv_i"]) == 12 and len(z["ctx_inv_i"]) == 3
    assert z["n_samples"] == 8 and z["n_invalid_samples_without_invalid_pair"] == 0
    cells = z["cell"].tolist()
    assert [cells.count(c) for c in ("intra_document:m5:k2", "cross_item:m3:k2", "intra_document:m2:k1", "cross_item:m2:k2")] == [7, 3, 1, 1]
    ctx = {rr.PAIRS[p] for p in z["ctx_pair"]}
    assert ctx == {(CH, DESC), (CH, SD), (DESC, SD)} and set(z["ctx_cell"].tolist()) == {"intra_document:m5:k2"}
    # each member: invalid spans in the invalid sample, the same fields (in canonical order) in the twin
    for a, b, c, d, p in zip(z["inv_i"], z["inv_j"], z["val_i"], z["val_j"], z["pair"]):
        assert (s["field"][a], s["field"][b]) == (s["field"][c], s["field"][d]) == rr.PAIRS[p]
        assert s["pair_id"][a] == s["pair_id"][c] and s["row_error_type"][a] != "" and s["row_error_type"][c] == ""
    st = rr.stack(z)
    assert st["y"].sum() == 12 and len(st["y"]) == 24 and (st["cell"][:12] == st["cell"][12:]).all()


def test_pair_features_concatenate_in_canonical_field_order():
    s, z = pairs_of()
    st = rr.stack(z)
    f = rr.pair_features(s["X"], st)
    assert f.shape == (24, 4)
    # cross_item m3: shown [AMT, CH, ED]; the (CH, AMT) pair is [X[CH span], X[AMT span]]
    k = next(n for n in range(24) if rr.PAIRS[st["pair"][n]] == (CH, AMT) and st["cell"][n] == "cross_item:m3:k2" and not st["y"][n])
    assert s["field"][st["i"][k]] == CH and f[k].tolist() == [*s["X"][st["i"][k]], *s["X"][st["j"][k]]]


def test_a_pair_not_judged_whole_is_refused():
    with pytest.raises(AssertionError, match="not judged whole"):
        pairs_of(ROWS[:-1])


def test_a_span_that_disagrees_with_the_row_file_is_refused():
    s = make_split()
    s["value"][0] = "zz"
    with pytest.raises(AssertionError, match="row file"):
        rr.build_pairs(s, rr.group_rows(s), make_data())


def test_a_twin_pair_that_is_no_items_is_refused():
    items = {**ITEMS, "doc-b": [{**ITEMS["doc-b"][0], CH: "10"}]}   # doc-b's twin (Weather, 9) is no item's now
    with pytest.raises(AssertionError, match="no item's"):
        pairs_of(data=make_data(items))
