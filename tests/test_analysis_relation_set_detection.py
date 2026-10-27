"""Unit tests for analysis/relation_set_detection.py on the relation_detection_mini line fixture (see
tests/test_analysis_relation_detection.py for its items and samples).

The fixture's only full (m = 5) record pair is doc-a intra_document:m5:k2: li0 shown [CH, DESC, SD, ED, AMT]
(spans 0-4 invalid, 5-9 twin), ED <- li1 (01/20/20), AMT <- li2 ($400.00). Its 26 subsets of sizes 2-5:
the 4 of {CH, DESC, SD} (sizes 2, 3) are still li0's -> context, ``hidden`` (ED and AMT are shown after
them); the other 22 contain ED or AMT and are no item's -> matched. The other three fixture pairs (m2, m3)
are dropped. ``make_split`` gives span ``i`` the activation ``X[i] = [2i, 2i + 1]``.
"""
from __future__ import annotations

import importlib.util
from math import comb
from pathlib import Path

import numpy as np
import pytest

from test_analysis_relation_detection import AMT, CH, DESC, ED, EQ, ITEMS, ROWS, SD, make_split, rr
from vrdu_common import item_keys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analysis_relation_set_detection", ROOT / "analysis" / "relation_set_detection.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)

SIZES = [2, 3, 4, 5]
M5 = [r for r in ROWS if r["cell"] == "intra_document:m5:k2"]


def make_data(rows=ROWS, items=ITEMS):
    return {"rows": {(r["pair_id"], r["valid"]): r for r in rows}, "eq": EQ,
            "item_keys": {d: item_keys(its, rr.LINE_FIELDS, EQ) for d, its in items.items()}}


def subsets_of(rows=ROWS, items=ITEMS, sizes=SIZES):
    s = make_split(rows)
    return s, rs.build_subsets(s, rr.group_rows(s), make_data(rows, items), sizes)


def reorder(rows, order):
    """The rows with their shown fields in ``order`` (pair ids unchanged: they are opaque keys)."""
    return [{**r, "fields": list(order)} for r in rows]


def test_subsets_hand_worked():
    s, z = subsets_of()
    assert z["n_records"] == 1 and z["n_samples"] == 8
    assert len(z["inv_rows"]) == 22 and len(z["ctx_inv_rows"]) == 4
    assert sum(comb(5, k) for k in SIZES) == 26
    assert set(z["ctx_fields"].tolist()) == {f"{CH},{DESC}", f"{CH},{SD}", f"{DESC},{SD}", f"{CH},{DESC},{SD}"}
    assert set(z["ctx_context_kind"].tolist()) == {"hidden"}
    assert set(z["cell"].tolist()) == {"intra_document:m5:k2"} and set(z["error_type"].tolist()) == {"intra_document"}
    # sizes: 10 + 10 + 5 + 1 = 26, minus 3 + 1 context
    assert [int((z["size"] == k).sum()) for k in SIZES] == [7, 9, 5, 1]
    # members in shown order, invalid record spans 0-4, twin 5-9
    k = z["fields"].tolist().index(f"{DESC},{ED},{AMT}")
    assert z["inv_rows"][k].tolist() == [1, 3, 4] and z["val_rows"][k].tolist() == [6, 8, 9]
    assert z["n_corrupted"][k] == 2 and z["last_corrupted"][k]
    k = z["fields"].tolist().index(f"{CH},{SD},{ED}")
    assert z["n_corrupted"][k] == 1 and z["last_corrupted"][k]
    k = z["fields"].tolist().index(f"{ED},{AMT}")
    assert z["n_corrupted"][k] == 2 and z["inv_rows"][k].tolist() == [3, 4]
    k = z["fields"].tolist().index(f"{CH},{DESC},{ED}")
    assert z["n_corrupted"][k] == 1 and z["last_corrupted"][k]
    k = z["fields"].tolist().index(f"{CH},{ED},{AMT}")
    assert z["n_corrupted"][k] == 2
    assert (z["n_corrupted"] >= 1).all() and (z["ctx_n_corrupted"] == 0).all()


def test_sizes_select_subsets():
    _, z = subsets_of(sizes=[5])
    assert len(z["inv_rows"]) == 1 and "ctx_inv_rows" not in z
    with pytest.raises(AssertionError):
        subsets_of(sizes=[1, 2])


def test_context_kinds_visible_and_corrupted_member():
    # ED shown first: the clean subsets of {CH, DESC, SD} now follow a corrupted field -> visible
    rows = reorder(M5, [ED, CH, DESC, SD, AMT])
    _, z = subsets_of(rows)
    assert len(z["ctx_inv_rows"]) == 4 and set(z["ctx_context_kind"].tolist()) == {"visible"}
    # an item with CH 7 and ED 01/20/20 makes (CH, ED) of the invalid record valid despite its corrupted ED
    extra = {CH: "7", DESC: None, SD: None, ED: "01/20/20", AMT: None}
    _, z = subsets_of(M5, {**ITEMS, "doc-a": ITEMS["doc-a"] + [extra]})
    kinds = dict(zip(z["ctx_fields"].tolist(), z["ctx_context_kind"].tolist()))
    assert kinds[f"{CH},{ED}"] == "corrupted_member" and len(z["inv_rows"]) == 21


def test_features_hand_worked():
    s, z = subsets_of()
    x = s["X"]
    k = z["fields"].tolist().index(f"{DESC},{ED},{AMT}")   # spans 1, 3, 4
    rows = [z["inv_rows"][k]]
    assert rs.subset_features(x, rows, "sum")[0].tolist() == [16, 19]
    assert rs.subset_features(x, rows, "last")[0].tolist() == [8, 9]
    assert rs.subset_features(x, rows, "context")[0].tolist() == [8, 10]
    st = rs.stack(z)
    assert len(st["rows"]) == 44 and st["y"].sum() == 22 and (st["fields"][:22] == st["fields"][22:]).all()
    with pytest.raises(AssertionError, match="unknown feature"):
        rs.subset_features(x, rows, "cond")


def test_prefix_match_dev_is_zero_only_for_equal_prefixes():
    s, z = subsets_of()
    dev, share = rs.prefix_match_dev(s["X"], z)        # arange activations: the twin's spans differ
    assert dev > 0 and share == 0.0
    x = s["X"].copy()
    x[5:10] = x[0:5]                                   # twin activations = invalid record's
    assert rs.prefix_match_dev(x, z) == (0.0, 1.0)
    x[7] *= 1.01                                       # SD in the twin: in 3 of the 4 hidden subsets
    dev, share = rs.prefix_match_dev(x, z)
    assert 0 < dev < 0.01 and share == 0.25


def test_paired_bootstrap_diff_of_identical_probes_is_zero():
    y = np.array([False, True] * 4)
    p = np.linspace(0.1, 0.9, 8)
    docs = np.repeat(["a", "b", "c", "d"], 2)
    assert rs.doc_bootstrap_auroc_diff(y, p, p, docs, 50, 0.9, 0) == (0.0, 0.0)


def test_a_twin_subset_that_is_no_items_is_refused():
    items = {**ITEMS, "doc-a": [{**ITEMS["doc-a"][0], DESC: "Film"}] + ITEMS["doc-a"][1:]}
    with pytest.raises(AssertionError, match="no item's"):
        subsets_of(items=items)


def test_a_pair_not_judged_whole_is_refused():
    s = make_split(ROWS[:-1])
    with pytest.raises(AssertionError, match="not judged whole"):
        rs.build_subsets(s, rr.group_rows(s), make_data(), SIZES)
