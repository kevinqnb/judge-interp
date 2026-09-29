"""Unit tests for analysis/relation_detection.py on a hand-built fixture.

Doc "a": three line items (fields channel, program_desc, program_start_date):
    line 0: ch=9, desc=6am, date=D1
    line 1: ch=9, desc=6am, date=D1        (duplicate of line 0 -> collapsed)
    line 2: ch=9, desc=7am, date=D1
    line 3: ch=5, desc=noon, date=null      (m=2)
Sources after collapse: A={9,6am,D1}, B={9,7am,D1}, C={5,noon}.
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
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)

CH, DESC, DATE = "channel", "program_desc", "program_start_date"


def tup(i, ch, desc, date):
    return {"row_key": ["a", i, None, 0], "valid": True, "invalid_fields": [],
            "entity_index": {CH: ch, DESC: desc, DATE: date}}


TUPLES = [tup(0, "9", "6am", "D1"), tup(1, "9", "6am", "D1"), tup(2, "9", "7am", "D1"), tup(3, "5", "noon", None),
          {**tup(4, "9", "junk", "D1"), "valid": False}]
KEYS = [(CH, "9"), (CH, "5"), (DESC, "6am"), (DESC, "7am"), (DESC, "noon"), (DATE, "D1")]


def make_split():
    n = len(KEYS)
    return {"doc_ids": np.array(["a"] * n), "entity_type": np.array([k[0] for k in KEYS]),
            "value": np.array([k[1] for k in KEYS]), "docs": ["a"], "tuple_membership": {"a": TUPLES}}


def test_collapse_and_items():
    src = rd.collapse_valid_sources(TUPLES)
    assert [s["line_indices"] for s in src] == [[0, 1], [2], [3]]
    assert src[2]["items"] == frozenset({(CH, "5"), (DESC, "noon")})


def test_join_multi_source_entities():
    info = rd.build_doc_info(make_split())["a"]
    matches = dict(zip(info["keys"], info["matches"]))
    assert matches[(CH, "9")] == [0, 1] and matches[(DATE, "D1")] == [0, 1]  # grey
    assert matches[(DESC, "6am")] == [0] and matches[(CH, "5")] == [2]


def test_join_detects_unrendered_source_item():
    s = make_split()
    for k in ("doc_ids", "entity_type", "value"):
        s[k] = s[k][:-1]  # drop (DATE, D1)
    with pytest.raises(AssertionError):
        rd.build_doc_info(s)


def test_6am_to_7am_swap_is_never_labelled_invalid():
    """Swapping 6am->7am while keeping ch=9,D1 lands on source B: must not be emitted as invalid."""
    info = rd.build_doc_info(make_split())
    z = rd.sample_pairs(info, ["a"], 400, np.random.default_rng(0))
    rows = {k: i for i, k in enumerate(KEYS)}
    inv = np.flatnonzero(~z["label"])
    assert len(inv) == 400 and z["label"].sum() == 400 and z["stats"]["swap_still_valid"] > 0
    sources = [s["items"] for s in info["a"]["sources"]]
    for i in inv:
        items = {KEYS[z["e_row"][i]]} | {KEYS[r] for r in z["other_rows"][i] if r >= 0}
        assert not any(items <= s for s in sources)
    for i in np.flatnonzero(z["label"]):
        items = {KEYS[z["e_row"][i]]} | {KEYS[r] for r in z["other_rows"][i] if r >= 0}
        assert any(items <= s for s in sources)
    assert z["stats"]["rejected_m_lt_2"] == 0  # all sources here have m>=2
    assert (z["k"] >= 1).all() and (z["k"] <= z["m"] - 1).all() and (z["q"] <= z["k"]).all()


def test_m1_tuple_is_rejected_and_counted():
    t = [tup(0, "9", None, None), tup(1, "5", "noon", "D2")]
    keys = [(CH, "9"), (CH, "5"), (DESC, "noon"), (DATE, "D2")]
    split = {"doc_ids": np.array(["a"] * 4), "entity_type": np.array([k[0] for k in keys]),
             "value": np.array([k[1] for k in keys]), "docs": ["a"], "tuple_membership": {"a": t}}
    z = rd.sample_pairs(rd.build_doc_info(split), ["a"], 50, np.random.default_rng(1))
    assert z["stats"]["rejected_m_lt_2"] > 0 and z["label"].sum() == 50


def test_relation_features_hand_values():
    """e=channel (1,0); others: program_desc (0,2) -> cos 0, program_start_date (-3,0) -> cos -1."""
    x = np.array([[1, 0], [0, 2], [-3, 0], [1, 1]], dtype=np.float32)
    et = np.array([CH, DESC, DATE, "sub_amount"])
    f, _ = rd.relation_features(x, et, np.array([0, 3]), np.array([[1, 2, -1, -1], [0, -1, -1, -1]]), absent_fill=0.0)
    f1, pres = rd.relation_features(x, et, np.array([0]), np.array([[1, -1, -1, -1]]), absent_fill=1.0)
    assert pres.sum() == 1 and (f1[~pres] == 1.0).all() and f1[pres][0] == 0.0
    assert f.shape == (2, 10)
    i_cd, i_cs = rd.RELATION_INDEX[(CH, DESC)], rd.RELATION_INDEX[(CH, DATE)]
    i_ca = rd.RELATION_INDEX[(CH, "sub_amount")]
    np.testing.assert_allclose(f[0, i_cd], 0.0, atol=1e-12)
    np.testing.assert_allclose(f[0, i_cs], -1.0)
    np.testing.assert_allclose(f[1, i_ca], 1 / np.sqrt(2))  # e=sub_amount (1,1) vs channel (1,0)
    assert (np.delete(f[0], [i_cd, i_cs]) == 0).all() and np.count_nonzero(f[1]) == 1


def test_filled_slots_identical_between_valid_and_invalid():
    """z and z' share types, so their non-zero patterns must match (no label leak via presence)."""
    info = rd.build_doc_info(make_split())
    z = rd.sample_pairs(info, ["a"], 200, np.random.default_rng(3))
    assert z["stats"]["no_swap_possible"] + z["stats"]["swap_still_valid"] > 0
    x = np.random.default_rng(0).standard_normal((len(KEYS), 4)).astype(np.float32)
    et = np.array([k[0] for k in KEYS])
    f, pres = rd.relation_features(x, et, z["e_row"], z["other_rows"], absent_fill=0.0)
    for pid in np.unique(z["pair_id"]):
        idx = np.flatnonzero(z["pair_id"] == pid)
        assert len(idx) == 2 and z["label"][idx].sum() == 1
        assert np.array_equal(pres[idx[0]], pres[idx[1]])
    assert (pres.sum(axis=1) == z["k"]).all()


def test_determinism():
    info = rd.build_doc_info(make_split())
    a = rd.sample_pairs(info, ["a"], 100, np.random.default_rng([0, 1]))
    b = rd.sample_pairs(info, ["a"], 100, np.random.default_rng([0, 1]))
    for k in ("e_row", "other_rows", "label", "k", "q"):
        assert np.array_equal(a[k], b[k])


def test_type_pattern_identical_across_classes():
    """Paired design: the multiset of filled-slot patterns is the same for valid and invalid."""
    info = rd.build_doc_info(make_split())
    z = rd.sample_pairs(info, ["a"], 300, np.random.default_rng(4))
    et = np.array([k[0] for k in KEYS])
    _, pres = rd.relation_features(np.eye(len(KEYS), 8, dtype=np.float32) + 1, et, z["e_row"], z["other_rows"], absent_fill=0.0)
    pv, pi = pres[z["label"]], pres[~z["label"]]
    assert sorted(map(tuple, pv.tolist())) == sorted(map(tuple, pi.tolist()))
