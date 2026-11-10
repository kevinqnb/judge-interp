"""Unit tests for analysis/relation_grouping_compare.py (model-free)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analysis_relation_grouping_compare", ROOT / "analysis" / "relation_grouping_compare.py")
gc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gc)

EX = [{"doc": "d1", "pair_ids": ["d1|0|a", "d1|1|b"]}]


def _npz(path, pids, cells, win, twin):
    np.savez(path, pair_id=np.array(pids), cell=np.array(cells), win_frac=np.array(win), twin_win_frac=np.array(twin),
             contrast=np.array(win) - np.array(twin))
    return path


def test_fixed_examples_returns_the_named_prompts_in_order():
    prompts = [{"pair_id": "d1|1|b", "doc": "d1"}, {"pair_id": "d1|0|a", "doc": "d1"}, {"pair_id": "d2|0|a", "doc": "d2"}]
    (ex,) = gc.fixed_examples(EX)(prompts, ["x"], 5, 3, None)
    assert ex["doc"] == "d1" and [q["pair_id"] for q in ex["prompts"]] == ["d1|0|a", "d1|1|b"]


def test_fixed_examples_refuses_a_missing_prompt():
    with pytest.raises(AssertionError, match="not among the kept prompts"):
        gc.fixed_examples(EX)([{"pair_id": "d1|0|a", "doc": "d1"}], ["x"], 5, 3, None)


def test_compare_pairs_baseline_and_new_by_pair_id(tmp_path):
    b = _npz(tmp_path / "b.npz", ["d1|1|b", "d1|0|a", "zz"], ["c2", "c1", "c"], [0.6, 0.7, 0.1], [0.5, 0.5, 0.1])
    n = _npz(tmp_path / "n.npz", ["d1|0|a", "d1|1|b"], ["c1", "c2"], [0.9, 0.2], [0.5, 0.4])
    (r0, r1) = gc.compare(EX, {"m": b}, {"m": n})["m"]
    assert r0["pair_id"] == "d1|0|a" and (r0["baseline_win_frac"], r0["grouping_win_frac"]) == (0.7, 0.9)
    assert r1["cell"] == "c2" and (r1["baseline_twin_win_frac"], r1["grouping_twin_win_frac"]) == (0.5, 0.4)
    assert r0["grouping_contrast"] == pytest.approx(0.4)


def test_compare_refuses_a_missing_or_repeated_prompt(tmp_path):
    b = _npz(tmp_path / "b.npz", ["d1|0|a", "d1|1|b"], ["c1", "c2"], [0.6, 0.7], [0.5, 0.5])
    short = _npz(tmp_path / "s.npz", ["d1|0|a"], ["c1"], [0.9], [0.5])
    with pytest.raises(AssertionError, match="appears"):
        gc.compare(EX, {"m": b}, {"m": short})
    dup = _npz(tmp_path / "d.npz", ["d1|0|a", "d1|0|a", "d1|1|b"], ["c1", "c1", "c2"], [0.9, 0.9, 0.1], [0.5, 0.5, 0.5])
    with pytest.raises(AssertionError, match="appears"):
        gc.compare(EX, {"m": b}, {"m": dup})


def _split(docs, vals):
    n = len(docs)
    d = {"doc_ids": np.array(docs), "value": np.array(vals)}
    for k in gc.ROW_KEYS:
        d.setdefault(k, np.array(["x"] * n))
    return d


def test_check_same_rows_accepts_the_filtered_original_and_refuses_changes():
    orig = _split(["a", "a", "b", "c"], ["1", "2", "3", "4"])
    new = _split(["a", "a", "c"], ["1", "2", "4"])
    assert gc.check_same_rows(new, orig) == 3
    with pytest.raises(AssertionError, match="value"):
        gc.check_same_rows(_split(["a", "a", "c"], ["1", "9", "4"]), orig)
    with pytest.raises(AssertionError):
        gc.check_same_rows(_split(["a", "a", "c", "c"], ["1", "2", "4", "4"]), orig)


def test_compare_records_cross_item_prompts_absent_from_both_files(tmp_path):
    ex = [{"doc": "d1", "pair_ids": ["d1|0|intra_document:k1|f", "d1|1|cross_item:k2|f"]}]
    b = _npz(tmp_path / "b.npz", ["d1|0|intra_document:k1|f"], ["intra_document:k1"], [0.6], [0.5])
    n = _npz(tmp_path / "n.npz", ["d1|0|intra_document:k1|f"], ["intra_document:k1"], [0.8], [0.5])
    r0, r1 = gc.compare(ex, {"m": b}, {"m": n})["m"]
    assert r0["eligible"] and r0["grouping_win_frac"] == 0.8
    assert r1 == {"pair_id": "d1|1|cross_item:k2|f", "cell": "cross_item:k2", "eligible": False}
    only_new = _npz(tmp_path / "o.npz", ["d1|0|intra_document:k1|f", "d1|1|cross_item:k2|f"], ["intra_document:k1", "cross_item:k2"], [0.8, 0.1], [0.5, 0.5])
    with pytest.raises(AssertionError, match="appears"):
        gc.compare(ex, {"m": b}, {"m": only_new})
    ex2 = [{"doc": "d1", "pair_ids": ["d1|9|intra_document:k1|f"]}]
    with pytest.raises(AssertionError, match="not a cross_item"):
        gc.compare(ex2, {"m": b}, {"m": n})
