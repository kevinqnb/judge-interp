"""Unit + tiny end-to-end tests for data/vrdu/build_vrdu_manifest.py.

Hand-built rows. Line base rows (document, line_index) and the (error_type, k) rows each has:

  (A,0)  inter 1,2   intra 1,2   cross 1,2    -> full support for k_values [1, 2]
  (A,1)  inter 1,2   intra 1     cross 1,2    -> intra lacks k=2: NOT in support
  (B,0)  inter 1,2   intra 1,2   cross 1,2    -> full support
  (C,0)  inter 1,2   intra 1,2   cross 2      -> cross lacks k=1: NOT in support

so the common support is {(A,0), (B,0)}; each selects 1 valid + 3 types x 2 k = 7 rows, 14 in all,
and every (error_type, k) cell has 2 rows. Main is the same shape with two error types.
"""

import json
import random
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "vrdu"))
import build_vrdu_manifest as m

LINE_HAS = {
    ("A", 0): {"inter_document": [1, 2], "intra_document": [1, 2], "cross_type": [1, 2]},
    ("A", 1): {"inter_document": [1, 2], "intra_document": [1], "cross_type": [1, 2]},
    ("B", 0): {"inter_document": [1, 2], "intra_document": [1, 2], "cross_type": [1, 2]},
    ("C", 0): {"inter_document": [1, 2], "intra_document": [1, 2], "cross_type": [2]},
}
MAIN_HAS = {
    ("A",): {"inter_document": [1, 2], "cross_type": [1, 2]},
    ("B",): {"inter_document": [1, 2], "cross_type": [1]},
    ("C",): {"inter_document": [1, 2], "cross_type": [1, 2]},
}


def make_rows(dataset, has):
    rows = []
    for bid, types in has.items():
        ident = {"document_id": bid[0], **({"line_index": bid[1]} if dataset == "line" else {})}
        rows.append({**ident, "valid": True, "error_type": None, "k": 0})
        for et, ks in types.items():
            rows.extend({**ident, "valid": False, "error_type": et, "k": k} for k in ks)
    return rows


def test_common_support_requires_every_type_at_every_k():
    rows = make_rows("line", LINE_HAS)
    assert m.common_support(rows, "line", [1, 2]) == [("A", 0), ("B", 0)]
    assert m.common_support(rows, "line", [1]) == [("A", 0), ("A", 1), ("B", 0)]   # C lacks cross k=1
    assert m.common_support(make_rows("main", MAIN_HAS), "main", [1, 2]) == [("A",), ("C",)]


def test_select_split_pairs_every_cell_on_the_same_base_rows():
    rows = make_rows("line", LINE_HAS)
    keys, info = m.select_split(rows, "line", [1, 2], None, random.Random(0))
    assert info["n_common_support"] == info["n_base_rows"] == 2 and len(keys) == 14
    assert set(info["cells"].values()) == {2} and len(info["cells"]) == 7
    for et in ("inter_document", "intra_document", "cross_type"):
        for k in (1, 2):   # the same two base rows in every cell, at exactly that k
            assert {(key[0], key[1]) for key in keys if key[2] == et and key[3] == k} == {("A", 0), ("B", 0)}
    assert [key for key in keys if key[2] is None] == [["A", 0, None, 0], ["B", 0, None, 0]]


def test_select_split_subsample_is_seeded_and_inside_the_support():
    rows = make_rows("line", LINE_HAS)
    a, _ = m.select_split(rows, "line", [1, 2], 1, random.Random("s"))
    b, _ = m.select_split(rows, "line", [1, 2], 1, random.Random("s"))
    assert a == b and len(a) == 7 and {(k[0], k[1]) for k in a} <= {("A", 0), ("B", 0)}


def test_select_split_refuses_an_empty_support_or_oversized_request():
    rows = make_rows("line", LINE_HAS)
    with pytest.raises(AssertionError):
        m.select_split(rows, "line", [1, 2, 3], None, random.Random(0))   # nobody has k=3
    with pytest.raises(AssertionError):
        m.select_split(rows, "line", [1, 2], 3, random.Random(0))         # only 2 in support
    with pytest.raises(AssertionError):
        m.select_split(rows, "line", [2, 1], None, random.Random(0))      # unsorted k_values


def test_verify_split_entry_catches_unequal_cells():
    rows = make_rows("main", MAIN_HAS)
    keys, info = m.select_split(rows, "main", [1, 2], None, random.Random(0))
    entry = {"k_values": [1, 2], **info, "rows": keys}
    m.verify_split_entry(entry, "main")
    with pytest.raises(AssertionError):
        m.verify_split_entry({**entry, "rows": keys[:-1]}, "main")


def test_apply_manifest_selects_exactly_the_named_rows():
    rows = make_rows("line", LINE_HAS)
    keys, _ = m.select_split(rows, "line", [1, 2], None, random.Random(0))
    chosen = m.apply_manifest(rows, "line", {tuple(k) for k in keys})
    assert len(chosen) == 14 and all((r["document_id"], r["line_index"]) in {("A", 0), ("B", 0)} for r in chosen)
    with pytest.raises(AssertionError):
        m.apply_manifest(rows[:-1], "line", {tuple(k) for k in keys} | {("Z", 0, None, 0)})


@pytest.fixture
def built(tmp_path):
    for dataset, has in (("main", MAIN_HAS), ("line", LINE_HAS)):
        (tmp_path / dataset).mkdir()
        for split in m.SPLITS:
            (tmp_path / dataset / f"{split}.json").write_text(json.dumps(make_rows(dataset, has)))
    cfg = {"id": "t", "params": {"data_root": str(tmp_path), "manifest": {
        "seed": 0,
        "main": {"k_values": [1], "n_base_rows": {"train": None, "test": 2}},
        "line": {"k_values": [1, 2], "n_base_rows": {"train": None, "test": None}},
    }}}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    m.main(["--config", str(cfg_path)])
    return tmp_path


def test_end_to_end_manifest_round_trips_and_has_equal_cells(built):
    keys = m.load_manifest_split(built, "line", "train")
    assert len(keys) == 14
    main_test = m.load_manifest_split(built, "main", "test")
    assert len(main_test) == 2 * (1 + 2 * 1)   # 2 base rows x (valid + inter k=1 + cross k=1)
    rows = json.loads((built / "line" / "train.json").read_text())
    assert len(m.apply_manifest(rows, "line", keys)) == 14


def test_manifest_goes_stale_when_the_invalids_change(built):
    path = built / "line" / "train.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(AssertionError, match="stale"):
        m.load_manifest_split(built, "line", "train")
    m.load_manifest_split(built, "line", "test")   # the untouched split still loads


def test_manifest_is_deterministic(built, tmp_path):
    first = (built / m.MANIFEST_FILE).read_text()
    m.main(["--config", str(built / "cfg.yaml")])
    again = json.loads((built / m.MANIFEST_FILE).read_text())
    assert again["datasets"] == json.loads(first)["datasets"]
