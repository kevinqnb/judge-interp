"""Unit + tiny end-to-end tests for data/vrdu/build_vrdu_line.py.

Fixture (``make_base``), per split: document ``<P>A`` with 3 line items and document ``<P>B`` with 1,
every value distinct across the whole document, all dates parsing, and every start date before every
end date (so no corruption can break date order). Expected by construction, with cells
m in [2, 3], max_k = 2, n_per_cell = 3:

  - 6 cells: intra (2,1) (2,2) (3,1) (3,2), cross_item (2,2) (3,2); 6 x 3 x 2 = 36 samples per split.
  - intra cells: only A's 3 items qualify (B has no donor), so each intra cell is exactly A's 3 items.
  - cross_item cells: 4 eligible items in 2 documents; the round-robin draw takes B's item in its first
    pass, so B's item is in every cross_item cell.
  - intra k=3 is infeasible (A has only 2 donors per item); n_per_cell = 4 is short in every intra cell.
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "vrdu"))
import build_vrdu_line as b  # noqa: E402

KINDS = {"name": ["advertiser", "agency", "property", "tv_address", "product"],
         "date": ["flight_from", "flight_to", "program_start_date", "program_end_date"],
         "amount": ["gross_amount", "sub_amount"]}
DATES = ("program_start_date", "program_end_date")
PIVOT = 50
EQ = b.make_eq(b.field_kinds(KINDS), PIVOT)
LINE = {"m_range": [2, 3], "max_k": 2, "n_per_cell": {"train": 3, "test": 3}}
PREFIX = {"train": "T", "test": "S"}


def item(doc, li, **override):
    return {"document_id": doc, "line_index": li, "valid": True, "channel": f"{doc}c{li}", "program_desc": f"{doc}d{li}",
            "program_start_date": f"01/0{li + 1}/20", "program_end_date": f"02/0{li + 1}/20",
            "sub_amount": f"{(li + 1) * 100}", **override}


def make_base(split):
    p = PREFIX[split]
    return [item(f"{p}A", li) for li in range(3)] + [item(f"{p}B", 0)]


def write_root(root, line=LINE):
    (root / "line").mkdir(parents=True, exist_ok=True)
    for split in b.SPLITS:
        (root / "line" / f"{split}_base.json").write_text(json.dumps(make_base(split)))
    cfg = {"id": "t", "seed": 0, "params": {"data_root": str(root), "dates": {"line_fields": list(DATES), "year_pivot": PIVOT},
                                            "cross_type": {"kinds": KINDS}, "line": line}}
    path = root / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def build(split="test", line=LINE):
    return b.build_split(make_base(split), line, split, 0, EQ, DATES, PIVOT)


# ---------------------------------------------------------------- cells, draw


def test_make_cells():
    assert b.make_cells([2, 3], 2) == [("intra_document", 2, 1), ("intra_document", 2, 2), ("intra_document", 3, 1),
                                       ("intra_document", 3, 2), ("cross_item", 2, 2), ("cross_item", 3, 2)]
    assert len(b.make_cells([2, 5], 3)) == 15   # 11 intra + 4 cross_item: the -06 grid
    with pytest.raises(AssertionError):
        b.make_cells([1, 3], 2)


def test_round_robin_takes_every_document_first():
    by_doc = {"A": list(range(5)), "B": ["x"]}
    for seed in range(5):
        assert "x" in b.draw_round_robin(by_doc, 2, random.Random(seed))
    with pytest.raises(ValueError, match="only 6 candidates"):
        b.draw_round_robin(by_doc, 7, random.Random(0))


# ---------------------------------------------------------------- known answers


def test_intra_rejects_a_sample_that_is_another_item():
    # X and Y share channel: corrupting sub_amount on {channel, sub_amount} shows exactly Y (the label rule).
    x = item("D", 0, channel="same", sub_amount="100")
    y = item("D", 1, channel="same", sub_amount="200")
    for seed in range(10):
        assert b.intra_corruption(x, [x, y], ("channel", "sub_amount"), 1, random.Random(seed), EQ, DATES, PIVOT, 0.0) is None
        assert b.intra_corruption(x, [x, y], ("program_desc", "sub_amount"), 1, random.Random(seed), EQ, DATES, PIVOT, 0.0) is not None


def test_intra_keeps_date_order():
    # Y's start (01/10) is after X's end (01/05): only the end date can be corrupted.
    x = item("D", 0, program_start_date="01/01/20", program_end_date="01/05/20")
    y = item("D", 1, program_start_date="01/10/20", program_end_date="01/20/20")
    for seed in range(10):
        values, sources = b.intra_corruption(x, [x, y], DATES, 1, random.Random(seed), EQ, DATES, PIVOT, 0.0)
        assert values == {"program_start_date": "01/01/20", "program_end_date": "01/20/20"} and sources == {"program_end_date": 1}


def test_cross_item_rejects_a_date_reversing_swap():
    x = item("D", 0, program_start_date="01/01/20", program_end_date="01/05/20")
    assert b.cross_item_corruption(x, [x], DATES, random.Random(0), EQ, DATES, PIVOT) is None
    frag = item("D", 0, program_start_date="/01/20", program_end_date="01/05/20")   # a fragment has no order
    values, swap = b.cross_item_corruption(frag, [frag], DATES, random.Random(0), EQ, DATES, PIVOT)
    assert values == {"program_start_date": "01/05/20", "program_end_date": "/01/20"} and swap == DATES


def test_cross_item_skips_noop_swaps():
    x = item("D", 0, channel="7", program_desc="7")
    assert b.cross_item_corruption(x, [x], ("channel", "program_desc"), random.Random(0), EQ, DATES, PIVOT) is None


# ---------------------------------------------------------------- the fixture split


def test_fixture_counts_and_cells():
    samples, stats = build()
    assert len(samples) == 36
    by_cell = Counter(s["cell"] for s in samples)
    assert set(by_cell.values()) == {6}
    for s in samples:
        if s["cell"].startswith("intra_document"):
            assert s["document_id"] == "SA"
    for cid in ("cross_item:m2:k2", "cross_item:m3:k2"):
        assert "SB" in {s["document_id"] for s in samples if s["cell"] == cid}
        assert stats["cells"][cid]["eligible_items"] == 4 and stats["cells"][cid]["n_docs"] == 2
    for cid in ("intra_document:m2:k1", "intra_document:m3:k2"):
        assert stats["cells"][cid]["eligible_items"] == 3 and stats["cells"][cid]["n_docs"] == 1


def test_twins_match_their_invalid_sample_and_item():
    samples, _ = build()
    base = {(r["document_id"], r["line_index"]): r for r in make_base("test")}
    pairs = {}
    for s in samples:
        pairs.setdefault(s["pair_id"], {})[s["valid"]] = s
    assert len(pairs) == 18
    for p in pairs.values():
        bad, good = p[False], p[True]
        assert bad["fields"] == good["fields"] and list(good["values"]) == good["fields"]
        r = base[(good["document_id"], good["line_index"])]
        assert good["values"] == {f: r[f] for f in good["fields"]}
        assert 1 <= bad["k"] == len(bad["invalid_fields"]) <= bad["m"] == len(bad["fields"])   # k counted within the subset


def test_classes_have_identical_structure_in_every_cell():
    samples, _ = build()
    for cid in {s["cell"] for s in samples}:
        def shape(valid):
            return Counter((s["document_id"], s["line_index"], tuple(s["fields"])) for s in samples
                           if s["cell"] == cid and s["valid"] == valid)
        assert shape(True) == shape(False)


def test_infeasible_cells_raise():
    with pytest.raises(ValueError, match=r"intra_document:m2:k1: only 3 candidates"):
        build(line={**LINE, "n_per_cell": {"train": 4, "test": 4}})
    with pytest.raises(ValueError, match=r"intra_document:m3:k3: only 0 candidates"):
        build(line={**LINE, "m_range": [3, 3], "max_k": 3})


def test_verify_catches_tampering():
    samples, _ = build()
    base, cells = make_base("test"), b.make_cells(LINE["m_range"], LINE["max_k"])
    b.verify_split(samples, base, cells, 3, EQ, DATES, PIVOT)
    twin = next(s for s in samples if s["valid"])
    twin["values"][twin["fields"][0]] = "tampered"
    with pytest.raises(AssertionError):
        b.verify_split(samples, base, cells, 3, EQ, DATES, PIVOT)
    with pytest.raises(AssertionError):
        b.verify_split(build()[0][2:], base, cells, 3, EQ, DATES, PIVOT)


# ---------------------------------------------------------------- end to end


def test_main_is_deterministic(tmp_path):
    outs = []
    for run in ("a", "b"):
        cfg = write_root(tmp_path / run)
        b.main(["--config", str(cfg)])
        outs.append({split: (tmp_path / run / "line" / f"{split}.json").read_bytes() for split in b.SPLITS})
        build_file = json.loads((tmp_path / run / b.BUILD_FILE).read_text())
        assert build_file["cells"] == [b.cell_id(c) for c in b.make_cells([2, 3], 2)]
        assert set(build_file["sha256"]) == {"line/train.json", "line/test.json"}
    assert outs[0] == outs[1]
    train = json.loads(outs[0]["train"])
    assert len(train) == 36 and {s["document_id"] for s in train} == {"TA", "TB"}
