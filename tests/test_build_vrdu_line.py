"""Unit + tiny end-to-end tests for data/vrdu/build_vrdu_line.py.

Fixture (``make_base``), per split: document ``<P>A`` with 6 line items (the last has no program_desc, so
m = 4; the others m = 5) and document ``<P>B`` with 2 (m = 5), every value distinct across the whole document,
all dates parsing, and every start date before every end date (so no corruption can break date order).
Expected by construction, with m_range = [2, 5], max_k = 2, n_per_cell = 2:

  - 3 cells: intra k2, intra k1, cross_item k2 (in that draw order); 3 x 2 x 2 = 12 samples per split.
  - intra k2: only A's items qualify (B's items have 1 donor each), so 6 candidates; n_per_cell = 7 is short.
  - every item is in at most one pair, shown with all its non-null fields in LINE_FIELDS order.
  - subsets: an m = 5 parent has 10 + 10 + 5 + 1 = 26 subsets of size 2-5, an m = 4 parent has 6 + 4 + 1 = 11.
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
LINE = {"m_range": [2, 5], "max_k": 2, "n_per_cell": {"train": 2, "test": 2}, "subset_min_size": 2}
PREFIX = {"train": "T", "test": "S"}


def item(doc, li, **override):
    return {"document_id": doc, "line_index": li, "valid": True, "channel": f"{doc}c{li}", "program_desc": f"{doc}d{li}",
            "program_start_date": f"01/0{li + 1}/20", "program_end_date": f"02/0{li + 1}/20",
            "sub_amount": f"{(li + 1) * 100}", **override}


def make_base(split):
    p = PREFIX[split]
    return ([item(f"{p}A", li) for li in range(5)] + [item(f"{p}A", 5, program_desc=None)]
            + [item(f"{p}B", li) for li in range(2)])


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
    assert b.make_cells([2, 5], 3) == [("intra_document", 3), ("intra_document", 2), ("intra_document", 1), ("cross_item", 2)]
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
    assert len(samples) == 12
    assert Counter(s["cell"] for s in samples) == {"intra_document:k2": 4, "intra_document:k1": 4, "cross_item:k2": 4}
    for s in samples:
        if s["cell"] == "intra_document:k2":
            assert s["document_id"] == "SA"
    assert stats["cells"]["intra_document:k2"]["eligible_items"] == 6
    assert stats["cells"]["intra_document:k1"]["n_invalid"] == 2


def test_twins_match_their_invalid_sample_and_item():
    samples, _ = build()
    base = {(r["document_id"], r["line_index"]): r for r in make_base("test")}
    pairs = {}
    for s in samples:
        pairs.setdefault(s["pair_id"], {})[s["valid"]] = s
    assert len(pairs) == 6
    items = Counter((p[True]["document_id"], p[True]["line_index"]) for p in pairs.values())
    assert max(items.values()) == 1, "an item is in two pairs"
    for p in pairs.values():
        bad, good = p[False], p[True]
        r = base[(good["document_id"], good["line_index"])]
        assert bad["fields"] == good["fields"] == [f for f in b.LINE_FIELDS if r[f] is not None]   # full length, standard order
        assert {f: good[f] for f in b.LINE_FIELDS} == {f: r[f] for f in b.LINE_FIELDS}
        assert 1 <= bad["k"] == len(bad["invalid_fields"]) <= bad["m"] == len(bad["fields"])


def test_classes_have_identical_structure_in_every_cell():
    samples, _ = build()
    for cid in {s["cell"] for s in samples}:
        def shape(valid):
            return Counter((s["document_id"], s["line_index"], tuple(s["fields"])) for s in samples
                           if s["cell"] == cid and s["valid"] == valid)
        assert shape(True) == shape(False)


def test_infeasible_cells_raise():
    with pytest.raises(ValueError, match=r"intra_document:k2: only 6 candidates"):
        build(line={**LINE, "n_per_cell": {"train": 7, "test": 7}})
    with pytest.raises(ValueError, match=r"intra_document:k3: only 0 candidates"):
        build(line={**LINE, "m_range": [3, 3], "max_k": 3})


def test_verify_catches_tampering():
    samples, _ = build()
    base, cells = make_base("test"), b.make_cells(LINE["m_range"], LINE["max_k"])
    args = (base, cells, LINE["m_range"], 2, EQ, DATES, PIVOT)
    b.verify_split(samples, *args)
    twin = next(s for s in samples if s["valid"])
    twin[twin["fields"][0]] = "tampered"
    with pytest.raises(AssertionError):
        b.verify_split(samples, *args)
    with pytest.raises(AssertionError):
        b.verify_split(build()[0][2:], *args)
    reordered = build()[0]
    reordered[0]["fields"] = reordered[0]["fields"][::-1]
    reordered[1]["fields"] = reordered[1]["fields"][::-1]
    with pytest.raises(AssertionError):
        b.verify_split(reordered, *args)


# ---------------------------------------------------------------- subsets


def parent_pair():
    """Items X, Y of document D that share program_desc; the invalid parent is X with Y's channel (k = 1)."""
    x = item("D", 0, channel="cx", program_desc="same", sub_amount="100")
    y = item("D", 1, channel="cy", program_desc="same", sub_amount="200")
    by_id = {("D", 0): x, ("D", 1): y}
    fields = list(b.LINE_FIELDS)
    cand = {"item": x, "fields": fields, "values": {f: y[f] if f == "channel" else x[f] for f in fields},
            "field_errors": {f: {"error_type": b.INTRA_DOCUMENT, "source": b.line_source(f, 1, None)} if f == "channel" else None
                             for f in fields}}
    bad, good = b.make_pair(cand, (b.INTRA_DOCUMENT, 1))
    return [bad, good], [x, y], by_id


def test_subsets_known_answer():
    parents, doc, _ = parent_pair()
    bad_rows = b.subset_rows(parents[0], 0, doc, 2, EQ)
    good_rows = b.subset_rows(parents[1], 1, doc, 2, EQ)
    assert len(bad_rows) == len(good_rows) == 26
    assert Counter(r["size"] for r in bad_rows) == {2: 10, 3: 10, 4: 5, 5: 1}
    assert all(r["valid"] and r["error_type"] is None for r in good_rows)
    for r in bad_rows:
        assert r["fields"] == [f for f in b.LINE_FIELDS if f in r["fields"]]
        assert r["invalid_fields"] == (["channel"] if "channel" in r["fields"] else [])
        assert r["error_type"] == (b.INTRA_DOCUMENT if "channel" in r["fields"] else None)
        if "channel" not in r["fields"]:
            assert r["valid"]                      # X's own values
    # the 15 subsets that show channel: all invalid except those that happen to equal Y (channel + program_desc only)
    shown = [r for r in bad_rows if "channel" in r["fields"]]
    assert len(shown) == 15
    assert {tuple(r["fields"]) for r in shown if r["valid"]} == {("channel", "program_desc")}
    only = next(r for r in shown if r["valid"])
    assert only["error_type"] == b.INTRA_DOCUMENT   # error type present although the label is valid: they may disagree
    assert next(r for r in bad_rows if r["size"] == 5)["valid"] is False   # the full subset keeps its parent's label
    assert next(r for r in good_rows if r["size"] == 5)["valid"] is True


def test_subsets_fewer_fields():
    x = item("D", 0, program_desc=None)
    parent, = [r for r in b.make_pair({"item": x, "fields": [f for f in b.LINE_FIELDS if f != "program_desc"],
                                       "values": {f: x[f] for f in b.LINE_FIELDS if f != "program_desc"},
                                       "field_errors": {f: None for f in b.LINE_FIELDS if f != "program_desc"}},
                                      (b.INTRA_DOCUMENT, 1)) if r["valid"]]
    assert len(b.subset_rows(parent, 0, [x], 2, EQ)) == 6 + 4 + 1


def test_verify_subsets_catches_tampering():
    samples, _ = build()
    base = make_base("test")
    rows, stats = b.build_subsets(samples, base, 2, EQ)
    assert len(rows) == stats["n_subsets"] == sum(26 if s["m"] == 5 else 11 for s in samples)
    b.verify_subsets(rows, samples, base, 2, EQ)
    flipped = [dict(r) for r in rows]
    flipped[0]["valid"] = not flipped[0]["valid"]
    with pytest.raises(AssertionError):
        b.verify_subsets(flipped, samples, base, 2, EQ)
    with pytest.raises(AssertionError):
        b.verify_subsets(rows[1:], samples, base, 2, EQ)
    wrong_parent = [dict(r) for r in rows]
    wrong_parent[0]["parent_valid"] = not wrong_parent[0]["parent_valid"]
    with pytest.raises(AssertionError):
        b.verify_subsets(wrong_parent, samples, base, 2, EQ)


# ---------------------------------------------------------------- end to end


def test_main_is_deterministic(tmp_path):
    outs = []
    for run in ("a", "b"):
        cfg = write_root(tmp_path / run)
        b.main(["--config", str(cfg)])
        files = [f"{split}{suffix}.json" for split in b.SPLITS for suffix in ("", "_subsets")]
        outs.append({f: (tmp_path / run / "line" / f).read_bytes() for f in files})
        build_file = json.loads((tmp_path / run / b.BUILD_FILE).read_text())
        assert build_file["cells"] == [b.cell_id(c) for c in b.make_cells([2, 5], 2)]
        assert set(build_file["sha256"]) == {f"line/{f}" for f in files}
    assert outs[0] == outs[1]
    train = json.loads(outs[0]["train.json"])
    assert len(train) == 12 and {s["document_id"] for s in train} == {"TA", "TB"}
    subsets = json.loads(outs[0]["train_subsets.json"])
    assert {r["parent_id"] for r in subsets} == {s["sample_id"] for s in train}
