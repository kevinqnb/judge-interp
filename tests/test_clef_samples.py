"""Tests for experiments/clef_samples.py, on the line build of test_build_vrdu_line.py's fixture and on the
hand-built relation_detection_mini line fixture (see tests/fixtures/make_line_fixtures.py)."""
import json
import sys
from pathlib import Path

import pytest

import test_build_vrdu_line as tl
from dataset_specs import LINE
from judge_interp import clef

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))
import clef_samples as cs  # noqa: E402

MINI = json.loads((REPO_ROOT / "tests/fixtures/relation_detection_mini/vrdu/line/train.json").read_text())


@pytest.fixture
def data_root(tmp_path):
    tl.b.main(["--config", str(tl.write_root(tmp_path))])
    return tmp_path


def test_conditional_primary_is_the_first_field_and_entities_follow_the_row():
    for row in MINI:
        out = cs.to_sample(row, "clef_relation_conditional")
        assert out["primary"]["field"] == row["fields"][0]
        assert [c["field"] for c in out["conditions"]] == row["fields"][1:]
        for e in (out["primary"], *out["conditions"]):
            assert e["value"] == row[e["field"]]
            assert e["valid"] == (e["field"] not in row["invalid_fields"])
        assert (out["error_type"], out["pair_id"], out["cell"]) == (row["error_type"] or "", row["pair_id"], row["cell"])


def test_known_answer_cross_item_pair():
    # doc-a li2 = {8, News, 01/21/20, 01/31/20, $400.00}, all five fields shown;
    # channel <-> sub_amount swapped in the invalid sample, original values in its twin
    pair = [r for r in MINI if r["cell"] == "cross_item:k2" and r["document_id"] == "doc-a"]
    bad, good = sorted(pair, key=lambda r: r["valid"])
    one = cs.to_sample(bad, "clef_record_oneshot")
    assert one["error_type"] == "cross_item" and one["pair_id"] == good["pair_id"]
    assert one["entities"] == [{"field": "channel", "value": "$400.00", "valid": False},
                               {"field": "program_desc", "value": "News", "valid": True},
                               {"field": "program_start_date", "value": "01/21/20", "valid": True},
                               {"field": "program_end_date", "value": "01/31/20", "valid": True},
                               {"field": "sub_amount", "value": "8", "valid": False}]
    twin = cs.to_sample(good, "clef_record_oneshot")
    assert twin["error_type"] == "" and twin["cell"] == "cross_item:k2"
    assert [e["value"] for e in twin["entities"]] == ["8", "News", "01/21/20", "01/31/20", "$400.00"] and all(e["valid"] for e in twin["entities"])


def test_oneshot_keeps_the_field_order_and_label():
    for row in MINI:
        out = cs.to_sample(row, "clef_record_oneshot")
        assert [e["field"] for e in out["entities"]] == row["fields"]
        assert all(e["valid"] for e in out["entities"]) == row["valid"]


def test_malformed_rows_are_refused():
    row = MINI[0]
    with pytest.raises(AssertionError, match="entity validity"):
        cs.to_sample({**row, "valid": not row["valid"]}, "clef_record_oneshot")
    hidden = next(r for r in MINI if len(r["fields"]) < 5)
    f = next(f for f in tl.b.LINE_FIELDS if f not in hidden["fields"])
    with pytest.raises(AssertionError, match="hidden field"):
        cs.to_sample({**hidden, f: "x"}, "clef_record_oneshot")


def test_build_writes_files_that_clef_loads_and_turns_into_units(data_root, tmp_path):
    out_dir = tmp_path / "out"
    prov = cs.build(data_root, "test", out_dir)
    assert sorted(prov["files"]) == ["conditional-test.json", "oneshot-test.json"]
    rows = json.loads((data_root / "line" / "test.json").read_text())
    for name, f in prov["files"].items():
        samples = clef.load_samples(out_dir / name, LINE, f["task"])
        assert [s["sample_id"] for s in samples] == [r["sample_id"] for r in rows]
        assert f["n_samples"] == len(rows) == 2 * f["n_valid"]
        # every pair carries over whole: one invalid and one valid sample with the same cell
        by_pair = {}
        for s in samples:
            by_pair.setdefault(s["pair_id"], []).append(s)
        assert all(len(p) == 2 and p[0]["cell"] == p[1]["cell"] and sorted(bool(s["error_type"]) for s in p) == [False, True]
                   for p in by_pair.values())
        contexts = {x["document_id"]: "text " + x["document_id"] for x in samples}
        units = (clef.conditional_units if f["task"] == "clef_relation_conditional" else clef.oneshot_units)(samples, LINE, contexts)
        assert sum(u["label_valid"] for u in units) == f["n_valid"]
    first = (out_dir / "oneshot-test.json").read_text()
    cs.build(data_root, "test", out_dir)
    assert (out_dir / "oneshot-test.json").read_text() == first


def test_build_refuses_a_stale_split_file(data_root, tmp_path):
    rows = json.loads((data_root / "line" / "test.json").read_text())
    (data_root / "line" / "test.json").write_text(json.dumps(rows[:-1]))
    with pytest.raises(AssertionError, match="stale"):
        cs.build(data_root, "test", tmp_path / "out")
