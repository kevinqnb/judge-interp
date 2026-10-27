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
        out = cs.to_sample(row, "clef_relation_conditional", "pairs")
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
    one = cs.to_sample(bad, "clef_record_oneshot", "pairs")
    assert one["error_type"] == "cross_item" and one["pair_id"] == good["pair_id"]
    assert one["entities"] == [{"field": "channel", "value": "$400.00", "valid": False},
                               {"field": "program_desc", "value": "News", "valid": True},
                               {"field": "program_start_date", "value": "01/21/20", "valid": True},
                               {"field": "program_end_date", "value": "01/31/20", "valid": True},
                               {"field": "sub_amount", "value": "8", "valid": False}]
    twin = cs.to_sample(good, "clef_record_oneshot", "pairs")
    assert twin["error_type"] == "" and twin["cell"] == "cross_item:k2"
    assert [e["value"] for e in twin["entities"]] == ["8", "News", "01/21/20", "01/31/20", "$400.00"] and all(e["valid"] for e in twin["entities"])


def test_oneshot_keeps_the_field_order_and_label():
    for row in MINI:
        out = cs.to_sample(row, "clef_record_oneshot", "pairs")
        assert [e["field"] for e in out["entities"]] == row["fields"]
        assert all(e["valid"] for e in out["entities"]) == row["valid"]


def test_malformed_rows_are_refused():
    row = MINI[0]
    with pytest.raises(AssertionError, match="entity validity"):
        cs.to_sample({**row, "valid": not row["valid"]}, "clef_record_oneshot", "pairs")
    hidden = next(r for r in MINI if len(r["fields"]) < 5)
    f = next(f for f in tl.b.LINE_FIELDS if f not in hidden["fields"])
    with pytest.raises(AssertionError, match="hidden field"):
        cs.to_sample({**hidden, f: "x"}, "clef_record_oneshot", "pairs")


def test_build_writes_files_that_clef_loads_and_turns_into_units(data_root, tmp_path):
    out_dir = tmp_path / "out"
    prov = cs.build(data_root, "test", "pairs", out_dir)
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
    cs.build(data_root, "test", "pairs", out_dir)
    assert (out_dir / "oneshot-test.json").read_text() == first


def test_build_refuses_a_stale_split_file(data_root, tmp_path):
    rows = json.loads((data_root / "line" / "test.json").read_text())
    (data_root / "line" / "test.json").write_text(json.dumps(rows[:-1]))
    with pytest.raises(AssertionError, match="stale"):
        cs.build(data_root, "test", "pairs", tmp_path / "out")


# ---------------------------------------------------------------- subsets


def test_subset_label_is_the_subsets_own_valid_not_the_entity_flags():
    parents, doc, _ = tl.parent_pair()   # X and Y share program_desc; the invalid parent shows Y's channel
    rows = tl.b.subset_rows(parents[0], 0, doc, 2, tl.EQ)
    same_as_y = next(r for r in rows if r["fields"] == ["channel", "program_desc"])
    assert same_as_y["valid"] and same_as_y["invalid_fields"] == ["channel"]
    for task, key in (("clef_record_oneshot", "entities"), ("clef_relation_conditional", None)):
        out = cs.to_sample(same_as_y, task, "subsets")
        ents = out[key] if key else [out["primary"], *out["conditions"]]
        assert out["label_valid"] is True and [e["valid"] for e in ents] == [False, True]   # corrupted entity, valid label
        assert out["sample_id"] == same_as_y["subset_id"] and out["error_type"] == "intra_document"
    clean = next(r for r in rows if r["fields"] == ["program_desc", "sub_amount"])
    assert cs.to_sample(clean, "clef_record_oneshot", "subsets")["label_valid"] is True
    shows = next(r for r in rows if r["fields"] == ["channel", "sub_amount"])
    assert not shows["valid"] and cs.to_sample(shows, "clef_record_oneshot", "subsets")["label_valid"] is False
    with pytest.raises(AssertionError, match="invalid subset without a corrupted field"):
        cs.to_sample({**clean, "valid": False}, "clef_record_oneshot", "subsets")
    with pytest.raises(AssertionError, match="field_errors"):
        cs.to_sample({**clean, "field_errors": {}}, "clef_record_oneshot", "subsets")


def test_build_subsets_writes_files_that_clef_loads(data_root, tmp_path):
    out_dir = tmp_path / "out"
    prov = cs.build(data_root, "test", "subsets", out_dir)
    assert sorted(prov["files"]) == ["conditional-test-subsets.json", "oneshot-test-subsets.json"] and prov["source"] == "subsets"
    rows = json.loads((data_root / "line" / "test_subsets.json").read_text())
    for name, f in prov["files"].items():
        samples = clef.load_samples(out_dir / name, LINE, f["task"])
        assert [s["sample_id"] for s in samples] == [r["subset_id"] for r in rows]
        assert [s["label_valid"] for s in samples] == [r["valid"] for r in rows]
        assert f["n_samples"] == len(rows) and f["n_valid"] == sum(r["valid"] for r in rows)
        assert f["n_valid"] != len(rows) // 2   # subsets are not class-balanced
        contexts = {x["document_id"]: "text " + x["document_id"] for x in samples}
        units = (clef.conditional_units if f["task"] == "clef_relation_conditional" else clef.oneshot_units)(samples, LINE, contexts)
        assert [u["label_valid"] for u in units] == [r["valid"] for r in rows]
    assert (out_dir / "provenance-test-subsets.json").is_file()


def test_build_subsets_refuses_a_stale_file(data_root, tmp_path):
    rows = json.loads((data_root / "line" / "test_subsets.json").read_text())
    (data_root / "line" / "test_subsets.json").write_text(json.dumps(rows[:-1]))
    with pytest.raises(AssertionError, match="stale"):
        cs.build(data_root, "test", "subsets", tmp_path / "out")
