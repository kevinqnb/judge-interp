"""Tests for experiments/clef_samples.py: routing the shared line subsets to the two Clef sample tasks.

Data: the hand-built rows of test_build_vrdu_subsets.py (every row has channel="c", program_desc="d",
program_start_date="s", program_end_date="e"; the inter_document k=1 row corrupts channel, k=2 corrupts
channel + program_desc), turned into subsets by the real builder. The converted files go through the
real ``clef.load_samples`` and unit builders.
"""
import json
import sys
from pathlib import Path

import pytest

import test_build_vrdu_subsets as ts
from dataset_specs import LINE
from judge_interp import clef

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import clef_samples as cs  # noqa: E402

VALUES = {"channel": "c", "program_desc": "d", "program_start_date": "s", "program_end_date": "e"}


@pytest.fixture
def data_root(tmp_path):
    (tmp_path / "line").mkdir()
    for split, docs in ts.SPLIT_DOCS.items():
        (tmp_path / "line" / f"{split}.json").write_text(json.dumps(ts.make_rows(docs)))
    ts.b.main(["--config", str(ts.write_config(tmp_path, n_replicates=2, n_per_class={"train": 6, "test": 4}))])
    return tmp_path


def test_conditional_primary_is_the_first_field_and_labels_follow_the_row(data_root):
    subsets = ts.s.load_subsets(data_root, "test", 0)
    for sub in subsets:
        out = cs.to_sample(sub, "clef_relation_conditional")
        assert out["primary"]["field"] == sub["fields"][0]
        assert [c["field"] for c in out["conditions"]] == sub["fields"][1:]
        for e in (out["primary"], *out["conditions"]):
            assert e["value"] == VALUES[e["field"]]
            assert e["valid"] == (e["field"] not in sub["row"]["invalid_fields"])
        assert out["error_type"] == (sub["row"]["error_type"] or "")


def test_oneshot_keeps_the_field_order_and_label(data_root):
    for sub in ts.s.load_subsets(data_root, "test", 0):
        out = cs.to_sample(sub, "clef_record_oneshot")
        assert [e["field"] for e in out["entities"]] == sub["fields"]
        assert all(e["valid"] for e in out["entities"]) == sub["label_valid"]


def test_known_answer_sample_from_a_corrupted_row():
    row = ts.make_rows(["A"])[2]   # inter_document k=2: channel and program_desc corrupted
    fields = ["program_end_date", "channel", "program_start_date", "program_desc"]
    sub = {"sample_id": ts.s.make_sample_id(row, fields), "document_id": "A", "fields": fields, "label_valid": False,
           "error_type": "inter_document", "row": row}
    cond = cs.to_sample(sub, "clef_relation_conditional")
    assert cond["error_type"] == "inter_document"
    assert cond["primary"] == {"field": "program_end_date", "value": "e", "valid": True}
    assert cond["conditions"] == [{"field": "channel", "value": "c", "valid": False},
                                  {"field": "program_start_date", "value": "s", "valid": True},
                                  {"field": "program_desc", "value": "d", "valid": False}]
    one = cs.to_sample(sub, "clef_record_oneshot")
    assert [(e["field"], e["valid"]) for e in one["entities"]] == [
        ("program_end_date", True), ("channel", False), ("program_start_date", True), ("program_desc", False)]


def test_a_validity_label_that_disagrees_with_the_row_is_refused(data_root):
    sub = ts.s.load_subsets(data_root, "test", 0)[0]
    with pytest.raises(AssertionError, match="entity validity"):
        cs.to_sample({**sub, "label_valid": not sub["label_valid"]}, "clef_record_oneshot")


def test_build_writes_files_that_clef_loads_and_turns_into_units(data_root, tmp_path):
    out_dir = tmp_path / "out"
    prov = cs.build(data_root, "test", [0, 1], out_dir)
    assert sorted(prov["files"]) == sorted(f"{p}-test-rep{r}.json" for p in ("conditional", "oneshot") for r in (0, 1))
    for name, f in prov["files"].items():
        task = f["task"]
        samples = clef.load_samples(out_dir / name, LINE, task)
        assert len(samples) == f["n_samples"] == 8 and f["n_valid"] == 4
        contexts = {s["document_id"]: "text " + s["document_id"] for s in samples}
        units = (clef.conditional_units if task == "clef_relation_conditional" else clef.oneshot_units)(samples, LINE, contexts)
        assert sum(u["label_valid"] for u in units) == f["n_valid"]
    # the two replicates differ, and the build is deterministic
    first = (out_dir / "oneshot-test-rep0.json").read_text()
    assert first != (out_dir / "oneshot-test-rep1.json").read_text()
    cs.build(data_root, "test", [0, 1], out_dir)
    assert (out_dir / "oneshot-test-rep0.json").read_text() == first


def test_build_refuses_a_stale_subsets_file(data_root, tmp_path):
    rows = json.loads((data_root / "line" / "test.json").read_text())
    (data_root / "line" / "test.json").write_text(json.dumps(rows[:-1]))
    with pytest.raises(AssertionError, match="stale"):
        cs.build(data_root, "test", [0], tmp_path / "out")
