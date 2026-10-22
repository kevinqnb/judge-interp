"""Unit tests for judge_interp.clef's model-free parts (no Clef weights are loaded).

Fixtures: tests/fixtures/entity_detection_mini (main + line) and relation_detection_mini (line), hand-built;
see the docstrings of tests/test_entity_detection.py / test_relation_detection.py.
The release code ``joint_schema_model.py`` is imported from the HF cache at the pinned revision;
tokenization is faked with a whitespace tokenizer.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from judge_interp import clef
from judge_interp import entity_detection as edr
from judge_interp import relation_detection as rdr
from dataset_specs import LINE, SPECS
from judge_interp.model_config import load_clef_model_config

LINE_FIELDS = list(LINE.fields)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CFG = load_clef_model_config(REPO_ROOT / "experiments" / "model-configs" / "clef.yaml")
ALL_LINE = ["inter_document", "intra_document", "cross_type"]

def _entity_items(dataset):
    strata = {"valid": None, "inter_document": None, "cross_type": None}
    if dataset == "line":
        strata = {"valid": None, "cross_type": None}
    p = dict(data_root="tests/fixtures/entity_detection_mini", split="train",
             error_types=["inter_document", "cross_type"], rows_per_stratum=strata, context_token_limit=None)
    return edr.build_items(SPECS[dataset], p, 0, REPO_ROOT, None)


def _relation_items():
    p = dict(data_root="tests/fixtures/relation_detection_mini", split="train", error_types=ALL_LINE,
             rows_per_stratum={"valid": None, **{t: None for t in ALL_LINE}}, context_token_limit=None)
    return rdr.build_items(LINE, p, 0, REPO_ROOT, None)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[len(w) for w in text.split()])


@pytest.fixture(scope="module")
def jsm():
    rc = MODEL_CFG["release_code"]
    return clef.load_release_code(MODEL_CFG["model"], MODEL_CFG["revision"], rc["file"], rc["sha256"])


def test_load_release_code_refuses_a_hash_mismatch():
    rc = MODEL_CFG["release_code"]
    with pytest.raises(ValueError, match="sha256"):
        clef.load_release_code(MODEL_CFG["model"], MODEL_CFG["revision"], rc["file"], "0" * 64)


# ------------------------------------------------------------------ entity
def test_entity_units_one_question_per_nonnull_field():
    items = _entity_items("main")
    contexts = {it["document_id"]: it["context"] for it in items}
    units = clef.entity_units(items, SPECS["main"], contexts)
    assert len(units) == len(items)
    for u, it in zip(units, items):
        assert list(u["record"]["questions"]) == [sp["field"] for sp in it["spans"]]
        assert len(u["spans"]) == len(u["record"]["questions"])
        assert all(q["type"] == "noul" for q in u["record"]["questions"].values())
        assert it["context"] in u["record"]["state"]
    m1 = [u for u in units if u["doc_id"] == "doc-a" and u["spans"][0]["row_error_type"] == ""][0]
    assert "agency" not in m1["record"]["questions"]  # null in M1
    assert "agency" not in m1["record"]["state"].split("## EXTRACTED RECORD\n")[1]  # null field: no line
    assert 'advertiser: "Acme"' in m1["record"]["state"]  # plain value: no pad-space convention here
    q = m1["record"]["questions"]["property"]["instructions"]
    assert "`property`" in q and SPECS["main"].field_descriptions["property"] in q


# ------------------------------------------------------------------ relation rows
def test_relation_units_hand_worked():
    # relation_detection_mini line train, in item order (doc, line, stratum, k):
    #   doc-a line 0: cross_type (end date wrong), intra_document (end date + amount wrong), valid
    #   doc-a line 1: inter_document (channel wrong), valid -- program_desc is null in both
    #   doc-b line 0: valid
    items = _relation_items()
    contexts = {it["document_id"]: it["context"] for it in items}
    units = clef.relation_units(items, LINE, contexts)
    assert [u["label_valid"] for u in units] == [False, False, True, False, True, True]
    assert [u["row_error_type"] for u in units] == ["cross_type", "intra_document", "", "inter_document", "", ""]
    assert [u["invalid_fields"] for u in units] == [
        ["program_end_date"], ["program_end_date", "sub_amount"], [], ["channel"], [], []]
    assert [len(u["fields"]) for u in units] == [5, 5, 5, 4, 4, 5]
    for u in units:
        lines = u["record"]["state"].split("## EXTRACTED RECORD\n")[1].split("\n")
        assert [ln.split(":")[0] for ln in lines] == u["fields"]  # every non-null field, canonical order, no null line
        assert "null" not in u["record"]["state"].split("## EXTRACTED RECORD\n")[1]
        assert list(u["record"]["questions"]) == ["related"]
        text = u["record"]["questions"]["related"]["instructions"]
        for f in LINE_FIELDS:
            assert (f"- {f}:" in text) == (f in u["fields"])
        assert [int(i) for i in u["rows"] if i >= 0] == [LINE_FIELDS.index(f) for f in u["fields"]]
        assert len(u["rows"]) == len(LINE_FIELDS)
        if u["row_error_type"] == "":
            assert u["label_valid"]
    assert "program_desc" not in units[3]["record"]["state"]  # null field omitted


def test_relation_question_matches_the_qwen_framing():
    text = clef.relation_question(["channel", "sub_amount"], LINE)["instructions"]
    assert "same single item" in text and "Do not judge whether an individual value is real" in text
    assert "serves as the field" not in text and "not what its field denotes" not in text


def test_criterion_two_points_at_the_field_list_only_where_there_is_one():
    clef_q = clef.entity_question("property", SPECS["main"])["instructions"]
    assert "as this field is described" in clef_q and "list of fields" not in clef_q
    qwen = edr.build_instructions(SPECS["main"])
    assert "as described in the list of fields above" in qwen and "as this field is described" not in qwen


def test_entity_question_forbids_outside_knowledge_and_state_has_no_null_line():
    items = _entity_items("main")
    contexts = {it["document_id"]: it["context"] for it in items}
    for u in clef.entity_units(items, SPECS["main"], contexts):
        assert "null" not in u["record"]["state"].split("## EXTRACTED RECORD\n")[1]
        for q in u["record"]["questions"].values():
            assert "Do not use outside knowledge" in q["instructions"]


# ------------------------------------------------------------------ context control
def test_swap_contexts_never_returns_own_text():
    ctx = {"a": "A", "b": "B", "c": "C", "d": "D"}
    sw = clef.swap_contexts(ctx, 0)
    assert set(sw) == set(ctx) and all(sw[d] != ctx[d] for d in ctx)
    assert sorted(sw.values()) == sorted(ctx.values())  # a permutation of the texts
    assert sw == clef.swap_contexts(ctx, 0)
    with pytest.raises(AssertionError):
        clef.swap_contexts({"a": "A"}, 0)


# ------------------------------------------------------------------ truncation guard
def _record(n_state_words):
    return {"id": "r", "state": " ".join(["w"] * n_state_words),
            "questions": {"q": {"type": "noul", "instructions": "Is it?"}}}


def test_release_encode_truncates_silently_but_encode_checked_refuses(jsm):
    tok = FakeTokenizer()
    fixed, total = clef.count_tokens(jsm, tok, _record(100))
    assert total == fixed + 100
    raw = jsm.encode_record(tok, _record(100), max_length=fixed + 10)
    assert len(raw.input_ids) == fixed + 10  # the hazard: 90 state tokens dropped, no error
    with pytest.raises(ValueError, match="would be truncated"):
        clef.encode_checked(jsm, tok, _record(100), fixed + 10)
    enc = clef.encode_checked(jsm, tok, _record(100), total)
    assert len(enc.input_ids) == total and enc.questions[0].option_ids == clef.OPTION_IDS


def test_filter_overlong_modes(jsm):
    tok = FakeTokenizer()
    fixed, _ = clef.count_tokens(jsm, tok, _record(1))
    mk = lambda n: {"record": _record(n), "doc_id": f"d{n}"}
    kept, excl = clef.filter_overlong([mk(5), mk(50)], jsm, tok, fixed + 10, "exclude")
    assert [u["doc_id"] for u in kept] == ["d5"] and [u["doc_id"] for u in excl] == ["d50"]
    assert kept[0]["n_tokens"] == fixed + 5
    with pytest.raises(ValueError, match="exceed max_length"):
        clef.filter_overlong([mk(5), mk(50)], jsm, tok, fixed + 10, "error")
    with pytest.raises(AssertionError):
        clef.filter_overlong([mk(5)], jsm, tok, fixed + 10, "drop")
