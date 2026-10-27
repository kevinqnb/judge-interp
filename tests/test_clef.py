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
from dataset_specs import LINE, SPECS
from judge_interp.model_config import load_clef_model_config

LINE_FIELDS = list(LINE.fields)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CFG = load_clef_model_config(REPO_ROOT / "experiments" / "model-configs" / "clef.yaml")

def _entity_items(dataset):
    strata = {"valid": None, "inter_document": None, "cross_type": None}
    if dataset == "line":
        strata = {"valid": None, "cross_type": None}
    p = dict(data_root="tests/fixtures/entity_detection_mini", split="train",
             error_types=["inter_document", "cross_type"], rows_per_stratum=strata, context_token_limit=None)
    return edr.build_items(SPECS[dataset], p, 0, REPO_ROOT, None)


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


# ------------------------------------------------------------------ conditional / one-shot samples
# tests/fixtures/clef_samples_mini (hand-built over relation_detection_mini): sample order, and who is
# valid, are stated in the files. doc-a's lines: 0 = (7, Movie, 01/05/20, 01/10/20, $100.00),
# 1 = (8, -, 01/11/20, 01/20/20, $250.00).
SAMPLES = REPO_ROOT / "tests" / "fixtures" / "clef_samples_mini"
SAMPLE_DATA_ROOT = "tests/fixtures/relation_detection_mini"


def _contexts(samples):
    from judge_interp.prompts import load_context
    return {d: load_context(REPO_ROOT / SAMPLE_DATA_ROOT, LINE, d) for d in {s["document_id"] for s in samples}}


def test_conditional_units_hand_worked():
    samples = clef.load_samples(SAMPLES / "conditional.json", LINE, "clef_relation_conditional")
    units = clef.conditional_units(samples, LINE, _contexts(samples))
    assert [u["sample_id"] for u in units] == ["c1", "c2", "c3", "c4"]
    # label = primary AND every condition valid; c3 pins "valid primary, corrupted condition" -> false
    assert [u["label_valid"] for u in units] == [True, False, False, True]
    assert [u["primary_valid"] for u in units] == [True, False, True, True]
    assert [u["invalid_conditions"] for u in units] == [[], [], ["program_end_date"], []]
    assert [u["invalid_fields"] for u in units] == [[], ["sub_amount"], ["program_end_date"], []]
    # shown order: primary first, then the conditions in the order given (not canonical order)
    assert [u["fields"] for u in units][2] == ["channel", "sub_amount", "program_end_date"]
    assert [u["fields"] for u in units][3] == ["program_desc", "program_end_date", "channel"]
    u = units[2]
    state = u["record"]["state"]
    assert samples[2]["document_id"] == "doc-a" and _contexts(samples)["doc-a"] in state
    assert state.split("## PRIMARY ENTITY\n")[1] == (
        'channel: "7"\n\n## CONDITIONS\nsub_amount: "$100.00"\nprogram_end_date: "01/05/20"')
    assert list(u["record"]["questions"]) == ["q"] and u["record"]["questions"]["q"]["type"] == "noul"
    assert u["primary"] == "channel" and u["conditions"] == ["sub_amount", "program_end_date"]


def test_oneshot_units_hand_worked():
    samples = clef.load_samples(SAMPLES / "oneshot.json", LINE, "clef_record_oneshot")
    units = clef.oneshot_units(samples, LINE, _contexts(samples))
    assert [u["label_valid"] for u in units] == [True, False, True, False]
    assert [u["invalid_fields"] for u in units] == [[], ["channel"], [], ["program_end_date", "sub_amount"]]
    assert units[0]["fields"] == ["sub_amount", "channel", "program_desc"]  # order given, not canonical
    assert units[0]["record"]["state"].split("## EXTRACTED RECORD\n")[1] == 'sub_amount: "$100.00"\nchannel: "7"\nprogram_desc: "Movie"'
    assert [u["error_type"] for u in units] == ["", "intra_document", "", "intra_document"]
    assert [u["pair_id"] for u in units] == ["p5", "p6", "p7", "p8"] and units[2]["cell"] == "cross_item:m2:k2"


def test_conditional_question_names_primary_and_lists_conditions():
    text = clef.conditional_question("channel", ["sub_amount", "program_end_date"], LINE)["instructions"]
    assert f"`channel`: {LINE.field_descriptions['channel']}" in text
    assert f"- `sub_amount`: {LINE.field_descriptions['sub_amount']}" in text and "- `program_end_date`:" in text
    assert "- `program_start_date`" not in text and "- `channel`" not in text  # only the present conditions
    assert "same single item as every one of the conditions" in text and "not related to one or more" in text
    assert "The field names are part of the association" in text and "listed under the wrong field" in text
    assert "as given" not in text  # conditions are judged too, not assumed correct
    assert "Do not use outside knowledge" in text and "unclear evidence means false" in text
    for bad in (["channel"], [], ["sub_amount", "sub_amount"], ["nope"]):
        with pytest.raises(AssertionError):
            clef.conditional_question("channel", bad, LINE)


def test_oneshot_question_has_entity_criteria_and_relation_and_points_at_the_field_list():
    text = clef.oneshot_question(["channel", "sub_amount"], LINE)["instructions"]
    assert "ENTIRE record" in text and "(1) Explicit support" in text and "(2) Type correspondence" in text
    assert "(3) Relation" in text and "same single item" in text
    assert "as described in the list of fields above" in text and "as this field is described" not in text
    assert text.index("- channel:") < text.index("(1) Explicit support")  # the list it points at comes before
    assert "- program_desc:" not in text and "Do not use outside knowledge" in text and "unclear evidence fails" in text
    with pytest.raises(AssertionError):
        clef.oneshot_question(["channel"], LINE)


def test_load_samples_refuses_malformed_files(tmp_path):
    import json
    good = json.loads((SAMPLES / "oneshot.json").read_text())

    def bad(mutate, match):
        data = json.loads(json.dumps(good))
        mutate(data)
        path = tmp_path / "s.json"
        path.write_text(json.dumps(data))
        with pytest.raises(AssertionError, match=match):
            clef.load_samples(path, LINE, "clef_record_oneshot")

    bad(lambda d: d.append(dict(d[0])), "not unique")
    bad(lambda d: d[0]["entities"].append(dict(d[0]["entities"][0])), "occurs twice")
    bad(lambda d: d[0]["entities"][0].update(field="nope"), "not a field")
    bad(lambda d: d[0]["entities"][0].update(valid="yes"), "types")
    bad(lambda d: d[0]["entities"].pop() and d[0]["entities"].pop(), "at least two")
    bad(lambda d: d[0].pop("error_type"), "keys")
    bad(lambda d: d[0].pop("pair_id"), "keys")
    bad(lambda d: d[0].update(cell=None), "must be strings")
    bad(lambda d: d[0].pop("label_valid"), "keys")
    bad(lambda d: d[0].update(label_valid="yes"), "must be a bool")
    bad(lambda d: d[0].update(label_valid=False), "every entity valid but label_valid false")   # o1: all entities valid
    bad(lambda d: d.clear(), "non-empty")
    cond = SAMPLES / "conditional.json"
    with pytest.raises(AssertionError, match="keys"):  # a conditional file under the one-shot task
        clef.load_samples(cond, LINE, "clef_record_oneshot")


def test_sample_columns_hand_worked():
    samples = clef.load_samples(SAMPLES / "conditional.json", LINE, "clef_relation_conditional")
    units = clef.conditional_units(samples, LINE, _contexts(samples))
    for u in units:
        u["context_doc"] = u["doc_id"]
    results = [{"logits": np.array([[l, 0.0]], dtype=np.float32), "p_true": np.array([1 / (1 + np.exp(-l))]), "n_tokens": 10 + i}
               for i, l in enumerate([2.0, -1.0, 0.5, 3.0])]
    cols = clef._sample_columns(units, results)
    assert list(cols["sample_id"]) == ["c1", "c2", "c3", "c4"]
    assert list(cols["pair_id"]) == ["p1", "p2", "p3", "p4"] and list(cols["cell"])[3] == "cross_item:m3:k2"
    assert list(cols["label_valid"]) == [True, False, False, True]
    assert list(cols["primary_valid"]) == [True, False, True, True]
    assert list(cols["invalid_conditions"]) == ["", "", "program_end_date", ""]
    assert list(cols["fields"])[2] == "channel|sub_amount|program_end_date" and list(cols["size"]) == [3, 3, 3, 3]
    assert list(cols["n_invalid_fields"]) == [0, 1, 1, 0] and list(cols["n_tokens"]) == [10, 11, 12, 13]
    s = clef.summarize(cols)
    assert (s["n_valid"], s["n_invalid"]) == (2, 2) and s["mean_p_true_valid"] > s["mean_p_true_invalid"]


@pytest.mark.parametrize("task,fname", [("clef_relation_conditional", "conditional.json"), ("clef_record_oneshot", "oneshot.json")])
def test_build_units_sample_tasks_and_context_control(task, fname):
    p = dict(data_root=SAMPLE_DATA_ROOT, samples_path=f"tests/fixtures/clef_samples_mini/{fname}", context_control="none")
    units, m = clef.build_units(task, p, 0, LINE, REPO_ROOT)
    assert m == {"n_items": 4, "n_units_before_filter": 4}
    assert all(u["context_doc"] == u["doc_id"] for u in units)
    ctx = _contexts([{"document_id": "doc-a"}, {"document_id": "doc-b"}])
    assert all(ctx[u["doc_id"]] in u["record"]["state"] for u in units)
    swapped, _ = clef.build_units(task, {**p, "context_control": "other_document"}, 0, LINE, REPO_ROOT)
    for u, w in zip(units, swapped):
        assert w["context_doc"] != w["doc_id"] and ctx[w["context_doc"]] in w["record"]["state"] and ctx[w["doc_id"]] not in w["record"]["state"]
        assert w["label_valid"] == u["label_valid"] and w["sample_id"] == u["sample_id"]


def test_task_params_are_exact():
    assert clef.TASKS == ("clef_entity_row", "clef_relation_conditional", "clef_record_oneshot")
    assert "clef_relation_row" not in clef.TASK_PARAMS
    assert "samples_path" in clef.TASK_PARAMS["clef_record_oneshot"] and "rows_per_stratum" not in clef.TASK_PARAMS["clef_record_oneshot"]
    assert "samples_path" not in clef.TASK_PARAMS["clef_entity_row"]


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


def test_a_sample_label_is_its_own_not_the_entity_flags(tmp_path):
    """A corrupted entity with label_valid true (a subset equal to a real item) stays valid; invalid_fields keeps the entity."""
    import json
    data = json.loads((SAMPLES / "oneshot.json").read_text())
    o2 = next(s for s in data if s["sample_id"] == "o2")
    assert not all(e["valid"] for e in o2["entities"]) and o2["label_valid"] is False
    o2["label_valid"] = True
    path = tmp_path / "s.json"
    path.write_text(json.dumps(data))
    samples = clef.load_samples(path, LINE, "clef_record_oneshot")
    contexts = {d: "x" for d in {s["document_id"] for s in samples}}
    (u,) = [u for u in clef.oneshot_units(samples, LINE, contexts) if u["sample_id"] == "o2"]
    assert u["label_valid"] is True and u["invalid_fields"] != []
