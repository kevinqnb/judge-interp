"""Unit tests for judge_interp.jev (no API call: the TypeSafe client is replaced by a fake).

Fixtures: tests/fixtures/entity_detection_mini (main), relation_detection_mini (line) and
clef_samples_mini, hand-built; see the docstrings of tests/test_clef.py.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from judge_interp import clef, jev
from judge_interp import entity_detection as edr
from judge_interp.model_config import load_jev_model_config
from judge_interp.prompts import load_context
from dataset_specs import LINE, SPECS

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = REPO_ROOT / "tests" / "fixtures" / "clef_samples_mini"
SAMPLE_DATA_ROOT = "tests/fixtures/relation_detection_mini"
MODEL = "jev-1.14.0"


def _contexts(samples):
    return {d: load_context(REPO_ROOT / SAMPLE_DATA_ROOT, LINE, d) for d in {s["document_id"] for s in samples}}


class FakeClient:
    """Answers every question with ``p_for(name, call_index)``; records each call."""

    def __init__(self, p_for, model=MODEL, input_tokens=7, drop=None):
        self.p_for, self.model, self.input_tokens, self.drop, self.calls = p_for, model, input_tokens, drop, []

    def system_one(self, state, questions, *, model):
        self.calls.append((state, questions, model))
        names = [n for n in questions if n != self.drop]
        answers = {n: SimpleNamespace(noul=self.p_for(n, len(self.calls))) for n in names}
        return SimpleNamespace(model=self.model, answers=answers, nouls=answers, request_id=f"req-{len(self.calls)}",
                               usage=SimpleNamespace(input_tokens=self.input_tokens, output_tokens=1))


def _scorer(client):
    s = object.__new__(jev.JevScorer)
    s.model, s.client = MODEL, client
    return s


# ------------------------------------------------------------------ entity
def test_entity_units_one_question_per_nonnull_field_and_a_json_state():
    strata = {"valid": None, "inter_document": None, "cross_type": None}
    p = dict(data_root="tests/fixtures/entity_detection_mini", split="train", error_types=["inter_document", "cross_type"],
             rows_per_stratum=strata, context_token_limit=None)
    items = edr.build_items(SPECS["main"], p, 0, REPO_ROOT, None)
    contexts = {it["document_id"]: it["context"] for it in items}
    units = jev.entity_units(items, SPECS["main"], contexts)
    assert len(units) == len(items)
    for u, it in zip(units, items):
        fields = [sp["field"] for sp in it["spans"]]
        assert list(u["record"]["questions"]) == fields
        assert list(u["record"]["state"]) == ["context", "extracted_record"]
        assert list(u["record"]["state"]["extracted_record"]) == fields
        assert u["record"]["state"]["context"] == it["context"]
        assert all(q.type == "noul" for q in u["record"]["questions"].values())
    m1 = [u for u in units if u["doc_id"] == "doc-a" and u["spans"][0]["row_error_type"] == ""][0]
    assert "agency" not in m1["record"]["questions"] and "agency" not in m1["record"]["state"]["extracted_record"]
    assert m1["record"]["state"]["extracted_record"]["advertiser"] == "Acme"
    q = m1["record"]["questions"]["property"].instructions
    assert "`extracted_record.property`" in q and SPECS["main"].field_descriptions["property"] in q
    assert "as this field is described" in q and "outside knowledge" in q


# ------------------------------------------------------------------ samples
def test_conditional_units_hand_worked():
    samples = clef.load_samples(SAMPLES / "conditional.json", LINE, "clef_relation_conditional")
    units = jev.conditional_units(samples, LINE, _contexts(samples))
    assert [u["label_valid"] for u in units] == [True, False, False, True]   # c3: valid primary, corrupted condition -> false
    assert [u["invalid_conditions"] for u in units] == [[], [], ["program_end_date"], []]
    u = units[2]
    assert u["record"]["state"] == {
        "context": _contexts(samples)["doc-a"], "primary_entity": {"channel": "7"},
        "conditions": {"sub_amount": "$100.00", "program_end_date": "01/05/20"}}
    assert list(u["record"]["state"]["conditions"]) == ["sub_amount", "program_end_date"]   # order given
    assert list(u["record"]["questions"]) == ["q"]
    q = u["record"]["questions"]["q"].instructions
    assert "`channel`" in q and "`conditions.program_end_date`" in q and "`conditions.sub_amount`" in q


def test_oneshot_units_hand_worked():
    samples = clef.load_samples(SAMPLES / "oneshot.json", LINE, "clef_record_oneshot")
    units = jev.oneshot_units(samples, LINE, _contexts(samples))
    assert [u["label_valid"] for u in units] == [True, False, True, False]
    assert [u["invalid_fields"] for u in units] == [[], ["channel"], [], ["program_end_date", "sub_amount"]]
    assert list(units[0]["record"]["state"]["extracted_record"].items()) == [
        ("sub_amount", "$100.00"), ("channel", "7"), ("program_desc", "Movie")]
    assert "(3) Relation" in units[0]["record"]["questions"]["q"].instructions


@pytest.mark.parametrize("task,fname", [("jev_relation_conditional", "conditional.json"), ("jev_record_oneshot", "oneshot.json")])
def test_build_units_sample_tasks_and_context_control(task, fname):
    p = dict(data_root=SAMPLE_DATA_ROOT, samples_path=f"tests/fixtures/clef_samples_mini/{fname}", context_control="none")
    units, m = jev.build_units(task, p, 0, LINE, REPO_ROOT)
    assert m == {"n_items": 4, "n_units": 4}
    ctx = _contexts([{"document_id": "doc-a"}, {"document_id": "doc-b"}])
    assert all(u["context_doc"] == u["doc_id"] and u["record"]["state"]["context"] == ctx[u["doc_id"]] for u in units)
    swapped, _ = jev.build_units(task, {**p, "context_control": "other_document"}, 0, LINE, REPO_ROOT)
    for u, w in zip(units, swapped):
        assert w["context_doc"] != w["doc_id"] and w["record"]["state"]["context"] == ctx[w["context_doc"]]
        assert w["label_valid"] == u["label_valid"] and w["sample_id"] == u["sample_id"]


def test_sample_columns_match_clef_minus_logits():
    samples = clef.load_samples(SAMPLES / "conditional.json", LINE, "clef_relation_conditional")
    units = jev.conditional_units(samples, LINE, _contexts(samples))
    for u in units:
        u["context_doc"] = u["doc_id"]
    results = [{"p_true": np.array([p]), "n_tokens": n} for p, n in zip([0.9, 0.1, 0.2, 0.8], [10, 11, 12, 13])]
    cols = jev._sample_columns(units, results)
    assert "logit_true" not in cols and list(cols["p_true"]) == [0.9, 0.1, 0.2, 0.8] and list(cols["n_tokens"]) == [10, 11, 12, 13]
    assert list(cols["label_valid"]) == [True, False, False, True] and list(cols["fields"])[2] == "channel|sub_amount|program_end_date"
    s = clef.summarize(cols)
    assert (s["n_valid"], s["n_invalid"]) == (2, 2) and s["mean_p_true_valid"] > s["mean_p_true_invalid"]


def test_task_params_are_exact():
    assert jev.TASKS == ("jev_entity_row", "jev_relation_conditional", "jev_record_oneshot")
    assert "overlong_docs" not in jev.TASK_PARAMS["jev_entity_row"] and "device" not in jev.TASK_PARAMS["jev_entity_row"]
    assert "samples_path" in jev.TASK_PARAMS["jev_record_oneshot"] and "rows_per_stratum" not in jev.TASK_PARAMS["jev_record_oneshot"]
    assert "samples_path" not in jev.TASK_PARAMS["jev_entity_row"]
    assert set(jev._CLEF_TASK) == set(jev.TASKS)


# ------------------------------------------------------------------ scorer
def test_scorer_reads_p_true_in_question_order_and_pins_the_model():
    names = {"b": 0.25, "a": 0.75}
    c = FakeClient(lambda n, i: names[n])
    r = _scorer(c).score({"id": "r", "state": {"x": 1}, "questions": {"b": jev.Noul(instructions="?"), "a": jev.Noul(instructions="?")}})
    assert r["p_true"].tolist() == [0.25, 0.75] and r["n_tokens"] == 7 and r["request_id"] == "req-1"
    assert c.calls[0][2] == MODEL and set(r["efficiency"]) == set(jev.EFFICIENCY_KEYS)


@pytest.mark.parametrize("client", [
    FakeClient(lambda n, i: 0.5, model="jev-1.13.0"),   # answered by another release
    FakeClient(lambda n, i: 1.5),                       # not a probability
    FakeClient(lambda n, i: float("nan")),
    FakeClient(lambda n, i: 0.5, drop="b"),             # a question unanswered
])
def test_scorer_refuses_a_bad_response(client):
    with pytest.raises(AssertionError):
        _scorer(client).score({"id": "r", "state": {}, "questions": {"a": jev.Noul(instructions="?"), "b": jev.Noul(instructions="?")}})


def test_scorer_refuses_missing_api_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        jev.JevScorer(MODEL, {"max_retries": 0, "backoff_max": 1.0, "timeout": 1.0})


def test_known_answer_check_passes_and_fails():
    ok = _scorer(FakeClient(lambda n, i: 0.9 if n == "above_1000" else 0.1)).known_answer_check()
    assert ok == {"known_answer_p_yes": 0.9, "known_answer_p_no": 0.1}
    with pytest.raises(AssertionError, match="known-answer"):
        _scorer(FakeClient(lambda n, i: 0.9)).known_answer_check()


def _units(n):
    return [{"record": {"id": f"r{i}", "state": {}, "questions": {"q": jev.Noul(instructions="?")}}} for i in range(n)]


def test_determinism_check_reports_the_drift_and_enforces_atol():
    drift = FakeClient(lambda n, i: 0.5 + (0.01 if i > 3 else 0.0))   # the 4th call (first rescoring) differs by 0.01
    results, d = jev._score_units(_scorer(drift), _units(3), 1, 0.05, verbose=False)
    assert len(results) == 3 and d == pytest.approx(0.01)
    with pytest.raises(AssertionError, match="determinism_atol"):
        jev._score_units(_scorer(FakeClient(lambda n, i: 0.5 + (0.01 if i > 3 else 0.0))), _units(3), 1, 0.001, verbose=False)
    _, d0 = jev._score_units(_scorer(FakeClient(lambda n, i: 0.5)), _units(2), 2, 0.0, verbose=False)
    assert d0 == 0.0


def test_efficiency_metrics_hand_worked():
    eff = lambda s, i, o, q: {"seconds_request": s, "input_tokens": i, "output_tokens": o, "n_questions": q}
    m, recs = jev.efficiency_metrics([{"efficiency": eff(1.0, 10, 1, 2)}, {"efficiency": eff(3.0, 30, 2, 1)}])
    assert m["eff_n_requests"] == 2 and m["eff_seconds_request_mean"] == 2.0 and m["eff_seconds_request_max"] == 3.0
    assert (m["eff_input_tokens_total"], m["eff_output_tokens_total"], m["eff_n_questions_total"]) == (40, 3, 3)
    assert set(recs) == set(jev.EFFICIENCY_KEYS)


# ------------------------------------------------------------------ run / model config
def test_run_end_to_end_with_a_fake_client(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    # plumbing only: one request per sample in file order; the fake's P(true) is not an oracle for the labels
    labels = {"c1": True, "c2": False, "c3": False, "c4": True}
    fake = FakeClient(lambda n, i: 0.5)
    sc = _scorer(fake)
    monkeypatch.setattr(jev, "JevScorer", lambda model, retry: sc)

    def p_for(name, i):
        state = fake.calls[-1][0]
        return 0.9 if "program_end_date" not in state["conditions"] or state["primary_entity"] != {"channel": "7"} else 0.1
    fake.p_for = p_for
    cfg = {"task": "jev_relation_conditional", "seed": 0, "params": dict(
        data_root=SAMPLE_DATA_ROOT, samples_path="tests/fixtures/clef_samples_mini/conditional.json", context_control="none",
        determinism_check_n=2, determinism_atol=0.0, known_answer_check=False)}
    m = jev.run(cfg, LINE, tmp_path, REPO_ROOT, {"model": MODEL, "retry": {}})
    cols = np.load(tmp_path / "artifacts" / "jev_predictions.npz")
    assert list(cols["sample_id"]) == list(labels) and m["n_predictions"] == 4
    assert len(fake.calls) == 4 + 2   # one request per sample, plus the rescoring
    assert m["determinism_max_abs_diff"] == 0.0 and m["eff_n_requests"] == 4 and m["eff_input_tokens_total"] == 28
    assert (tmp_path / "artifacts" / "efficiency_records.npz").exists() and (tmp_path / "artifacts" / "request_ids.json").exists()


def test_jev_model_config_is_valid_and_aliases_are_refused(tmp_path):
    cfg = load_jev_model_config(REPO_ROOT / "experiments" / "model-configs" / "jev.yaml")
    assert cfg["model"] == MODEL
    bad = tmp_path / "bad.yaml"
    bad.write_text("model: jev-latest\nretry: {max_retries: 1, backoff_max: 1.0, timeout: 1.0}\n")
    with pytest.raises(ValueError, match="alias"):
        load_jev_model_config(bad)
    bad.write_text("model: jev-1.14.0\nretry: {max_retries: 1, backoff_max: 1.0}\n")
    with pytest.raises(KeyError):
        load_jev_model_config(bad)
