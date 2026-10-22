"""Unit tests for judge_interp.clef's model-free parts (no Clef weights are loaded).

Fixtures: tests/fixtures/entity_detection_mini (main + line) and relation_detection_mini (line), hand-built;
see the docstrings of tests/test_entity_detection.py / test_relation_detection.py.
The release code ``joint_schema_model.py`` is imported from the HF cache at the pinned revision;
tokenization is faked with a whitespace tokenizer.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from judge_interp import clef
from judge_interp import entity_detection as edr
from judge_interp import relation_detection as rdr
from dataset_specs import LINE, SPECS

LINE_FIELDS = list(LINE.fields)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL, REVISION = "Cloudflare/clef-flash", "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3"
ALL_LINE = ["inter_document", "intra_document", "cross_type"]

_spec = importlib.util.spec_from_file_location(
    "analysis_relation_detection", REPO_ROOT / "analysis" / "relation_detection.py"
)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


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
    return clef.load_release_code(MODEL, REVISION)


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
    assert 'agency: null' in m1["record"]["state"]
    assert 'advertiser: "Acme"' in m1["record"]["state"]  # plain value: no pad-space convention here
    q = m1["record"]["questions"]["property"]["instructions"]
    assert "`property`" in q and SPECS["main"].field_descriptions["property"] in q


# ------------------------------------------------------------------ relation subsets
def test_group_rows_matches_the_probe():
    flat = clef.flatten_items(_relation_items())
    groups = clef.group_rows(flat, LINE)
    assert groups == probe.group_rows(
        {k: flat[k] for k in ("doc_ids", "row_ordinal", "field", "field_position", "row_error_type")})


def test_draw_samples_hand_worked():
    # One 3-field row (spans 0-2, span 1 invalid) and one 1-field row (too small).
    # Subsets of size >= 2 of 3 fields: 3 pairs + 1 triple = 4, so 10 requested yields 4.
    groups = [
        {"doc": "a", "ordinal": 0, "rows": [0, 1, 2], "fields": ["x", "y", "z"], "error_type": "cross_type"},
        {"doc": "a", "ordinal": 1, "rows": [3], "fields": ["x"], "error_type": ""},
    ]
    label_valid = np.array([True, False, True, True])
    s = clef.draw_samples(groups, label_valid, 10, 2, np.random.default_rng(0), 5)
    assert s["stats"] == {"n_rows": 2, "n_rows_too_small": 1, "n_rows_fewer_subsets_than_requested": 1}
    got = sorted(tuple(r for r in row if r >= 0) for row in s["rows"].tolist())
    assert got == [(0, 1), (0, 1, 2), (0, 2), (1, 2)]
    for row, label, size in zip(s["rows"].tolist(), s["label"], s["size"]):
        members = [r for r in row if r >= 0]
        assert size == len(members) and label == (1 not in members)
    again = clef.draw_samples(groups, label_valid, 2, 2, np.random.default_rng(0), 5)
    same = clef.draw_samples(groups, label_valid, 2, 2, np.random.default_rng(0), 5)
    assert np.array_equal(again["rows"], same["rows"])


def test_draw_relation_samples_is_balanced_and_seed_deterministic():
    flat = clef.flatten_items(_relation_items())
    p = dict(samples_per_row=4, min_subset_size=2, n_per_class=3, subset_rng_stream=1)
    a = clef.draw_relation_samples(flat, p, 0, LINE)
    b = clef.draw_relation_samples(flat, p, 0, LINE)
    assert int(a["label"].sum()) == 3 and int((~a["label"]).sum()) == 3
    for k in ("rows", "label", "size", "doc", "ordinal", "error_type"):
        assert np.array_equal(a[k], b[k]), k


def test_subset_label_is_all_valid_and_state_has_only_the_subset():
    items = _relation_items()
    flat = clef.flatten_items(items)
    p = dict(samples_per_row=40, min_subset_size=2, n_per_class=2, subset_rng_stream=1)
    samples = clef.draw_relation_samples(flat, p, 0, LINE)
    contexts = {it["document_id"]: it["context"] for it in items}
    units = clef.relation_units(flat, samples, contexts, LINE)
    assert len(units) == 4 and sum(u["label_valid"] for u in units) == 2
    for u in units:
        assert u["label_valid"] == (u["invalid_fields"] == [])
        lines = u["record"]["state"].split("## EXTRACTED RECORD\n")[1].split("\n")
        assert [ln.split(":")[0] for ln in lines] == u["fields"]  # only the subset, canonical order
        assert list(u["record"]["questions"]) == ["related"]
        text = u["record"]["questions"]["related"]["instructions"]
        for f in LINE_FIELDS:
            assert (f"- {f}:" in text) == (f in u["fields"])
        if u["row_error_type"] == "":
            assert u["label_valid"]


def test_balance_hard_errors_when_a_class_is_too_small():
    flat = clef.flatten_items(_relation_items())
    p = dict(samples_per_row=1, min_subset_size=2, n_per_class=10**6, subset_rng_stream=1)
    with pytest.raises(ValueError, match="requested"):
        clef.draw_relation_samples(flat, p, 0, LINE)


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
