"""Unit tests for judge_interp.relation_detection's model-free parts.

Fixture: tests/fixtures/entity_relation_mini (see test_relation_prompts.py).
Hand-worked-out for train / min_valid_per_type=1: doc-x has channel{7, 13},
program_desc{Evening News, Late Movie}, sub_amount{$500.00} (plus dates);
doc-y has one value per type. All five LINE_FIELDS render for both documents.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from judge_interp import relation_detection as rd
from judge_interp.prompts import LINE_FIELDS

FIXTURE = "tests/fixtures/entity_relation_mini"
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = {"data_root": FIXTURE, "split": "train", "min_valid_per_type": 1, "min_types_per_doc": 2,
          "context_token_limit": None, "doc_subset": None}


def test_instructions_describe_every_type_and_have_no_cue():
    text = rd.build_instructions(LINE_FIELDS)
    for f in LINE_FIELDS:
        assert f"- {f}: {rd.TYPE_DESCRIPTIONS[f]}" in text
    assert "valid?" not in text
    assert "(<type 1 index>, <type 2 index>, <type 3 index>, <type 4 index>, <type 5 index>)" in text
    sub = rd.build_instructions(["channel", "sub_amount"])
    assert "program_desc" not in sub and "(<type 1 index>, <type 2 index>)" in sub


@pytest.mark.parametrize("bad", [["channel"], ["channel", "channel"], ["channel", "nope"]])
def test_instructions_reject_bad_types(bad):
    with pytest.raises(ValueError):
        rd.build_instructions(bad)


def test_render_hand_worked_example():
    r = rd.render_lists({"program_desc": ["Late Movie", "News"], "channel": ["7"]})
    assert r["text"] == (
        'TYPE 1 ENTITIES (channel):\n1. "7"\n\n'
        'TYPE 2 ENTITIES (program_desc):\n1. "Late Movie"\n2. "News"'
    )
    assert r["entity_types"] == ["channel", "program_desc"]
    assert "valid?" not in r["text"]
    got = [(s["type_num"], s["list_position"], r["text"][s["char_start"]:s["char_end"]]) for s in r["spans"]]
    assert got == [(1, 1, "7"), (2, 1, "Late Movie"), (2, 2, "News")]


def test_span_excludes_quotes_and_handles_escapes():
    r = rd.render_lists({"program_desc": ['a"b\nc', "é"]})
    s0, s1 = r["spans"]
    assert r["text"][s0["char_start"]:s0["char_end"]] == 'a\\"b\\nc'  # ends on the real last char
    assert r["text"][s1["char_start"]:s1["char_end"]] == "\\u00e9"
    assert r["text"][s0["char_start"] - 1] == '"' and r["text"][s0["char_end"]] == '"'


def test_duplicate_values_across_types_get_distinct_spans():
    r = rd.render_lists({"program_start_date": ["01/02"], "program_end_date": ["01/02"]})
    a, b = r["spans"]
    assert r["text"][a["char_start"]:a["char_end"]] == r["text"][b["char_start"]:b["char_end"]]
    assert a["char_start"] != b["char_start"]


def test_render_errors():
    with pytest.raises(ValueError):
        rd.render_lists({})
    with pytest.raises(ValueError):
        rd.render_lists({"advertiser": ["x"]})
    with pytest.raises(ValueError):
        rd.render_lists({"channel": []})
    with pytest.raises(AssertionError):
        rd.render_lists({"channel": [""]})


def test_build_items_fixture():
    items, tuples, skipped = rd.build_items(PARAMS, 0, REPO_ROOT)
    assert skipped == []
    assert [it["document_id"] for it in items] == ["doc-x", "doc-y"]
    assert set(tuples) == {"doc-x", "doc-y"}
    x = items[0]
    assert [len(it["spans"]) for it in items] == [7, 5]
    by_type = {}
    for s in x["spans"]:
        by_type.setdefault(s["entity_type"], []).append(x["query"][s["char_start"]:s["char_end"]])
    assert sorted(by_type["channel"]) == ["13", "7"]
    assert sorted(by_type["program_desc"]) == ["Evening News", "Late Movie"]
    assert by_type["sub_amount"] == ["$500.00"]
    for it in items:  # spans live in the query, never the context
        assert it["query"].count("TYPE ") == len(set(s["type_num"] for s in it["spans"]))


def test_build_items_deterministic_and_doc_subset():
    a, _, _ = rd.build_items(PARAMS, 0, REPO_ROOT)
    b, _, _ = rd.build_items(PARAMS, 0, REPO_ROOT)
    assert a == b
    sub, tuples, _ = rd.build_items({**PARAMS, "doc_subset": {"n": 1, "seed": 0}}, 0, REPO_ROOT)
    assert len(sub) == 1 and set(tuples) == {sub[0]["document_id"]}
    with pytest.raises(ValueError):
        rd.build_items({**PARAMS, "doc_subset": {"n": 5, "seed": 0}}, 0, REPO_ROOT)


def test_context_token_limit_needs_tokenizer():
    with pytest.raises(ValueError):
        rd.build_items({**PARAMS, "context_token_limit": 100}, 0, REPO_ROOT)


def _result(doc_ids, types, nums, pos):
    return {"doc_ids": np.array(doc_ids), "entity_type": np.array(types),
            "type_num": np.array(nums), "list_position": np.array(pos)}


def test_invariants_accept_good_reject_bad():
    good = _result(["d"] * 4, ["channel", "channel", "sub_amount", "sub_amount"], [1, 1, 2, 2], [1, 2, 1, 2])
    rd.check_list_invariants(good)
    with pytest.raises(AssertionError):
        rd.check_list_invariants(_result(["d"] * 2, ["channel"] * 2, [1, 1], [1, 3]))
    with pytest.raises(AssertionError):
        rd.check_list_invariants(_result(["d"] * 2, ["channel", "sub_amount"], [1, 3], [1, 1]))


def test_span_last_token_is_entity_not_closing_quote():
    """Real Qwen tokenizer: the token read for each entry ends the value, not the quote."""
    from transformers import AutoTokenizer

    from judge_interp.representationlm import RepresentationLM

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    r = rd.render_lists({"channel": ["7", "13"], "program_desc": ["Late Movie", "News"], "sub_amount": ["$500.00"]})
    enc = tok(r["text"], add_special_tokens=False, return_offsets_mapping=True)
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    idxs = []
    for s in r["spans"]:
        i = RepresentationLM._last_token_covering(offsets, s["char_end"] - 1)
        idxs.append(i)
        assert offsets[i][0] >= s["char_start"] - 1  # token begins at or after the opening quote
        assert r["text"][offsets[i][0]:s["char_end"]].strip('"').endswith(str(s["value"])[-1])
    assert len(set(idxs)) == len(idxs)


def test_min_types_per_doc_gate():
    items, _, skipped = rd.build_items({**PARAMS, "min_types_per_doc": 5}, 0, REPO_ROOT)
    assert len(items) == 2 and skipped == []  # both fixture docs render all 5 types
    with pytest.raises(ValueError, match="no document"):
        rd.build_items({**PARAMS, "min_types_per_doc": 6}, 0, REPO_ROOT)
    with pytest.raises(ValueError, match="min_types_per_doc"):
        rd.build_items({**PARAMS, "min_types_per_doc": 1}, 0, REPO_ROOT)
