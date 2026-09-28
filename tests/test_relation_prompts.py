"""Unit tests for judge_interp.relation_prompts -- model-free, login-node only.

Uses the same fixture as test_entity_prompts.py
(tests/fixtures/entity_relation_mini). doc-x has 2 line items (line_index 0, 1)
plus one intra_document-corrupted variant of line_index=1 (program_desc
swapped from line_index=0's "Evening News") -- the invalid-tuple ground truth
-- and two inter_document decoy rows, which must NOT show up as tuples (they
aren't real line items, just entity-detection decoys). doc-y has a single line
item and no corruption at all.
"""
from __future__ import annotations

import json

import pytest

from judge_interp import relation_prompts
from judge_interp.entity_prompts import LINE_FIELDS

FIXTURE = "tests/fixtures/entity_relation_mini"


def test_lists_are_valid_only_and_cover_all_fields_with_enough_entities():
    result = relation_prompts.build_relation_lists(FIXTURE, "train", seed=0, min_valid_per_type=1)
    # Every LINE_FIELDS type has >=1 valid value for both docs in this fixture,
    # and min_invalid_per_list=0 relaxes the invalid-count gate entirely, so
    # nothing here is skipped.
    assert result["skipped"] == []
    assert set(result["lists"]) == {(doc, field) for doc in ("doc-x", "doc-y") for field in LINE_FIELDS}
    for entities in result["lists"].values():
        assert all(label for _, label in entities)

    assert sorted(result["lists"][("doc-x", "channel")]) == sorted([("7", True), ("13", True)])
    assert sorted(result["lists"][("doc-x", "program_desc")]) == sorted(
        [("Evening News", True), ("Late Movie", True)]
    )
    assert result["lists"][("doc-x", "sub_amount")] == [("$500.00", True)]


def test_tuples_include_valid_and_intra_document_only():
    result = relation_prompts.build_relation_lists(FIXTURE, "train", seed=0, min_valid_per_type=1)

    doc_x_tuples = result["tuples"]["doc-x"]
    assert len(doc_x_tuples) == 3  # 2 valid line items + 1 intra_document invalid variant
    valid_tuples = [t for t in doc_x_tuples if t["valid"]]
    invalid_tuples = [t for t in doc_x_tuples if not t["valid"]]
    assert len(valid_tuples) == 2
    assert len(invalid_tuples) == 1

    invalid = invalid_tuples[0]
    assert invalid["row_key"] == ("doc-x", 1, "intra_document", 1)
    assert invalid["invalid_fields"] == ["program_desc"]
    assert invalid["entity_index"]["channel"] == "13"
    # The swapped-in value is itself a real entity of the document (line_index=0's).
    assert invalid["entity_index"]["program_desc"] == "Evening News"

    doc_y_tuples = result["tuples"]["doc-y"]
    assert len(doc_y_tuples) == 1
    assert doc_y_tuples[0]["valid"] is True


def test_inter_document_rows_never_become_tuples():
    result = relation_prompts.build_relation_lists(FIXTURE, "train", seed=0, min_valid_per_type=1)
    row_keys = {t["row_key"] for t in result["tuples"]["doc-x"]}
    assert not any(rk[2] == "inter_document" for rk in row_keys)


def test_entity_index_covers_all_line_fields_including_null():
    result = relation_prompts.build_relation_lists(FIXTURE, "train", seed=0, min_valid_per_type=1)
    line0 = next(t for t in result["tuples"]["doc-x"] if t["row_key"] == ("doc-x", 0, None, 0))
    assert set(line0["entity_index"]) == set(LINE_FIELDS)
    assert line0["entity_index"]["sub_amount"] is None


def test_high_min_valid_per_type_skips_singleton_fields():
    result = relation_prompts.build_relation_lists(FIXTURE, "train", seed=0, min_valid_per_type=2)
    # doc-x's dates/sub_amount each have only 1 distinct valid value.
    assert ("doc-x", "program_start_date") not in result["lists"]
    assert ("doc-x", "sub_amount") not in result["lists"]
    assert ("doc-x", "channel") in result["lists"]
    assert ("doc-x", "program_desc") in result["lists"]
    # Tuple bookkeeping is unaffected by which lists render.
    assert len(result["tuples"]["doc-x"]) == 3


# ---------------------------------------------------------------------------
# render_relation_prompt


def test_render_relation_prompt_headers_in_fixed_type_order():
    result = relation_prompts.build_relation_lists(FIXTURE, "train", seed=0, min_valid_per_type=1)
    rendered = relation_prompts.render_relation_prompt(_doc_lists(result, "doc-x"))
    text = rendered["text"]
    headers = [f"TYPE {i + 1} ENTITIES ({field}):" for i, field in enumerate(LINE_FIELDS)]
    positions = [text.index(h) for h in headers]
    assert positions == sorted(positions)


def test_render_relation_prompt_spans_locate_exact_substrings():
    result = relation_prompts.build_relation_lists(FIXTURE, "train", seed=0, min_valid_per_type=1)
    doc_lists = _doc_lists(result, "doc-x")
    rendered = relation_prompts.render_relation_prompt(doc_lists)
    text = rendered["text"]
    for field, spans in rendered["spans"].items():
        for span in spans:
            assert text[span["content_start"] : span["content_end"]] == json.dumps(span["value"])
            assert text[span["cue_start"] : span["cue_end"]] == "valid?"


def test_render_relation_prompt_omits_missing_field_section():
    doc_lists = {"channel": [("7", True)]}
    rendered = relation_prompts.render_relation_prompt(doc_lists)
    assert "program_desc" not in rendered["text"]
    assert set(rendered["spans"]) == {"channel"}


def test_render_relation_prompt_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        relation_prompts.render_relation_prompt({})


def test_render_relation_prompt_unknown_field_raises():
    with pytest.raises(ValueError, match="LINE_FIELDS"):
        relation_prompts.render_relation_prompt({"not_a_real_field": [("x", True)]})


def _doc_lists(result: dict, doc_id: str) -> dict:
    return {field: entities for (d, field), entities in result["lists"].items() if d == doc_id}
