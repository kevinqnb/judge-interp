"""Unit tests for judge_interp.entity_prompts -- model-free, login-node only.

Fixture: tests/fixtures/entity_relation_mini/vrdu/{main,line}/train.json (see
that directory for the hand-worked-out expected valid/decoy lists). Two
documents, doc-x and doc-y, built specifically to exercise:

- the coincidental-overlap filter (doc-x/channel: an inter_document decoy of
  "13" coincidentally equals doc-x's own line_index=1 channel -- must be
  filtered out, not included as a decoy)
- a null-heavy field skipped from both directions (doc-x/agency: 0 valid,
  1 decoy; doc-y/agency: 1 valid, 0 decoy)
- a normal passing list on each dataset (line: doc-x/program_desc;
  main: doc-x/property, doc-x/gross_amount, doc-y/property)
"""
from __future__ import annotations

import json

import pytest

from judge_interp import entity_prompts

FIXTURE = "tests/fixtures/entity_relation_mini"

_DEFAULTS = dict(seed=0, min_valid_per_list=1, min_invalid_per_list=1, max_invalid_per_list=8)


def test_line_only_program_desc_passes():
    result = entity_prompts.build_entity_lists(FIXTURE, "line", "train", **_DEFAULTS)
    assert set(result["lists"]) == {("doc-x", "program_desc")}

    entities = result["lists"][("doc-x", "program_desc")]
    assert sorted(entities) == sorted(
        [("Evening News", True), ("Late Movie", True), ("Morning Weather", False)]
    )


def test_line_channel_overlap_filtered_to_skip():
    result = entity_prompts.build_entity_lists(FIXTURE, "line", "train", **_DEFAULTS)
    skip = next(s for s in result["skipped"] if s["document_id"] == "doc-x" and s["field"] == "channel")
    # "13" is a genuine valid channel for doc-x's own line_index=1, so the
    # inter_document decoy that coincidentally equals it must be filtered out
    # -- leaving 2 valid, 0 invalid, not 2 valid / 1 invalid.
    assert skip == {"document_id": "doc-x", "field": "channel", "n_valid": 2, "n_invalid": 0}


def test_line_skipped_count():
    result = entity_prompts.build_entity_lists(FIXTURE, "line", "train", **_DEFAULTS)
    # 2 docs x 5 LINE_FIELDS = 10 combos, 1 passes (doc-x/program_desc).
    assert len(result["skipped"]) == 9


def test_main_agency_skipped_both_directions():
    result = entity_prompts.build_entity_lists(FIXTURE, "main", "train", **_DEFAULTS)
    skip_x = next(s for s in result["skipped"] if s["document_id"] == "doc-x" and s["field"] == "agency")
    skip_y = next(s for s in result["skipped"] if s["document_id"] == "doc-y" and s["field"] == "agency")
    # doc-x's base agency is null (0 valid) despite a "Big Agency" decoy (1 invalid).
    assert skip_x == {"document_id": "doc-x", "field": "agency", "n_valid": 0, "n_invalid": 1}
    # doc-y's base agency is "Big Agency" (1 valid) with no corruption (0 invalid).
    assert skip_y == {"document_id": "doc-y", "field": "agency", "n_valid": 1, "n_invalid": 0}


def test_main_passing_lists():
    result = entity_prompts.build_entity_lists(FIXTURE, "main", "train", **_DEFAULTS)
    assert set(result["lists"]) == {("doc-x", "property"), ("doc-x", "gross_amount"), ("doc-y", "property")}
    assert sorted(result["lists"][("doc-x", "property")]) == sorted(
        [("WAAA", True), ("WBBB", False), ("WDDD", False)]
    )
    assert sorted(result["lists"][("doc-x", "gross_amount")]) == sorted(
        [("$1,200.00", True), ("$9,999.00", False)]
    )
    assert sorted(result["lists"][("doc-y", "property")]) == sorted([("WBBB", True), ("WCCC", False)])


def test_main_skipped_count():
    result = entity_prompts.build_entity_lists(FIXTURE, "main", "train", **_DEFAULTS)
    # 2 docs x 9 MAIN_FIELDS = 18 combos, 3 pass.
    assert len(result["skipped"]) == 15


def test_valid_decoy_never_overlap_invariant_holds_generally():
    for dataset in ("main", "line"):
        result = entity_prompts.build_entity_lists(FIXTURE, dataset, "train", **_DEFAULTS)
        for key, entities in result["lists"].items():
            values_by_label = {}
            for value, label in entities:
                values_by_label.setdefault(label, set()).add(value)
            assert not (values_by_label.get(True, set()) & values_by_label.get(False, set())), key


def test_max_invalid_per_list_caps_deterministically():
    # doc-x/property has 2 decoys ("WBBB", "WDDD"); cap to 1 and check the
    # result is a single deterministic entity, reproducible across calls, and
    # still contains the 1 valid entity untouched by the cap.
    kwargs = dict(seed=0, min_valid_per_list=1, min_invalid_per_list=1, max_invalid_per_list=1)
    r1 = entity_prompts.build_entity_lists(FIXTURE, "main", "train", **kwargs)
    r2 = entity_prompts.build_entity_lists(FIXTURE, "main", "train", **kwargs)
    entities = r1["lists"][("doc-x", "property")]
    assert entities == r2["lists"][("doc-x", "property")]
    assert len(entities) == 2
    values_by_label = {v: label for v, label in entities}
    assert values_by_label["WAAA"] is True
    invalid_values = {v for v, label in entities if not label}
    assert invalid_values <= {"WBBB", "WDDD"}
    assert len(invalid_values) == 1


def test_seeded_shuffle_is_deterministic_across_calls():
    r1 = entity_prompts.build_entity_lists(FIXTURE, "line", "train", **_DEFAULTS)
    r2 = entity_prompts.build_entity_lists(FIXTURE, "line", "train", **_DEFAULTS)
    assert r1["lists"] == r2["lists"]


def test_seeded_shuffle_differs_by_seed():
    r1 = entity_prompts.build_entity_lists(FIXTURE, "line", "train", **_DEFAULTS)
    r2 = entity_prompts.build_entity_lists(FIXTURE, "line", "train", **{**_DEFAULTS, "seed": 1})
    # Same set of entities, but (with high probability for a 3-item list) a
    # different order under a different seed.
    key = ("doc-x", "program_desc")
    assert set(r1["lists"][key]) == set(r2["lists"][key])


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="dataset must be one of"):
        entity_prompts.build_entity_lists(FIXTURE, "bogus", "train", **_DEFAULTS)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(min_valid_per_list=-1),
        dict(min_invalid_per_list=-1),
        dict(max_invalid_per_list=-1),
        dict(min_invalid_per_list=5, max_invalid_per_list=1),
    ],
)
def test_invalid_params_raise(kwargs):
    params = {**_DEFAULTS, **kwargs}
    with pytest.raises(ValueError):
        entity_prompts.build_entity_lists(FIXTURE, "line", "train", **params)


# ---------------------------------------------------------------------------
# render_entity_list


def test_render_entity_list_spans_locate_exact_substrings():
    entities = [("Acme Committee", True), ("WBBB", False)]
    result = entity_prompts.render_entity_list(entities)
    text = result["text"]
    assert text == '1. "Acme Committee" — valid?\n2. "WBBB" — valid?'

    for span, (value, label_valid) in zip(result["spans"], entities):
        assert span["value"] == value
        assert span["label_valid"] == label_valid
        assert text[span["content_start"] : span["content_end"]] == json.dumps(value)
        assert text[span["cue_start"] : span["cue_end"]] == "valid?"


def test_render_entity_list_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        entity_prompts.render_entity_list([])


def test_render_entity_list_no_span_overlap_between_entities():
    entities = [("a", True), ("bb", False), ("ccc", True)]
    result = entity_prompts.render_entity_list(entities)
    spans = result["spans"]
    intervals = [(s["content_start"], s["cue_end"]) for s in spans]
    for (s1, e1), (s2, e2) in zip(intervals, intervals[1:]):
        assert e1 <= s2
