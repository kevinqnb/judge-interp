from __future__ import annotations

import pytest

from judge_interp.relation_instructions import build_relation_detection_instructions


def test_includes_all_entity_types_and_count():
    out = build_relation_detection_instructions(["channel", "program_desc", "sub_amount"])
    assert "channel, program_desc, sub_amount" in out
    assert "3 types" in out
    assert "type 3 index" in out


def test_output_format_stated():
    out = build_relation_detection_instructions(["channel", "program_desc"])
    assert "(<type 1 index>, <type 2 index>)" in out


def test_shared_entity_language_present():
    out = build_relation_detection_instructions(["channel", "program_desc"])
    assert "more than one valid tuple" in out


@pytest.mark.parametrize(
    "bad",
    [
        [],
        ["channel"],
        ["channel", "channel"],
        "channel,program_desc",
        None,
    ],
)
def test_bad_entity_types_raises(bad):
    with pytest.raises(ValueError):
        build_relation_detection_instructions(bad)
