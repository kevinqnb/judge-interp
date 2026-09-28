from __future__ import annotations

import pytest

from judge_interp.entity_instructions import build_entity_detection_instructions
from judge_interp.instruction_prompts import _EXPLICIT_CRITERION


def test_includes_entity_type_and_criterion():
    out = build_entity_detection_instructions("channel")
    assert "channel" in out
    assert _EXPLICIT_CRITERION.format(n=1) in out


def test_output_format_stated():
    out = build_entity_detection_instructions("channel")
    assert "<n>: true" in out or "`<n>: true`" in out


def test_no_cohesion_language():
    out = build_entity_detection_instructions("channel")
    assert "cohesion" not in out.lower()


@pytest.mark.parametrize("bad", ["", 123, None, ["channel"]])
def test_bad_entity_type_raises(bad):
    with pytest.raises(TypeError):
        build_entity_detection_instructions(bad)


def test_different_entity_types_differ():
    a = build_entity_detection_instructions("channel")
    b = build_entity_detection_instructions("program_desc")
    assert a != b
