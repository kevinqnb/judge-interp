"""Unit tests for judge_interp.instruction_prompts."""
import pytest

from judge_interp.instruction_prompts import build_instructions


def test_cohesion_clause_toggles():
    with_cohesion = build_instructions(include_cohesion=True)
    without = build_instructions(include_cohesion=False)

    # The worked address example is the cohesion criterion's body.
    assert "63 Chestnut St." in with_cohesion
    assert "Cohesion" in with_cohesion
    assert "63 Chestnut St." not in without
    assert "Cohesion" not in without

    # The explicit-support criterion is in both.
    assert "Explicit support" in with_cohesion
    assert "Explicit support" in without


def test_verdict_format_always_stated():
    for flag in (True, False):
        text = build_instructions(include_cohesion=flag)
        assert "`true`" in text and "`false`" in text
        assert "exactly one token" in text


def test_criteria_numbered_from_one():
    assert build_instructions(include_cohesion=True).count("(1)") >= 1
    assert "(2)" in build_instructions(include_cohesion=True)
    assert "(2)" not in build_instructions(include_cohesion=False)


def test_non_bool_rejected():
    with pytest.raises(TypeError):
        build_instructions(1)
    with pytest.raises(TypeError):
        build_instructions("yes")
