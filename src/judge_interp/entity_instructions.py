"""Instruction strings for the entity-detection judge.

A judge instance is given the instructions built here, the full OCR
``## CONTEXT:``, and a numbered list of candidate values for ONE entity type in
``## QUERY:`` (see ``entity_prompts.render_entity_list``), and must judge each
candidate independently: is this value genuinely, explicitly present in the
document for this field?

Unlike the whole-record judge (``instruction_prompts.py``), there is no
cohesion criterion here -- entity detection asks only whether a value is
supported by the document at all, not whether it belongs with any other field's
value. Relation detection (``relation_instructions.py``) is where cohesion
across fields is judged.
"""
from __future__ import annotations

from judge_interp.instruction_prompts import _EXPLICIT_CRITERION

_HEADER = """You are validating candidate values for a single field, extracted from a document by an information-extraction pipeline. The field is: {entity_type}.

You are given:
1) In ## CONTEXT: the full OCR text of the source document.
2) In ## QUERY: a numbered list of candidate values for this field. Each line ends with a cue, `— valid?`.

For each candidate independently, decide whether it is a VALID value for this field, per this criterion:

{criterion}

Judge only against the context. Do not use outside knowledge about the entities involved, and do not guess when the evidence is unclear — unclear evidence is a failed criterion. Judge every candidate independently: one candidate's validity has no bearing on another's.

For each numbered candidate, answer with exactly one line in the form `<n>: true` or `<n>: false`, in order, one line per candidate."""


def build_entity_detection_instructions(entity_type: str) -> str:
    """Assemble the entity-detection judge instruction block.

    Args:
        entity_type: the field name being judged (e.g. ``"channel"``), shown
            verbatim to the judge so it knows what kind of value it's seeing.

    Returns:
        The instruction string, ready to pass as the ``instructions`` block.

    Raises:
        TypeError: ``entity_type`` is not a non-empty ``str``.
    """
    if not isinstance(entity_type, str) or not entity_type:
        raise TypeError(f"entity_type must be a non-empty str, got {entity_type!r}")

    criterion = _EXPLICIT_CRITERION.format(n=1)
    return _HEADER.format(entity_type=entity_type, criterion=criterion)
