"""Instruction strings for the relation-detection judge.

A judge instance is given the instructions built here, the full OCR
``## CONTEXT:``, and, in ``## QUERY:``, one numbered entity list per field type
(see ``relation_prompts.render_relation_prompt``) -- every entity shown is
individually a genuine value from the document (entity-level validity is
``entity_instructions.py``'s task, not this one). The judge must find the
m-tuples -- one entity per type -- that are mutually supported by the
document: e.g., for VRDU line items, which channel, program description, pair
of dates, and sub-amount actually belong to the same broadcast.
"""
from __future__ import annotations

_HEADER = """You are validating relationships between entities extracted from a document by an information-extraction pipeline. The entities come in {n_types} types: {entity_types}.

You are given:
1) In ## CONTEXT: the full OCR text of the source document.
2) In ## QUERY: one numbered list per entity type. Every entity listed is a genuine value drawn from the document for its type -- you are not judging whether an individual entity is real, only which combinations of them belong together.

A valid m-tuple selects exactly one entity from each type such that all selected entities are mutually related in the document: they must all describe the same real-world thing (e.g., for VRDU line items, a channel, its program description, its start and end dates, and its sub-amount all belong to the same broadcast). An entity individually present in the context but combined with an entity it does not belong with fails this criterion, even though both entities are themselves genuine.

Some entities may belong to more than one valid tuple (e.g. the same value shared across multiple true groupings) -- report every valid tuple you find, not just one per entity.

List every valid tuple you find, one per line, in the form `({index_format})`, using the numbered index from each type's list."""


def build_relation_detection_instructions(entity_types: list[str]) -> str:
    """Assemble the relation-detection judge instruction block.

    Args:
        entity_types: the field names being related, in the exact order their
            numbered lists appear in ``## QUERY:`` (must match
            ``relation_prompts.render_relation_prompt``'s section order).

    Returns:
        The instruction string, ready to pass as the ``instructions`` block.

    Raises:
        ValueError: ``entity_types`` has fewer than 2 entries (a "relation"
            needs at least 2 types) or a duplicate entry.
    """
    if not isinstance(entity_types, list) or len(entity_types) < 2:
        raise ValueError(f"entity_types must be a list of >= 2 field names, got {entity_types!r}")
    if len(set(entity_types)) != len(entity_types):
        raise ValueError(f"entity_types has duplicate(s): {entity_types!r}")

    index_format = ", ".join(f"<type {i} index>" for i in range(1, len(entity_types) + 1))
    return _HEADER.format(
        n_types=len(entity_types), entity_types=", ".join(entity_types), index_format=index_format
    )
