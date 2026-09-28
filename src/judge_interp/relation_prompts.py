"""Build and render per-document relation-detection prompts for the ``line``
dataset: a valid-only entity list per field type, plus ground-truth tuple
membership drawn from the existing rows (no new corruption synthesis).

Pure text / list work only -- no torch, no nnsight -- mirrors ``prompts.py``
and ``entity_prompts.py``'s login-node-testable discipline.

A valid m-tuple is one line item's own field values -- by construction they
already co-occur validly (``error_type: null`` rows). An invalid m-tuple is an
``intra_document``-corrupted row: its fields are individually real entities of
the *same* document, but 1..len(LINE_FIELDS) of them were swapped from a
different line item, so the combination isn't mutually supported. This is only
meaningful for ``line`` -- ``main`` has one entity per type per document, so
there's no relational structure to test (see ``entity_prompts`` for that
dataset's entity-level task).

Entities are presented once per type (not once per candidate tuple): the
document context ``D`` -- given in full, before any entity list -- is the
channel through which an entity's own representation can encode its true
relational identity, independent of list position. See devlog/build note for
why this design was kept over re-rendering one block per candidate tuple.
"""
from __future__ import annotations

from pathlib import Path

from judge_interp.entity_prompts import LINE_FIELDS, build_entity_lists, render_entity_list
from judge_interp.prompts import load_split


def build_relation_lists(data_root: str | Path, split: str, seed: int, min_valid_per_type: int) -> dict:
    """Aggregate ``line`` rows into per-document, per-type valid-only entity
    lists, plus ground-truth tuple membership.

    Reuses ``entity_prompts.build_entity_lists`` for the entity lists
    themselves (with ``min_invalid_per_list=0, max_invalid_per_list=0``, which
    forces every list to be valid-only regardless of how many decoys exist for
    that field -- this task does not need a second aggregation path). A
    (document, field) list absent from the result (fewer than
    ``min_valid_per_type`` distinct valid values) is simply omitted from that
    document's rendering -- see ``render_relation_prompt`` -- not treated as
    disqualifying the whole document.

    Returns:
        ``{
          "lists": {(document_id, field): [(value, True), ...]},
          "tuples": {document_id: [{"row_key", "valid", "invalid_fields",
                                     "entity_index": {field: value}}, ...]},
          "skipped": [{"document_id", "field", "n_valid"}, ...],
        }``
        ``entity_index`` maps every ``LINE_FIELDS`` name to that row's exact
        value (``None`` for a null field) -- the join key back to ``lists``.
        ``tuples`` includes every document with at least one row, even one
        with no surviving entity list at all (an empty ``lists`` entry for it)
        -- ground-truth tuple bookkeeping doesn't depend on rendering.

    Raises:
        ValueError: ``min_valid_per_type`` not a non-negative int.
    """
    entities_result = build_entity_lists(
        data_root,
        "line",
        split,
        seed=seed,
        min_valid_per_list=min_valid_per_type,
        min_invalid_per_list=0,
        max_invalid_per_list=0,
    )
    lists = entities_result["lists"]
    for key, entities in lists.items():
        assert all(label for _, label in entities), f"{key}: relation lists must be valid-only"

    rows = load_split(data_root, "line", split)
    by_doc: dict[str, list[dict]] = {}
    for r in rows:
        by_doc.setdefault(r["document_id"], []).append(r)

    tuples: dict[str, list[dict]] = {}
    for doc_id, doc_rows in by_doc.items():
        candidates = [r for r in doc_rows if r["valid"] or r["error_type"] == "intra_document"]
        tuples[doc_id] = [
            {
                "row_key": (doc_id, r["line_index"], r["error_type"], r["k"]),
                "valid": bool(r["valid"]),
                "invalid_fields": list(r["invalid_fields"]),
                "entity_index": {field: r[field] for field in LINE_FIELDS},
            }
            for r in candidates
        ]

    return {"lists": lists, "tuples": tuples, "skipped": entities_result["skipped"]}


def render_relation_prompt(doc_lists: dict[str, list[tuple[str, bool]]]) -> dict:
    """Render a document's per-type valid entity lists as the ``## QUERY:`` block.

    Args:
        doc_lists: ``{field: [(value, True), ...]}`` for this document, one
            entry per ``LINE_FIELDS`` type that had enough valid entities to
            include (a type absent from ``doc_lists`` is omitted entirely, not
            rendered as an empty section).

    Returns:
        ``{"text": str, "spans": {field: [entity-span-dict, ...]}}`` -- each
        field's spans are exactly ``render_entity_list``'s per-entity span
        dicts (``content_start``/``content_end``/etc.), offset into the
        combined ``text`` rather than into that field's own section alone.

    Raises:
        ValueError: ``doc_lists`` is empty, or a field is not in
            ``entity_prompts.LINE_FIELDS``.
    """
    if not doc_lists:
        raise ValueError("doc_lists is empty")
    unknown = set(doc_lists) - set(LINE_FIELDS)
    if unknown:
        raise ValueError(f"doc_lists has field(s) outside LINE_FIELDS: {sorted(unknown)}")

    sections: list[str] = []
    spans: dict[str, list[dict]] = {}
    offset = 0
    type_num = 0
    for field in LINE_FIELDS:  # fixed type order -- see module docstring
        if field not in doc_lists:
            continue
        type_num += 1
        header = f"TYPE {type_num} ENTITIES ({field}):\n"
        rendered = render_entity_list(doc_lists[field])

        content_offset = offset + len(header)
        for span in rendered["spans"]:
            span["content_start"] += content_offset
            span["content_end"] += content_offset
            span["cue_start"] += content_offset
            span["cue_end"] += content_offset
        spans[field] = rendered["spans"]

        section = header + rendered["text"]
        sections.append(section)
        offset += len(section) + 2  # +2 for the "\n\n" section separator below

    return {"text": "\n\n".join(sections), "spans": spans}
