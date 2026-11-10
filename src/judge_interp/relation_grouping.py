"""Relation grouping: the relation-detection read-out under instructions that ask for a grouping.

Same prompts, rows, rendering, labels and read-out as ``relation_detection`` (hidden states at each
entity's closing quote, one prompt per row); only the instruction block differs. Instead of one verdict on
the record, the model is asked to sort the entities into groups (one group per item of the document) by a
greedy rule, so the representations may encode group membership. Nothing is decoded.

``params.document_ids`` restricts the run to every row of those documents (all error types and cells, valid and
corrupted rows), so that per-(document, field) centring in the analysis still rests on all rows of each document.
The item set is that of ``relation_detection`` filtered by document, so rows are identical to the original run's.
"""
from __future__ import annotations

from pathlib import Path

from judge_interp.dataset_spec import DatasetSpec
from judge_interp.relation_detection import (  # noqa: F401  (re-exported)
    ROW_PARAMS,
    RelationDetectionLM,
    run_rows,
)
from judge_interp.relation_detection import build_items as _build_relation_items
from judge_interp.relation_detection import _HEADER as _DETECTION_HEADER

# The shared part (role, inputs, field list) is the detection header up to its second paragraph after the field list;
# the split marker is the first words of the paragraph the grouping instructions replace.
_REPLACED_FROM = "The document may describe many items of the kind this record describes."
_GROUPING_TAIL = """The document may describe many similar items of the kind this record describes. Two entities are related if they are values of the same single item in the document, each under the field that denotes it for that item. An entity that is a genuine value of the document but belongs to a different item, or is listed under the wrong field, is not related to the others.

Your task is to sort the entities into groups, so that each group holds exactly the entities that belong to one item. Go through the entities in the order they are listed:
1. Put the first entity in group 1.
2. For each later entity, check the existing groups in order (group 1, then group 2, and so on). Put it in the first group where it is related to all of the entities. If there is no such group, start a new group with the next number.

Judge only against the context. Do not use outside knowledge, and do not guess when the evidence is unclear: an entity with unclear evidence is not related.

Answer with one line per entity, in the order listed, in the form `field: group N`."""

assert _REPLACED_FROM in _DETECTION_HEADER
_HEADER = _DETECTION_HEADER[: _DETECTION_HEADER.index(_REPLACED_FROM)] + _GROUPING_TAIL

TASK_PARAMS = {"relation_grouping": (*ROW_PARAMS, "document_ids")}


def build_instructions(spec: DatasetSpec) -> str:
    """Build the grouping instruction block (fixed for a run), one description line per field.

    Raises:
        ValueError: the dataset does not support the relation tasks.
    """
    spec.require_relation()
    types = "\n".join(f"- {f}: {spec.field_descriptions[f]}" for f in spec.fields)
    return _HEADER.format(
        record_description=spec.record_description, context_description=spec.context_description, types=types,
    )


def build_items(spec: DatasetSpec, params: dict, seed: int, repo_root: Path, tokenizer=None) -> list[dict]:
    """``relation_detection`` items (same ``rows_per_stratum`` draw) with the grouping instructions, for ``params.document_ids`` only.

    Raises:
        ValueError: a requested document has no row, or ``document_ids`` is empty.
    """
    from judge_interp import row_detection

    ids = list(params["document_ids"])
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("document_ids must be a non-empty list without duplicates")
    items = row_detection.build_items(spec, params, seed, repo_root, tokenizer, build_instructions(spec))
    kept = [it for it in items if it["document_id"] in set(ids)]
    missing = set(ids) - {it["document_id"] for it in kept}
    if missing:
        raise ValueError(f"no row for document(s) {sorted(missing)}")
    return kept


class RelationGroupingLM(RelationDetectionLM):
    """``RowDetectionLM`` for the grouping instructions."""


def run(cfg: dict, spec: DatasetSpec, run_dir: Path, repo_root: Path, model_cfg: None) -> dict:
    """Run the task: see ``row_detection.run_rows``. ``model_cfg`` must be ``None``."""
    assert model_cfg is None
    return run_rows(cfg, spec, run_dir, repo_root, build_items, RelationGroupingLM)
