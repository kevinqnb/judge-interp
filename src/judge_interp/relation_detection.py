"""Relation detection: hidden states read at each entity's closing quote, one prompt per row.

Same pipeline as ``entity_detection``, with different instructions. ``build_instructions``
asks the judge whether the non-null fields of one row all describe the same item among the
several similar items a document may contain (for VRDU, the same line item of the order),
i.e. whether the entities belong together, each value being what its field denotes for that
item (a value of the right item listed under the wrong field fails the relation). The judge gives one verdict on the record as a
whole, not one per entity; representations are still collected at every entity's closing
quote and labelled per entity. Only datasets whose spec has ``supports_relation`` are
accepted.

Item building, rendering, the model-side read-out, label checks and metrics are shared with
``entity_detection`` and live in ``row_detection`` (re-exported here). Rendering and labels
are as there: each non-null value is a JSON string followed by one pad space before its
closing quote, the query ends with a newline, and labels are per entity.
Layers are whatever the config lists; no logits are materialised (``logits_to_keep=1``).
"""
from __future__ import annotations

from pathlib import Path

from judge_interp.dataset_spec import DatasetSpec
from judge_interp.row_detection import (  # noqa: F401  (re-exported: the task's public API)
    SPAN_META_KEYS,
    ROW_PARAMS,
    RowDetectionLM,
    check_row,
    check_row_invariants,
    compute_metrics,
    render_row,
    row_index,
    select_rows,
)
from judge_interp.row_detection import build_items as _build_items
from judge_interp.row_detection import run_rows


_HEADER = """You are validating the relationships between entities extracted from a document by an information-extraction pipeline. The entities are the values of one record: {record_description}.

You are given:
1) In ## CONTEXT: the full {context_description}.
2) In ## EXTRACTED RECORD: the extracted record. It has one line per extracted field, in the form `field: value`, each line ending with a newline. A value is a JSON string in double quotes. Fields that were not extracted are not listed. Each string value is followed by one space before its closing quote; that padding space is not part of the value. The closing quote and the newline after it end the entity.

The fields that may appear are:
{types}

The document may describe many items of the kind this record describes. A valid record is one whose entities are all related: taken together, they describe the same single item in the document. If one or more entities belong to a different item than the others, the record as a whole is not valid, even though each of those entities may be a genuine value of the document.

The field names are part of the relation. Each value must be what its field denotes for that item, as described in the list of fields above. A value that belongs to the right item but is listed under the wrong field (it is actually the value of a different field) breaks the relation, and the record is not valid.

Do not judge entities one by one; judge the record as a whole: do all its values, each under its own field, belong together as one item? Judge only against the context. Do not use outside knowledge about the entities involved, and do not guess when the evidence is unclear — unclear evidence is a failed criterion. Give one verdict for the whole record.

Answer with exactly one line: `related: true` if all the entities of the record belong together as one item, each under its correct field, or `related: false` if they do not."""

def build_instructions(spec: DatasetSpec) -> str:
    """Build the instruction block, which is fixed for a run.

    Args:
        spec: the dataset; must have ``supports_relation``.

    Returns:
        The instruction text, with one description line per field.

    Raises:
        ValueError: the dataset does not support the relation tasks.
    """
    spec.require_relation()
    types = "\n".join(f"- {f}: {spec.field_descriptions[f]}" for f in spec.fields)
    return _HEADER.format(
        record_description=spec.record_description, context_description=spec.context_description, types=types,
    )


def build_items(spec: DatasetSpec, params: dict, seed: int, repo_root: Path, tokenizer=None) -> list[dict]:
    """Build one ``RelationDetectionLM.collect_rows`` item per selected row.

    See ``row_detection.build_items``; ``tokenizer`` must be given if and only if
    ``params["context_token_limit"]`` is not ``None``.

    Raises:
        ValueError: ``spec`` does not support the relation tasks, or as in
            ``row_detection.build_items``.
    """
    return _build_items(spec, params, seed, repo_root, tokenizer, build_instructions(spec))


class RelationDetectionLM(RowDetectionLM):
    """``RowDetectionLM`` for the relation-detection instructions."""


TASK_PARAMS = {"relation_detection": ROW_PARAMS}


def run(cfg: dict, spec: DatasetSpec, run_dir: Path, repo_root: Path, model_cfg: None) -> dict:
    """Run the task: see ``row_detection.run_rows``. ``model_cfg`` must be ``None`` (no model config)."""
    assert model_cfg is None
    return run_rows(cfg, spec, run_dir, repo_root, build_items, RelationDetectionLM)
