"""Entity detection: hidden states read at each entity's closing quote, one prompt per dataset row.

``build_instructions`` states the judging task: for each non-null field of a row, decide
whether the value is explicitly supported by the context and is the right kind of value for
that field. Everything else -- item building, rendering, the model-side read-out, label
checks and metrics -- is shared with ``relation_detection`` and lives in ``row_detection``;
its public functions are re-exported here.

The task is dataset-independent: field names and descriptions, the record and context
wording, the error types and the file layout come from a ``DatasetSpec``.
"""
from __future__ import annotations

from pathlib import Path

from judge_interp.dataset_spec import DatasetSpec
from judge_interp.row_detection import (  # noqa: F401  (re-exported: the task's public API)
    SPAN_META_KEYS,
    RowDetectionLM,
    check_row,
    check_row_invariants,
    compute_metrics,
    render_row,
    row_index,
    select_rows,
)
from judge_interp.row_detection import build_items as _build_items


_HEADER = """You are validating a record extracted from a document by an information-extraction pipeline.

You are given:
1) In ## CONTEXT: the full {context_description}.
2) In ## QUERY: the extracted record, which is {record_description}. It has one line per field, in the form `field: value`. A value is a JSON string in double quotes, or `null` when the field was not extracted. Each string value is followed by one space before its closing quote; that padding space is not part of the value.

The fields are:
{types}

For each field whose value is not null, decide independently whether the value is a VALID value for that field in this document. A value is valid only if BOTH criteria below hold:

{criteria}

Judge only against the context. Do not use outside knowledge about the entities involved, and do not guess when the evidence is unclear — unclear evidence is a failed criterion. Judge every field independently: one field's validity has no bearing on another's.

For each field with a value, answer with exactly one line in the form `<field>: true` or `<field>: false`, in the order shown. Do not answer for fields whose value is null."""

EXPLICIT_CRITERION = """({n}) Explicit support. Every non-null value must appear explicitly in the context. A value may differ from the context only in trivial surface formatting (whitespace, line breaks, letter case, and equivalent number forms such as 10 vs 10.0 or 1,000 vs 1000). A value produced by paraphrasing, summarising, inferring, rounding, unit conversion, or any other transformation of the text is NOT explicit support. A `null` value always satisfies this criterion."""

_TYPE_CRITERION = """({n}) Type correspondence. The field name is a claim about what the value is, and it must hold in this document: the value must be what that field denotes here, as described in the list of fields above. Appearing somewhere in the context is not enough. A value that is present in the context but is actually the value of a different field fails this criterion, even though it is a genuine entity of the document. So does a value that belongs to a different part of the document than the record being judged.{hint} Check each value against the role its field requires, not merely against the text."""


def build_criteria(spec: DatasetSpec) -> str:
    """The numbered validity criteria of the instructions (also used by ``clef``)."""
    return "\n".join([
        EXPLICIT_CRITERION.format(n=1),
        _TYPE_CRITERION.format(
            n=2, hint="" if spec.type_confusion_hint is None else f" For example, {spec.type_confusion_hint}."
        ),
    ])


def build_instructions(spec: DatasetSpec) -> str:
    """The instruction block for one dataset (fixed for a run)."""
    types = "\n".join(f"- {f}: {spec.field_descriptions[f]}" for f in spec.fields)
    return _HEADER.format(
        record_description=spec.record_description, context_description=spec.context_description, types=types,
        criteria=build_criteria(spec),
    )


def build_items(spec: DatasetSpec, params: dict, seed: int, repo_root: Path, tokenizer=None) -> list[dict]:
    """Build one ``EntityDetectionLM.collect_rows`` item per selected row.

    See ``row_detection.build_items``; ``tokenizer`` must be given if and only if
    ``params["context_token_limit"]`` is not ``None``.
    """
    return _build_items(spec, params, seed, repo_root, tokenizer, build_instructions(spec))


class EntityDetectionLM(RowDetectionLM):
    """``RowDetectionLM`` for the entity-detection instructions."""
