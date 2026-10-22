"""Shared data access and prompt helpers for the judge tasks.

Dataset-independent: everything that depends on the dataset (field lists, descriptions,
error types, file layout) comes from a ``DatasetSpec`` (see ``dataset_spec``). Provides
``load_split`` / ``load_context`` readers, ``render_query`` for the ``## EXTRACTED RECORD:`` block,
``resolve_layers`` for config layer lists, and ``sub_seed`` for deterministic per-key
seeds.

No torch or nnsight is imported, so this module is testable on a login node.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from judge_interp.dataset_spec import DatasetSpec

# Row keys that describe the row rather than the extraction -- never shown to the
# judge (they would leak the label). ``error_type`` is null on valid rows and one
# of the dataset's error types on invalid ones; ``field_errors`` labels each field.
NON_FIELD_KEYS: frozenset[str] = frozenset(
    {"document_id", "line_index", "valid", "k", "num_invalid_fields", "invalid_fields",
     "error_type", "field_errors"}
)

LAST_LAYER_TOKEN = "last"


def sub_seed(seed: int, *parts: str) -> int:
    """Derive a deterministic seed from a base seed and string parts.

    Uses SHA-256 rather than ``hash``, which is randomised per process for ``str``.

    Args:
        seed: base seed.
        *parts: strings identifying the draw (e.g. dataset, split, stratum).

    Returns:
        An int in ``[0, 2**63)`` suitable for ``random.Random``.
    """
    key = "|".join([str(seed), *parts]).encode()
    return int(hashlib.sha256(key).hexdigest(), 16) % (2**63)


def load_split(data_root: str | Path, spec: DatasetSpec, split: str, error_types: list[str]) -> list[dict]:
    """Read the rows of one split: every valid row plus the selected invalid rows.

    Args:
        data_root: directory the spec's ``split_path`` is relative to.
        spec: the dataset.
        split: ``"train"`` or ``"test"``.
        error_types: error types of invalid rows to keep; an empty list keeps valid rows only.

    Returns:
        The rows of ``<data_root>/<spec.split_path>`` that are valid or whose
        ``error_type`` is in ``error_types``.

    Raises:
        ValueError: unknown split or error type, or duplicate error types.
        FileNotFoundError: the split file does not exist.
        AssertionError: the file is empty, has a row with an unrecognised error type, or
            has no valid row after filtering.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    if not isinstance(error_types, list):
        raise ValueError(f"error_types must be a list, got {error_types!r}")
    unknown = set(error_types) - set(spec.error_types)
    if unknown or len(set(error_types)) != len(error_types):
        raise ValueError(
            f"error_types {error_types!r} must be distinct members of {list(spec.error_types)} for {spec.name!r}"
        )
    path = Path(data_root) / spec.split_path.format(split=split)
    if not path.is_file():
        raise FileNotFoundError(f"no such split file: {path}")
    with open(path) as f:
        rows = json.load(f)
    assert isinstance(rows, list) and rows, f"{path}: expected a non-empty list"
    kept = [r for r in rows if r["valid"] or r["error_type"] in error_types]
    assert all(r["error_type"] in spec.error_types for r in rows if not r["valid"]), (
        f"{path}: invalid row with an unrecognised error_type"
    )
    assert any(r["valid"] for r in kept), f"{path}: no valid rows"
    return kept


def load_context(data_root: str | Path, spec: DatasetSpec, document_id: str) -> str:
    """Read the full context text of one document.

    Args:
        data_root: directory the spec's ``context_path`` is relative to.
        spec: the dataset.
        document_id: document identifier (the ``{document_id}`` of ``spec.context_path``).

    Returns:
        The context text.

    Raises:
        FileNotFoundError: no context file for the document.
        AssertionError: the text is empty.
    """
    path = Path(data_root) / spec.context_path.format(document_id=document_id)
    if not path.is_file():
        raise FileNotFoundError(f"no context text for document {document_id!r}: {path}")
    text = path.read_text()
    assert text.strip(), f"context text for document {document_id!r} is empty: {path}"
    return text


def render_query(row: dict, spec: DatasetSpec) -> str:
    """Render a row as the ``## EXTRACTED RECORD:`` block.

    Emits only the non-null fields of ``spec``, in canonical order, one ``field: <json>``
    line each; a null field is omitted. JSON encoding keeps a value that contains a
    newline on one line.

    Args:
        row: a dataset row; label and identity keys are never emitted.
        spec: the dataset.

    Returns:
        The newline-joined query text, without a trailing newline.

    Raises:
        AssertionError: the row lacks a field of ``spec``, or a field collides with a
            label/identity key.
    """
    fields = spec.fields
    leaked = NON_FIELD_KEYS.intersection(fields)
    assert not leaked, f"whitelisted query fields collide with non-field keys: {sorted(leaked)}"
    missing = [f for f in fields if f not in row]
    assert not missing, f"row is missing whitelisted field(s): {missing}"

    lines = [f"{f}: {json.dumps(row[f])}" for f in fields if row[f] is not None]
    return "\n".join(lines)


def resolve_layers(layers: list, n_layers: int) -> list[int]:
    """Validate a config layer list against the model depth.

    Layer convention: ``0`` is the token-embedding output, ``1..n_layers-1`` is the
    residual stream after that many transformer blocks (pre-norm), and ``n_layers`` is
    the post-final-norm state. Row L2 norms are not comparable across that last boundary.

    Args:
        layers: ints and/or the string ``"last"`` (resolved to ``n_layers``).
        n_layers: number of transformer blocks in the model.

    Returns:
        The distinct layer indices, sorted ascending.

    Raises:
        ValueError: empty list, unrecognised or out-of-range entry, duplicate (after
            resolving ``"last"``), or ``n_layers < 1``.
    """
    if not isinstance(layers, list) or not layers:
        raise ValueError(f"layers must be a non-empty list, got {layers!r}")
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")

    seen: set[int] = set()
    for entry in layers:
        if entry == LAST_LAYER_TOKEN:
            value = n_layers
        elif isinstance(entry, bool) or not isinstance(entry, (int, np.integer)):
            raise ValueError(f"layer entry {entry!r} is not an int or {LAST_LAYER_TOKEN!r}")
        else:
            value = int(entry)
        if not (0 <= value <= n_layers):
            raise ValueError(
                f"layer {value} out of range for a {n_layers}-block model "
                f"(valid: 0 = embeddings .. {n_layers} = post-final-norm)"
            )
        if value in seen:
            raise ValueError(f"duplicate layer {value} in {layers!r}")
        seen.add(value)
    return sorted(seen)
