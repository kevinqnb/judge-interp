"""Shared data access and prompt helpers for the judge tasks.

Provides the dataset schema (field lists imported from ``scripts/build_vrdu_dataset.py``),
``load_split`` / ``load_ocr_context`` readers for ``data/vrdu/``, ``render_query`` for the
``## QUERY:`` block, per-field descriptions (``TYPE_DESCRIPTIONS``) used in the
instructions, ``resolve_layers`` for config layer lists, and ``sub_seed`` for
deterministic per-key seeds.

No torch or nnsight is imported, so this module is testable on a login node.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# build_vrdu_dataset.py is a script, not an installed module; its own directory
# is where it expects to be imported from (build_vrdu_invalids.py does the same).
_SCRIPTS_DIR = str(REPO_ROOT / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import build_vrdu_dataset as _schema  # noqa: E402

MAIN_FIELDS: list[str] = list(_schema.MAIN_FIELDS)
LINE_FIELDS: list[str] = list(_schema.LINE_FIELDS)
LINE_CARRIED_FIELDS: list[str] = list(_schema.LINE_CARRIED_FIELDS)

# The fields shown to the judge, in a fixed order, per dataset: ``LINE_CARRIED_FIELDS``
# (document-level fields repeated on each line row) followed by the line-item fields.
QUERY_FIELDS: dict[str, list[str]] = {
    "main": MAIN_FIELDS,
    "line": LINE_CARRIED_FIELDS + LINE_FIELDS,
}

# Row keys that describe the row rather than the extraction — never shown to the
# judge (they would leak the label). ``error_type`` is null on valid rows and one
# of the dataset's error types on invalid ones; ``field_errors`` labels each field.
NON_FIELD_KEYS: frozenset[str] = frozenset(
    {"document_id", "line_index", "valid", "k", "num_invalid_fields", "invalid_fields",
     "error_type", "field_errors"}
)

# Invalid-row error types each dataset's file contains (see build_vrdu_invalids.py).
ERROR_TYPES: dict[str, tuple[str, ...]] = {
    "main": ("inter_document", "cross_type"),
    "line": ("inter_document", "intra_document", "cross_type"),
}

LAST_LAYER_TOKEN = "last"

# What each field denotes, shown to the judge in the instructions.
TYPE_DESCRIPTIONS: dict[str, str] = {
    "advertiser": "the political campaign, candidate or committee that the advertising is bought for",
    "agency": "the advertising agency that placed the buy on the advertiser's behalf",
    "property": "the TV station (typically its call letters) that sells the airtime and receives the order",
    "tv_address": "the mailing address of that TV station (street, city, state, ZIP; may span several lines)",
    "product": "the campaign, candidate, or race description that the buy is booked under",
    "contract_num": "the contract, order, or estimate identifier of the whole buy",
    "flight_from": "the start date of the whole buy's flight (its overall air period)",
    "flight_to": "the end date of the whole buy's flight (its overall air period)",
    "gross_amount": "the total gross dollar amount of the whole contract",
    "channel": "the station or channel that this line item (spot) airs on",
    "program_desc": "the program, time slot, or daypart that this line item airs in",
    "program_start_date": "the start date of this line item's air period",
    "program_end_date": "the end date of this line item's air period",
    "sub_amount": "the dollar amount (subtotal) of this line item",
}
assert set(TYPE_DESCRIPTIONS) == set(MAIN_FIELDS) | set(LINE_FIELDS), "type descriptions must cover exactly the main and line fields"


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


def load_split(data_root: str | Path, dataset: str, split: str, error_types: list[str]) -> list[dict]:
    """Read the rows of one split: every valid row plus the selected invalid rows.

    Args:
        data_root: directory containing ``vrdu/``.
        dataset: ``"main"`` or ``"line"``.
        split: ``"train"`` or ``"test"``.
        error_types: error types of invalid rows to keep; an empty list keeps valid rows only.

    Returns:
        The rows of ``<data_root>/vrdu/<dataset>/<split>.json`` that are valid or whose
        ``error_type`` is in ``error_types``.

    Raises:
        ValueError: unknown dataset, split or error type, or duplicate error types.
        FileNotFoundError: the split file does not exist.
        AssertionError: the file is empty, has a row with an unrecognised error type, or
            has no valid row after filtering.
    """
    if dataset not in QUERY_FIELDS:
        raise ValueError(f"dataset must be one of {sorted(QUERY_FIELDS)}, got {dataset!r}")
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    if not isinstance(error_types, list):
        raise ValueError(f"error_types must be a list, got {error_types!r}")
    unknown = set(error_types) - set(ERROR_TYPES[dataset])
    if unknown or len(set(error_types)) != len(error_types):
        raise ValueError(
            f"error_types {error_types!r} must be distinct members of {list(ERROR_TYPES[dataset])} for {dataset!r}"
        )
    path = Path(data_root) / "vrdu" / dataset / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no such split file: {path}")
    with open(path) as f:
        rows = json.load(f)
    assert isinstance(rows, list) and rows, f"{path}: expected a non-empty list"
    kept = [r for r in rows if r["valid"] or r["error_type"] in error_types]
    assert all(r["error_type"] in ERROR_TYPES[dataset] for r in rows if not r["valid"]), (
        f"{path}: invalid row with an unrecognised error_type"
    )
    assert any(r["valid"] for r in kept), f"{path}: no valid rows"
    return kept


def load_ocr_context(data_root: str | Path, document_id: str) -> str:
    """Read the full OCR text of one document.

    Args:
        data_root: directory containing ``vrdu/``.
        document_id: document identifier (file stem under ``vrdu/ocr/``).

    Returns:
        The OCR text.

    Raises:
        FileNotFoundError: no OCR file for the document.
        AssertionError: the text is empty.
    """
    path = Path(data_root) / "vrdu" / "ocr" / f"{document_id}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"no OCR text for document {document_id!r}: {path}")
    text = path.read_text()
    assert text.strip(), f"OCR text for document {document_id!r} is empty: {path}"
    return text


def render_query(row: dict, dataset: str) -> str:
    """Render a row as the ``## QUERY:`` block.

    Emits only the fields of ``dataset``, in canonical order, one ``field: <json>`` line
    each (``null`` for a missing value). JSON encoding keeps a value that contains a
    newline (e.g. ``tv_address``) on one line.

    Args:
        row: a dataset row; label and identity keys are never emitted.
        dataset: ``"main"`` or ``"line"``.

    Returns:
        The newline-joined query text, without a trailing newline.

    Raises:
        ValueError: unknown ``dataset``.
        AssertionError: the row lacks a field of ``dataset``.
    """
    if dataset not in QUERY_FIELDS:
        raise ValueError(f"dataset must be one of {sorted(QUERY_FIELDS)}, got {dataset!r}")
    fields = QUERY_FIELDS[dataset]

    leaked = NON_FIELD_KEYS.intersection(fields)
    assert not leaked, f"whitelisted query fields collide with non-field keys: {sorted(leaked)}"
    missing = [f for f in fields if f not in row]
    assert not missing, f"row is missing whitelisted field(s): {missing}"

    lines = [f"{f}: {json.dumps(row[f])}" for f in fields]
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
