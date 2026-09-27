"""Turn a dataset row into the judge's ``## QUERY:`` block, and helpers around it.

Pure text / list work only — no torch, no nnsight — so the unit tests here run
on a login node without a GPU. Anything that touches the model lives in
``representationlm.py``.

The canonical field lists are imported from ``scripts/build_vrdu_dataset.py``,
the same single source of truth ``scripts/build_vrdu_invalids.py`` uses, so the
query can never drift from the schema the data was built against.
"""
from __future__ import annotations

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

# The fields shown to the judge, in a fixed order, per dataset. Line rows carry
# the six document-level fields plus the five line-item fields.
QUERY_FIELDS: dict[str, list[str]] = {
    "main": MAIN_FIELDS,
    "line": LINE_CARRIED_FIELDS + LINE_FIELDS,
}

# Row keys that describe the row rather than the extraction — never shown to the
# judge (they would leak the label). ``error_type`` is line-only (null on valid
# rows; "inter_document" / "intra_document" on invalid ones).
NON_FIELD_KEYS: frozenset[str] = frozenset(
    {"document_id", "line_index", "valid", "k", "num_invalid_fields", "invalid_fields", "error_type"}
)

LAST_LAYER_TOKEN = "last"


def load_split(data_root: str | Path, dataset: str, split: str) -> list[dict]:
    """Read ``<data_root>/vrdu/<dataset>/<split>.json`` (valid + invalid rows)."""
    if dataset not in QUERY_FIELDS:
        raise ValueError(f"dataset must be one of {sorted(QUERY_FIELDS)}, got {dataset!r}")
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    path = Path(data_root) / "vrdu" / dataset / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no such split file: {path}")
    with open(path) as f:
        rows = json.load(f)
    assert isinstance(rows, list) and rows, f"{path}: expected a non-empty list"
    return rows


def load_ocr_context(data_root: str | Path, document_id: str) -> str:
    """Read the full OCR text for one document. Raises if missing or empty."""
    path = Path(data_root) / "vrdu" / "ocr" / f"{document_id}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"no OCR text for document {document_id!r}: {path}")
    text = path.read_text()
    assert text.strip(), f"OCR text for document {document_id!r} is empty: {path}"
    return text


def row_key(row: dict, dataset: str) -> tuple:
    """Identity of a row within a split: (document_id[, line_index[, error_type]], k).

    Line rows are keyed on ``error_type`` too: since 2026-09-15 the same
    ``(document_id, line_index, k)`` can carry both an inter-document and an
    intra-document invalid variant, so ``k`` alone no longer disambiguates.
    """
    if dataset == "main":
        return (row["document_id"], row["k"])
    return (row["document_id"], row["line_index"], row["error_type"], row["k"])


def render_query(row: dict, dataset: str) -> str:
    """Render a row as the judge's ``## QUERY:`` block.

    Emits exactly the whitelisted fields for ``dataset``, in canonical order, one
    ``field: <json>`` line each (``null`` for a missing value). Values are
    JSON-encoded so a value containing a newline (e.g. ``tv_address``) stays on
    one line and is unambiguous.

    Raises:
        ValueError: unknown ``dataset``.
        AssertionError: the row is missing a whitelisted field, or a whitelisted
            field name collides with a non-field (label/identity) key.
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
    """Validate a config layer list against model depth; return it sorted ascending.

    Layer convention (matches ``../../coastal/scholarlm``'s representation LM):
    ``0`` is the token-embedding output, ``1..n_layers-1`` is the residual stream
    after that many transformer blocks (pre-norm), and ``n_layers`` is the
    post-final-norm state — the vector the unembedding sees. The string
    ``"last"`` resolves to ``n_layers``.

    Norm-space caveat: layer ``n_layers`` is post-final-norm; every other layer
    is a raw pre-norm residual. Row L2 norms are not comparable across that
    boundary.

    Raises:
        ValueError: empty list, an unrecognised entry, an out-of-range entry, or
            a duplicate (after resolving ``"last"``).
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
