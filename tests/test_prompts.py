"""Unit tests for judge_interp.prompts (pure text / schema helpers)."""
import json
from pathlib import Path

import pytest

from judge_interp import prompts

FIXTURE = Path(__file__).parent / "fixtures" / "judge_interp_mini"


# --- render_query ---------------------------------------------------------


def _main_row():
    return json.loads((FIXTURE / "vrdu" / "main" / "train.json").read_text())[0]


def _line_row():
    return json.loads((FIXTURE / "vrdu" / "line" / "train.json").read_text())[0]


def test_render_query_main_fields_and_order():
    q = prompts.render_query(_main_row(), "main")
    names = [line.split(":", 1)[0] for line in q.splitlines()]
    assert names == prompts.MAIN_FIELDS


def test_render_query_line_fields_and_order():
    q = prompts.render_query(_line_row(), "line")
    names = [line.split(":", 1)[0] for line in q.splitlines()]
    assert prompts.LINE_CARRIED_FIELDS == []
    assert names == prompts.LINE_FIELDS


def test_render_query_null_and_multiline_values():
    q = prompts.render_query(_main_row(), "main")
    lines = dict(line.split(": ", 1) for line in q.splitlines())
    assert lines["agency"] == "null"          # json.dumps(None)
    assert lines["product"] == "null"
    # tv_address contains a newline in the source; it must stay on one line.
    assert len(q.splitlines()) == len(prompts.MAIN_FIELDS)
    assert lines["tv_address"] == json.dumps("63 Chestnut St.\nBoston, MA 02101")


def test_render_query_no_label_keys_leak():
    q = prompts.render_query(_main_row(), "main")
    for bad in ("valid", "num_invalid_fields", "invalid_fields", "document_id", "error_type"):
        assert bad not in q


def test_render_query_unknown_dataset():
    with pytest.raises(ValueError):
        prompts.render_query(_main_row(), "sections")


def test_render_query_missing_field_is_hard_error():
    row = _main_row()
    del row["gross_amount"]
    with pytest.raises(AssertionError):
        prompts.render_query(row, "main")


# --- load_ocr_context ----------------------------------------------------


def test_load_ocr_context_reads_file():
    text = prompts.load_ocr_context(FIXTURE, "doc-alpha")
    assert "63 Chestnut St." in text


def test_load_ocr_context_missing():
    with pytest.raises(FileNotFoundError):
        prompts.load_ocr_context(FIXTURE, "doc-nope")


def test_load_ocr_context_empty():
    with pytest.raises(AssertionError):
        prompts.load_ocr_context(FIXTURE, "doc-empty")


# --- load_split --------------------------------------------------------


def test_load_split_reads_rows():
    rows = prompts.load_split(FIXTURE, "main", "train")
    assert len(rows) == 3
    assert rows[0]["document_id"] == "doc-alpha"


def test_load_split_bad_dataset_or_split():
    with pytest.raises(ValueError):
        prompts.load_split(FIXTURE, "sections", "train")
    with pytest.raises(ValueError):
        prompts.load_split(FIXTURE, "main", "dev")


def test_load_split_missing_file():
    with pytest.raises(FileNotFoundError):
        prompts.load_split(FIXTURE / "nope", "main", "train")


# --- row_key ----------------------------------------------------------


def test_row_key_shapes():
    assert prompts.row_key({"document_id": "d", "k": 3}, "main") == ("d", 3)
    row = {"document_id": "d", "line_index": 2, "error_type": "intra_document", "k": 3}
    assert prompts.row_key(row, "line") == ("d", 2, "intra_document", 3)


# --- resolve_layers -------------------------------------------------------


def test_resolve_layers_last_token():
    assert prompts.resolve_layers(["last"], 28) == [28]
    assert prompts.resolve_layers([0, 8, "last"], 28) == [0, 8, 28]


def test_resolve_layers_sorts_and_dedups_nothing_extra():
    assert prompts.resolve_layers([16, 0, 8], 28) == [0, 8, 16]


def test_resolve_layers_rejects():
    with pytest.raises(ValueError):
        prompts.resolve_layers([], 28)
    with pytest.raises(ValueError):
        prompts.resolve_layers([29], 28)          # out of range
    with pytest.raises(ValueError):
        prompts.resolve_layers([-1], 28)
    with pytest.raises(ValueError):
        prompts.resolve_layers([28, "last"], 28)  # duplicate after resolving
    with pytest.raises(ValueError):
        prompts.resolve_layers([1.5], 28)         # non-int
    with pytest.raises(ValueError):
        prompts.resolve_layers([True], 28)        # bool is not a layer
