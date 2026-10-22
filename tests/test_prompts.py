"""Unit tests for judge_interp.prompts (pure text / schema helpers)."""
import json
from pathlib import Path

import pytest

from judge_interp import prompts
from dataset_specs import LINE, MAIN

FIXTURE = Path(__file__).parent / "fixtures" / "judge_interp_mini"


# --- render_query ---------------------------------------------------------


def _main_row():
    return json.loads((FIXTURE / "vrdu" / "main" / "train.json").read_text())[0]


def _line_row():
    return json.loads((FIXTURE / "vrdu" / "line" / "train.json").read_text())[0]


def test_render_query_main_fields_and_order():
    q = prompts.render_query(_main_row(), MAIN)
    names = [line.split(":", 1)[0] for line in q.splitlines()]
    assert names == [f for f in MAIN.fields if f not in ("agency", "product")]  # null in this row: omitted


def test_render_query_line_fields_and_order():
    q = prompts.render_query(_line_row(), LINE)
    names = [line.split(":", 1)[0] for line in q.splitlines()]
    assert names == [f for f in LINE.fields if f != "sub_amount"]  # null in this row: omitted


def test_render_query_null_and_multiline_values():
    q = prompts.render_query(_main_row(), MAIN)
    lines = dict(line.split(": ", 1) for line in q.splitlines())
    assert "agency" not in lines and "product" not in lines  # null fields have no line
    assert "null" not in q
    # tv_address contains a newline in the source; it must stay on one line.
    assert len(q.splitlines()) == len(MAIN.fields) - 2
    assert lines["tv_address"] == json.dumps("63 Chestnut St.\nBoston, MA 02101")


def test_render_query_no_label_keys_leak():
    q = prompts.render_query(_main_row(), MAIN)
    for bad in ("valid", "num_invalid_fields", "invalid_fields", "document_id", "error_type"):
        assert bad not in q


def test_render_query_missing_field_is_hard_error():
    row = _main_row()
    del row["gross_amount"]
    with pytest.raises(AssertionError):
        prompts.render_query(row, MAIN)


# --- load_context ---------------------------------------------------------


def test_load_context_reads_file():
    text = prompts.load_context(FIXTURE, MAIN, "doc-alpha")
    assert "63 Chestnut St." in text


def test_load_context_missing():
    with pytest.raises(FileNotFoundError):
        prompts.load_context(FIXTURE, MAIN, "doc-nope")


def test_load_context_empty():
    with pytest.raises(AssertionError):
        prompts.load_context(FIXTURE, MAIN, "doc-empty")


# --- load_split --------------------------------------------------------


def test_load_split_reads_rows():
    rows = prompts.load_split(FIXTURE, MAIN, "train", ["inter_document"])
    assert len(rows) == 3
    assert rows[0]["document_id"] == "doc-alpha"


def test_load_split_selects_error_types_and_always_keeps_valid_rows():
    # the fixture's main file holds 3 valid/inter rows plus one cross_type row
    def kinds(error_types):
        return sorted((r["document_id"], str(r["error_type"])) for r in prompts.load_split(FIXTURE, MAIN, "train", error_types))
    assert kinds([]) == [("doc-alpha", "None"), ("doc-beta", "None")]
    assert kinds(["inter_document"]) == [("doc-alpha", "None"), ("doc-alpha", "inter_document"), ("doc-beta", "None")]
    assert kinds(["cross_type"]) == [("doc-alpha", "None"), ("doc-alpha", "cross_type"), ("doc-beta", "None")]
    assert len(kinds(["inter_document", "cross_type"])) == 4


def test_load_split_rejects_bad_error_types():
    for bad in (["intra_document"], ["nope"], ["cross_type", "cross_type"], "cross_type", None):
        with pytest.raises(ValueError, match="error_types"):
            prompts.load_split(FIXTURE, MAIN, "train", bad)


def test_load_split_bad_split():
    with pytest.raises(ValueError):
        prompts.load_split(FIXTURE, MAIN, "dev", [])


def test_load_split_missing_file():
    with pytest.raises(FileNotFoundError):
        prompts.load_split(FIXTURE / "nope", MAIN, "train", [])


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
