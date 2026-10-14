"""Unit tests for judge_interp.relation_detection_row's model-free parts.

Fixture: tests/fixtures/relation_row_mini (hand-built, line only). Rows:

    doc-a  line 0: valid; intra_document k=2 (program_end_date, sub_amount <- line 1);
                   cross_type k=1 (program_end_date <- program_start_date)
           line 1: valid (program_desc null); inter_document k=1 (channel from another document)
    doc-b  line 0: valid
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from judge_interp import relation_detection_row as rr
from judge_interp.prompts import LINE_FIELDS, load_split, render_query

FIXTURE = "tests/fixtures/relation_row_mini"
REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_LINE = ["inter_document", "intra_document", "cross_type"]
ALL_STRATA = {"valid": None, "inter_document": None, "intra_document": None, "cross_type": None}


def _rows():
    return load_split(REPO_ROOT / FIXTURE, "line", "train", ALL_LINE)


def _params(**kw):
    p = dict(
        data_root=FIXTURE, dataset="line", split="train", error_types=ALL_LINE,
        rows_per_stratum=dict(ALL_STRATA), context_token_limit=None,
    )
    p.update(kw)
    return p


def _by(items, doc, error_type, line_index):
    hit = [it for it in items if it["document_id"] == doc
           and it["spans"][0]["row_error_type"] == error_type
           and it["spans"][0]["line_index"] == line_index]
    assert len(hit) == 1
    return hit[0]


def test_render_row_hand_worked():
    r = [x for x in _rows() if x["document_id"] == "doc-a" and x["valid"] and x["line_index"] == 0][0]
    out = rr.render_row(r)
    assert out["text"] == (
        'channel: "7 "\nprogram_desc: "Movie "\nprogram_start_date: "01/05/20 "\n'
        'program_end_date: "01/10/20 "\nsub_amount: "$100.00 "\n'
    )
    assert [f for f, _, _ in out["spans"]] == LINE_FIELDS
    # `channel: "7 "`: name 0-6, ': ' 7-8, open quote 9, '7' 10, pad 11, close quote 12
    assert out["spans"][0] == ("channel", 12, 13)
    for _, a, b in out["spans"]:
        assert out["text"][a:b] == '"'
        assert out["text"][a - 1] == " " and out["text"][b] == "\n"  # pad before, newline after
    assert out["spans"][-1][2] == len(out["text"]) - 1  # last line terminated like the others


def test_render_row_null_field_has_no_span_and_escaped_quote():
    r = copy.deepcopy([x for x in _rows() if x["valid"] and x["program_desc"] is None][0])
    out = rr.render_row(r)
    assert out["text"].splitlines()[1] == "program_desc: null"
    assert "program_desc" not in [f for f, _, _ in out["spans"]]
    r["channel"] = 'Say "hi"'
    out = rr.render_row(r)
    f, a, b = out["spans"][0]
    assert out["text"].splitlines()[0] == 'channel: "Say \\"hi\\" "'
    assert out["text"][a - 3:b] == '\\" "'  # escaped quote, pad space, then the closing quote


def test_render_row_differs_from_render_query_only_by_padding():
    for r in _rows():
        text = rr.render_row(r)["text"]
        assert text.endswith("\n") and not text.endswith("\n\n")
        plain = [ln[:-2] + '"' if ln.endswith(' "') else ln for ln in text[:-1].split("\n")]
        assert "\n".join(plain) == render_query(r, "line")
        assert text.count("\n") == len(LINE_FIELDS)  # only line-item fields, no main fields
        assert "advertiser" not in text and "gross_amount" not in text


def test_labels_and_items():
    items = rr.build_items(_params(), 0, REPO_ROOT)
    assert len(items) == 6
    assert [(it["document_id"], it["row_ordinal"]) for it in items] == [
        ("doc-a", 0), ("doc-a", 1), ("doc-a", 2), ("doc-a", 3), ("doc-a", 4), ("doc-b", 0),
    ]
    assert all(it["instructions"] == items[0]["instructions"] for it in items)

    intra = _by(items, "doc-a", "intra_document", 0)
    got = {s["field"]: (s["label_valid"], s["field_error_type"], s["source_field"], s["source_line_index"])
           for s in intra["spans"]}
    assert got["program_end_date"] == (False, "intra_document", "program_end_date", 1)
    assert got["sub_amount"] == (False, "intra_document", "sub_amount", 1)
    assert got["channel"] == (True, "", "", -1)
    assert [s["row_k"] for s in intra["spans"]] == [2] * 5  # a mixed row: 3 correct, 2 wrong
    assert sum(not s["label_valid"] for s in intra["spans"]) == 2

    cross = _by(items, "doc-a", "cross_type", 0)
    bad = [s for s in cross["spans"] if not s["label_valid"]]
    assert [(s["field"], s["source_field"], s["source_line_index"], s["exchanged_with"]) for s in bad] == [
        ("program_end_date", "program_start_date", 0, "")
    ]

    inter = _by(items, "doc-a", "inter_document", 1)
    assert len(inter["spans"]) == 4  # program_desc null
    bad = [s for s in inter["spans"] if not s["label_valid"]]
    assert [(s["field"], s["value"], s["source_scope"], s["source_line_index"]) for s in bad] == [
        ("channel", "9", "other_document", -1)
    ]

    valid = _by(items, "doc-a", "", 0)
    assert all(s["label_valid"] and s["row_k"] == 0 and s["dataset"] == "line" for s in valid["spans"])


def test_only_line_dataset_is_supported():
    with pytest.raises(ValueError, match="'line' only"):
        rr.build_items(_params(dataset="main"), 0, REPO_ROOT)


def test_check_row_rejects_inconsistent_labels():
    good = [r for r in _rows() if r["error_type"] == "intra_document"][0]
    rr.check_row(good)
    for mutate in (
        lambda r: r["field_errors"].update(sub_amount=None),  # drops a label, k disagrees
        lambda r: r.update(error_type=None),
        lambda r: r.update(k=1),
        lambda r: r["field_errors"]["sub_amount"].update(error_type="inter_document"),
        lambda r: r.update(sub_amount=None),  # corrupted field with null value
        lambda r: r["field_errors"]["sub_amount"]["source"].pop("line_index"),
        lambda r: r["field_errors"]["sub_amount"]["source"].update(field="not_a_field"),
    ):
        bad = copy.deepcopy(good)
        mutate(bad)
        with pytest.raises(AssertionError):
            rr.check_row(bad)


def test_stratified_subset_determinism_and_errors():
    rows = _rows()
    p = {"valid": 2, "inter_document": 1, "intra_document": 1, "cross_type": 1}
    a = rr.select_rows(rows, "train", p, 0)
    assert a == rr.select_rows(rows, "train", p, 0)
    assert len(a) == 5 and sum(not r["valid"] for r in a) == 3
    with pytest.raises(ValueError, match="requested but only"):
        rr.select_rows(rows, "train", {**p, "valid": 4}, 0)
    with pytest.raises(ValueError, match="strata present"):
        rr.select_rows(rows, "train", {"valid": 1}, 0)
    with pytest.raises(ValueError, match="int >= 1"):
        rr.select_rows(rows, "train", {**p, "valid": 0}, 0)


def test_instructions_are_relational_and_name_only_line_fields():
    text = rr.build_instructions()
    for f in LINE_FIELDS:
        assert f"- {f}:" in text
    for f in ("advertiser", "agency", "gross_amount", "flight_from"):
        assert f"- {f}:" not in text
    assert "same line item" in text
    assert "padding space" in text and "closing quote" in text and "newline" in text
    assert "{" not in text and "}" not in text  # no format placeholder left unfilled


def test_context_token_limit_needs_tokenizer():
    with pytest.raises(ValueError, match="tokenizer"):
        rr.build_items(_params(context_token_limit=10), 0, REPO_ROOT)


def _fake_result():
    """Two prompts: valid (3 spans) and intra_document k=2 (3 spans)."""
    return {
        "doc_ids": np.array(["d"] * 6),
        "row_ordinal": np.array([0, 0, 0, 1, 1, 1]),
        "row_k": np.array([0, 0, 0, 2, 2, 2]),
        "row_error_type": np.array([""] * 3 + ["intra_document"] * 3),
        "field_error_type": np.array(["", "", "", "intra_document", "intra_document", ""]),
        "label_valid": np.array([True, True, True, False, False, True]),
        "field_position": np.array([1, 2, 3, 1, 2, 3]),
        "exact_end_alignment": np.array([True, True, False, True, True, True]),
        "token_id": np.array([1, 1, 2, 1, 1, 2]),
        "prompt_n_tokens": np.array([10] * 3 + [12] * 3),
    }


def test_invariants_accept_good_and_reject_bad():
    r = _fake_result()
    rr.check_row_invariants(r)
    m = rr.compute_metrics(r)
    assert (m["n_spans"], m["n_prompts"], m["n_invalid_spans"], m["n_spans_intra_document"]) == (6, 2, 2, 2)
    assert m["n_rows_with_mixed_entities"] == 1
    assert m["n_distinct_quote_token_ids"] == 2
    for key, val in (
        ("row_k", np.array([0, 0, 0, 1, 1, 1])),
        ("field_error_type", np.array(["", "", "", "intra_document", "", ""])),
        ("label_valid", np.array([True, True, True, False, True, True])),
        ("field_position", np.array([1, 2, 3, 1, 3, 2])),
    ):
        bad = {**r, key: val}
        with pytest.raises(AssertionError):
            rr.check_row_invariants(bad)
