"""Unit tests for judge_interp.relation_detection's model-free parts.

Fixture: tests/fixtures/relation_detection_mini (line only, written by tests/fixtures/make_line_fixtures.py).
Four minimal pairs (an invalid row and its clean twin, the item's full set of non-null fields in standard order):

    intra_document:k2  doc-a li0, all five fields; end date <- li1, sub_amount <- li2
    intra_document:k1  doc-a li1, four fields (program_desc is null); channel <- li0
    cross_item:k2      doc-a li2, all five fields; channel <-> sub_amount
    cross_item:k2      doc-b li0, all five fields; program_desc <-> channel
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from judge_interp import relation_detection as rr
from judge_interp.prompts import load_split, render_query
from dataset_specs import LINE, MAIN

LINE_FIELDS = list(LINE.fields)

FIXTURE = "tests/fixtures/relation_detection_mini"
REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_LINE = ["intra_document", "cross_item"]
CELLS = ["cross_item:k2", "intra_document:k1", "intra_document:k2"]
ALL_STRATA = {c: None for c in CELLS}


def _rows():
    return load_split(REPO_ROOT / FIXTURE, LINE, "train", ALL_LINE)


def _row(cell, valid, doc="doc-a"):
    (r,) = [x for x in _rows() if x["cell"] == cell and x["valid"] == valid and x["document_id"] == doc]
    return r


def _params(**kw):
    p = dict(
        data_root=FIXTURE, split="train", error_types=ALL_LINE,
        rows_per_stratum=dict(ALL_STRATA), context_token_limit=None,
    )
    p.update(kw)
    return p


def _by(items, cell, valid, doc="doc-a"):
    hit = [it for it in items if it["spans"][0]["cell"] == cell and (it["spans"][0]["row_error_type"] == "") == valid
           and it["document_id"] == doc]
    assert len(hit) == 1
    return hit[0]


def test_render_row_hand_worked():
    out = rr.render_row(_row("intra_document:k2", True), LINE)
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


def test_render_row_shows_the_fields_in_the_row_order():
    good, bad = _row("cross_item:k2", True), _row("cross_item:k2", False)
    assert rr.render_row(good, LINE)["text"] == (
        'channel: "8 "\nprogram_desc: "News "\nprogram_start_date: "01/21/20 "\nprogram_end_date: "01/31/20 "\n'
        'sub_amount: "$400.00 "\n')
    assert rr.render_row(bad, LINE)["text"] == (
        'channel: "$400.00 "\nprogram_desc: "News "\nprogram_start_date: "01/21/20 "\nprogram_end_date: "01/31/20 "\n'
        'sub_amount: "8 "\n')
    shuffled = copy.deepcopy(bad)   # the renderer follows the row's own `fields` order
    shuffled["fields"] = ["sub_amount", "channel", "program_end_date", "program_desc", "program_start_date"]
    assert [f for f, _, _ in rr.render_row(shuffled, LINE)["spans"]] == shuffled["fields"]
    wrong = copy.deepcopy(good)
    wrong["fields"] = ["channel", "sub_amount"]   # not exactly the non-null fields
    with pytest.raises(AssertionError, match="non-null fields"):
        rr.render_row(wrong, LINE)


def test_render_row_hidden_fields_have_no_span_and_escaped_quote():
    r = copy.deepcopy(_row("intra_document:k1", True))   # li1: program_desc is null, so it is not shown
    out = rr.render_row(r, LINE)
    assert out["text"] == ('channel: "8 "\nprogram_start_date: "01/11/20 "\nprogram_end_date: "01/20/20 "\n'
                           'sub_amount: "$250.00 "\n')
    assert [f for f, _, _ in out["spans"]] == ["channel", "program_start_date", "program_end_date", "sub_amount"]
    r["channel"] = 'Say "hi"'
    out = rr.render_row(r, LINE)
    f, a, b = out["spans"][0]
    assert out["text"].splitlines()[0] == 'channel: "Say \\"hi\\" "'
    assert out["text"][a - 3:b] == '\\" "'  # escaped quote, pad space, then the closing quote


def test_render_row_differs_from_render_query_only_by_padding():
    for r in _rows():
        text = rr.render_row(r, LINE)["text"]
        assert text.endswith("\n") and not text.endswith("\n\n")
        plain = [ln[:-2] + '"' if ln.endswith(' "') else ln for ln in text[:-1].split("\n")]
        assert "\n".join(plain) == render_query(r, LINE)
        assert text.count("\n") == len(r["fields"]) == r["m"]  # only the shown fields
        assert "advertiser" not in text and "gross_amount" not in text


def test_labels_and_items():
    items = rr.build_items(LINE, _params(), 0, REPO_ROOT)
    assert len(items) == 8
    # sorted by (document, line_index, cell, k): each pair's twin (k=0) right before its invalid row
    assert [(it["document_id"], it["row_ordinal"], it["spans"][0]["cell"], it["spans"][0]["row_k"]) for it in items] == [
        ("doc-a", 0, "intra_document:k2", 0), ("doc-a", 1, "intra_document:k2", 2),
        ("doc-a", 2, "intra_document:k1", 0), ("doc-a", 3, "intra_document:k1", 1),
        ("doc-a", 4, "cross_item:k2", 0), ("doc-a", 5, "cross_item:k2", 2),
        ("doc-b", 0, "cross_item:k2", 0), ("doc-b", 1, "cross_item:k2", 2),
    ]
    assert all(it["instructions"] == items[0]["instructions"] for it in items)
    for it in items:
        assert len({s["pair_id"] for s in it["spans"]}) == 1 and it["spans"][0]["pair_id"].endswith("+".join(
            s["field"] for s in it["spans"]))

    intra = _by(items, "intra_document:k2", False)
    got = {s["field"]: (s["label_valid"], s["field_error_type"], s["source_field"], s["source_line_index"])
           for s in intra["spans"]}
    assert got["program_end_date"] == (False, "intra_document", "program_end_date", 1)
    assert got["sub_amount"] == (False, "intra_document", "sub_amount", 2)
    assert got["channel"] == (True, "", "", -1)
    assert [s["row_k"] for s in intra["spans"]] == [2] * 5  # a mixed row: 3 correct, 2 wrong

    cross = _by(items, "cross_item:k2", False)
    assert [(s["field"], s["field_position"], s["label_valid"]) for s in cross["spans"]] == [
        ("channel", 1, False), ("program_desc", 2, True), ("program_start_date", 3, True),
        ("program_end_date", 4, True), ("sub_amount", 5, False)]
    bad = [s for s in cross["spans"] if not s["label_valid"]]
    assert [(s["field"], s["value"], s["source_field"], s["source_line_index"], s["exchanged_with"]) for s in bad] == [
        ("channel", "$400.00", "sub_amount", 2, "sub_amount"), ("sub_amount", "8", "channel", 2, "channel")]

    twin = _by(items, "intra_document:k2", True)
    assert all(s["label_valid"] and s["row_k"] == 0 and s["dataset"] == "line" for s in twin["spans"])
    assert twin["spans"][0]["pair_id"] == intra["spans"][0]["pair_id"]


def test_dataset_without_relation_support_is_rejected():
    with pytest.raises(ValueError, match="does not support the relation tasks"):
        rr.build_items(MAIN, _params(), 0, REPO_ROOT)
    with pytest.raises(ValueError, match="does not support the relation tasks"):
        rr.build_instructions(MAIN)


def test_check_row_rejects_inconsistent_labels():
    good = _row("intra_document:k2", False)
    rr.check_row(good, LINE)
    rr.check_row(_row("cross_item:k2", False), LINE)
    for mutate in (
        lambda r: r["field_errors"].update(sub_amount=None),  # drops a label, k disagrees
        lambda r: r.update(error_type=None),
        lambda r: r.update(k=1),
        lambda r: r["field_errors"]["sub_amount"].update(error_type="cross_item"),
        lambda r: r.update(sub_amount=None),  # corrupted field with null value
        lambda r: r["field_errors"]["sub_amount"]["source"].pop("line_index"),
        lambda r: r["field_errors"]["sub_amount"]["source"].update(field=None),  # vocabulary is not checked here
        lambda r: r.update(m=4),
        lambda r: r.update(fields=r["fields"][:-1]),
        lambda r: r.update(cell=None),
    ):
        bad = copy.deepcopy(good)
        mutate(bad)
        with pytest.raises(AssertionError):
            rr.check_row(bad, LINE)
    swap = copy.deepcopy(_row("cross_item:k2", False))
    swap["field_errors"]["channel"]["source"].pop("exchanged_with")   # a swap must name its partner
    with pytest.raises(AssertionError):
        rr.check_row(swap, LINE)


def test_pairs_are_drawn_and_filtered_whole():
    rows = _rows()
    p = {c: 1 for c in CELLS}
    a = rr.select_rows(rows, LINE, "train", p, 0)
    assert a == rr.select_rows(rows, LINE, "train", p, 0)
    assert len(a) == 6 and sum(not r["valid"] for r in a) == 3
    assert {r["pair_id"] for r in a if r["valid"]} == {r["pair_id"] for r in a if not r["valid"]}
    with pytest.raises(ValueError, match="requested but only"):
        rr.select_rows(rows, LINE, "train", {**p, "cross_item:k2": 3}, 0)
    with pytest.raises(ValueError, match="strata present"):
        rr.select_rows(rows, LINE, "train", {"valid": 1}, 0)
    with pytest.raises(ValueError, match="int >= 1"):
        rr.select_rows(rows, LINE, "train", {**p, "cross_item:k2": 0}, 0)
    intra = load_split(REPO_ROOT / FIXTURE, LINE, "train", ["intra_document"])   # cross_item pairs dropped, twins too
    assert len(intra) == 4 and {r["cell"] for r in intra} == {"intra_document:k1", "intra_document:k2"}
    with pytest.raises(AssertionError, match="pair .* is not one invalid"):
        rr.select_rows(rows[:-1], LINE, "train", p, 0)




def test_instructions_are_relational_and_name_only_line_fields():
    text = rr.build_instructions(LINE)
    for f in LINE_FIELDS:
        assert f"- {f}:" in text
    for f in ("advertiser", "agency", "gross_amount", "flight_from"):
        assert f"- {f}:" not in text
    assert "same single item" in text
    assert "The field names are part of the relation" in text and "listed under the wrong field" in text
    assert "related: true" in text and "related: false" in text
    assert "<field>" not in text  # one verdict for the record, not one per field
    assert "padding space" in text and "closing quote" in text and "newline" in text
    assert "{" not in text and "}" not in text  # no format placeholder left unfilled


def test_context_token_limit_needs_tokenizer():
    with pytest.raises(ValueError, match="tokenizer"):
        rr.build_items(LINE, _params(context_token_limit=10), 0, REPO_ROOT)


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
        "token_id": np.array([1, 1, 1, 1, 1, 1]),
        "prompt_n_tokens": np.array([10] * 3 + [12] * 3),
    }


def test_invariants_accept_good_and_reject_bad():
    r = _fake_result()
    rr.check_row_invariants(r)
    m = rr.compute_metrics(r, LINE)
    assert (m["n_spans"], m["n_prompts"], m["n_invalid_spans"], m["n_spans_intra_document"]) == (6, 2, 2, 2)
    assert m["n_rows_with_mixed_entities"] == 1
    assert m["read_token_id"] == 1
    with pytest.raises(AssertionError, match="different tokens"):  # one entity read at another token
        rr.compute_metrics({**r, "token_id": np.array([1, 1, 2, 1, 1, 1])}, LINE)
    for key, val in (
        ("row_k", np.array([0, 0, 0, 1, 1, 1])),
        ("field_error_type", np.array(["", "", "", "intra_document", "", ""])),
        ("label_valid", np.array([True, True, True, False, True, True])),
        ("field_position", np.array([1, 2, 3, 1, 3, 2])),
    ):
        bad = {**r, key: val}
        with pytest.raises(AssertionError):
            rr.check_row_invariants(bad)
