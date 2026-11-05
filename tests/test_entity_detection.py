"""Unit tests for judge_interp.entity_detection's model-free parts.

Fixture: tests/fixtures/entity_detection_mini (hand-built). Rows:

    main  doc-a  M1 valid (agency null)
                 M2 cross_type k=2: advertiser<-product, product<-advertiser (exchanged)
                 M3 inter_document k=1: property = WZZZ
          doc-b  valid
    line  doc-a  L0 valid; L1 valid (program_desc null);
                 L0 cross_type k=1: program_end_date <- program_start_date
          doc-b  valid
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from judge_interp import entity_detection as er
from judge_interp.prompts import load_split, render_query
from dataset_specs import LINE, MAIN, SPECS

FIXTURE = "tests/fixtures/entity_detection_mini"
REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_MAIN = ["inter_document", "cross_type"]
ALL_LINE = ["intra_document", "cross_item"]
# The line fixture (tests/fixtures/make_line_fixtures.py) has one minimal pair per cell.
LINE_CELLS = ["cross_item:k2", "intra_document:k1", "intra_document:k2"]


def _rows(dataset):
    return load_split(REPO_ROOT / FIXTURE, SPECS[dataset], "train", ALL_MAIN if dataset == "main" else ALL_LINE)


def _params(dataset, **kw):
    p = dict(
        data_root=FIXTURE, split="train",
        error_types=ALL_MAIN,
        rows_per_stratum={"valid": None, "inter_document": None, "cross_type": None},
        context_token_limit=None,
    )
    if dataset == "main":
        p["read_fields"] = "all"
    if dataset == "line":
        p["error_types"], p["rows_per_stratum"] = ALL_LINE, dict.fromkeys(LINE_CELLS)
    p.update(kw)
    return p


def test_render_row_hand_worked():
    r = _rows("main")[0]  # M1
    out = er.render_row(r, MAIN)
    assert out["text"] == (
        'advertiser: "Acme "\nproperty: "WAAA "\ntv_address: "1 Main St\\nBoston "\n'
        'product: "Race 1 "\ncontract_num: "111 "\nflight_from: "01/01/20 "\nflight_to: "01/31/20 "\n'
        'gross_amount: "$500.00 "\n'
    )
    assert [f for f, _, _ in out["spans"]] == [
        "advertiser", "property", "tv_address", "product", "contract_num",
        "flight_from", "flight_to", "gross_amount",
    ]  # agency is null: no line, no span
    for _, a, b in out["spans"]:
        assert out["text"][a:b] == '"'
        assert out["text"][a - 1] == " " and out["text"][b] == "\n"  # pad before, newline after
    # `advertiser: "Acme "`: name 0-9, ': ' 10-11, open quote 12, Acme 13-16, pad 17, close quote 18
    assert out["spans"][0] == ("advertiser", 18, 19)
    # the last line is terminated like the others
    assert out["spans"][-1][2] == len(out["text"]) - 1


def test_render_row_escaped_inner_quote():
    r = copy.deepcopy(_rows("line")[0])
    r["program_desc"] = 'Say "hi"'
    out = er.render_row(r, LINE)
    f, a, b = out["spans"][1]
    assert f == "program_desc"
    assert out["text"].splitlines()[1] == 'program_desc: "Say \\"hi\\" "'
    assert out["text"][a - 3:b] == '\\" "'  # escaped quote, pad space, then the closing quote


def test_render_row_differs_from_render_query_only_by_padding():
    for ds in ("main", "line"):
        for r in _rows(ds):
            text = er.render_row(r, SPECS[ds])["text"]
            assert text.endswith("\n") and not text.endswith("\n\n")
            plain = [ln[:-2] + '"' if ln.endswith(' "') else ln for ln in text[:-1].split("\n")]
            assert "\n".join(plain) == render_query(r, SPECS[ds])


def test_line_prompt_has_no_main_fields():
    # a line row shows only its own `fields` (2, 3 or 5 here), in that order
    for r in _rows("line"):
        text = render_query(r, LINE)
        assert [ln.split(":", 1)[0] for ln in text.split("\n")] == r["fields"]
        assert "advertiser" not in text and "gross_amount" not in text and "null" not in text


def test_labels_from_field_errors():
    items = er.build_items(MAIN, _params("main"), 0, REPO_ROOT)
    assert [(it["document_id"], it["row_ordinal"]) for it in items] == [
        ("doc-a", 0), ("doc-a", 1), ("doc-a", 2), ("doc-a", 3), ("doc-a", 4), ("doc-b", 0),
    ]
    # doc-a rows sorted by (stratum, k, patch target): cross_type k=1 patched (advertiser, product), cross_type k=2,
    # inter_document, valid
    by = {(it["document_id"], it["spans"][0]["row_error_type"], it["spans"][0]["patch_target"]): it for it in items}
    cross = by[("doc-a", "cross_type", "")]
    got = {s["field"]: (s["label_valid"], s["field_error_type"], s["source_field"], s["exchanged_with"])
           for s in cross["spans"]}
    assert got["advertiser"] == (False, "cross_type", "product", "product")
    assert got["product"] == (False, "cross_type", "advertiser", "advertiser")
    assert got["property"] == (True, "", "", "")
    assert [s["row_k"] for s in cross["spans"]] == [2] * 8
    assert sum(not s["label_valid"] for s in cross["spans"]) == 2

    inter = by[("doc-a", "inter_document", "")]
    bad = [s for s in inter["spans"] if not s["label_valid"]]
    assert [(s["field"], s["value"], s["source_scope"], s["source_line_index"]) for s in bad] == [
        ("property", "WZZZ", "other_document", -1)
    ]
    valid = by[("doc-a", "", "")]
    assert all(s["label_valid"] and s["row_error_type"] == "" for s in valid["spans"])
    assert len(valid["spans"]) == 8  # agency is null
    assert all(s["patch_target"] == "" and s["parent_k"] == s["row_k"] for it in items for s in it["spans"] if it is not by[("doc-a", "cross_type", "advertiser")] and it is not by[("doc-a", "cross_type", "product")])

    # patched row: the cross_type k=2 row with only advertiser restored; every field read
    pa = by[("doc-a", "cross_type", "advertiser")]
    assert [s["row_k"] for s in pa["spans"]] == [1] * 8 and [s["parent_k"] for s in pa["spans"]] == [2] * 8
    got = {s["field"]: s["label_valid"] for s in pa["spans"]}
    assert got["advertiser"] and not got["product"] and sum(not v for v in got.values()) == 1
    assert next(s for s in pa["spans"] if s["field"] == "advertiser")["value"] != next(s for s in cross["spans"] if s["field"] == "advertiser")["value"]
    assert "advertiser" in pa["query"] and pa["query"] != cross["query"]


def test_target_only_reads_just_the_target_of_a_patched_row():
    items = er.build_items(MAIN, _params("main", read_fields="target_only"), 0, REPO_ROOT)
    patched = [it for it in items if it["spans"][0]["patch_target"]]
    assert len(patched) == 2
    for it in patched:
        (sp,) = it["spans"]
        assert sp["field"] == sp["patch_target"] and sp["label_valid"] and sp["row_k"] == 1 and sp["parent_k"] == 2
        assert sp["field_position"] > 1 or sp["field"] == "advertiser"   # the position in the full row is kept
    assert all(len(it["spans"]) in (8, 9) for it in items if not it["spans"][0]["patch_target"])


def test_read_fields_is_required_iff_the_file_has_patched_rows():
    p = _params("main")
    del p["read_fields"]
    with pytest.raises(AssertionError, match="read_fields"):
        er.build_items(MAIN, p, 0, REPO_ROOT)
    with pytest.raises(AssertionError, match="read_fields"):
        er.build_items(LINE, _params("line", read_fields="all"), 0, REPO_ROOT)
    with pytest.raises(AssertionError, match="read_fields"):
        er.build_items(MAIN, _params("main", read_fields="some"), 0, REPO_ROOT)


def test_line_items_and_null_field():
    items = er.build_items(LINE, _params("line"), 0, REPO_ROOT)
    assert len(items) == 8
    n_spans = sorted(len(it["spans"]) for it in items)
    assert n_spans == [4, 4, 5, 5, 5, 5, 5, 5]  # each pair: twin and invalid row show the same fields
    cross = [it for it in items if it["spans"][0]["row_error_type"] == "cross_item" and it["spans"][0]["cell"] == "cross_item:k2"
             and it["document_id"] == "doc-a"][0]
    bad = [s for s in cross["spans"] if not s["label_valid"]]
    assert [(s["field"], s["source_field"], s["source_line_index"], s["exchanged_with"], s["line_index"])
            for s in bad] == [("channel", "sub_amount", 2, "sub_amount", 2), ("sub_amount", "channel", 2, "channel", 2)]


def test_check_row_rejects_inconsistent_labels():
    good = _rows("main")[1]  # M2
    er.check_row(good, MAIN)
    for mutate in (
        lambda r: r["field_errors"].update(property=None, advertiser=None),  # drops labels, k disagrees
        lambda r: r.update(error_type=None),
        lambda r: r.update(k=1),
        lambda r: r["field_errors"]["advertiser"].update(error_type="inter_document"),
        lambda r: r.update(advertiser=None),  # corrupted field with null value
        lambda r: r["field_errors"]["advertiser"]["source"].pop("exchanged_with"),
    ):
        bad = copy.deepcopy(good)
        mutate(bad)
        with pytest.raises(AssertionError):
            er.check_row(bad, MAIN)


def test_stratified_subset_determinism_and_errors():
    rows = _rows("main")
    p = {"valid": 2, "inter_document": 1, "cross_type": 1}
    a = er.select_rows(rows, MAIN, "train", p, 0)
    assert a == er.select_rows(rows, MAIN, "train", p, 0)
    assert len(a) == 6 and sum(not r["valid"] for r in a) == 4   # the cross_type k=2 row brings its 2 patched rows
    assert sum(r["patch_target"] is not None for r in a) == 2 and {r["parent_k"] for r in a if r["patch_target"]} == {2}
    only_inter = er.select_rows(rows, MAIN, "train", {"valid": 2, "inter_document": 1, "cross_type": 1}, 0)
    assert only_inter == a
    one = er.select_rows(rows, MAIN, "train", {"valid": 1, "inter_document": 1, "cross_type": 1}, 0)
    assert sum(r["patch_target"] is not None for r in one) == 2
    with pytest.raises(ValueError, match="requested but only"):
        er.select_rows(rows, MAIN, "train", {**p, "valid": 3}, 0)
    with pytest.raises(ValueError, match="strata present"):
        er.select_rows(rows, MAIN, "train", {"valid": 1}, 0)
    with pytest.raises(ValueError, match="int >= 1"):
        er.select_rows(rows, MAIN, "train", {**p, "valid": 0}, 0)


def test_instructions_name_every_field_of_the_dataset_only():
    main, line = er.build_instructions(MAIN), er.build_instructions(LINE)
    for f in ("advertiser", "gross_amount"):
        assert f"- {f}:" in main and f"- {f}:" not in line
    for f in ("channel", "sub_amount"):
        assert f"- {f}:" in line and f"- {f}:" not in main
    assert "Type correspondence" in main
    assert "{n}" not in main and "{hint}" not in line


def test_context_token_limit_needs_tokenizer():
    with pytest.raises(ValueError, match="tokenizer"):
        er.build_items(MAIN, _params("main", context_token_limit=10), 0, REPO_ROOT)


def _fake_result():
    """Two prompts: valid (3 spans) and cross_type k=2 (3 spans)."""
    return {
        "doc_ids": np.array(["d"] * 6),
        "row_ordinal": np.array([0, 0, 0, 1, 1, 1]),
        "row_k": np.array([0, 0, 0, 2, 2, 2]),
        "parent_k": np.array([0, 0, 0, 2, 2, 2]),
        "patch_target": np.array([""] * 6),
        "field": np.array(["advertiser", "property", "product"] * 2),
        "row_error_type": np.array(["", "", "", "cross_type"] * 1 + ["cross_type"] * 2),
        "field_error_type": np.array(["", "", "", "cross_type", "cross_type", ""]),
        "label_valid": np.array([True, True, True, False, False, True]),
        "field_position": np.array([1, 2, 3, 1, 2, 3]),
        "token_id": np.array([1, 1, 1, 1, 1, 1]),
        "prompt_n_tokens": np.array([10] * 3 + [12] * 3),
    }


def test_invariants_accept_good_and_reject_bad():
    r = _fake_result()
    er.check_row_invariants(r)
    m = er.compute_metrics(r, MAIN)
    assert (m["n_spans"], m["n_prompts"], m["n_invalid_spans"], m["n_spans_cross_type"]) == (6, 2, 2, 2)
    assert m["n_rows_with_mixed_entities"] == 1
    assert m["read_token_id"] == 1
    with pytest.raises(AssertionError, match="different tokens"):  # one entity read at another token
        er.compute_metrics({**r, "token_id": np.array([1, 1, 2, 1, 1, 1])}, MAIN)
    for key, val in (
        ("row_k", np.array([0, 0, 0, 1, 1, 1])),
        ("field_error_type", np.array(["", "", "", "cross_type", "", ""])),
        ("label_valid", np.array([True, True, True, False, True, True])),
        ("field_position", np.array([1, 2, 3, 1, 3, 2])),
    ):
        bad = {**r, key: val}
        with pytest.raises(AssertionError):
            er.check_row_invariants(bad)


def _patched_result(target_only):
    """A valid row (3 spans), a cross_type k=3 row (3 spans, 3 wrong... shown as 2 wrong + 1 right), and the patched row of its
    first wrong field `advertiser` (row_k 1, parent_k 2)."""
    r = _fake_result()   # prompt 1: cross_type k=2, advertiser + property wrong, product right
    if target_only:      # prompt 2: patched row read at its target (advertiser, position 1) alone
        add = dict(row_k=[1], parent_k=[2], patch_target=["advertiser"], row_error_type=["cross_type"], field_error_type=[""],
                   label_valid=[True], field_position=[1], field=["advertiser"], token_id=[1], prompt_n_tokens=[12], row_ordinal=[2])
    else:                # prompt 2: patched row, all spans: property still wrong
        add = dict(row_k=[1] * 3, parent_k=[2] * 3, patch_target=["advertiser"] * 3, row_error_type=["cross_type"] * 3,
                   field_error_type=["", "cross_type", ""], label_valid=[True, False, True], field_position=[1, 2, 3],
                   field=["advertiser", "property", "product"], token_id=[1] * 3, prompt_n_tokens=[12] * 3, row_ordinal=[2] * 3)
    n = len(add["row_k"])
    add["doc_ids"] = ["d"] * n
    return {k: np.concatenate([v, np.array(add[k])]) for k, v in r.items()}


def test_invariants_for_patched_rows():
    for target_only in (False, True):
        r = _patched_result(target_only)
        er.check_row_invariants(r)
        bad_cases = {
            "parent_k": np.where(r["patch_target"] != "", 5, r["parent_k"]),   # parent_k != row_k + 1
            "label_valid": np.where((r["patch_target"] != "") & (r["field"] == "advertiser"), False, r["label_valid"]),   # target not valid
        }
        for key, val in bad_cases.items():
            with pytest.raises(AssertionError):
                er.check_row_invariants({**r, key: val})
    # all-spans patched row: its invalid count must equal row_k
    r = _patched_result(False)
    with pytest.raises(AssertionError, match="invalid-span count"):
        er.check_row_invariants({**r, "label_valid": np.where(r["patch_target"] != "", True, r["label_valid"]),
                                 "field_error_type": np.where(r["patch_target"] != "", "", r["field_error_type"])})
