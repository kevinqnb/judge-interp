"""Unit tests for judge_interp.relation_grouping (model-free): instruction text, document filter, items equal relation_detection's."""
from __future__ import annotations

from pathlib import Path

import pytest

from judge_interp import relation_detection as rd
from judge_interp import relation_grouping as rg
from dataset_specs import LINE

FIXTURE = "tests/fixtures/relation_detection_mini"
REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_LINE = ["intra_document", "cross_item"]
CELLS = ["cross_item:k2", "intra_document:k1", "intra_document:k2"]


def _params(**kw):
    p = dict(data_root=FIXTURE, split="train", error_types=ALL_LINE, rows_per_stratum={c: None for c in CELLS},
             context_token_limit=None, document_ids=["doc-a"])
    p.update(kw)
    return p


def test_instructions_share_the_header_and_replace_the_verdict():
    g, d = rg.build_instructions(LINE), rd.build_instructions(LINE)
    shared = d[: d.index("The document may describe many items")]
    assert g.startswith(shared) and "- channel:" in g
    assert "related: true" not in g and "Do not judge entities one by one" not in g
    assert "first group where it is related to all of the entities" in g
    assert "{" not in g and "}" not in g   # no unformatted placeholder


def test_items_are_the_detection_items_of_the_requested_documents_with_new_instructions():
    got = rg.build_items(LINE, _params(), 0, REPO_ROOT)
    ref = [it for it in rd.build_items(LINE, _params(), 0, REPO_ROOT) if it["document_id"] == "doc-a"]
    assert len(got) == len(ref) > 0
    assert {it["document_id"] for it in got} == {"doc-a"}
    for a, b in zip(got, ref):
        assert a["instructions"] == rg.build_instructions(LINE) != b["instructions"]
        assert {k: v for k, v in a.items() if k != "instructions"} == {k: v for k, v in b.items() if k != "instructions"}


def test_document_ids_are_validated():
    with pytest.raises(ValueError, match="no row for document"):
        rg.build_items(LINE, _params(document_ids=["doc-a", "nope"]), 0, REPO_ROOT)
    with pytest.raises(ValueError, match="non-empty"):
        rg.build_items(LINE, _params(document_ids=[]), 0, REPO_ROOT)
    with pytest.raises(ValueError, match="duplicates"):
        rg.build_items(LINE, _params(document_ids=["doc-a", "doc-a"]), 0, REPO_ROOT)


def test_task_params_add_document_ids():
    assert rg.TASK_PARAMS["relation_grouping"] == (*rd.TASK_PARAMS["relation_detection"], "document_ids")
