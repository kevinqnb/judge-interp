"""Unit tests for judge_interp.entity_detection's model-free parts.

Fixture: tests/fixtures/entity_detection_mini (3 documents), hand-worked-out:

    doc-a  advertiser{Acme} property{WAAA} contract_num{111}
           channel{7}  (same value on both line items -> counted once)
           program_desc{Movie, News}                              m = 6
    doc-b  advertiser{Bravo} property{WBBB} contract_num{222}
           channel{9} program_desc{Weather}                       m = 5
    doc-c  advertiser{Acme} property{WCCC} contract_num{333}
           channel{7} program_desc{Sports}                        m = 5

The fixture also holds invalid rows (property WZZZ, channel 99) that must never
appear as ground truth. doc-a's OCR contains the word "Sports".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from judge_interp import entity_detection as ed

FIXTURE = "tests/fixtures/entity_detection_mini"
REPO_ROOT = Path(__file__).resolve().parents[1]
M_BY_DOC = {"doc-a": 6, "doc-b": 5, "doc-c": 5}


@pytest.fixture(scope="module")
def truth():
    return ed.load_ground_truth(FIXTURE, "train")


@pytest.fixture(scope="module")
def pools(truth):
    return ed.build_decoy_pools(truth)


def test_entity_types_are_main_plus_line():
    assert len(ed.TYPE_NAMES) == 14
    assert ed.TYPE_NAMES[0] == "advertiser" and ed.TYPE_NAMES[-1] == "sub_amount"


def test_ground_truth_hand_counts(truth):
    assert set(truth) == set(M_BY_DOC)
    for doc, m in M_BY_DOC.items():
        assert sum(len(v) for v in truth[doc].values()) == m
        assert set(truth[doc]) == set(ed.TYPE_NAMES)
    assert truth["doc-a"]["channel"] == ["7"]
    assert truth["doc-a"]["program_desc"] == ["Movie", "News"]
    assert truth["doc-a"]["property"] == ["WAAA"]  # not the invalid row's WZZZ
    assert truth["doc-a"]["agency"] == []


def test_decoy_pools(pools):
    assert pools["channel"] == {"7": ["doc-a", "doc-c"], "9": ["doc-b"]}
    assert "99" not in pools["channel"] and "WZZZ" not in pools["property"]


def _draw(truth, pools, doc, j, seed=0, k_min=0):
    return ed.sample_prompt_entries(doc, truth[doc], pools, j, seed, k_min)


def test_sample_invariants_over_many_draws(truth, pools):
    ks = set()
    for doc, m in M_BY_DOC.items():
        for j in range(60):
            d = _draw(truth, pools, doc, j)
            assert d["m"] == m and len(d["entries"]) == m and 0 <= d["k"] <= m
            ks.add(d["k"])
            assert sum(not e["label_valid"] for e in d["entries"]) == d["k"]
            for t in ed.TYPE_NAMES:
                own = truth[doc][t]
                of_type = [e for e in d["entries"] if e["entity_type"] == t]
                assert len(of_type) == len(own)  # decoys replace same-type slots
                kept = {e["value"] for e in of_type if e["label_valid"]}
                assert kept <= set(own)
                decoys = [e for e in of_type if not e["label_valid"]]
                assert len({e["value"] for e in decoys}) == len(decoys)
                for e in decoys:
                    assert e["value"] not in own
                    assert doc not in pools[t][e["value"]]
                    assert e["decoy_source_doc"] in pools[t][e["value"]]
    assert 0 in ks and 6 in ks  # both endpoints of [0, m] reachable


def test_k_zero_reproduces_ground_truth(truth, pools):
    hits = [d for j in range(200) if (d := _draw(truth, pools, "doc-b", j))["k"] == 0]
    assert hits
    got = sorted((e["entity_type"], e["value"]) for e in hits[0]["entries"])
    want = sorted((t, v) for t in ed.TYPE_NAMES for v in truth["doc-b"][t])
    assert got == want and all(e["label_valid"] for e in hits[0]["entries"])


def test_k_equals_m_leaves_no_own_value(truth, pools):
    for doc, m in M_BY_DOC.items():
        d = _draw(truth, pools, doc, 0, k_min=m)
        assert d["k"] == m
        assert not any(e["label_valid"] for e in d["entries"])
        for e in d["entries"]:
            assert e["value"] not in truth[doc][e["entity_type"]]


def test_seed_determinism_and_sensitivity(truth, pools):
    a = [_draw(truth, pools, "doc-a", j, seed=3) for j in range(10)]
    assert a == [_draw(truth, pools, "doc-a", j, seed=3) for j in range(10)]
    assert a != [_draw(truth, pools, "doc-a", j, seed=4) for j in range(10)]


def test_shuffled_not_in_canonical_order(truth, pools):
    orders = {tuple(e["entity_type"] for e in _draw(truth, pools, "doc-a", j)["entries"]) for j in range(20)}
    assert len(orders) > 1


def test_k_min_above_m_raises(truth, pools):
    with pytest.raises(ValueError, match="k_min"):
        _draw(truth, pools, "doc-a", 0, k_min=7)


def test_empty_decoy_pool_raises():
    solo = {"d": {t: [] for t in ed.TYPE_NAMES}}
    solo["d"]["channel"] = ["7"]
    with pytest.raises(ValueError, match="eligible"):
        ed.sample_prompt_entries("d", solo["d"], ed.build_decoy_pools(solo), 0, 0, k_min=1)


def test_render_entities_hand_worked():
    entries = [
        {"entity_type": "advertiser", "value": "Acme"},
        {"entity_type": "tv_address", "value": "63 Chestnut\nBoston"},
    ]
    r = ed.render_entities(entries)
    assert r["text"] == '1. [advertiser] "Acme"\n2. [tv_address] "63 Chestnut\\nBoston"'
    assert r["text"][slice(*r["spans"][0])] == "Acme"
    assert r["text"][slice(*r["spans"][1])] == "63 Chestnut\\nBoston"
    with pytest.raises(ValueError):
        ed.render_entities([])


_PARAMS = dict(
    data_root=FIXTURE, split="train", samples_per_doc=4, k_min=0,
    context_token_limit=None, doc_subset=None,
)


def test_build_items_shape_and_span_text():
    items = ed.build_items(_PARAMS, 0, REPO_ROOT)
    assert [(it["document_id"], it["sample_index"]) for it in items] == [
        (d, j) for d in sorted(M_BY_DOC) for j in range(4)
    ]
    for it in items:
        m = M_BY_DOC[it["document_id"]]
        assert len(it["spans"]) == m
        k = it["spans"][0]["k"]
        assert all(sp["m"] == m and sp["k"] == k for sp in it["spans"])
        assert sum(not sp["label_valid"] for sp in it["spans"]) == k
        for sp in it["spans"]:
            assert set(sp) == {"char_start", "char_end", *ed.SPAN_META_KEYS}
            assert it["query"][sp["char_start"]:sp["char_end"]] == json.dumps(sp["value"])[1:-1]
            assert it["query"].splitlines()[sp["list_position"] - 1].startswith(
                f"{sp['list_position']}. [{sp['entity_type']}] "
            )


def test_decoy_in_ocr_flag_matches_substring():
    items = ed.build_items({**_PARAMS, "samples_per_doc": 30}, 0, REPO_ROOT)
    seen = set()
    for it in items:
        for sp in it["spans"]:
            if sp["label_valid"]:
                assert sp["decoy_in_ocr"] is False
            else:
                assert sp["decoy_in_ocr"] == (sp["value"] in it["context"])
                seen.add((it["document_id"], sp["decoy_in_ocr"]))
    assert ("doc-a", True) in seen  # "Sports" drawn as a doc-a decoy at least once


def test_build_items_doc_subset_and_limit_checks():
    sub = ed.build_items({**_PARAMS, "doc_subset": {"n": 2, "seed": 0}}, 0, REPO_ROOT)
    assert len({it["document_id"] for it in sub}) == 2
    with pytest.raises(ValueError, match="doc_subset"):
        ed.build_items({**_PARAMS, "doc_subset": {"n": 9, "seed": 0}}, 0, REPO_ROOT)
    with pytest.raises(ValueError, match="tokenizer"):
        ed.build_items({**_PARAMS, "context_token_limit": 10}, 0, REPO_ROOT)


def test_instructions_name_every_type():
    text = ed.build_instructions()
    for t in ed.TYPE_NAMES:
        assert f"- {t}: {ed.TYPE_DESCRIPTIONS[t]}" in text
    assert "(1) Explicit support" in text and "(2) Type correspondence" in text
    assert "BOTH criteria" in text
    assert "{" not in text and "}" not in text  # no unfilled format fields


def _fake_result(valid, m, k, positions=None):
    n = len(valid)
    return {
        "doc_ids": np.asarray(["d"] * n),
        "sample_index": np.zeros(n, dtype=np.int64),
        "m": np.full(n, m, dtype=np.int64),
        "k": np.full(n, k, dtype=np.int64),
        "label_valid": np.asarray(valid, dtype=bool),
        "list_position": np.asarray(positions or range(1, n + 1), dtype=np.int64),
    }


def test_prompt_invariants_accept_good_and_reject_bad():
    ed.check_prompt_invariants(_fake_result([True, False, False], m=3, k=2))
    with pytest.raises(AssertionError, match="invalid count"):
        ed.check_prompt_invariants(_fake_result([True, False, False], m=3, k=1))
    with pytest.raises(AssertionError, match="row count"):
        ed.check_prompt_invariants(_fake_result([True, False, False], m=4, k=2))
    with pytest.raises(AssertionError, match="positions"):
        ed.check_prompt_invariants(_fake_result([True, False, False], m=3, k=2, positions=[1, 1, 3]))


def test_compute_metrics_hand_countable():
    r = _fake_result([True, False, False, True], m=4, k=2)
    r.update(
        exact_end_alignment=np.asarray([True, False, True, True]),
        decoy_in_ocr=np.asarray([False, True, False, False]),
        prompt_n_tokens=np.full(4, 100, dtype=np.int64),
    )
    out = ed.compute_metrics(r)
    assert out["n_spans"] == 4 and out["n_prompts"] == 1
    assert out["n_valid_spans"] == 2 and out["n_invalid_spans"] == 2
    assert out["mean_m"] == 4.0 and out["mean_k"] == 2.0 and out["mean_k_over_m"] == 0.5
    assert out["exact_end_alignment_rate"] == 0.75 and out["decoy_in_ocr_rate"] == 0.5
