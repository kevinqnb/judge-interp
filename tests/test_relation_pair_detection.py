"""Unit tests for judge_interp.relation_pair_detection's model-free parts.

Fixture: tests/fixtures/relation_pair_detection_mini (written by tests/fixtures/make_line_fixtures.py), doc-a only:
    li0 = {7, Movie, 01/05/20, 01/10/20, $100.00}; li1 = {8, null, 01/11/20, 01/20/20, $250.00};
    li2 = {8, News, 01/21/20, 01/31/20, $400.00}
    intra_document:k1  li2, five fields; program_start_date <- li1. Pairs: 10, invalid 3 (start with desc / end / amount);
                       (channel 8, start 01/11/20) is li1's pair: valid despite its corrupted member
    cross_item:k2      li0, five fields; channel <-> sub_amount. Pairs: 10, invalid 7 (every pair with a swapped field)
    intra_document:k1  li1, four fields; channel <- li0. Pairs: 6, invalid 3 (every pair with channel)
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from judge_interp import relation_pair_detection as rp
from judge_interp.prompts import load_split
from judge_interp.row_detection import READ_TOKEN
from dataset_specs import LINE, MAIN

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = "tests/fixtures/relation_pair_detection_mini"
CELLS = ["cross_item:k2", "intra_document:k1"]


def _params(**kw):
    p = dict(data_root=FIXTURE, split="train", error_types=list(LINE.error_types), rows_per_stratum=dict.fromkeys(CELLS),
             context_token_limit=None, subsets_path="vrdu/line/{split}_subsets.json")
    return p | kw


def _items(seed=0, **kw):
    return rp.build_items(LINE, _params(**kw), seed, REPO_ROOT)


def _row(cell, valid, line_index):
    (r,) = [x for x in load_split(REPO_ROOT / FIXTURE, LINE, "train", list(LINE.error_types))
            if x["cell"] == cell and x["valid"] == valid and x["line_index"] == line_index]
    return r


def _spans(items):
    return [sp for it in items for sp in it["spans"]]


def _result(items):
    """The aggregated columns ``collect_rows`` would build from these items (model columns faked)."""
    sp = _spans(items)
    out = {k: np.asarray([s[k] for s in sp]) for k in rp.PAIR_META_KEYS}
    out["doc_ids"] = np.asarray([it["document_id"] for it in items for _ in it["spans"]])
    out["token_id"] = np.full(len(sp), 7)
    out["prompt_n_tokens"] = np.full(len(sp), 100)
    return out


# ---------------------------------------------------------------- order and rendering
def test_pair_order_covers_every_pair_once_and_is_seeded_by_pair_id():
    fields = list(LINE.fields)
    order = rp.pair_order(fields, "pid-1", 0)
    assert len(order) == math.comb(5, 2)
    assert sorted(tuple(sorted(p)) for p in order) == sorted(tuple(sorted(p)) for p in __import__("itertools").combinations(fields, 2))
    assert order == rp.pair_order(fields, "pid-1", 0)            # deterministic
    orders = {tuple(rp.pair_order(fields, f"pid-{i}", 0)) for i in range(20)}
    assert len(orders) > 1                                         # depends on the pair id
    assert rp.pair_order(fields, "pid-1", 1) != order              # and on the seed
    flipped = [(a, b) for a, b in order if fields.index(a) > fields.index(b)]
    assert 0 < len(flipped) < len(order)                           # orientation is randomised (for this draw)


def test_render_pairs_hand_worked():
    row = _row("cross_item:k2", False, 0)   # channel "$100.00", sub_amount "7" (swapped)
    out = rp.render_pairs(row, [("sub_amount", "channel"), ("program_desc", "program_start_date")])
    assert out["text"] == ('sub_amount: "7", channel: "$100.00 "\n'
                           'program_desc: "Movie", program_start_date: "01/05/20 "\n')
    # line 1: 'sub_amount: ' 0-11, '"7"' 12-14, ', ' 15-16, 'channel: ' 17-25, '"$100.00 "' 26-35
    assert out["spans"][0] == (35, 36)
    assert out["spans"][1][1] == len(out["text"]) - 1
    for a, b in out["spans"]:
        assert out["text"][a - 1:b + 1] == READ_TOKEN


# ---------------------------------------------------------------- items and labels
def test_build_items_counts_and_twins_share_the_order():
    items = _items()
    assert len(items) == 6
    assert sorted(len(it["spans"]) for it in items) == [6, 6, 10, 10, 10, 10]
    by_sample = {it["spans"][0]["sample_id"]: it for it in items}
    for sid, it in by_sample.items():
        if sid.endswith("|invalid"):
            twin = by_sample[sid[: -len("invalid")] + "valid"]
            assert [(s["field_a"], s["field_b"]) for s in it["spans"]] == [(s["field_a"], s["field_b"]) for s in twin["spans"]]
    for it in items:
        assert it["query"].count("\n") == len(it["spans"]) == math.comb(it["spans"][0]["parent_m"], 2)
        assert [s["pair_position"] for s in it["spans"]] == list(range(1, len(it["spans"]) + 1))


def test_labels_are_the_subsets_files_including_a_valid_pair_with_a_corrupted_member():
    sp = _spans(_items())
    bad = [s for s in sp if not s["parent_valid"]]
    assert (sum(not s["label_valid"] for s in bad), sum(s["label_valid"] for s in bad)) == (13, 13)
    assert all(s["label_valid"] and s["n_corrupted"] == 0 for s in sp if s["parent_valid"])
    (hit,) = [s for s in sp if s["label_valid"] and s["n_corrupted"] > 0]
    assert {hit["field_a"], hit["field_b"]} == {"channel", "program_start_date"} and hit["line_index"] == 2
    assert {hit["value_a"], hit["value_b"]} == {"8", "01/11/20"}
    assert sum(s["n_corrupted"] == 2 for s in sp) == 1   # the swapped pair
    subsets = {r["subset_id"]: r for r in json.loads((REPO_ROOT / FIXTURE / "vrdu/line/train_subsets.json").read_text())}
    assert all(subsets[s["subset_id"]]["valid"] == s["label_valid"] for s in sp)


def test_the_label_follows_the_subsets_file_not_field_errors(tmp_path):
    root = tmp_path / "mini"
    shutil.copytree(REPO_ROOT / FIXTURE, root)
    path = root / "vrdu/line/train_subsets.json"
    rows = json.loads(path.read_text())
    target = next(r for r in rows if r["size"] == 2 and r["valid"] and r["invalid_fields"])
    target["valid"] = False
    path.write_text(json.dumps(rows))
    sp = _spans(rp.build_items(LINE, _params(data_root=str(root)), 0, REPO_ROOT))
    assert not next(s for s in sp if s["subset_id"] == target["subset_id"])["label_valid"]


def test_a_missing_pair_row_is_an_error(tmp_path):
    root = tmp_path / "mini"
    shutil.copytree(REPO_ROOT / FIXTURE, root)
    path = root / "vrdu/line/train_subsets.json"
    rows = json.loads(path.read_text())
    path.write_text(json.dumps([r for r in rows if r is not next(x for x in rows if x["size"] == 2)]))
    with pytest.raises(AssertionError, match="size-2 subsets"):
        rp.build_items(LINE, _params(data_root=str(root)), 0, REPO_ROOT)


def test_main_dataset_is_refused():
    with pytest.raises(ValueError, match="relation"):
        rp.build_instructions(MAIN)


def test_build_items_is_deterministic_and_the_seed_moves_the_order():
    assert _items(0) == _items(0)
    assert [it["query"] for it in _items(0)] != [it["query"] for it in _items(1)]


# ---------------------------------------------------------------- invariants and metrics
def test_invariants_hold_and_catch_broken_bookkeeping():
    items = _items()
    res = _result(items)
    rp.check_pair_invariants(res)
    for key, mutate, msg in [
        ("pair_position", lambda a: a.__setitem__(1, 5), "pair_position"),
        ("label_valid", lambda a: a.__setitem__(int(np.flatnonzero(res["parent_valid"])[0]), False), "clean parent"),
        ("n_corrupted", lambda a: a.__setitem__(0, a[0] + 1), "n_corrupted"),
    ]:
        broken = {k: v.copy() for k, v in res.items()}
        mutate(broken[key])
        with pytest.raises(AssertionError, match=msg):
            rp.check_pair_invariants(broken)


def test_compute_metrics_counts():
    m = rp.compute_metrics(_result(_items()), LINE)
    assert m["n_spans"] == 52 and m["n_prompts"] == 6
    assert m["n_spans_corrupted_parent"] == 26 and m["n_valid_spans_corrupted_parent"] == 13
    assert m["n_valid_spans_with_corrupted_member"] == 1 and m["n_spans_ncorr2"] == 1
    assert m["n_spans_intra_document"] == 16 and m["n_spans_cross_item"] == 10


# ---------------------------------------------------------------- tokenizers
END_OF_TURN = {"Qwen/Qwen2.5-0.5B-Instruct": "<|im_end|>", "meta-llama/Llama-3.1-8B-Instruct": "<|eot_id|>"}


@pytest.mark.parametrize("model", list(END_OF_TURN))
def test_every_pair_is_read_at_the_same_token(model):
    from transformers import AutoTokenizer

    lm = rp.RelationPairDetectionLM.__new__(rp.RelationPairDetectionLM)
    lm.tokenizer, lm.model_name = AutoTokenizer.from_pretrained(model), model
    for it in _items():
        ids, offsets, shift = lm._build_prompt_with_offsets(it["instructions"], it["context"], it["query"])
        reads = [lm._last_token_covering(offsets, shift + sp["char_start"]) for sp in it["spans"]]
        assert all(lm.tokenizer.decode([ids[i]]) == READ_TOKEN for i in reads)
        assert len(set(reads)) == len(reads)
        assert lm.tokenizer.decode([ids[reads[-1] + 1]]) == END_OF_TURN[model]


@pytest.mark.parametrize("model", list(END_OF_TURN))
def test_values_variant_every_pair_is_read_at_the_values_token(model):
    from transformers import AutoTokenizer

    lm = rp.RelationPairValuesLM.__new__(rp.RelationPairValuesLM)
    lm.tokenizer, lm.model_name = AutoTokenizer.from_pretrained(model), model
    assert lm.read_token == rp.VALUES_READ_TOKEN == ' ")\n'
    for it in rp.build_values_items(LINE, _params(), 0, REPO_ROOT):
        ids, offsets, shift = lm._build_prompt_with_offsets(it["instructions"], it["context"], it["query"])
        reads = [lm._last_token_covering(offsets, shift + sp["char_start"]) for sp in it["spans"]]
        assert all(lm.tokenizer.decode([ids[i]]) == lm.read_token for i in reads)
        assert len(set(reads)) == len(reads)
        assert lm.tokenizer.decode([ids[reads[-1] + 1]]) == END_OF_TURN[model]


# ---------------------------------------------------------------- relation_pair_values (values only)
def test_render_value_pairs_hand_worked():
    row = _row("cross_item:k2", False, 0)
    out = rp.render_value_pairs(row, [("sub_amount", "channel"), ("program_desc", "program_start_date")])
    assert out["text"] == ('("7", "$100.00 ")\n'
                           '("Movie", "01/05/20 ")\n')
    # line 1: '(' 0, '"7"' 1-3, ', ' 4-5, '"$100.00 "' 6-15, ')' 16
    assert out["spans"][0] == (15, 16)
    assert out["spans"][1][1] == len(out["text"]) - 2
    for a, b in out["spans"]:
        assert out["text"][a - 1:b + 2] == rp.VALUES_READ_TOKEN


def test_values_instructions_name_no_field_and_no_type_criterion():
    text = rp.build_values_instructions(LINE)
    assert not any(f in text for f in LINE.fields)
    assert "RELATED" in text and "Type correspondence" not in text and "Explicit support" not in text
    assert "Type correspondence" in rp.build_instructions(LINE)   # the full prompt still has it
    with pytest.raises(ValueError, match="relation"):
        rp.build_values_instructions(MAIN)


def test_values_items_equal_the_full_items_but_for_the_prompt():
    full, vals = _items(), rp.build_values_items(LINE, _params(), 0, REPO_ROOT)
    assert len(full) == len(vals)
    for a, b in zip(full, vals):
        assert (a["document_id"], a["row_ordinal"], a["context"]) == (b["document_id"], b["row_ordinal"], b["context"])
        assert a["instructions"] != b["instructions"] and a["query"] != b["query"]
        strip = lambda s: {k: v for k, v in s.items() if k not in ("char_start", "char_end")}
        assert [strip(s) for s in a["spans"]] == [strip(s) for s in b["spans"]]
        lines = b["query"].splitlines()
        assert all(ln == f'({json.dumps(s["value_a"])}, {json.dumps(s["value_b"])[:-1]} ")' for ln, s in zip(lines, b["spans"]))


def test_run_dispatches_on_the_task(monkeypatch):
    seen = {}
    monkeypatch.setattr(rp, "run_rows", lambda cfg, spec, run_dir, repo_root, build_fn, lm_class: seen.update(b=build_fn, c=lm_class))
    rp.run({"task": "relation_pair_values"}, LINE, None, None, None)
    assert (seen["b"], seen["c"]) == (rp.build_values_items, rp.RelationPairValuesLM)
    rp.run({"task": "relation_pair_detection"}, LINE, None, None, None)
    assert (seen["b"], seen["c"]) == (rp.build_items, rp.RelationPairDetectionLM)
    assert rp.RelationPairDetectionLM.read_token == READ_TOKEN


def test_fixture_config_loads_and_requires_subsets_path(tmp_path):
    import sys

    import yaml

    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    import run_experiment as run

    path = REPO_ROOT / "tests/fixtures/relation_pair_detection_tiny_e2e.yaml"
    cfg = run.load_config(path)
    assert cfg["task"] == "relation_pair_detection"
    p = tmp_path / path.name
    p.write_text(yaml.safe_dump({**cfg, "params": {k: v for k, v in cfg["params"].items() if k != "subsets_path"}}))
    with pytest.raises(KeyError, match="subsets_path"):
        run.load_config(p)
