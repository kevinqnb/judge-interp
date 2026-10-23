"""Unit + tiny end-to-end tests for data/vrdu/build_vrdu_subsets.py and data/vrdu/subsets.py.

Fixture (``make_rows``): 4 line items per document with all values distinct within the document and
sub_amount null. Per item: a valid row, inter_document k=1 and k=2, and intra_document k=1, with the
corrupted fields rotating by line index. No invalid subset of it matches a real item; the match rule
is tested on dedicated rows.
"""

import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "vrdu"))
import build_vrdu_subsets as b
import subsets as s

FIELDS4 = ["channel", "program_desc", "program_start_date", "program_end_date"]
SPLIT_DOCS = {"train": ["A", "B", "C"], "test": ["D", "E", "F"]}
KINDS = {"name": ["advertiser", "agency", "property", "tv_address", "product"],
         "date": ["flight_from", "flight_to", "program_start_date", "program_end_date"],
         "amount": ["gross_amount", "sub_amount"]}
PIVOT = 50
EQ = s.make_sample_eq({"kinds": KINDS, "year_pivot": PIVOT})
N_LINES = 4


def item(doc, li):
    return {"channel": f"{doc}c{li}", "program_desc": f"{doc}d{li}", "program_start_date": f"0{li + 1}/01/20",
            "program_end_date": f"0{li + 1}/09/20", "sub_amount": None}


def row(doc, li, et=None, k=0, bad=(), values=None):
    return {"document_id": doc, "line_index": li, "valid": et is None, "error_type": et, "k": k,
            "invalid_fields": list(bad), **(values or item(doc, li))}


def make_rows(docs, n_lines=N_LINES):
    rows = []
    for doc in docs:
        for li in range(n_lines):
            order = FIELDS4[li % 4:] + FIELDS4[:li % 4]
            rows.append(row(doc, li))
            for k in (1, 2):
                rows.append(row(doc, li, "inter_document", k, order[:k], {**item(doc, li), **{f: f"X{f}" for f in order[:k]}}))
            donor = item(doc, (li + 1) % n_lines)
            rows.append(row(doc, li, "intra_document", 1, order[:1], {**item(doc, li), order[0]: donor[order[0]]}))
    return rows


def write_data(root, rows_by_split=None, kinds=KINDS, year_pivot=PIVOT):
    (root / "line").mkdir(exist_ok=True)
    for split, docs in SPLIT_DOCS.items():
        rows = rows_by_split[split] if rows_by_split else make_rows(docs)
        (root / "line" / f"{split}.json").write_text(json.dumps(rows))
    snapshot = {"params": {"cross_type": {"kinds": kinds}, "dates": {"year_pivot": year_pivot}}}
    (root / "invalids.json").write_text(json.dumps({"build": {"config_snapshot": snapshot}}))
    return root


def write_config(root, **override):
    p = {"seed": 0, "source": "all_rows", "n_replicates": 3, "samples_per_row": 11, "min_subset_size": 2,
         "n_per_error_type": {"train": 4, "test": 3}, "balance_by": "field_set_error_type",
         "max_error_type_share_deviation": 0.2, **override}
    path = root / "cfg.yaml"
    path.write_text(yaml.safe_dump({"id": "t", "params": {"data_root": str(root), "subsets": p, "cross_type": {"kinds": KINDS},
                                                         "dates": {"year_pivot": PIVOT}}}))
    return path


def fs(*names):
    return frozenset(names)


# ---------------------------------------------------------------- ids, values, labels
def test_sample_id_round_trips_and_keeps_the_field_order():
    r = make_rows(["A"])[1]   # inter_document k=1
    sid = s.make_sample_id(r, ["program_end_date", "channel"])
    assert sid == "A|0|inter_document|1|program_end_date+channel"
    assert s.parse_sample_id(sid) == {"document_id": "A", "line_index": 0, "error_type": "inter_document", "k": 1,
                                      "fields": ["program_end_date", "channel"], "swap": None}
    assert s.parse_sample_id(s.make_sample_id(make_rows(["A"])[0], ["channel", "program_desc"]))["error_type"] is None


def test_cross_item_sample_id_round_trips_and_names_the_swap():
    valid = make_rows(["A"])[0]
    sid = s.make_sample_id(valid, ["program_end_date", "channel", "program_desc"], ("channel", "program_end_date"))
    assert sid == "A|0|cross_item|2|program_end_date+channel+program_desc|channel~program_end_date"
    assert s.parse_sample_id(sid) == {"document_id": "A", "line_index": 0, "error_type": "cross_item", "k": 2,
                                      "fields": ["program_end_date", "channel", "program_desc"],
                                      "swap": ("channel", "program_end_date")}
    with pytest.raises(AssertionError):   # the pair must be in LINE_FIELDS order
        s.make_sample_id(valid, ["channel", "program_end_date"], ("program_end_date", "channel"))
    with pytest.raises(AssertionError):   # a cross_item comes from a valid row only
        s.make_sample_id(make_rows(["A"])[1], ["channel", "program_end_date"], ("channel", "program_end_date"))
    with pytest.raises(AssertionError):   # a 6-part id must be cross_item
        s.parse_sample_id("A|0|valid|0|channel+program_desc|channel~program_desc")


def test_values_invalid_fields_and_label():
    r = make_rows(["A"])[2]   # inter_document k=2 on line 0: channel, program_desc
    assert s.sample_invalid_fields(r, ["program_end_date", "program_desc", "channel"]) == ["program_desc", "channel"]
    assert not s.sample_label_valid(r, ["channel", "program_end_date"])
    assert s.sample_label_valid(r, ["program_end_date", "program_start_date"])
    v = make_rows(["A"])[0]
    swapped = s.sample_values(v, ["program_desc", "channel", "program_start_date"], ("channel", "program_desc"))
    assert swapped == {"program_desc": "Ac0", "channel": "Ad0", "program_start_date": "01/01/20"}
    assert s.sample_invalid_fields(v, ["program_desc", "channel", "program_start_date"], ("channel", "program_desc")) == [
        "program_desc", "channel"]


def test_swap_is_noop_when_either_value_would_not_change():
    r = {**make_rows(["A"])[0], "program_end_date": "1/1/20"}   # start 01/01/20 == end up to reformat
    assert s.swap_is_noop(r, "program_start_date", "program_end_date", EQ)
    assert not s.swap_is_noop(r, "channel", "program_end_date", EQ)


# ---------------------------------------------------------------- draw, match rule, cross_item
def test_draw_gives_distinct_unordered_subsets_of_nonnull_fields_in_random_order():
    rows = make_rows(["A"])
    samples, stats = b.draw_samples(rows, 11, 2, np.random.default_rng(0))   # all 11 subsets of every row
    assert len(samples) == 11 * len(rows) and stats["n_rows_fewer_subsets_than_requested"] == 0
    for r in rows:
        sets = [frozenset(x["fields"]) for x in samples if x["row"] is r]
        assert len(set(sets)) == len(sets) == 11 and all(len(f) >= 2 and "sub_amount" not in f for f in sets)
    assert any(x["fields"] != sorted(x["fields"], key=FIELDS4.index) for x in samples), "order never differs from canonical"
    assert {x["label_valid"] for x in samples} == {True, False}
    assert all(x["values"] == {f: x["row"][f] for f in x["fields"]} for x in samples)


def test_draw_takes_all_subsets_when_fewer_exist_and_skips_too_small_rows():
    small = {**make_rows(["A"])[0], "program_start_date": None, "program_end_date": None}   # channel, program_desc: 1 subset
    tiny = {**make_rows(["B"])[0], "program_desc": None, "program_start_date": None, "program_end_date": None}   # 1 field
    samples, stats = b.draw_samples([small, tiny], 4, 2, np.random.default_rng(0))
    assert len(samples) == 1 and set(samples[0]["fields"]) == {"channel", "program_desc"}
    assert stats["n_rows_too_small"] == 1 and stats["n_rows_fewer_subsets_than_requested"] == 1


def test_drop_matching_removes_an_invalid_subset_that_is_another_items_and_keeps_the_rest():
    # Items 0 and 1 of document S share both dates and differ in channel / desc. An intra row of item 0 takes
    # item 1's channel: {channel, start, end} of it is exactly item 1's -> valid, dropped. {channel, desc}
    # still shows item 0's desc next to item 1's channel -> stays invalid.
    i0 = {**item("S", 0), "program_start_date": "05/01/20", "program_end_date": "05/09/20"}
    i1 = {**item("S", 1), "program_start_date": "5/1/20", "program_end_date": "05/09/20"}   # a reformat still matches
    rows = [row("S", 0, values=i0), row("S", 1, values=i1),
            row("S", 0, "intra_document", 1, ["channel"], {**i0, "channel": i1["channel"]})]
    items = s.items_by_document(rows)
    bad = rows[2]
    same = b.make_sample(bad, ["program_end_date", "channel", "program_start_date"])
    diff = b.make_sample(bad, ["channel", "program_desc"])
    clean = b.make_sample(rows[0], ["channel", "program_desc"])
    kept, dropped = b.drop_matching([same, diff, clean], items, EQ)
    assert kept == [diff, clean] and dropped == Counter({"intra_document": 1})


def test_cross_item_candidates_swap_a_shown_pair_of_clean_rows_only_and_skip_noop_pairs():
    v = {**make_rows(["A"])[0], "program_end_date": "01/01/20"}   # start == end: that pair is a no-op swap
    rows = [v, make_rows(["A"])[1]]
    items = s.items_by_document(rows)
    drawn = [b.make_sample(v, ["program_end_date", "program_start_date"]),             # only pair is a no-op
             b.make_sample(v, ["channel", "program_start_date", "program_end_date"]),   # 2 of 3 pairs eligible
             b.make_sample(rows[1], ["program_desc", "program_end_date"])]               # corrupted row: never a source
    cands, n_noop = b.cross_item_candidates(drawn, items, EQ, np.random.default_rng(0))
    assert n_noop == 2 and len(cands) == 1
    c = cands[0]
    assert c["error_type"] == "cross_item" and c["swap"] in {("channel", "program_start_date"), ("channel", "program_end_date")}
    assert c["fields"] == ["channel", "program_start_date", "program_end_date"] and not c["label_valid"]
    a, z = c["swap"]
    assert c["values"][a] == v[z] and c["values"][z] == v[a] and c["invalid_fields"] == [f for f in c["fields"] if f in c["swap"]]
    picks = Counter(b.cross_item_candidates(drawn, items, EQ, np.random.default_rng(seed))[0][0]["swap"] for seed in range(200))
    assert set(picks) == {("channel", "program_start_date"), ("channel", "program_end_date")}   # both eligible pairs drawn


# ---------------------------------------------------------------- allocation and balance
def test_target_counts_use_only_valid_rows_and_every_subset():
    rows = [{**make_rows(["A"])[0], "program_end_date": None},   # 3 fields: 3 pairs + 1 triple
            make_rows(["B"])[0],                                  # 4 fields: 11 subsets
            make_rows(["C"])[1]]                                  # invalid row: ignored
    t = b.target_field_set_counts(rows, 2)
    assert sum(t.values()) == 4 + 11 and t[fs("channel", "program_desc")] == 2 and t[fs(*FIELDS4)] == 1
    with pytest.raises(AssertionError):
        b.target_field_set_counts([make_rows(["C"])[1]], 2)


def test_allocate_sums_to_n_is_proportional_and_breaks_ties_by_set_order():
    a, c, d = fs("channel", "program_desc"), fs("channel", "program_start_date"), fs("program_desc", "program_start_date")
    assert b.allocate(Counter({a: 3, c: 1}), 4) == {a: 3, c: 1}
    assert b.allocate(Counter({a: 1, c: 1, d: 1}), 2) == {a: 1, c: 1, d: 0}, "equal remainders go to the first sets"
    big = b.allocate(Counter({a: 7, c: 5, d: 3}), 1000)
    assert sum(big.values()) == 1000 and abs(big[a] - 467) <= 1


def test_split_equally_gives_the_remainder_to_the_types_behind_on_the_running_total():
    running = Counter()
    types = ["intra_document", "cross_type", "inter_document"]
    assert b.split_equally(7, types, running) == {"cross_type": 3, "inter_document": 2, "intra_document": 2}   # tie: sorted
    assert b.split_equally(1, types, running) == {"cross_type": 0, "inter_document": 1, "intra_document": 0}
    assert b.split_equally(1, types, running) == {"cross_type": 0, "inter_document": 0, "intra_document": 1}
    assert running == Counter({"cross_type": 3, "inter_document": 3, "intra_document": 3})
    # many one-sample sets: the totals stay within one of each other (a fixed tie order would give all to one type)
    running = Counter()
    for _ in range(100):
        b.split_equally(1, types, running)
    assert max(running.values()) - min(running.values()) <= 1
    assert b.split_equally(0, ["a"], Counter()) == {"a": 0}


def _pools(split="train", seed=0):
    rows = b.source_rows(make_rows(SPLIT_DOCS[split]), "all_rows", None, split)
    items = s.items_by_document(rows)
    rng = np.random.default_rng(seed)
    drawn, _ = b.draw_samples(rows, 11, 2, rng)
    drawn, _ = b.drop_matching(drawn, items, EQ)
    cands, _ = b.cross_item_candidates(drawn, items, EQ, rng)
    return rows, drawn, cands


def test_balance_gives_exact_cross_item_and_class_counts_and_identical_field_set_histograms():
    rows, drawn, cands = _pools()
    target = b.target_field_set_counts(rows, 2)
    out = b.balance_field_set_error_type(drawn, cands, 4, ["inter_document", "intra_document"], target, np.random.default_rng(1))
    hist = lambda lab: Counter(frozenset(x["fields"]) for x in out if x["label_valid"] == lab)
    assert len(out) == 24 and hist(True) == hist(False) and sum(hist(True).values()) == 12
    assert sum(x["error_type"] == "cross_item" for x in out) == 4
    # per set: the row types split the set's row-type allocation equally (remainder to the first type)
    alloc_rows, alloc_ci = b.allocate(target, 8), b.allocate(target, 4)
    for f, n in alloc_rows.items():
        got = Counter(x["error_type"] for x in out if not x["label_valid"] and frozenset(x["fields"]) == f)
        assert got["cross_item"] == alloc_ci[f]
        assert got["inter_document"] + got["intra_document"] == n and abs(got["inter_document"] - got["intra_document"]) <= 1
    assert b.balance_field_set_error_type(drawn, cands, 4, ["inter_document", "intra_document"], target,
                                          np.random.default_rng(1)) == out


def test_balance_raises_on_a_short_cell():
    rows, drawn, cands = _pools()
    target = b.target_field_set_counts(rows, 2)
    with pytest.raises(ValueError, match="only"):
        b.balance_field_set_error_type(drawn, cands, 10**4, ["inter_document", "intra_document"], target, np.random.default_rng(1))
    with pytest.raises(ValueError, match="cross_item"):
        b.balance_field_set_error_type(drawn, [], 4, ["inter_document", "intra_document"], target, np.random.default_rng(1))


def test_same_seed_draws_identically_and_replicates_differ():
    rows = make_rows(SPLIT_DOCS["train"])
    items = s.items_by_document(rows)
    p = {"seed": 0, "samples_per_row": 11, "min_subset_size": 2, "n_per_error_type": {"train": 4, "test": 3},
         "max_error_type_share_deviation": 0.2}
    kind_of = b.field_kinds(KINDS)
    a = b.draw_replicate(rows, items, p, "train", 0, EQ, kind_of)
    assert a == b.draw_replicate(rows, items, p, "train", 0, EQ, kind_of)
    assert a["samples"] != b.draw_replicate(rows, items, p, "train", 1, EQ, kind_of)["samples"]


# ---------------------------------------------------------------- end to end
@pytest.fixture
def built(tmp_path):
    write_data(tmp_path)
    b.main(["--config", str(write_config(tmp_path))])
    return tmp_path


def test_end_to_end_replicates_load_balanced_and_join_to_rows(built):
    blob = json.loads((built / s.SUBSETS_FILE).read_text())
    assert blob["build"]["error_types"] == ["cross_item", "inter_document", "intra_document"]
    assert blob["build"]["equivalence"] == {"kinds": KINDS, "year_pivot": PIVOT}
    for split, per_type in (("train", 4), ("test", 3)):
        seen = []
        for r in range(3):
            samples = s.load_subsets(built, split, r)
            assert len(samples) == 2 * 3 * per_type and sum(x["label_valid"] for x in samples) == 3 * per_type
            assert sum(x["error_type"] == "cross_item" for x in samples) == per_type
            assert all(x["document_id"] in SPLIT_DOCS[split] for x in samples)
            for x in samples:
                assert x["values"] == s.sample_values(x["row"], x["fields"], x["swap"])
                assert x["label_valid"] == (not x["invalid_fields"])
            seen.append({x["sample_id"] for x in samples})
            stats = blob["line"][split]["replicates"][r]["stats"]
            assert sum(stats["cross_item_pairs"].values()) == per_type
            assert stats["n_dropped_matches_item"] == {}   # every value is distinct within a document
        assert len({frozenset(x) for x in seen}) == 3, "replicates are not distinct"


def test_replicates_are_unchanged_when_n_replicates_grows_and_the_file_is_deterministic(built):
    b.main(["--config", str(write_config(built, n_replicates=2))])
    two = json.loads((built / s.SUBSETS_FILE).read_text())
    b.main(["--config", str(write_config(built, n_replicates=3))])
    three = json.loads((built / s.SUBSETS_FILE).read_text())
    for split in ("train", "test"):
        assert three["line"][split]["replicates"][:2] == two["line"][split]["replicates"]
    b.main(["--config", str(write_config(built, n_replicates=3))])
    assert json.loads((built / s.SUBSETS_FILE).read_text())["line"] == three["line"]


def test_subsets_go_stale_when_the_rows_change(built):
    path = built / "line" / "train.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(AssertionError, match="stale"):
        s.load_subsets(built, "train", 0)
    s.load_subsets(built, "test", 0)   # the untouched split still loads


def test_loader_refuses_a_different_equivalence(built):
    write_data(built, year_pivot=60)   # rewrites the same rows, so only invalids.json changes
    with pytest.raises(AssertionError, match="equivalence"):
        s.load_subsets(built, "test", 0)


def test_loader_catches_a_sample_that_disagrees_with_its_row(built):
    blob = json.loads((built / s.SUBSETS_FILE).read_text())
    ids = blob["line"]["test"]["replicates"][0]["samples"]
    i = next(j for j, sid in enumerate(ids) if s.parse_sample_id(sid)["swap"] is None)
    ids[i] = ids[i].rsplit("|", 1)[0] + "|sub_amount+channel"   # sub_amount is null in every row
    (built / s.SUBSETS_FILE).write_text(json.dumps(blob))
    with pytest.raises(AssertionError):
        s.load_subsets(built, "test", 0)


def test_loader_catches_an_invalid_label_that_the_match_rule_says_is_valid(built):
    # Rewrite one corrupted row so its values become another item's, keep the subsets file pointing at it
    # (re-recording the hash, as a builder bug would): the loader's biconditional must fire.
    blob = json.loads((built / s.SUBSETS_FILE).read_text())
    samples = s.load_subsets(built, "test", 0)
    victim = next(x for x in samples if not x["label_valid"] and x["swap"] is None)
    rows = json.loads((built / "line" / "test.json").read_text())
    for r in rows:
        if (r["document_id"], r["line_index"], r["error_type"], r["k"]) == (
                victim["document_id"], victim["line_index"], victim["error_type"], victim["k"]):
            other = item(victim["document_id"], (victim["line_index"] + 2) % N_LINES)
            r.update({f: other[f] for f in FIELDS4})
    (built / "line" / "test.json").write_text(json.dumps(rows))
    blob["build"]["sha256"]["line/test.json"] = s.sha256_of(built / "line" / "test.json")
    (built / s.SUBSETS_FILE).write_text(json.dumps(blob))
    with pytest.raises(AssertionError, match="match rule"):
        s.load_subsets(built, "test", 0)


def test_loader_catches_a_drifting_error_type_share(built):
    path = built / s.SUBSETS_FILE
    blob = json.loads(path.read_text())
    # A fourth expected type that never occurs: every recorded type is now 1/3 vs an expected 1/4.
    blob["build"]["error_types"] = ["cross_item", "cross_type", "inter_document", "intra_document"]
    blob["build"]["subsets_params"]["max_error_type_share_deviation"] = 0.05
    path.write_text(json.dumps(blob))
    with pytest.raises(AssertionError):   # class sizes (4 types x n) and shares both disagree
        s.load_subsets(built, "train", 0)


def test_check_error_type_shares():
    ets = ["inter_document"] * 4 + ["intra_document"] * 3 + ["cross_type"] * 3
    exp = ["cross_type", "inter_document", "intra_document"]
    assert s.check_error_type_shares(ets, exp, 0.1) == {"cross_type": 0.3, "inter_document": 0.4, "intra_document": 0.3}
    with pytest.raises(AssertionError, match="inter_document"):
        s.check_error_type_shares(ets, exp, 0.05)
    with pytest.raises(AssertionError):
        s.check_error_type_shares(ets, exp[:2], 0.5)   # an error type outside the expected set


def test_config_rejects_missing_extra_and_bad_keys(built):
    cfg = write_config(built)
    good = yaml.safe_load(cfg.read_text())
    mutations = (lambda c: c["params"]["subsets"].pop("seed"),
                 lambda c: c["params"]["subsets"].update(n_per_class={"train": 1, "test": 1}),
                 lambda c: c["params"]["subsets"].update(source="bogus"),
                 lambda c: c["params"]["subsets"].update(balance_by="field_set"),
                 lambda c: c["params"]["subsets"].update(max_error_type_share_deviation=0),
                 lambda c: c["params"]["subsets"].update(n_per_error_type={"train": 0, "test": 1}),
                 lambda c: c["params"].pop("dates"))
    for mutate in mutations:
        bad = yaml.safe_load(yaml.safe_dump(good))
        mutate(bad)
        cfg.write_text(yaml.safe_dump(bad))
        with pytest.raises(AssertionError):
            b.load_config(cfg)
    cfg.write_text(yaml.safe_dump(good))
    b.load_config(cfg)


def test_builder_refuses_a_config_equivalence_that_differs_from_invalids_json(built):
    write_data(built, year_pivot=60)
    with pytest.raises(AssertionError, match="invalids.json"):
        b.main(["--config", str(write_config(built))])
