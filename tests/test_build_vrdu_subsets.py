"""Unit + tiny end-to-end tests for data/vrdu/build_vrdu_subsets.py and data/vrdu/subsets.py.

Hand-built line rows. Every base row (document, line_index) has channel / program_desc /
program_start_date / program_end_date set and sub_amount null, so it has 4 usable fields and
C(4,2) + C(4,3) + C(4,4) = 11 unordered subsets of size >= 2. Each base row gives three rows:
valid; inter_document k=1 (channel corrupted); inter_document k=2 (channel, program_desc corrupted).
A subset of an invalid row is valid iff it avoids the corrupted fields.
"""

import itertools
import json
from collections import Counter
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "vrdu"))
import build_vrdu_subsets as b
import subsets as s

BASE = {"channel": "c", "program_desc": "d", "program_start_date": "s", "program_end_date": "e", "sub_amount": None}
SPLIT_DOCS = {"train": ["A", "B", "C"], "test": ["D", "E", "F"]}


def make_rows(docs, n_lines=3):
    rows = []
    for doc in docs:
        for li in range(n_lines):
            for et, k, bad in ((None, 0, []), ("inter_document", 1, ["channel"]), ("inter_document", 2, ["channel", "program_desc"])):
                rows.append({"document_id": doc, "line_index": li, "valid": et is None, "error_type": et, "k": k,
                             "invalid_fields": bad, **BASE})
    return rows


def test_sample_id_round_trips_and_keeps_the_field_order():
    row = make_rows(["A"])[1]
    sid = s.make_sample_id(row, ["program_end_date", "channel"])
    assert sid == "A|0|inter_document|1|program_end_date+channel"
    assert s.parse_sample_id(sid) == {"document_id": "A", "line_index": 0, "error_type": "inter_document", "k": 1,
                                      "fields": ["program_end_date", "channel"]}
    valid = make_rows(["A"])[0]
    assert s.parse_sample_id(s.make_sample_id(valid, ["channel", "program_desc"]))["error_type"] is None


def test_label_is_valid_iff_no_field_is_corrupted_and_ignores_order():
    row = make_rows(["A"])[2]   # channel, program_desc corrupted
    assert not s.sample_label_valid(row, ["channel", "program_end_date"])
    assert not s.sample_label_valid(row, ["program_end_date", "program_desc", "program_start_date"])
    assert s.sample_label_valid(row, ["program_start_date", "program_end_date"])
    assert s.sample_label_valid(row, ["program_end_date", "program_start_date"])
    assert s.sample_label_valid(make_rows(["A"])[0], list(BASE)[:4])


def test_draw_gives_distinct_unordered_subsets_of_nonnull_fields_in_random_order():
    rows = make_rows(["A"])
    samples, stats = b.draw_samples(rows, 11, 2, np.random.default_rng(0))   # all 11 subsets of every row
    assert len(samples) == 11 * len(rows) and stats["n_rows_fewer_subsets_than_requested"] == 0
    for row in rows:
        mine = [x for x in samples if x["row"] is row]
        sets = [frozenset(x["fields"]) for x in mine]
        assert len(set(sets)) == len(sets) == 11
        assert all(len(f) >= 2 and "sub_amount" not in f for f in sets)
    canonical = [f for f in b.LINE_FIELDS]
    assert any(x["fields"] != sorted(x["fields"], key=canonical.index) for x in samples), "order never differs from canonical"
    assert {x["label_valid"] for x in samples} == {True, False}


def test_draw_takes_all_subsets_when_fewer_exist_and_skips_too_small_rows():
    small = {**make_rows(["A"])[0], "program_start_date": None, "program_end_date": None}   # channel, program_desc: 1 subset
    tiny = {**make_rows(["B"])[0], "program_desc": None, "program_start_date": None, "program_end_date": None}   # 1 field
    samples, stats = b.draw_samples([small, tiny], 4, 2, np.random.default_rng(0))
    assert len(samples) == 1 and set(samples[0]["fields"]) == {"channel", "program_desc"}
    assert stats["n_rows_too_small"] == 1 and stats["n_rows_fewer_subsets_than_requested"] == 1


def test_balance_is_exact_and_refuses_a_short_pool():
    samples, _ = b.draw_samples(make_rows(["A"]), 4, 2, np.random.default_rng(0))
    out = b.balance(samples, 5, np.random.default_rng(1))
    assert sum(x["label_valid"] for x in out) == 5 and sum(not x["label_valid"] for x in out) == 5
    with pytest.raises(ValueError, match="requested"):
        b.balance(samples, 10**6, np.random.default_rng(1))


def test_same_seed_draws_identically_and_replicates_differ():
    rows = make_rows(SPLIT_DOCS["train"])
    p = {"seed": 0, "samples_per_row": 4, "min_subset_size": 2, "n_per_class": {"train": 5, "test": 5}, "balance_by": "none",
         "max_error_type_share_deviation": 0.5}
    a, a2 = b.draw_replicate(rows, p, "train", 0), b.draw_replicate(rows, p, "train", 0)
    other, other_split = b.draw_replicate(rows, p, "train", 1), b.draw_replicate(rows, p, "test", 0)
    assert a == a2
    assert a["samples"] != other["samples"] and a["samples"] != other_split["samples"]


@pytest.fixture
def built(tmp_path):
    (tmp_path / "line").mkdir()
    for split, docs in SPLIT_DOCS.items():
        (tmp_path / "line" / f"{split}.json").write_text(json.dumps(make_rows(docs)))
    return tmp_path


def write_config(root, **override):
    p = {"seed": 0, "source": "all_rows", "n_replicates": 3, "samples_per_row": 4, "min_subset_size": 2,
         "n_per_class": {"train": 6, "test": 4}, "balance_by": "none", "max_error_type_share_deviation": 0.5, **override}
    path = root / "cfg.yaml"
    path.write_text(yaml.safe_dump({"id": "t", "params": {"data_root": str(root), "subsets": p}}))
    return path


def test_end_to_end_replicates_load_balanced_and_join_to_rows(built):
    b.main(["--config", str(write_config(built))])
    for split, n in (("train", 6), ("test", 4)):
        seen = []
        for r in range(3):
            samples = s.load_subsets(built, split, r)
            assert len(samples) == 2 * n and sum(x["label_valid"] for x in samples) == n
            assert all(x["row"]["document_id"] in SPLIT_DOCS[split] for x in samples)
            seen.append({x["sample_id"] for x in samples})
        assert len({frozenset(x) for x in seen}) == 3, "replicates are not distinct"


def test_replicates_are_unchanged_when_n_replicates_grows_and_the_file_is_deterministic(built):
    b.main(["--config", str(write_config(built, n_replicates=2))])
    two = json.loads((built / s.SUBSETS_FILE).read_text())
    b.main(["--config", str(write_config(built, n_replicates=3))])
    three = json.loads((built / s.SUBSETS_FILE).read_text())
    for split in ("train", "test"):
        assert three["line"][split]["replicates"][:2] == two["line"][split]["replicates"]
    b.main(["--config", str(write_config(built, n_replicates=3))])
    again = json.loads((built / s.SUBSETS_FILE).read_text())
    assert again["line"] == three["line"]


def test_subsets_go_stale_when_the_rows_change(built):
    b.main(["--config", str(write_config(built))])
    path = built / "line" / "train.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(AssertionError, match="stale"):
        s.load_subsets(built, "train", 0)
    s.load_subsets(built, "test", 0)   # the untouched split still loads


def test_loader_catches_a_sample_that_disagrees_with_its_row(built):
    b.main(["--config", str(write_config(built))])
    blob = json.loads((built / s.SUBSETS_FILE).read_text())
    ids = blob["line"]["test"]["replicates"][0]["samples"]
    ids[0] = ids[0].rsplit("|", 1)[0] + "|sub_amount+channel"   # sub_amount is null in every row
    (built / s.SUBSETS_FILE).write_text(json.dumps(blob))
    with pytest.raises(AssertionError):
        s.load_subsets(built, "test", 0)


def test_config_rejects_missing_extra_and_main_keys(built):
    cfg = write_config(built)
    good = yaml.safe_load(cfg.read_text())
    for mutate in (lambda p: p.pop("seed"), lambda p: p.update(main={"n_per_class": {}}), lambda p: p.update(source="bogus")):
        bad = yaml.safe_load(cfg.read_text())
        mutate(bad["params"]["subsets"])
        cfg.write_text(yaml.safe_dump(bad))
        with pytest.raises(AssertionError):
            b.load_config(cfg)
    cfg.write_text(yaml.safe_dump(good))
    b.load_config(cfg)


# ---------------------------------------------------------------- field-set balancing
FS_DOCS = {"train": [f"T{i}" for i in range(6)], "test": [f"E{i}" for i in range(6)]}
FIELDS4 = ["channel", "program_desc", "program_start_date", "program_end_date"]


def make_rows_fs(docs):
    """Per line, a valid row and inter / intra rows at k=1..3 whose corrupted fields start at a rotating field,
    so every field set has both valid and invalid samples somewhere in the pool."""
    rows = []
    for doc in docs:
        for li in range(4):
            order = FIELDS4[li:] + FIELDS4[:li]
            rows.append({"document_id": doc, "line_index": li, "valid": True, "error_type": None, "k": 0, "invalid_fields": [], **BASE})
            for et in ("inter_document", "intra_document"):
                for k in (1, 2, 3):
                    rows.append({"document_id": doc, "line_index": li, "valid": False, "error_type": et, "k": k,
                                 "invalid_fields": order[:k], **BASE})
    return rows


def fs(*names):
    return frozenset(names)


def test_target_counts_use_only_valid_rows_and_every_subset():
    rows = [{**make_rows(["A"])[0], "program_end_date": None},   # 3 fields: 3 pairs + 1 triple
            make_rows(["B"])[0],                                  # 4 fields: 11 subsets
            make_rows(["C"])[1]]                                  # invalid row: ignored
    t = b.target_field_set_counts(rows, 2)
    assert sum(t.values()) == 4 + 11 and t[fs("channel", "program_desc")] == 2 and t[fs(*FIELDS4)] == 1
    assert fs("channel", "program_desc", "program_start_date") in t
    with pytest.raises(AssertionError):
        b.target_field_set_counts([make_rows(["C"])[1]], 2)


def test_allocate_sums_to_n_is_proportional_and_breaks_ties_by_set_order():
    a, c, d = fs("channel", "program_desc"), fs("channel", "program_start_date"), fs("program_desc", "program_start_date")
    assert b.allocate(Counter({a: 3, c: 1}), 4) == {a: 3, c: 1}
    out = b.allocate(Counter({a: 1, c: 1, d: 1}), 2)
    assert out == {a: 1, c: 1, d: 0}, "equal remainders must go to the first sets in set order"
    big = b.allocate(Counter({a: 7, c: 5, d: 3}), 1000)
    assert sum(big.values()) == 1000 and abs(big[a] - 467) <= 1


def test_balance_field_set_gives_both_classes_the_same_per_set_counts_and_raises_on_a_short_pool():
    rows = b.source_rows(make_rows_fs(FS_DOCS["train"]), "all_rows", None, "train")
    drawn, _ = b.draw_samples(rows, 4, 2, np.random.default_rng(0))
    target = b.target_field_set_counts(rows, 2)
    out = b.balance_field_set(drawn, 22, target, np.random.default_rng(1))
    by = lambda lab: Counter(frozenset(x["fields"]) for x in out if x["label_valid"] == lab)
    assert len(out) == 44 and by(True) == by(False) and sum(by(True).values()) == 22
    assert b.balance_field_set(drawn, 22, target, np.random.default_rng(1)) == out
    with pytest.raises(ValueError, match="only"):
        b.balance_field_set(drawn, 10**6, target, np.random.default_rng(1))


@pytest.fixture
def built_fs(tmp_path):
    (tmp_path / "line").mkdir()
    for split, docs in FS_DOCS.items():
        (tmp_path / "line" / f"{split}.json").write_text(json.dumps(make_rows_fs(docs)))
    b.main(["--config", str(write_config(tmp_path, balance_by="field_set", n_per_class={"train": 22, "test": 11}))])
    return tmp_path


def test_end_to_end_field_set_balance_loads_with_equal_histograms_and_shares(built_fs):
    for split, n in (("train", 22), ("test", 11)):
        for r in range(3):
            samples = s.load_subsets(built_fs, split, r)
            hist = lambda lab: Counter(frozenset(x["fields"]) for x in samples if x["label_valid"] == lab)
            assert hist(True) == hist(False) and len(samples) == 2 * n
    blob = json.loads((built_fs / s.SUBSETS_FILE).read_text())
    assert blob["build"]["error_types"] == ["inter_document", "intra_document"]
    assert set(blob["line"]["train"]["replicates"][0]["stats"]["invalid_error_type_shares"]) == {"inter_document", "intra_document"}


def test_loader_catches_a_field_set_mismatch_and_a_drifting_error_type_share(built):
    b.main(["--config", str(write_config(built))])   # balance_by none: the histograms differ
    path = built / s.SUBSETS_FILE
    blob = json.loads(path.read_text())
    blob["build"]["subsets_params"]["balance_by"] = "field_set"
    path.write_text(json.dumps(blob))
    with pytest.raises(AssertionError, match="field-set histograms"):
        s.load_subsets(built, "train", 0)
    blob["build"]["subsets_params"]["balance_by"] = "none"
    blob["build"]["error_types"] = ["inter_document", "intra_document"]   # the data has only inter: share 1.0 vs 0.5
    blob["build"]["subsets_params"]["max_error_type_share_deviation"] = 0.1
    path.write_text(json.dumps(blob))
    with pytest.raises(AssertionError, match="share of"):
        s.load_subsets(built, "train", 0)


def test_check_error_type_shares():
    ets = ["inter_document"] * 4 + ["intra_document"] * 3 + ["cross_type"] * 3
    exp = ["cross_type", "inter_document", "intra_document"]
    assert s.check_error_type_shares(ets, exp, 0.1) == {"cross_type": 0.3, "inter_document": 0.4, "intra_document": 0.3}
    with pytest.raises(AssertionError, match="inter_document"):
        s.check_error_type_shares(ets, exp, 0.05)
    with pytest.raises(AssertionError):
        s.check_error_type_shares(ets, exp[:2], 0.5)   # an error type outside the expected set


def test_none_reproduces_the_label_only_balance(built):
    rows = b.source_rows(make_rows(SPLIT_DOCS["train"]), "all_rows", None, "train")
    p = {"seed": 0, "samples_per_row": 4, "min_subset_size": 2, "n_per_class": {"train": 6, "test": 4}, "balance_by": "none",
         "max_error_type_share_deviation": 0.5}
    rng = np.random.default_rng([0, 0, 2])
    drawn, _ = b.draw_samples(rows, 4, 2, rng)
    expect = [x["sample_id"] for x in b.balance(drawn, 6, rng)]
    assert b.draw_replicate(rows, p, "train", 2)["samples"] == expect


def test_config_rejects_a_bad_balance_mode_and_tolerance(built):
    cfg = write_config(built)
    for key, bad in (("balance_by", "size"), ("max_error_type_share_deviation", 0), ("max_error_type_share_deviation", 1.5)):
        c = yaml.safe_load(cfg.read_text())
        c["params"]["subsets"][key] = bad
        cfg.write_text(yaml.safe_dump(c))
        with pytest.raises(AssertionError):
            b.load_config(cfg)
