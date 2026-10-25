"""Unit + tiny end-to-end tests for data/vrdu/build_vrdu_main.py and data/vrdu/main_samples.py.

End to end: the hand-built mini corpus in tests/fixtures/vrdu_invalids_mini/. Its main bases have 9 train /
3 test documents (m1: agency null; m3: flight_from is the OCR fragment "05/20/"; plus the main rows of the
line documents L1-L7), each reaching every inter_document k in 1..9 (every field can be resampled from the
split-wide pool, nulls included). Its line bases are a source of cross_type values. cross_type rows were
checked equal to the -05 builder's (build_vrdu_invalids.py, since removed) on this corpus.

main_samples: hand-built rows of documents X, Y, Z where X's agency is null, so corrupting it is a null fill.
"""

import json
import random
import shutil
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "vrdu"))
import build_vrdu_main as b  # noqa: E402
import main_samples as ms  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "vrdu_invalids_mini"
CONFIG = REPO / "data" / "vrdu" / "build-configs" / "2026-10-24-vrdu-invalids-06.yaml"
KINDS = yaml.safe_load(CONFIG.read_text())["params"]["cross_type"]["kinds"]
PIVOT = 50
DATE_FIELDS = ["program_start_date", "program_end_date"]
EQ = b.make_eq(b.field_kinds(KINDS), PIVOT)
FRAG_RATE = 0.2


def run(builder, config_path, root):
    shutil.copytree(FIXTURE, root)
    config = yaml.safe_load(config_path.read_text())
    config["params"]["data_root"] = str(root)
    cfg = root / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(config))
    builder.main(["--config", str(cfg)])
    return {split: json.loads((root / "main" / f"{split}.json").read_text()) for split in b.SPLITS}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return run(b, CONFIG, tmp_path_factory.mktemp("main") / "vrdu")


# ---------------------------------------------------------------- builder


def test_counts_and_k_is_the_number_of_corrupted_fields(built):
    for split, n_docs in (("train", 9), ("test", 3)):
        rows = built[split]
        assert sum(r["valid"] for r in rows) == n_docs
        assert sorted(r["k"] for r in rows if r["error_type"] == "inter_document") == sorted(list(range(1, 10)) * n_docs)
        keys = [(r["document_id"], r["error_type"], r["k"]) for r in rows]
        assert len(keys) == len(set(keys))
        for r in rows:
            assert r["k"] == r["num_invalid_fields"] == len(r["invalid_fields"])


def test_null_fill_is_recorded(built):
    m1 = [r for r in built["train"] if r["document_id"] == "m1" and "agency" in r["invalid_fields"]]   # m1's agency is null
    assert m1 and all(r["field_errors"]["agency"]["source"]["null_fill"] for r in m1)
    for r in built["train"]:
        for f in r["invalid_fields"]:
            assert isinstance(r["field_errors"][f]["source"]["null_fill"], bool)


def test_cross_type_known_answer_for_main_document_m1(built):
    # m1: five name fields with distinct same-kind donors (agency is null and may receive one);
    # the dates can only be filled from each other and from_to would reverse -> at most one date;
    # gross_amount has no same-kind donor (m1 has no line items). So achievable k = 1..6.
    rows = [r for r in built["train"] if r["document_id"] == "m1" and r["error_type"] == "cross_type"]
    assert sorted(r["k"] for r in rows) == [1, 2, 3, 4, 5, 6]
    touched = {f for r in rows for f in r["invalid_fields"]}
    assert touched <= {"advertiser", "agency", "property", "tv_address", "product", "flight_from", "flight_to"}
    assert not touched & {"contract_num", "gross_amount"}


def test_every_row_labels_every_entity_explicitly(built):
    for split in b.SPLITS:
        for row in built[split]:
            assert list(row["field_errors"]) == b.MAIN_FIELDS
            wrong = [f for f in b.MAIN_FIELDS if row["field_errors"][f] is not None]
            assert sorted(wrong) == sorted(row["invalid_fields"]) and (row["error_type"] is None) == row["valid"]
            assert all(row["field_errors"][f]["error_type"] == row["error_type"] for f in wrong)


def test_seed_determinism(built, tmp_path):
    again = run(b, CONFIG, tmp_path / "vrdu")
    assert json.dumps(again) == json.dumps(built)


def test_config_requires_corrupt_from_null(tmp_path):
    config = yaml.safe_load(CONFIG.read_text())
    config["params"]["main"]["corrupt_from_null"] = False
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(config))
    with pytest.raises(AssertionError, match="corrupt_from_null"):
        b.load_config(cfg)


# ---------------------------------------------------------------- main_samples


def clean(doc, **override):
    row = {f: f"{doc}-{f}" for f in ms.MAIN_FIELDS}
    row.update(override)
    return {"document_id": doc, "valid": True, "error_type": None, "k": 0, "num_invalid_fields": 0,
            "invalid_fields": [], "field_errors": {f: None for f in ms.MAIN_FIELDS}, **row}


def corrupt(base, et, fields):
    row = {**base, "valid": False, "error_type": et, "k": len(fields), "num_invalid_fields": len(fields),
           "invalid_fields": list(fields), "field_errors": dict(base["field_errors"])}
    for f in fields:
        row[f] = f"wrong-{f}"
        row["field_errors"][f] = {"error_type": et, "source": {"null_fill": base[f] is None}}
    return row


X, Y, Z = clean("X", agency=None), clean("Y"), clean("Z")


def test_pairs_twin_by_document_and_null_fill_by_other_document():
    rows = [X, Y, Z, corrupt(X, "inter_document", ["agency"]), corrupt(Y, "inter_document", ["advertiser"]),
            corrupt(Z, "inter_document", ["advertiser"])]
    pairs = ms.main_entity_pairs(rows, seed=0)
    assert len(pairs) == 3
    same = [p for p in pairs if not p["null_fill"]]
    assert {(p["invalid"]["document_id"], p["valid"]["document_id"]) for p in same} == {("Y", "Y"), ("Z", "Z")}
    assert all(p["twin"] == "same_document" and p["valid"]["error_type"] is None and p["valid"]["k"] == 0 for p in same)
    (fill,) = [p for p in pairs if p["null_fill"]]
    assert fill["twin"] == "other_document" and fill["field"] == "agency" and fill["valid"]["document_id"] in {"Y", "Z"}


def test_null_fill_twin_is_another_document_with_the_field():
    w = clean("W", agency=None)
    rows = [X, Y, w, corrupt(X, "cross_type", ["agency"])]
    for seed in range(5):
        (fill,) = ms.main_entity_pairs(rows, seed)
        assert fill["valid"]["document_id"] == "Y"   # W has no agency, X is the document itself


def test_null_fill_without_any_other_document_raises():
    rows = [X, clean("W", agency=None), corrupt(X, "cross_type", ["agency"])]
    with pytest.raises(AssertionError, match="no other document has this field"):
        ms.main_entity_pairs(rows, seed=0)


def test_kept_pairs_are_the_uncorrupted_shown_fields():
    rows = [X, Y, corrupt(X, "inter_document", ["advertiser", "agency"])]
    kept = ms.kept_span_pairs(rows)
    assert sorted(p["field"] for p in kept) == sorted(f for f in ms.MAIN_FIELDS if f not in ("advertiser", "agency"))
    assert all(p["kept"]["document_id"] == p["clean"]["document_id"] == "X" for p in kept)


def test_pairs_on_the_built_corpus_are_balanced_per_stratum(built):
    for split in b.SPLITS:
        rows = built[split]
        pairs = ms.main_entity_pairs(rows, seed=0)
        assert len(pairs) == sum(r["k"] for r in rows if not r["valid"])   # one twin per corrupted span
        clean_rows = {r["document_id"]: r for r in rows if r["valid"]}
        for p in pairs:
            assert clean_rows[p["valid"]["document_id"]][p["field"]] is not None
            assert (p["twin"] == "same_document") == (p["valid"]["document_id"] == p["invalid"]["document_id"]) == (not p["null_fill"])


# ---------------------------------------------------------------- generation units (moved from the -05 tests)


def test_main_pools_keep_fragments_and_have_no_sentinels_nulls_dropped():
    rows = [{**{f: None for f in b.MAIN_FIELDS}, "advertiser": "a", "flight_from": "01/10/20"},
            {**{f: None for f in b.MAIN_FIELDS}, "advertiser": "a", "flight_from": "05/20/", "flight_to": "02/02/20"}]
    pools = b.main_pools(rows)
    assert pools["advertiser"] == ["a", "a"] and pools["agency"] == []
    assert pools["flight_from"] == ["01/10/20", "05/20/"]   # the fragment stays; nothing synthetic added
    assert pools["flight_to"] == ["02/02/20"]


def test_date_feasibility_ordered_pair_impossible_to_reorder():
    row = {"program_start_date": "02/01/20", "program_end_date": "02/05/20"}
    pools = {"program_start_date": ["02/01/20", "02/05/20"], "program_end_date": ["02/01/20", "02/05/20"]}
    assert b._date_feasibility(row, DATE_FIELDS, pools, PIVOT, EQ) == (True, True, False)  # each alone; not both


def test_date_feasibility_fragment_partner_does_not_block():
    row = {"program_start_date": "03/01/", "program_end_date": "03/31/20"}
    pools = {"program_start_date": ["03/31/20", "03/05/20"], "program_end_date": ["03/31/20", "03/05/20"]}
    assert b._date_feasibility(row, DATE_FIELDS, pools, PIVOT, EQ) == (True, True, True)


def test_date_feasibility_a_fragment_candidate_is_always_orderable():
    row = {"program_start_date": "05/01/20", "program_end_date": "05/10/20"}
    only_late = ["05/01/20", "06/01/20"]    # the only different start (06/01) is after the end
    assert b._date_feasibility(row, DATE_FIELDS, {"program_start_date": only_late, "program_end_date": ["05/10/20"]},
                               PIVOT, EQ)[0] is False
    assert b._date_feasibility(row, DATE_FIELDS, {"program_start_date": [*only_late, "/12/20"], "program_end_date": ["05/10/20"]},
                               PIVOT, EQ)[0] is True


def test_nondate_invalidatable_excludes_constant_field_and_reformat_only_alternatives():
    rows = [{"channel": "5", "program_desc": "A", "sub_amount": None},
            {"channel": "5", "program_desc": "B", "sub_amount": "$9"}]
    pools = {f: [r[f] for r in rows if r[f] is not None] for f in ["channel", "program_desc", "sub_amount"]}
    got = b._nondate_invalidatable(rows[0], b.LINE_FIELDS, DATE_FIELDS, pools, EQ)
    assert "channel" not in got and {"program_desc", "sub_amount"} <= set(got)
    row = {"channel": "5", "program_desc": "ISSUE", "sub_amount": "$1,234.00"}
    pools = {"channel": ["5"], "program_desc": ["Issue"], "sub_amount": ["1234"]}
    assert b._nondate_invalidatable(row, b.LINE_FIELDS, DATE_FIELDS, pools, EQ) == []


def test_achievable_counts_and_targets():
    assert b.achievable_counts(2, (True, True, True)) == {0, 1, 2, 3, 4}
    assert b.achievable_counts(0, (True, True, False)) == {0, 1}
    assert b.achievable_counts(1, (False, False, False)) == {0, 1}
    row = {"channel": "1", "program_desc": "d", "program_start_date": "01/01/20", "program_end_date": "01/10/20", "sub_amount": "$1"}
    pools = {"channel": ["1", "2"], "program_desc": ["d"], "sub_amount": ["$1", "$2"],   # desc has no differing value
             "program_start_date": ["01/01/20", "01/05/20"], "program_end_date": ["01/10/20", "01/15/20"]}
    # 2 non-date fields + both dates (alone or together) -> achievable {0..4}; requesting up to 5 caps at 4
    assert b.achievable_targets(row, b.LINE_FIELDS, DATE_FIELDS, pools, PIVOT, (1, 5), EQ) == [1, 2, 3, 4]


def test_sample_new_returns_a_differing_value_and_raises_without_one():
    assert b.sample_new(["x", "x", "y"], "x", random.Random(0), EQ, "program_desc") == "y"
    with pytest.raises(AssertionError):
        b.sample_new(["x", "x"], "x", random.Random(0), EQ, "program_desc")
    with pytest.raises(AssertionError):     # a case variant is not a different value
        b.sample_new(["X", "x"], "x", random.Random(0), EQ, "program_desc")


def test_make_invalid_row_caps_when_only_one_date_field_can_change_and_none_when_nothing_feasible():
    row = {"document_id": "d", "channel": "9", "program_desc": "Solo",
           "program_start_date": "02/01/20", "program_end_date": "02/05/20", "sub_amount": "$9"}
    shared = ["02/01/20", "02/05/20"]
    pools = {"channel": ["9"], "program_desc": ["Solo"], "sub_amount": ["$9"], "program_start_date": shared, "program_end_date": shared}
    new_row, changed = b.make_invalid_row(row, 3, b.LINE_FIELDS, DATE_FIELDS, pools, PIVOT, random.Random(1), EQ, FRAG_RATE)
    assert len(changed) == 1 and changed[0] in DATE_FIELDS
    assert b.parse_date(new_row["program_start_date"], PIVOT) <= b.parse_date(new_row["program_end_date"], PIVOT)
    row = {"document_id": "d", "channel": "7", "program_desc": "x",
           "program_start_date": "04/10/20", "program_end_date": "04/10/20", "sub_amount": "$5"}
    pools = {"channel": ["7"], "program_desc": ["x"], "sub_amount": ["$5"],
             "program_start_date": ["04/10/20"], "program_end_date": ["04/10/20"]}
    assert b.make_invalid_row(row, 2, b.LINE_FIELDS, DATE_FIELDS, pools, PIVOT, random.Random(0), EQ, FRAG_RATE) is None


def _truth(main_row, line_rows):
    return b.document_truth([{"document_id": "d", "valid": True, **main_row}],
                            [{"document_id": "d", "valid": True, **r} for r in line_rows], EQ)["d"]


def test_cross_type_candidates_same_kind_cross_dataset_and_exclusions():
    from collections import Counter
    kind_of = b.field_kinds(KINDS)
    main_row = {**{f: None for f in b.MAIN_FIELDS}, "advertiser": "A", "agency": "B", "gross_amount": "$5",
                "flight_from": "02/01/20", "flight_to": "03/01/20"}
    line_rows = [{"line_index": 0, "channel": "9", "program_desc": "x", "program_start_date": "02/01/20",
                  "program_end_date": "02/02/20", "sub_amount": "$9"},
                 {"line_index": 1, "channel": "9", "program_desc": "y", "program_start_date": "02/0",  # fragment
                  "program_end_date": "02/03/20", "sub_amount": None}]
    truth, counts = _truth(main_row, line_rows), Counter()

    def cands(target):
        return b.cross_type_candidates(target, KINDS, kind_of, main_row, line_rows, truth, counts, EQ, None)

    # amount: main gross_amount can take the line sub_amount and vice versa
    assert [(v, s["dataset"], s["field"], s["line_index"]) for v, s in cands("gross_amount")] == [("$9", "line", "sub_amount", 0)]
    assert [(v, s["dataset"], s["field"], s["line_index"]) for v, s in cands("sub_amount")] == [("$5", "main", "gross_amount", None)]
    # name: advertiser <- agency "B" only; a value already valid for the target is excluded
    assert sorted(v for v, _ in cands("advertiser")) == ["B"] and [v for v, _ in cands("agency")] == ["A"]
    # date: the fragment is an ordinary candidate; flight_from's own valid value 02/01/20 is not
    assert sorted(v for v, _ in cands("flight_from")) == ["02/0", "02/02/20", "02/03/20", "03/01/20"]
    assert counts["excluded_valid_for_target"] >= 1
    assert not {"contract_num", "channel", "program_desc"} & set(kind_of)


def test_cross_type_value_valid_for_target_elsewhere_in_document_is_excluded():
    from collections import Counter
    main_row = {**{f: None for f in b.MAIN_FIELDS}, "flight_from": "01/01/20", "flight_to": "02/01/20"}
    line_rows = [{"line_index": 0, "channel": None, "program_desc": None, "program_start_date": "01/01/20",
                  "program_end_date": "01/05/20", "sub_amount": None}]
    got = [(v, s["field"]) for v, s in b.cross_type_candidates(
        "flight_from", KINDS, b.field_kinds(KINDS), main_row, line_rows, _truth(main_row, line_rows), Counter(), EQ, None)]
    assert ("01/01/20", "program_start_date") not in got
    assert ("01/05/20", "program_end_date") in got and ("02/01/20", "flight_to") in got


def test_mark_exchanges_only_for_mutual_pulls_within_the_record():
    def src(dataset, field, line_index=None):
        return {"scope": "same_document", "dataset": dataset, "field": field, "line_index": line_index}
    sources = {"advertiser": src("main", "agency"), "agency": src("main", "advertiser"), "property": src("main", "advertiser")}
    b.mark_exchanges(sources, "main", {})
    assert sources["advertiser"]["exchanged_with"] == "agency" and sources["agency"]["exchanged_with"] == "advertiser"
    assert sources["property"]["exchanged_with"] is None
