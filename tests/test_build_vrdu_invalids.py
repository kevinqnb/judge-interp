"""Unit + tiny end-to-end tests for scripts/build_vrdu_invalids.py.

The fixture in tests/fixtures/vrdu_invalids_mini/ is a hand-built mini corpus
whose pools can be enumerated by inspection. Line documents also have a main row
(the builder joins them for the ``flight_from`` / ``flight_to`` page dates):

  main/train  m1 (agency null), m2, m3 (flight_from is the OCR fragment "05/20/");
              plus the main rows for L1, L2, L3, L5.
  main/test   m4, m5 + the main row for L4.
  line/train  L1  channel constant "5" across 3 items (never invalidatable);
              L2  one item, its 2 page dates cannot be reordered -> only one
                  date field is ever changed, k caps to 1;
              L3  one item, fragment start date -> only the start can change,
                  and only to the single other page date;
              L5  one item, both dates OCR fragments and no other page date ->
                  nothing to change, row skipped.
  line/test   L4  two items, everything invalidatable.
"""

import json
import random
import shutil
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_vrdu_invalids as b

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "vrdu_invalids_mini"
CONFIG = Path(__file__).resolve().parent.parent / "configs" / "2026-09-10-vrdu-invalids-01.yaml"
PIVOT = 50
DATE_FIELDS = ["program_start_date", "program_end_date"]


# --- parse_date ------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("01/10/20", (2020, 1, 10)),
        ("3/5/2019", (2019, 3, 5)),
        ("12/31/99", (1999, 12, 31)),
        ("05/20/", None),
        ("/12/20", None),
        ("11-06-2012", None),
        ("13/01/20", None),
        (None, None),
    ],
)
def test_parse_date(text, expected):
    assert b.parse_date(text, PIVOT) == expected


# --- pools ---------------------------------------------------------------


def test_main_pools_dates_drop_fragments_add_sentinels_others_drop_nulls():
    rows = [
        {"advertiser": "a", "agency": None, "property": None, "tv_address": None, "product": None,
         "contract_num": None, "flight_from": "01/10/20", "flight_to": None, "gross_amount": None},
        {"advertiser": "a", "agency": None, "property": None, "tv_address": None, "product": None,
         "contract_num": None, "flight_from": "05/20/", "flight_to": "02/02/20", "gross_amount": None},
    ]
    pools = b.main_pools(rows, ["01/01/00", "12/31/49"], ["flight_from", "flight_to"], PIVOT)
    assert pools["advertiser"] == ["a", "a"]
    assert pools["agency"] == []
    assert sorted(pools["flight_from"]) == sorted(["01/10/20", "01/01/00", "12/31/49"])
    assert sorted(pools["flight_to"]) == sorted(["02/02/20", "01/01/00", "12/31/49"])


def test_line_pools_dates_are_one_shared_page_pool():
    doc_rows = [
        {"channel": "5", "program_desc": "A", "program_start_date": "01/01/20",
         "program_end_date": "01/07/20", "sub_amount": None},
        {"channel": "5", "program_desc": "B", "program_start_date": "01/08/20",
         "program_end_date": "01/07/20", "sub_amount": "$9"},
    ]
    main_row = {"flight_from": "12/20/19", "flight_to": "02/28/"}  # to-date is a fragment, dropped
    pools = b.line_pools(doc_rows, main_row, ["flight_from", "flight_to"], DATE_FIELDS, PIVOT)
    assert pools["program_start_date"] is pools["program_end_date"]
    assert sorted(set(pools["program_start_date"])) == sorted(["01/01/20", "01/07/20", "01/08/20", "12/20/19"])
    assert pools["channel"] == ["5", "5"]
    assert pools["sub_amount"] == ["$9"]


# --- feasibility -------------------------------------------------------


def test_date_feasibility_ordered_pair_impossible_to_reorder():
    # page dates {02/01, 02/05}; row is start=02/01 end=02/05.
    row = {"program_start_date": "02/01/20", "program_end_date": "02/05/20"}
    pools = {"program_start_date": ["02/01/20", "02/05/20"], "program_end_date": ["02/01/20", "02/05/20"]}
    start_alone, end_alone, both = b._date_feasibility(row, DATE_FIELDS, pools, PIVOT)
    assert (start_alone, end_alone, both) == (True, True, False)  # each alone works; together cannot


def test_date_feasibility_fragment_start_blocks_end_alone():
    row = {"program_start_date": "03/01/", "program_end_date": "03/31/20"}
    pools = {"program_start_date": ["03/31/20"], "program_end_date": ["03/31/20"]}
    assert b._date_feasibility(row, DATE_FIELDS, pools, PIVOT) == (True, False, False)


def test_nondate_invalidatable_excludes_constant_field():
    rows = [{"channel": "5", "program_desc": "A", "sub_amount": None},
            {"channel": "5", "program_desc": "B", "sub_amount": "$9"}]
    pools = {f: [r[f] for r in rows if r[f] is not None] for f in ["channel", "program_desc", "sub_amount"]}
    got = b._nondate_invalidatable(rows[0], b.LINE_FIELDS, DATE_FIELDS, pools)
    assert "channel" not in got and {"program_desc", "sub_amount"} <= set(got)


def test_achievable_counts():
    assert b.achievable_counts(2, (True, True, True)) == {0, 1, 2, 3, 4}
    assert b.achievable_counts(0, (True, True, False)) == {0, 1}
    assert b.achievable_counts(1, (False, False, False)) == {0, 1}


# --- corruption units --------------------------------------------------


def test_sample_new_returns_a_differing_value_and_raises_without_one():
    assert b.sample_new(["x", "x", "y"], "x", random.Random(0)) == "y"
    with pytest.raises(AssertionError):
        b.sample_new(["x", "x"], "x", random.Random(0))


def test_make_invalid_row_caps_when_only_one_date_field_can_change():
    row = {"document_id": "d", "channel": "9", "program_desc": "Solo",
           "program_start_date": "02/01/20", "program_end_date": "02/05/20", "sub_amount": "$9"}
    shared = ["02/01/20", "02/05/20"]
    pools = {"channel": ["9"], "program_desc": ["Solo"], "sub_amount": ["$9"],
             "program_start_date": shared, "program_end_date": shared}
    new_row, changed = b.make_invalid_row(row, 3, b.LINE_FIELDS, DATE_FIELDS, pools, PIVOT, random.Random(1))
    assert len(changed) == 1 and changed[0] in DATE_FIELDS
    assert b.parse_date(new_row["program_start_date"], PIVOT) <= b.parse_date(new_row["program_end_date"], PIVOT)


def test_make_invalid_row_returns_none_when_nothing_feasible():
    row = {"document_id": "d", "channel": "7", "program_desc": "x",
           "program_start_date": "04/10/20", "program_end_date": "04/10/20", "sub_amount": "$5"}
    pools = {"channel": ["7"], "program_desc": ["x"], "sub_amount": ["$5"],
             "program_start_date": ["04/10/20"], "program_end_date": ["04/10/20"]}
    assert b.make_invalid_row(row, 2, b.LINE_FIELDS, DATE_FIELDS, pools, PIVOT, random.Random(0)) is None


# --- tiny end-to-end -------------------------------------------------


@pytest.fixture
def built(tmp_path):
    data_root = tmp_path / "vrdu"
    shutil.copytree(FIXTURE, data_root)
    config = yaml.safe_load(CONFIG.read_text())
    config["params"]["data_root"] = str(data_root)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    b.main(["--config", str(cfg_path)])

    def load(rel):
        return json.loads((data_root / rel).read_text())

    return {
        "main_train": load("main/train.json"), "main_test": load("main/test.json"),
        "line_train": load("line/train.json"), "line_test": load("line/test.json"),
        "provenance": load("invalids.json")["build"],
        "data_root": data_root, "cfg_path": cfg_path,
    }


def _split(rows):
    return [r for r in rows if r["valid"]], [r for r in rows if not r["valid"]]


ALL = [
    ("main_train", ["document_id"], b.MAIN_FIELDS, ["flight_from", "flight_to"]),
    ("main_test", ["document_id"], b.MAIN_FIELDS, ["flight_from", "flight_to"]),
    ("line_train", ["document_id", "line_index"], [*b.LINE_CARRIED_FIELDS, *b.LINE_FIELDS], DATE_FIELDS),
    ("line_test", ["document_id", "line_index"], [*b.LINE_CARRIED_FIELDS, *b.LINE_FIELDS], DATE_FIELDS),
]


def test_counts_and_provenance(built):
    assert len(built["main_train"]) == 7 * 10
    assert len(built["main_test"]) == 3 * 10
    assert len(built["line_train"]) == 6 + (3 * 4 + 4 + 4)  # L1 3 items, L2, L3; L5 skipped
    assert len(built["line_test"]) == 2 + 2 * 4

    prov = built["provenance"]
    assert prov["skipped_rows"]["line"]["train"] == [{"document_id": "L5", "line_index": 0}]
    assert prov["skipped_rows"]["main"]["train"] == [] and prov["capped_rows"]["main"]["train"] == []
    line_capped = {(c["document_id"], c["k"], c["realized"]) for c in prov["capped_rows"]["line"]["train"]}
    assert line_capped == {("L2", 2, 1), ("L2", 3, 1), ("L2", 4, 1),
                           ("L3", 2, 1), ("L3", 3, 1), ("L3", 4, 1)}
    assert prov["line_date_pool_sizes"]["train"] == {
        "min": 0, "median": 1.5, "max": 5, "docs_with_no_page_date": 1, "line_rows_with_no_page_date": 1
    }
    assert prov["invalid_field_freq"]["line"]["train"]["channel"] == 0  # constant in L1, single elsewhere
    assert prov["python"] and prov["config_id"] == "2026-09-10-vrdu-invalids-01"


def test_every_invalid_row_differs_in_exactly_its_listed_fields(built):
    for key, id_keys, fields, _ in ALL:
        base_name = key.replace("_train", "/train_base.json").replace("_test", "/test_base.json")
        base_by_id = {tuple(r[k] for k in id_keys): r
                      for r in json.loads((built["data_root"] / base_name).read_text())}
        valid, invalid = _split(built[key])
        assert len(valid) == len(base_by_id)
        for row in invalid:
            base = base_by_id[tuple(row[k] for k in id_keys)]
            changed = sorted(f for f in fields if row[f] != base[f])
            assert changed == sorted(row["invalid_fields"])
            assert 1 <= row["num_invalid_fields"] == len(changed) <= row["k"]
            assert all(row[f] is not None for f in changed)


def test_rows_uniquely_keyed_by_id_and_k(built):
    for key, id_keys, _, _ in ALL:
        ids = [(*[r[k] for k in id_keys], r["k"]) for r in built[key]]
        assert len(ids) == len(set(ids))


def test_date_order_holds_on_every_touched_row(built):
    for key, id_keys, _, date_fields in ALL:
        _, invalid = _split(built[key])
        for row in invalid:
            if not ({date_fields[0], date_fields[1]} & set(row["invalid_fields"])):
                continue
            a, z = row[date_fields[0]], row[date_fields[1]]
            if a is None or z is None:
                continue
            assert b.parse_date(a, PIVOT) is not None and b.parse_date(z, PIVOT) is not None
            assert b.parse_date(a, PIVOT) <= b.parse_date(z, PIVOT)


def test_line_date_corruptions_are_real_page_dates(built):
    """No sentinels on the line side: every corrupted line date already appears on the page."""
    for key in ("line_train", "line_test"):
        base_name = key.replace("_train", "/train_base.json").replace("_test", "/test_base.json")
        line_base = json.loads((built["data_root"] / base_name).read_text())
        main_name = key.replace("line", "main").replace("_train", "/train_base.json").replace("_test", "/test_base.json")
        main_by_id = {r["document_id"]: r for r in json.loads((built["data_root"] / main_name).read_text())}
        by_doc = {}
        for r in line_base:
            by_doc.setdefault(r["document_id"], []).append(r)
        _, invalid = _split(built[key])
        for row in invalid:
            page = {r[f] for r in by_doc[row["document_id"]] for f in DATE_FIELDS if r[f]}
            page |= {main_by_id[row["document_id"]][f] for f in ("flight_from", "flight_to")
                     if main_by_id[row["document_id"]][f]}
            for f in DATE_FIELDS:
                if f in row["invalid_fields"]:
                    assert row[f] in page, (row["document_id"], row["line_index"], f, row[f])
            assert "01/01/00" not in (row[DATE_FIELDS[0]], row[DATE_FIELDS[1]])


def test_forced_single_candidate_cases(built):
    _, line_inv = _split(built["line_train"])
    # L1 li0/li2 end on 01/07/20; the only page date <= that (and != the start) is 01/07/20.
    for row in line_inv:
        if row["document_id"] == "L1" and row["line_index"] in (0, 2) and row["invalid_fields"] == ["program_start_date"]:
            assert row["program_start_date"] == "01/07/20"
    # L3: start can only become the single other page date.
    l3 = [r for r in line_inv if r["document_id"] == "L3"]
    assert len(l3) == 4
    for row in l3:
        assert row["invalid_fields"] == ["program_start_date"] and row["program_start_date"] == "03/31/20"
    # L2: start-only -> 02/05/20, end-only -> 02/01/20 (each the sole candidate).
    for row in (r for r in line_inv if r["document_id"] == "L2"):
        assert row["num_invalid_fields"] == 1
        if row["invalid_fields"] == ["program_start_date"]:
            assert row["program_start_date"] == "02/05/20"
        else:
            assert row["invalid_fields"] == ["program_end_date"] and row["program_end_date"] == "02/01/20"


def test_constant_channel_never_invalidated(built):
    _, line_inv = _split(built["line_train"])
    for row in (r for r in line_inv if r["document_id"] == "L1"):
        assert "channel" not in row["invalid_fields"] and row["channel"] == "5"


def test_main_test_corruptions_stay_within_the_test_split_pool(built):
    train_base = json.loads((built["data_root"] / "main/train_base.json").read_text())
    test_base = json.loads((built["data_root"] / "main/test_base.json").read_text())
    _, invalid = _split(built["main_test"])
    for field in ("advertiser", "property", "product"):
        train_vals = {r[field] for r in train_base if r[field] is not None}
        test_vals = {r[field] for r in test_base if r[field] is not None}
        for row in invalid:
            if field in row["invalid_fields"]:
                assert row[field] in test_vals and row[field] not in (train_vals - test_vals)


def test_fragment_flight_from_row_keeps_the_pair_verifiable(built):
    _, invalid = _split(built["main_train"])
    for row in (r for r in invalid if r["document_id"] == "m3"):
        # flight_to is only ever changed together with the fragment flight_from
        if "flight_to" in row["invalid_fields"]:
            assert "flight_from" in row["invalid_fields"]
        if "flight_from" not in row["invalid_fields"]:
            assert row["flight_from"] == "05/20/"  # fragment retained untouched


def test_num_invalid_fields_distribution(built):
    dist = built["provenance"]["num_invalid_fields_dist"]
    as_ints = lambda d: {int(k): v for k, v in d.items()}
    assert as_ints(dist["main"]["train"]) == {i: 7 for i in range(10)}
    assert as_ints(dist["main"]["test"]) == {i: 3 for i in range(10)}
    assert as_ints(dist["line"]["train"]) == {0: 6, 1: 11, 2: 3, 3: 3, 4: 3}
    assert as_ints(dist["line"]["test"]) == {0: 2, 1: 2, 2: 2, 3: 2, 4: 2}


def test_seed_determinism(built, tmp_path):
    other = tmp_path / "vrdu2"
    shutil.copytree(FIXTURE, other)
    config = yaml.safe_load(built["cfg_path"].read_text())
    config["params"]["data_root"] = str(other)
    cfg2 = tmp_path / "cfg2.yaml"
    cfg2.write_text(yaml.safe_dump(config))
    b.main(["--config", str(cfg2)])
    for name in ["main/train.json", "main/test.json", "line/train.json", "line/test.json"]:
        assert (built["data_root"] / name).read_bytes() == (other / name).read_bytes()
