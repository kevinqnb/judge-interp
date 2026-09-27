"""Unit + tiny end-to-end tests for scripts/build_vrdu_invalids.py.

The fixture in tests/fixtures/vrdu_invalids_mini/ is a hand-built mini corpus.
Line documents also have a main row (the builder cross-checks every line
document id exists in the main base, even though the new line pipeline no
longer reads main-row values):

  main/train  m1 (agency null), m2, m3 (flight_from is the OCR fragment "05/20/");
              plus the main rows for L1, L2, L3, L5, L6, L7.
  main/test   m4, m5 + the main row for L4.

  line/train  L1  3 items, channel constant "5" (never invalidatable); intra
                  donor count = 2 per row -> intra caps at 2.
              L2  1 item -> 0 intra donors -> intra skipped entirely.
              L3  1 item, fragment start date -> still fully feasible for
                  inter-document (abundant external pool); intra skipped
                  (0 donors) same as L2.
              L5  1 item, both dates OCR fragments -> intra skipped (0 donors).
              L6  6 items, every item's fields entirely distinct from every
                  other item's (no two items share a value) -> intra reaches
                  the full k=5 for every one of its 6 rows, and no (field,
                  donor) pair ever risks reproducing another item verbatim.
              L7  3 items, likewise mutually distinct -> intra caps at 2 (only
                  2 donors) for every row -- the donor-scarcity stop-and-label
                  path, not the duplicate-avoidance path.
  line/test   L4  2 items -> intra caps at 1 each (1 donor); inter-document is
                  fully skipped for both (L4 is the only document with line
                  items in the test split, so the external pool is empty).

These counts are recorded as regression values below; they were derived by
reasoning about the fixture's construction, then confirmed by running the
builder and inspecting the output (not fit to whatever the code happened to
produce).
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
CONFIG = Path(__file__).resolve().parent.parent / "configs" / "2026-09-15-vrdu-invalids-02.yaml"
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


def test_line_inter_pools_excludes_own_document():
    rows = [
        {"document_id": "A", "line_index": 0, "channel": "1", "program_desc": "x",
         "program_start_date": "01/01/20", "program_end_date": "01/05/20", "sub_amount": "$1"},
        {"document_id": "A", "line_index": 1, "channel": "1b", "program_desc": "x2",
         "program_start_date": "01/02/20", "program_end_date": "01/06/20", "sub_amount": "$1b"},
        {"document_id": "B", "line_index": 0, "channel": "2", "program_desc": "y",
         "program_start_date": "02/01/20", "program_end_date": "02/05/20", "sub_amount": "$2"},
    ]
    pools = b.line_inter_pools_by_doc(rows, DATE_FIELDS, PIVOT)
    assert set(pools) == {"A", "B"}
    assert pools["A"]["channel"] == ["2"]              # only B's row, A's own two excluded
    assert pools["B"]["channel"] == ["1", "1b"]
    assert sorted(pools["A"]["program_start_date"]) == sorted(["02/01/20", "02/05/20"])
    assert pools["A"]["program_start_date"] is pools["A"]["program_end_date"]  # shared date pool


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


def test_achievable_targets_filters_to_range_and_excludes_infeasible_field():
    row = {"channel": "1", "program_desc": "d",
           "program_start_date": "01/01/20", "program_end_date": "01/10/20", "sub_amount": "$1"}
    pools = {
        "channel": ["1", "2"], "program_desc": ["d"], "sub_amount": ["$1", "$2"],  # desc has no differing value
        "program_start_date": ["01/01/20", "01/05/20"], "program_end_date": ["01/10/20", "01/15/20"],
    }
    # feasible_nondate = {channel, sub_amount} = 2; both date-alone options and "both" are feasible
    # (see achievable_counts) -> achievable = {0,1,2,3,4}; requesting up to 5 caps at 4.
    targets = b.achievable_targets(row, b.LINE_FIELDS, DATE_FIELDS, pools, PIVOT, (1, 5))
    assert targets == [1, 2, 3, 4]


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


# --- build_intra_chain ---------------------------------------------------


def _l6_style_donors():
    """Five donors, each differing from the base row in ALL five fields, and
    distinct from each other too -- so every donor is eligible for every field
    and reaching the full chain of 5 is guaranteed regardless of draw order."""
    return [
        {"channel": "2", "program_desc": "Show Two", "program_start_date": "05/02/20",
         "program_end_date": "05/11/20", "sub_amount": "$110.00"},
        {"channel": "3", "program_desc": "Show Three", "program_start_date": "05/03/20",
         "program_end_date": "05/12/20", "sub_amount": "$120.00"},
        {"channel": "4", "program_desc": "Show Four", "program_start_date": "05/04/20",
         "program_end_date": "05/13/20", "sub_amount": "$130.00"},
        {"channel": "5", "program_desc": "Show Five", "program_start_date": "05/05/20",
         "program_end_date": "05/14/20", "sub_amount": "$140.00"},
        {"channel": "6", "program_desc": "Show Six", "program_start_date": "05/06/20",
         "program_end_date": "05/15/20", "sub_amount": "$150.00"},
    ]


def test_build_intra_chain_reaches_max_with_fully_distinct_donors():
    base = {"channel": "1", "program_desc": "Base Show",
            "program_start_date": "05/01/20", "program_end_date": "05/10/20", "sub_amount": "$100.00"}
    donors = _l6_style_donors()
    chain = b.build_intra_chain(base, donors, b.LINE_FIELDS, DATE_FIELDS, PIVOT, 5, random.Random(0))
    assert [len(fields) for _row, fields in chain] == [1, 2, 3, 4, 5]
    # nested: each step's fields are a superset of the previous step's
    for i in range(len(chain) - 1):
        assert set(chain[i][1]) < set(chain[i + 1][1])
    final_row, final_fields = chain[-1]
    assert final_fields == sorted(b.LINE_FIELDS)
    for field in b.LINE_FIELDS:
        assert final_row[field] != base[field]
        assert any(final_row[field] == donor[field] for donor in donors)  # came from some donor


def test_build_intra_chain_caps_at_donor_scarcity():
    base = {"channel": "8", "program_desc": "Cap Base",
            "program_start_date": "07/01/20", "program_end_date": "07/10/20", "sub_amount": "$700.00"}
    donors = _l6_style_donors()[:2]  # only 2 distinct donors available
    chain = b.build_intra_chain(base, donors, b.LINE_FIELDS, DATE_FIELDS, PIVOT, 5, random.Random(0))
    assert [len(fields) for _row, fields in chain] == [1, 2]  # stops at 2, not assumed to be 5


def test_build_intra_chain_no_donors_is_empty():
    base = {"channel": "1", "program_desc": "d",
            "program_start_date": "01/01/20", "program_end_date": "01/05/20", "sub_amount": "$1"}
    assert b.build_intra_chain(base, [], b.LINE_FIELDS, DATE_FIELDS, PIVOT, 5, random.Random(0)) == []


def test_build_intra_chain_declines_a_would_be_duplicate_pair_instead_of_raising():
    # The only donor differs from base in exactly one field (channel); borrowing
    # it would reproduce the donor verbatim, so that (field, donor) pair must be
    # excluded rather than aborting the whole chain.
    base = {"channel": "1", "program_desc": "A",
            "program_start_date": "01/01/20", "program_end_date": "01/10/20", "sub_amount": "$1"}
    donor = {"channel": "2", "program_desc": "A",
             "program_start_date": "01/01/20", "program_end_date": "01/10/20", "sub_amount": "$1"}
    chain = b.build_intra_chain(base, [donor], b.LINE_FIELDS, DATE_FIELDS, PIVOT, 5, random.Random(0))
    assert chain == []


def test_build_intra_chain_never_reproduces_a_near_duplicate_donor():
    # donor_a differs from base ONLY in "channel" -- borrowing it while every
    # other field still holds its original value would recreate donor_a
    # verbatim, so that (channel, donor_a) pair must be excluded whenever it
    # would actually produce that duplicate. donor_b is fully distinct and
    # collides with nothing, so progress is still made through it (and,
    # depending on step order, "channel" via donor_a can validly open up too,
    # once some other field has already diverged from donor_a's matching
    # values -- the point of this test is that no *emitted* row ever equals
    # donor_a's or donor_b's tuple, not that the chain length is fixed).
    base = {"channel": "1", "program_desc": "A",
            "program_start_date": "01/01/20", "program_end_date": "01/10/20", "sub_amount": "$1"}
    donor_a = {"channel": "2", "program_desc": "A",
               "program_start_date": "01/01/20", "program_end_date": "01/10/20", "sub_amount": "$1"}
    donor_b = {"channel": "9", "program_desc": "Z",
               "program_start_date": "09/01/20", "program_end_date": "09/10/20", "sub_amount": "$9"}
    chain = b.build_intra_chain(base, [donor_a, donor_b], b.LINE_FIELDS, DATE_FIELDS, PIVOT, 5, random.Random(0))
    assert chain  # progress is still made via donor_b despite donor_a's collision risk
    for row, _fields in chain:
        got = tuple(row[f] for f in b.LINE_FIELDS)
        assert got != tuple(donor_a[f] for f in b.LINE_FIELDS)
        assert got != tuple(donor_b[f] for f in b.LINE_FIELDS)


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


def test_main_dataset_generation_is_unchanged(built):
    # 9 docs in the train main base (m1,m2,m3,L1,L2,L3,L5,L6,L7), each achieving
    # the full k=0..9 (agency etc. always resamplable from the split-wide pool).
    assert len(built["main_train"]) == 9 * 10
    assert len(built["main_test"]) == 3 * 10
    prov = built["provenance"]
    assert prov["skipped_rows"]["main"]["train"] == [] and prov["capped_rows"]["main"]["train"] == []


def test_line_counts_and_provenance(built):
    valid, invalid = _split(built["line_train"])
    assert len(valid) == 15  # L1(3) + L2(1) + L3(1) + L5(1) + L6(6) + L7(3)
    inter = [r for r in invalid if r["error_type"] == "inter_document"]
    intra = [r for r in invalid if r["error_type"] == "intra_document"]
    assert len(inter) == 75   # 15 rows x k=1..5, all achievable via the abundant external pool
    assert len(intra) == 42   # L1: 3x2 + L7: 3x2 + L6: 6x5 = 6+6+30

    valid_t, invalid_t = _split(built["line_test"])
    assert len(valid_t) == 2
    assert sum(1 for r in invalid_t if r["error_type"] == "inter_document") == 0
    assert sum(1 for r in invalid_t if r["error_type"] == "intra_document") == 2

    prov = built["provenance"]
    assert prov["config_id"] == "2026-09-15-vrdu-invalids-02"
    dist = prov["num_invalid_fields_dist"]["line"]["train"]
    assert {int(k): v for k, v in dist["inter_document"].items()} == {1: 15, 2: 15, 3: 15, 4: 15, 5: 15}
    assert {int(k): v for k, v in dist["intra_document"].items()} == {1: 12, 2: 12, 3: 6, 4: 6, 5: 6}


def test_line_skipped_and_capped_rows(built):
    prov = built["provenance"]
    skipped_intra_train = {(r["document_id"], r["line_index"]) for r in prov["skipped_rows"]["line"]["train"]["intra_document"]}
    assert skipped_intra_train == {("L2", 0), ("L3", 0), ("L5", 0)}
    assert prov["skipped_rows"]["line"]["train"]["inter_document"] == []

    # test split: L4 is the only line document, so inter-document has no
    # external pool at all -- both rows skipped.
    skipped_inter_test = {(r["document_id"], r["line_index"]) for r in prov["skipped_rows"]["line"]["test"]["inter_document"]}
    assert skipped_inter_test == {("L4", 0), ("L4", 1)}

    capped_intra_train = {(c["document_id"], c["line_index"]): c["reached"] for c in prov["capped_rows"]["line"]["train"]["intra_document"]}
    assert capped_intra_train == {
        ("L1", 0): 2, ("L1", 1): 2, ("L1", 2): 2,
        ("L7", 0): 2, ("L7", 1): 2, ("L7", 2): 2,
    }
    capped_intra_test = {(c["document_id"], c["line_index"]): c["reached"] for c in prov["capped_rows"]["line"]["test"]["intra_document"]}
    assert capped_intra_test == {("L4", 0): 1, ("L4", 1): 1}


def test_l6_line_index_0_reaches_full_k5_for_both_error_types(built):
    rows = [r for r in built["line_train"] if r["document_id"] == "L6" and r["line_index"] == 0]
    by_type = {(r["error_type"], r["k"]): r for r in rows}
    assert {"inter_document", "intra_document"} == {t for t, _ in by_type if t is not None}
    for error_type in ("inter_document", "intra_document"):
        ks = sorted(k for t, k in by_type if t == error_type)
        assert ks == [1, 2, 3, 4, 5]
        top = by_type[(error_type, 5)]
        assert sorted(top["invalid_fields"]) == sorted(b.LINE_FIELDS)
        for field in b.LINE_FIELDS:
            assert top[field] != next(r for r in rows if r["k"] == 0)[field]


def test_rows_uniquely_keyed_and_k_equals_num_invalid_fields(built):
    for rows in (built["line_train"], built["line_test"]):
        keys = [(r["document_id"], r["line_index"], r["error_type"], r["k"]) for r in rows]
        assert len(keys) == len(set(keys))
        for r in rows:
            assert r["k"] == r["num_invalid_fields"]
    for rows in (built["main_train"], built["main_test"]):
        keys = [(r["document_id"], r["k"]) for r in rows]
        assert len(keys) == len(set(keys))


def test_every_invalid_line_row_differs_in_exactly_its_listed_fields(built):
    base_by_id = {(r["document_id"], r["line_index"]): r
                  for r in json.loads((built["data_root"] / "line/train_base.json").read_text())}
    _, invalid = _split(built["line_train"])
    for row in invalid:
        base = base_by_id[(row["document_id"], row["line_index"])]
        changed = sorted(f for f in b.LINE_FIELDS if row[f] != base[f])
        assert changed == sorted(row["invalid_fields"])
        assert row["k"] == row["num_invalid_fields"] == len(changed) >= 1
        assert all(row[f] is not None for f in changed)


def test_date_order_holds_on_every_touched_line_row(built):
    _, invalid = _split(built["line_train"])
    invalid += _split(built["line_test"])[1]
    for row in invalid:
        if not ({"program_start_date", "program_end_date"} & set(row["invalid_fields"])):
            continue
        a, z = row["program_start_date"], row["program_end_date"]
        if a is None or z is None:
            continue
        assert b.parse_date(a, PIVOT) is not None and b.parse_date(z, PIVOT) is not None
        assert b.parse_date(a, PIVOT) <= b.parse_date(z, PIVOT)


def test_intra_document_values_always_come_from_the_same_document(built):
    line_base = json.loads((built["data_root"] / "line/train_base.json").read_text())
    by_doc = {}
    for r in line_base:
        by_doc.setdefault(r["document_id"], []).append(r)
    _, invalid = _split(built["line_train"])
    for row in (r for r in invalid if r["error_type"] == "intra_document"):
        doc_values = {f: {r[f] for r in by_doc[row["document_id"]]} for f in b.LINE_FIELDS}
        for field in row["invalid_fields"]:
            assert row[field] in doc_values[field]


def test_constant_channel_never_invalidated_intra_document(built):
    # channel is constant ("5") across all of L1's own line items, so
    # intra-document (donors = other L1 rows) can never supply a differing
    # value; inter-document draws from OTHER documents and legitimately can.
    _, invalid = _split(built["line_train"])
    for row in (r for r in invalid if r["document_id"] == "L1" and r["error_type"] == "intra_document"):
        assert "channel" not in row["invalid_fields"] and row["channel"] == "5"


def test_no_intra_document_row_duplicates_another_valid_row_verbatim(built):
    line_base = json.loads((built["data_root"] / "line/train_base.json").read_text())
    by_doc = {}
    for r in line_base:
        by_doc.setdefault(r["document_id"], []).append(r)
    _, invalid = _split(built["line_train"])
    for row in (r for r in invalid if r["error_type"] == "intra_document"):
        full = tuple(row[f] for f in b.LINE_FIELDS)
        others = {tuple(r[f] for f in b.LINE_FIELDS) for r in by_doc[row["document_id"]]
                  if r["line_index"] != row["line_index"]}
        assert full not in others


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
