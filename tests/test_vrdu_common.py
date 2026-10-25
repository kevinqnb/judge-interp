"""Unit tests for data/vrdu/vrdu_common.py: dates, the equivalence key, the label rule and fragments
(moved from the -05 test_build_vrdu_invalids.py with the helpers)."""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "vrdu"))
import vrdu_common as c  # noqa: E402

PIVOT = 50
DATE_FIELDS = ["program_start_date", "program_end_date"]
KIND_OF = {"flight_from": "date", "flight_to": "date", "program_start_date": "date",
           "program_end_date": "date", "gross_amount": "amount", "sub_amount": "amount"}
EQ = c.make_eq(KIND_OF, PIVOT)


@pytest.mark.parametrize("text,expected", [
    ("01/10/20", (2020, 1, 10)), ("3/5/2019", (2019, 3, 5)), ("12/31/99", (1999, 12, 31)),
    ("05/20/", None), ("/12/20", None), ("11-06-2012", None), ("13/01/20", None), (None, None),
])
def test_parse_date(text, expected):
    assert c.parse_date(text, PIVOT) == expected


def test_pair_ok_enforces_order_only_when_both_parse():
    assert c.pair_ok((2020, 1, 1), (2020, 1, 1)) and c.pair_ok((2020, 1, 1), (2020, 2, 1))
    assert not c.pair_ok((2020, 2, 1), (2020, 1, 1))
    assert c.pair_ok(None, (2020, 1, 1)) and c.pair_ok((2020, 2, 1), None) and c.pair_ok(None, None)


@pytest.mark.parametrize("field,a,b_", [
    ("program_start_date", "2/18/20", "02/18/20"), ("program_start_date", "2/18/2020", "02/18/20"),
    ("sub_amount", "$1,234.00", "1234"), ("program_desc", "ISSUE", "Issue"),
    ("program_desc", "Priorities  USA", "priorities usa"), ("program_start_date", "/08/20", " /08/20 "),
])
def test_equivalence_key_treats_reformats_as_the_same_value(field, a, b_):
    assert EQ(field, a) == EQ(field, b_)


@pytest.mark.parametrize("field,a,b_", [
    ("program_start_date", "2/18/20", "2/19/20"), ("program_start_date", "/08/20", "/09/20"),
    ("sub_amount", "$1,234.00", "$1,234.50"),
    ("channel", "5", "05"),       # no kind -> text: an identifier is not read as a number
    ("program_desc", "Show A", "Show B"),
])
def test_equivalence_key_keeps_different_values_apart(field, a, b_):
    assert EQ(field, a) != EQ(field, b_)


def test_equivalence_key_none_is_none():
    assert EQ("sub_amount", None) is None


ITEMS = [
    {"line_index": 0, "channel": "5", "program_desc": "News", "program_start_date": "2/18/20",
     "program_end_date": "02/24/20", "sub_amount": "$100.00"},
    {"line_index": 1, "channel": "5", "program_desc": None, "program_start_date": "02/18/20",
     "program_end_date": "02/24/20", "sub_amount": "$250.00"},
]


@pytest.mark.parametrize("values, expected", [
    ({"channel": "5", "program_desc": "News"}, True),                     # item 0
    ({"channel": "5", "program_start_date": "02/18/20"}, True),           # both items (reformat of item 0's 2/18/20)
    ({"program_desc": "news", "sub_amount": "100"}, True),                # case / amount format folded
    ({"program_desc": "News", "sub_amount": "$250.00"}, False),           # desc of item 0, amount of item 1
    ({"channel": "5", "program_desc": "Sports"}, False),                  # no item has that desc
    ({"program_desc": "News", "program_end_date": "02/24/20", "sub_amount": "$250.00"}, False),
    ({"channel": "5", "sub_amount": "$250.00"}, True),                    # item 1; its null desc is not shown
])
def test_matches_item(values, expected):
    assert c.matches_item(values, ITEMS, EQ) is expected
    keys = c.item_keys(ITEMS, c.LINE_FIELDS, EQ)
    assert c.matches_keys({f: EQ(f, v) for f, v in values.items()}, keys) is expected


def test_matches_item_an_item_null_matches_no_value():
    assert not c.matches_item({"program_desc": "Anything", "sub_amount": "$250.00"}, ITEMS, EQ)


def test_matches_item_refuses_a_null_or_empty_shown_value():
    with pytest.raises(AssertionError):
        c.matches_item({"channel": None}, ITEMS, EQ)
    with pytest.raises(AssertionError):
        c.matches_item({}, ITEMS, EQ)


def test_nonnull_values_drops_nulls_only():
    assert c.nonnull_values(ITEMS[1], c.LINE_FIELDS) == {
        "channel": "5", "program_start_date": "02/18/20", "program_end_date": "02/24/20", "sub_amount": "$250.00"}


def test_pick_fragment_neutral_draws_fragments_at_the_target_rate_not_the_eligible_rate():
    # 1 real + 2 fragments eligible: a uniform pick would take a fragment 2/3 of the time; the class is
    # drawn at the target rate (0.2) instead.
    eligible = ["r0", "/01/20", "/02/20"]
    frag = lambda v: v.startswith("/")
    rng = random.Random(0)
    n = 3000
    hits = sum(frag(c.pick_fragment_neutral(eligible, 0.2, frag, rng)) for _ in range(n))
    assert 0.15 < hits / n < 0.25      # expected 0.20, sd ~0.007


def test_pick_fragment_neutral_forces_the_only_available_class():
    frag = lambda v: v.startswith("/")
    rng = random.Random(0)
    assert {c.pick_fragment_neutral(["/01/20"], 0.2, frag, rng) for _ in range(20)} == {"/01/20"}
    assert {c.pick_fragment_neutral(["r1"], 0.9, frag, rng) for _ in range(20)} == {"r1"}
    with pytest.raises(AssertionError):
        c.pick_fragment_neutral([], 0.2, frag, rng)
    with pytest.raises(AssertionError):
        c.pick_fragment_neutral(["r1"], 1.5, frag, rng)


def test_fragment_rate_is_the_share_of_non_null_date_values_that_do_not_parse():
    rows = [{"program_start_date": "01/01/20", "program_end_date": "/05/20"},
            {"program_start_date": "02/01/", "program_end_date": None},
            {"program_start_date": "03/01/20", "program_end_date": "03/09/20"}]
    assert c.fragment_rate(rows, DATE_FIELDS, PIVOT) == 2 / 5     # "/05/20" and "02/01/" of 5 values
    with pytest.raises(AssertionError):
        c.fragment_rate([{"program_start_date": None, "program_end_date": None}], DATE_FIELDS, PIVOT)


def test_field_kinds_inverts_and_refuses_a_field_in_two_kinds():
    assert c.field_kinds({"date": ["flight_from"], "amount": ["sub_amount"]}) == {"flight_from": "date", "sub_amount": "amount"}
    with pytest.raises(AssertionError):
        c.field_kinds({"date": ["flight_from"], "amount": ["flight_from"]})
    with pytest.raises(AssertionError):
        c.field_kinds({"date": ["nope"]})
