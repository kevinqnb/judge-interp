"""Unit + tiny end-to-end tests for scripts/build_vrdu_dataset.py.

The fixture in tests/fixtures/vrdu_mini/ is a hand-built 3-document corpus whose
expected output can be checked by inspection:

  aaa  clean main fields; 2 line items, the two annotation entries listing their
       fields in different orders, and the second item positioned earlier in the
       text so line_index has to reorder them.
  bbb  no line items; a `property` conflict resolved by majority ("X-TV" x2 vs
       "X" x1) and a `flight_from` conflict that is a tie, resolved to the
       earliest occurrence ("05/01/" at offset 100, not "05/01/20" at 200).
  ccc  only `advertiser` and a single line item carrying only program_desc.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_vrdu_dataset as b  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "vrdu_mini"


def occ(text, start, end):
    return [text, [0, 0, 0, 0, 0], [[start, end]]]


# --- select_main_value -------------------------------------------------------


def test_select_main_value_unanimous():
    value, conflict = b.select_main_value([occ("KAAA-TV\n", 5, 12), occ("KAAA-TV\n", 40, 47)])
    assert value == "KAAA-TV"
    assert conflict is None


def test_select_main_value_majority_wins():
    value, conflict = b.select_main_value(
        [occ("X-TV\n", 10, 14), occ("X-TV\n", 50, 54), occ("X\n", 90, 91)]
    )
    assert value == "X-TV"
    assert conflict == {
        "chosen": "X-TV",
        "occurrences": [{"text": "X-TV", "count": 2}, {"text": "X", "count": 1}],
    }


def test_select_main_value_tie_breaks_to_earliest():
    value, conflict = b.select_main_value([occ("05/01/20\n", 200, 208), occ("05/01/", 100, 106)])
    assert value == "05/01/"
    assert conflict["chosen"] == "05/01/"
    # distinct occurrences ordered by first appearance (earliest offset first)
    assert [o["text"] for o in conflict["occurrences"]] == ["05/01/", "05/01/20"]


def test_empty_after_strip_is_loud():
    with pytest.raises(AssertionError):
        b.select_main_value([occ("   \n", 1, 5)])


# --- extract_line_items ----------------------------------------------------


def test_line_items_reordered_by_offset_and_key_maps_per_entry():
    record = _record("aaa")
    items = b.extract_line_items(record["annotations"])
    assert len(items) == 2

    # "Evening News" entry sits at offset 40, so it comes first.
    assert items[0] == {
        "channel": "7",
        "program_desc": "Evening News",
        "program_start_date": "03/01/20",
        "program_end_date": None,
        "sub_amount": None,
    }
    assert items[1] == {
        "channel": "9",
        "program_desc": "Morning News",
        "program_start_date": None,
        "program_end_date": None,
        "sub_amount": None,
    }


def test_line_items_empty_when_none_annotated():
    assert b.extract_line_items(_record("bbb")["annotations"]) == []


# --- extract_main_fields ---------------------------------------------------


def test_main_fields_absent_are_null_but_present():
    values, conflicts = b.extract_main_fields(_record("ccc")["annotations"])
    assert set(values) == set(b.MAIN_FIELDS)
    assert values["advertiser"] == "Cee LLC"
    assert all(values[f] is None for f in b.MAIN_FIELDS if f != "advertiser")
    assert conflicts == {}


def test_main_fields_conflicts_recorded():
    values, conflicts = b.extract_main_fields(_record("bbb")["annotations"])
    assert values["property"] == "X-TV"
    assert values["flight_from"] == "05/01/"
    assert set(conflicts) == {"property", "flight_from"}


# --- assign_splits -------------------------------------------------------


def test_assign_splits_is_deterministic_and_a_partition():
    ids = [f"doc{i:03d}" for i in range(50)]
    a = b.assign_splits(ids, test_frac=0.2, seed=0)
    c = b.assign_splits(list(reversed(ids)), test_frac=0.2, seed=0)
    assert a == c  # independent of input order
    assert set(a) == set(ids)
    assert sum(v == "test" for v in a.values()) == 10
    assert b.assign_splits(ids, test_frac=0.2, seed=1) != a


# --- tiny end-to-end -----------------------------------------------------


def test_end_to_end(tmp_path):
    out = tmp_path / "vrdu"
    b.main(
        [
            "--dataset-jsonl", str(FIXTURE / "dataset.jsonl"),
            "--pdf-dir", str(FIXTURE / "pdfs"),
            "--out-dir", str(out),
            "--test-frac", "0.34",
            "--seed", "0",
            "--skip-known-answer-checks",
        ]
    )

    assert sorted(p.name for p in (out / "ocr").glob("*.txt")) == ["aaa.txt", "bbb.txt", "ccc.txt"]
    assert (out / "ocr" / "aaa.txt").read_text(encoding="utf-8") == "DOC AAA OCR TEXT\n"

    directory = json.loads((out / "directory.json").read_text(encoding="utf-8"))
    assert directory["build"]["n_docs"] == 3
    assert directory["build"]["n_train"] + directory["build"]["n_test"] == 3
    assert len(directory["build"]["git_sha"]) == 40
    assert isinstance(directory["build"]["git_dirty"], bool)
    by_id = {d["document_id"]: d for d in directory["documents"]}
    assert by_id["aaa"]["n_line_items"] == 2
    assert by_id["bbb"]["n_line_items"] == 0
    assert set(by_id["bbb"]["value_conflicts"]) == {"property", "flight_from"}
    assert by_id["ccc"]["main_fields_present"] == ["advertiser"]

    main_rows = json.loads((out / "main" / "train_base.json").read_text(encoding="utf-8")) + json.loads(
        (out / "main" / "test_base.json").read_text(encoding="utf-8")
    )
    assert len(main_rows) == 3
    assert all(r["valid"] is True for r in main_rows)
    assert all(set(r) == {"document_id", "valid", *b.MAIN_FIELDS} for r in main_rows)
    bbb = next(r for r in main_rows if r["document_id"] == "bbb")
    assert bbb["property"] == "X-TV"
    assert bbb["agency"] is None

    line_rows = json.loads((out / "line" / "train_base.json").read_text(encoding="utf-8")) + json.loads(
        (out / "line" / "test_base.json").read_text(encoding="utf-8")
    )
    assert len(line_rows) == 3
    assert all(set(r) == {"document_id", "line_index", "valid", *b.LINE_CARRIED_FIELDS, *b.LINE_FIELDS} for r in line_rows)
    ccc_line = next(r for r in line_rows if r["document_id"] == "ccc")
    assert ccc_line["line_index"] == 0
    assert ccc_line["advertiser"] == "Cee LLC"  # carried document-level context
    assert ccc_line["property"] is None
    assert ccc_line["program_desc"] == "Late Show"
    assert ccc_line["channel"] is None

    aaa_lines = sorted((r for r in line_rows if r["document_id"] == "aaa"), key=lambda r: r["line_index"])
    assert [r["program_desc"] for r in aaa_lines] == ["Evening News", "Morning News"]


# --- helpers -----------------------------------------------------------


def _record(doc_id):
    for line in (FIXTURE / "dataset.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["filename"] == f"{doc_id}.pdf":
            return record
    raise KeyError(doc_id)
