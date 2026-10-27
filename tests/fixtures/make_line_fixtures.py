"""Write the hand-built line fixtures (``*/vrdu/line/train.json``) in the -07 row format.

Each fixture is a list of hand-picked minimal pairs, each over one item's full set of non-null fields in
``LINE_FIELDS`` order, one pair per item. A pair is built with ``build_vrdu_line.make_pair`` (the builder's
own row format) and the whole file is checked with ``build_vrdu_line.verify_split`` against the listed line
items, so every row is a correct sample of its cell.

    uv run --python 3.12 python tests/fixtures/make_line_fixtures.py

relation_detection_mini and entity_detection_mini (same rows):
    doc-a li0 = {7, Movie, 01/05/20, 01/10/20, $100.00}; li1 = {8, null, 01/11/20, 01/20/20, $250.00};
          li2 = {8, News, 01/21/20, 01/31/20, $400.00}
    doc-b li0 = {9, Weather, 02/02/20, 02/09/20, $300.00}
    intra_document:k2  doc-a li0, all five fields; end date <- li1, sub_amount <- li2
    cross_item:k2      doc-a li2, all five fields; channel <-> sub_amount
    intra_document:k1  doc-a li1, its four non-null fields; channel <- li0
    cross_item:k2      doc-b li0, all five fields; program_desc <-> channel
judge_interp_mini:
    doc-beta li0 = {7, Evening News, 02/10/20, 02/17/20, null}
    cross_item:k2      the four non-null fields; channel <-> program_desc
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data" / "vrdu"))
import build_vrdu_line as b  # noqa: E402

FIXTURES = Path(__file__).resolve().parent
KINDS = {"name": ["advertiser", "agency", "property", "tv_address", "product"],
         "date": ["flight_from", "flight_to", "program_start_date", "program_end_date"],
         "amount": ["gross_amount", "sub_amount"]}
DATES, PIVOT = ("program_start_date", "program_end_date"), 50


def item(doc, li, channel, desc, start, end, amount):
    return {"document_id": doc, "line_index": li, "valid": True, "channel": channel, "program_desc": desc,
            "program_start_date": start, "program_end_date": end, "sub_amount": amount}


def intra(it, fields, donors, by_id):
    """Pair of ``it`` over ``fields`` with each field in ``donors`` taken from that donor item."""
    values = {f: by_id[(it["document_id"], donors[f])][f] if f in donors else it[f] for f in fields}
    errors = {f: {"error_type": b.INTRA_DOCUMENT, "source": b.line_source(f, donors[f], None)} if f in donors else None
              for f in fields}
    return b.make_pair({"item": it, "fields": fields, "values": values, "field_errors": errors},
                       (b.INTRA_DOCUMENT, len(donors)))


def swap(it, fields, a, c):
    """Pair of ``it`` over ``fields`` with fields ``a`` and ``c`` swapped."""
    values = {f: it[c] if f == a else it[a] if f == c else it[f] for f in fields}
    li = it["line_index"]
    errors = {f: {"error_type": b.CROSS_ITEM, "source": b.line_source(c if f == a else a, li, c if f == a else a)}
              if f in (a, c) else None for f in fields}
    return b.make_pair({"item": it, "fields": fields, "values": values, "field_errors": errors}, (b.CROSS_ITEM, 2))


def write(name, items, pairs):
    eq = b.make_eq(b.field_kinds(KINDS), PIVOT)
    rows = [r for p in pairs for r in p]
    for cell in sorted({(r["error_type"], r["k"]) for r in rows if not r["valid"]}, key=b.cell_id):   # a cell may hold several pairs
        in_cell = [r for r in rows if r["cell"] == b.cell_id(cell)]
        b.verify_split(in_cell, items, [cell], [2, 5], len(in_cell) // 2, eq, DATES, PIVOT)
    path = FIXTURES / name / "vrdu" / "line" / "train.json"
    path.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    print(f"{path.relative_to(ROOT)}: {len(rows)} rows")


def main():
    items = [item("doc-a", 0, "7", "Movie", "01/05/20", "01/10/20", "$100.00"),
             item("doc-a", 1, "8", None, "01/11/20", "01/20/20", "$250.00"),
             item("doc-a", 2, "8", "News", "01/21/20", "01/31/20", "$400.00"),
             item("doc-b", 0, "9", "Weather", "02/02/20", "02/09/20", "$300.00")]
    by_id = {(i["document_id"], i["line_index"]): i for i in items}
    a0, a1, a2, b0 = items
    pairs = [intra(a0, list(b.LINE_FIELDS), {"program_end_date": 1, "sub_amount": 2}, by_id),
             swap(a2, list(b.LINE_FIELDS), "channel", "sub_amount"),
             intra(a1, [f for f in b.LINE_FIELDS if a1[f] is not None], {"channel": 0}, by_id),
             swap(b0, list(b.LINE_FIELDS), "program_desc", "channel")]
    for name in ("relation_detection_mini", "entity_detection_mini"):
        write(name, items, pairs)

    beta = item("doc-beta", 0, "7", "Evening News", "02/10/20", "02/17/20", None)
    write("judge_interp_mini", [beta], [swap(beta, [f for f in b.LINE_FIELDS if beta[f] is not None], "channel", "program_desc")])


if __name__ == "__main__":
    main()
