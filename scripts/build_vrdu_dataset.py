"""Organize the VRDU ad-buy-form corpus into this repo's dataset layout.

Reads the upstream ``dataset.jsonl`` (one row per document: OCR text plus
bounding-box annotations) and writes, under ``--out-dir``:

    directory.json            split assignment + per-document metadata
    ocr/<document_id>.txt      raw OCR text, one file per document
    main/{train,test}_base.json   one datapoint per document (9 document-level fields)
    line/{train,test}_base.json   one datapoint per line item (6 carried fields + 5 line fields)

Every datapoint written here is a *correct* extraction, so it gets ``valid: true``.
Invalid datapoints (deliberately corrupted values) are added by later steps, not here.

The VRDU annotation model records each field at *every* bounding box it appears in.
For a document-level field those occurrences are meant to be the same value; when the
OCR disagrees between boxes we pick by majority vote over the stripped strings, with
the earliest occurrence in the document breaking ties, and record every distinct
occurrence in ``directory.json`` so the choice stays reversible.

No cluster job: this is local data preparation. Run it directly.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# --- Fixed VRDU ad-buy-form schema (not run-varying; see meta.json in the corpus) ---

# Document-level fields: at most one value per document.
MAIN_FIELDS = [
    "advertiser",
    "agency",
    "property",
    "tv_address",
    "product",
    "contract_num",
    "flight_from",
    "flight_to",
    "gross_amount",
]

# Line-item fields: repeated, one value per line item.
LINE_FIELDS = [
    "channel",
    "program_desc",
    "program_start_date",
    "program_end_date",
    "sub_amount",
]

# Document-level fields copied onto every line-item datapoint for context.
LINE_CARRIED_FIELDS = [
    "advertiser",
    "agency",
    "property",
    "tv_address",
    "product",
    "contract_num",
]

# Known-answer checks (rung 3): fixed properties of the upstream corpus. If the
# upstream file ever changes, these fire and the run stops rather than writing a
# quietly different dataset.
EXPECTED_N_DOCS = 641
EXPECTED_N_LINE_ITEMS = 9163
EXPECTED_DOCS_WITHOUT_LINE_ITEMS = 5
EXPECTED_MAIN_NONNULL = {
    "advertiser": 635,
    "agency": 283,
    "property": 595,
    "tv_address": 535,
    "product": 607,
    "contract_num": 624,
    "flight_from": 540,
    "flight_to": 538,
    "gross_amount": 629,
}
EXPECTED_LINE_NONNULL = {
    "channel": 7262,
    "program_desc": 8164,
    "program_start_date": 8482,
    "program_end_date": 7460,
    "sub_amount": 6836,
}


# --- Annotation parsing ---------------------------------------------------------
#
# annotations: list of [key, value] pairs.
#   key is a str          -> document-level field; value is a list of occurrences.
#   key is a list of str  -> one line-item group; value is a list of instances,
#                            each instance a list of occurrences aligned to key.
#   occurrence = [text, bbox, segments]; segments = [[start, end], ...] into the
#   OCR text. We order occurrences by their earliest segment start.


def occurrence_offset(occurrence):
    """Earliest OCR character offset an occurrence points at."""
    _text, _bbox, segments = occurrence
    assert segments, f"occurrence has no segments: {occurrence!r}"
    return min(start for start, _end in segments)


def _clean(text):
    """Strip surrounding whitespace only; internal newlines (e.g. in tv_address) stay."""
    stripped = text.strip()
    assert stripped, f"annotation value is empty after stripping: {text!r}"
    return stripped


def select_main_value(occurrences):
    """Pick the value for one document-level field from its occurrences.

    Returns ``(value, conflict)`` where ``conflict`` is None when every occurrence
    agrees, otherwise a dict recording the chosen value and all distinct
    occurrences (ordered by first appearance, with counts).
    """
    assert occurrences, "select_main_value called with no occurrences"
    entries = [(_clean(occ[0]), occurrence_offset(occ)) for occ in occurrences]

    earliest_offset = {}
    for text, offset in entries:
        if text not in earliest_offset or offset < earliest_offset[text]:
            earliest_offset[text] = offset
    counts = Counter(text for text, _offset in entries)

    if len(counts) == 1:
        return entries[0][0], None

    top = max(counts.values())
    chosen = min(
        (text for text, count in counts.items() if count == top),
        key=lambda text: earliest_offset[text],
    )
    distinct_by_appearance = sorted(counts, key=lambda text: earliest_offset[text])
    conflict = {
        "chosen": chosen,
        "occurrences": [{"text": text, "count": counts[text]} for text in distinct_by_appearance],
    }
    return chosen, conflict


def extract_main_fields(annotations):
    """Return ``(values, conflicts)`` for the document-level fields.

    ``values`` maps every MAIN_FIELD to its value or None. ``conflicts`` maps the
    subset of fields whose occurrences disagreed to their conflict record.
    """
    occurrences_by_field = defaultdict(list)
    entry_count = Counter()
    for key, value in annotations:
        if isinstance(key, str):
            entry_count[key] += 1
            occurrences_by_field[key].extend(value)

    values = {field: None for field in MAIN_FIELDS}
    conflicts = {}
    for field, occurrences in occurrences_by_field.items():
        assert field in MAIN_FIELDS, f"unexpected document-level field {field!r}"
        assert entry_count[field] == 1, (
            f"field {field!r} appears in {entry_count[field]} annotation entries; "
            "the majority-vote rule assumes a single entry per document-level field"
        )
        value, conflict = select_main_value(occurrences)
        values[field] = value
        if conflict is not None:
            conflicts[field] = conflict
    return values, conflicts


def extract_line_items(annotations):
    """Return the document's line items, ordered by position in the OCR text.

    Each item is a dict mapping every LINE_FIELD to its value or None.
    """
    items = []
    for key, value in annotations:
        if isinstance(key, str):
            continue
        assert isinstance(key, list) and all(isinstance(k, str) for k in key)
        for field in key:
            assert field in LINE_FIELDS, f"unexpected line-item field {field!r}"
        for instance in value:
            assert len(instance) == len(key), (
                f"line-item instance has {len(instance)} occurrences for key {key!r}"
            )
            fields = {field: None for field in LINE_FIELDS}
            for field, occ in zip(key, instance):
                fields[field] = _clean(occ[0])
            offset = min(occurrence_offset(occ) for occ in instance)
            items.append((offset, fields))

    items.sort(key=lambda pair: pair[0])
    return [fields for _offset, fields in items]


# --- Document assembly ---------------------------------------------------------


def document_id(filename):
    assert filename.endswith(".pdf"), f"unexpected filename {filename!r}"
    return filename[: -len(".pdf")]


def load_documents(dataset_jsonl):
    """Parse every row of the upstream dataset into a plain dict."""
    documents = []
    with open(dataset_jsonl, encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            record = json.loads(raw)
            assert set(record) >= {"filename", "ocr", "annotations"}, (
                f"row {line_no} missing expected keys: {sorted(record)}"
            )
            ocr_text = record["ocr"]["text"]
            assert ocr_text, f"row {line_no} has empty OCR text"
            values, conflicts = extract_main_fields(record["annotations"])
            line_items = extract_line_items(record["annotations"])
            documents.append(
                {
                    "document_id": document_id(record["filename"]),
                    "filename": record["filename"],
                    "ocr_text": ocr_text,
                    "main_values": values,
                    "conflicts": conflicts,
                    "line_items": line_items,
                }
            )
    return documents


def assign_splits(doc_ids, test_frac, seed):
    """Deterministically label each document 'train' or 'test'.

    Sorting first makes the result independent of input order; the seeded shuffle
    is the only source of randomness.
    """
    assert 0.0 < test_frac < 1.0, f"test_frac must be in (0, 1), got {test_frac}"
    ordered = sorted(doc_ids)
    assert len(ordered) == len(set(ordered)), "duplicate document ids"
    random.Random(seed).shuffle(ordered)
    n_test = round(len(ordered) * test_frac)
    assert 0 < n_test < len(ordered), "split leaves train or test empty"
    test_ids = set(ordered[:n_test])
    return {doc_id: ("test" if doc_id in test_ids else "train") for doc_id in doc_ids}


# --- Output -------------------------------------------------------------------


def _git(*args):
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent,
    )
    assert result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.strip()}"
    return result.stdout


def git_state():
    """Commit the build ran from, and whether the working tree had uncommitted changes.

    ``directory.json``'s build block is the only reproducibility record for this
    dataset, so a dirty tree (the script itself not yet committed, say) must be
    visible in it.
    """
    return _git("rev-parse", "HEAD").strip(), bool(_git("status", "--porcelain").strip())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-jsonl", required=True, type=Path, help="upstream dataset.jsonl")
    parser.add_argument("--pdf-dir", required=True, type=Path, help="directory of source PDFs")
    parser.add_argument("--out-dir", required=True, type=Path, help="destination, e.g. data/vrdu")
    parser.add_argument("--test-frac", required=True, type=float, help="fraction of documents held out for test")
    parser.add_argument("--seed", required=True, type=int, help="seed for the train/test shuffle")
    parser.add_argument(
        "--skip-known-answer-checks",
        action="store_true",
        help="skip the fixed-corpus assertions (use only on a non-standard input)",
    )
    args = parser.parse_args(argv)

    assert args.dataset_jsonl.is_file(), f"no such file: {args.dataset_jsonl}"
    assert args.pdf_dir.is_dir(), f"no such directory: {args.pdf_dir}"

    documents = load_documents(args.dataset_jsonl)
    doc_ids = [doc["document_id"] for doc in documents]
    assert len(doc_ids) == len(set(doc_ids)), "duplicate document ids in dataset.jsonl"

    pdf_ids = {p.stem for p in args.pdf_dir.glob("*.pdf")}
    assert set(doc_ids) == pdf_ids, (
        "document ids and PDF files do not match: "
        f"{len(set(doc_ids) - pdf_ids)} rows without a PDF, "
        f"{len(pdf_ids - set(doc_ids))} PDFs without a row"
    )

    if not args.skip_known_answer_checks:
        assert len(documents) == EXPECTED_N_DOCS, f"expected {EXPECTED_N_DOCS} docs, got {len(documents)}"

    splits = assign_splits(doc_ids, args.test_frac, args.seed)

    out_dir = args.out_dir
    ocr_dir = out_dir / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "main").mkdir(parents=True, exist_ok=True)
    (out_dir / "line").mkdir(parents=True, exist_ok=True)

    # OCR text, one file per document.
    for doc in documents:
        (ocr_dir / f"{doc['document_id']}.txt").write_text(doc["ocr_text"], encoding="utf-8")
    written = {p.stem for p in ocr_dir.glob("*.txt")}
    assert written == set(doc_ids), "ocr/ directory does not match the document set"

    # Datapoints.
    main_rows = {"train": [], "test": []}
    line_rows = {"train": [], "test": []}
    for doc in documents:
        split = splits[doc["document_id"]]
        main_row = {"document_id": doc["document_id"], "valid": True}
        main_row.update({field: doc["main_values"][field] for field in MAIN_FIELDS})
        assert set(main_row) == {"document_id", "valid", *MAIN_FIELDS}
        main_rows[split].append(main_row)

        for line_index, item in enumerate(doc["line_items"]):
            line_row = {"document_id": doc["document_id"], "line_index": line_index, "valid": True}
            line_row.update({field: doc["main_values"][field] for field in LINE_CARRIED_FIELDS})
            line_row.update({field: item[field] for field in LINE_FIELDS})
            assert set(line_row) == {"document_id", "line_index", "valid", *LINE_CARRIED_FIELDS, *LINE_FIELDS}
            line_rows[split].append(line_row)

    for split in ("train", "test"):
        (out_dir / "main" / f"{split}_base.json").write_text(
            json.dumps(main_rows[split], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (out_dir / "line" / f"{split}_base.json").write_text(
            json.dumps(line_rows[split], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    # directory.json
    directory_docs = []
    for doc in documents:
        directory_docs.append(
            {
                "document_id": doc["document_id"],
                "split": splits[doc["document_id"]],
                "source_pdf": str(args.pdf_dir / doc["filename"]),
                "ocr_chars": len(doc["ocr_text"]),
                "n_line_items": len(doc["line_items"]),
                "main_fields_present": [f for f in MAIN_FIELDS if doc["main_values"][f] is not None],
                "value_conflicts": doc["conflicts"],
            }
        )
    n_train = sum(1 for d in directory_docs if d["split"] == "train")
    n_test = len(directory_docs) - n_train
    sha, dirty = git_state()
    directory = {
        "build": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_sha": sha,
            "git_dirty": dirty,
            "script": "scripts/build_vrdu_dataset.py",
            "source": str(args.dataset_jsonl.resolve()),
            "seed": args.seed,
            "test_frac": args.test_frac,
            "n_docs": len(directory_docs),
            "n_train": n_train,
            "n_test": n_test,
            "n_line_items": sum(d["n_line_items"] for d in directory_docs),
            "main_fields": MAIN_FIELDS,
            "line_fields": LINE_FIELDS,
            "line_carried_fields": LINE_CARRIED_FIELDS,
        },
        "documents": directory_docs,
    }
    (out_dir / "directory.json").write_text(json.dumps(directory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # --- Post-write verification ---------------------------------------------
    all_main = main_rows["train"] + main_rows["test"]
    all_line = line_rows["train"] + line_rows["test"]
    assert len(all_main) == len(documents)
    assert {r["document_id"] for r in all_main} == set(doc_ids)
    assert {r["document_id"] for r in all_line} <= set(doc_ids)
    assert len(all_line) == sum(d["n_line_items"] for d in directory_docs)

    main_nonnull = {f: sum(1 for r in all_main if r[f] is not None) for f in MAIN_FIELDS}
    line_nonnull = {f: sum(1 for r in all_line if r[f] is not None) for f in LINE_FIELDS}
    docs_without_line_items = sum(1 for d in directory_docs if d["n_line_items"] == 0)
    n_conflicts = sum(len(d["value_conflicts"]) for d in directory_docs)

    if not args.skip_known_answer_checks:
        assert len(all_line) == EXPECTED_N_LINE_ITEMS, (len(all_line), EXPECTED_N_LINE_ITEMS)
        assert docs_without_line_items == EXPECTED_DOCS_WITHOUT_LINE_ITEMS
        assert main_nonnull == EXPECTED_MAIN_NONNULL, main_nonnull
        assert line_nonnull == EXPECTED_LINE_NONNULL, line_nonnull

    print(f"documents: {len(documents)}  train: {n_train}  test: {n_test}")
    print(f"line items: {len(all_line)}  train: {len(line_rows['train'])}  test: {len(line_rows['test'])}")
    print(f"documents with no line items: {docs_without_line_items}")
    print(f"documents with >=1 value conflict: {sum(1 for d in directory_docs if d['value_conflicts'])}")
    print(f"total conflicted fields: {n_conflicts}")
    print(f"main non-null:  {main_nonnull}")
    print(f"line non-null:  {line_nonnull}")
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
