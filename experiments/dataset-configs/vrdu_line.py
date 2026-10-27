"""Dataset config: VRDU ad-buy-form, line items (many rows per document).

Defines ``DATASET`` for ``judge_interp.dataset_spec.load_dataset_spec``. The field list
comes from ``data/vrdu/build_vrdu_dataset.py`` (single source of truth for the schema); the
descriptions, wording and file layout below are this dataset's. A row is one line item shown with all
its non-null fields in standard order (``data/vrdu/build_vrdu_line.py``), so the relation tasks apply: the
entities of a row should all belong to the same line item. Rows come in minimal pairs (an invalid
row and its clean twin) and are drawn pair by pair.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data" / "vrdu"))

import build_vrdu_dataset as schema  # noqa: E402

from judge_interp.dataset_spec import DatasetSpec  # noqa: E402

FIELD_DESCRIPTIONS = {
    "channel": "the station or channel that this line item (spot) airs on",
    "program_desc": "the program, time slot, or daypart that this line item airs in",
    "program_start_date": "the start date of this line item's air period",
    "program_end_date": "the end date of this line item's air period",
    "sub_amount": "the dollar amount (subtotal) of this line item",
}

DATASET = DatasetSpec(
    name="line",
    fields=tuple(schema.LINE_FIELDS),
    field_descriptions=FIELD_DESCRIPTIONS,
    error_types=("intra_document", "cross_item"),
    record_description="one line item (one broadcast spot) of the advertising order of a document",
    context_description="OCR text of the source document",
    type_confusion_hint=(
        "a date or a dollar amount that belongs to another line item, or to the order as a whole, "
        "is invalid for this line item even though it is a genuine value of the document"
    ),
    row_index_key="line_index",
    split_path="vrdu/line/{split}.json",
    context_path="vrdu/ocr/{document_id}.txt",
    supports_relation=True,
    ordered_fields=True,
    pair_key="pair_id",
)
