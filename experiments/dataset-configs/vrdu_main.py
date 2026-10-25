"""Dataset config: VRDU ad-buy-form, document-level fields (one row per document).

Defines ``DATASET`` for ``judge_interp.dataset_spec.load_dataset_spec``. The field list
comes from ``data/vrdu/build_vrdu_dataset.py`` (single source of truth for the schema); the
descriptions, wording and file layout below are this dataset's.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data" / "vrdu"))

import build_vrdu_dataset as schema  # noqa: E402

from judge_interp.dataset_spec import DatasetSpec  # noqa: E402

FIELD_DESCRIPTIONS = {
    "advertiser": "the political campaign, candidate or committee that the advertising is bought for",
    "agency": "the advertising agency that placed the buy on the advertiser's behalf",
    "property": "the TV station (typically its call letters) that sells the airtime and receives the order",
    "tv_address": "the mailing address of that TV station (street, city, state, ZIP; may span several lines)",
    "product": "the campaign, candidate, or race description that the buy is booked under",
    "contract_num": "the contract, order, or estimate identifier of the whole buy",
    "flight_from": "the start date of the whole buy's flight (its overall air period)",
    "flight_to": "the end date of the whole buy's flight (its overall air period)",
    "gross_amount": "the total gross dollar amount of the whole contract",
}

DATASET = DatasetSpec(
    name="main",
    fields=tuple(schema.MAIN_FIELDS),
    field_descriptions=FIELD_DESCRIPTIONS,
    error_types=("inter_document", "cross_type"),
    record_description="the advertising order of a document (one record per document)",
    context_description="OCR text of the source document",
    type_confusion_hint=(
        "a value under `advertiser` whose text is really the name of the agency, the TV station, "
        "or a product description is invalid, as is a date or a dollar amount that belongs to "
        "another field of the record"
    ),
    row_index_key=None,
    split_path="vrdu/main/{split}.json",
    context_path="vrdu/ocr/{document_id}.txt",
    supports_relation=False,
    ordered_fields=False,
    pair_key=None,
)
