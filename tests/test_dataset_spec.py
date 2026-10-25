"""Unit tests for judge_interp.dataset_spec and the VRDU dataset configs."""
import dataclasses

import pytest

from judge_interp.dataset_spec import DatasetSpec, load_dataset_spec
from dataset_specs import LINE, MAIN


def test_vrdu_specs_match_the_build_schema():
    import build_vrdu_dataset as schema  # on sys.path via the dataset configs

    assert list(MAIN.fields) == schema.MAIN_FIELDS and list(LINE.fields) == schema.LINE_FIELDS
    assert (MAIN.supports_relation, LINE.supports_relation) == (False, True)
    assert MAIN.row_index_key is None and LINE.row_index_key == "line_index"
    assert (MAIN.ordered_fields, MAIN.pair_key) == (False, None) and (LINE.ordered_fields, LINE.pair_key) == (True, "pair_id")
    assert LINE.error_types == ("intra_document", "cross_item") and MAIN.error_types == ("inter_document", "cross_type")


def test_require_relation():
    LINE.require_relation()
    with pytest.raises(ValueError, match="does not support the relation tasks"):
        MAIN.require_relation()


@pytest.mark.parametrize("change", [
    {"name": ""},
    {"fields": ()},
    {"fields": ["advertiser"]},  # list, not tuple
    {"field_descriptions": {"advertiser": "x"}},  # keys != fields
    {"split_path": "vrdu/main/train.json"},  # no {split}
    {"context_path": "vrdu/ocr/doc.txt"},  # no {document_id}
    {"type_confusion_hint": ""},
    {"supports_relation": 1},
    {"ordered_fields": None},
    {"pair_key": ""},
])
def test_invalid_specs_are_rejected(change):
    with pytest.raises(ValueError):
        dataclasses.replace(MAIN, **change)


def test_spec_has_no_defaults():
    with pytest.raises(TypeError):
        DatasetSpec(name="x")


def test_load_dataset_spec_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset_spec(tmp_path / "nope.py")
    (tmp_path / "empty.py").write_text("X = 1\n")
    with pytest.raises(AttributeError, match="DATASET"):
        load_dataset_spec(tmp_path / "empty.py")
    (tmp_path / "wrong.py").write_text("DATASET = 1\n")
    with pytest.raises(TypeError, match="DatasetSpec"):
        load_dataset_spec(tmp_path / "wrong.py")
