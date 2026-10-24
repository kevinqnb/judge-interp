"""Unit + tiny end-to-end tests for data/vrdu/build_vrdu_main.py and data/vrdu/main_samples.py.

End to end: the mini corpus in tests/fixtures/vrdu_invalids_mini/ (see test_build_vrdu_invalids.py). Its
main bases have 9 train / 3 test documents, each reaching every inter_document k in 1..9 (every field can
be resampled from the split-wide pool, nulls included). cross_type uses the -05 code and RNG stream
unchanged, so its rows must equal those of build_vrdu_invalids.py (-05) on the same corpus.

main_samples: hand-built rows of documents X, Y, Z where X's agency is null, so corrupting it is a null fill.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "vrdu"))
import build_vrdu_invalids as old  # noqa: E402  (-05 builder: regression reference for cross_type)
import build_vrdu_main as b  # noqa: E402
import main_samples as ms  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "vrdu_invalids_mini"
CONFIG = REPO / "data" / "vrdu" / "build-configs" / "2026-10-24-vrdu-invalids-06.yaml"
OLD_CONFIG = REPO / "data" / "vrdu" / "build-configs" / "2026-10-22-vrdu-invalids-05.yaml"


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


def test_cross_type_and_valid_rows_equal_the_05_builder(built, tmp_path):
    ref = run(old, OLD_CONFIG, tmp_path / "vrdu05")
    for split in b.SPLITS:
        for keep in (lambda r: r["valid"], lambda r: r["error_type"] == "cross_type"):
            assert [r for r in built[split] if keep(r)] == [r for r in ref[split] if keep(r)]


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
