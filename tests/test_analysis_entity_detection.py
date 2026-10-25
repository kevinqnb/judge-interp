"""Unit tests for analysis/entity_detection.py on tiny hand-built fixtures.

Brier fixture (7 test spans):
    idx  valid  prob
     0   T      0.9
     1   T      0.2   (wrongly called invalid)
     2   T      0.8
     3   F      0.1
     4   F      0.7   (missed)
     5   F      0.3
     6   F      0.4
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "analysis_entity_detection", Path(__file__).resolve().parents[1] / "analysis" / "entity_detection.py"
)
er = importlib.util.module_from_spec(spec)
spec.loader.exec_module(er)

Y = np.array([True, True, True, False, False, False, False])
P = np.array([0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 0.4])


def test_brier_known_answers():
    y = np.array([True, False, True, False])
    b, ref, bss = er.brier_skill_score(y, y.astype(float), 0.5)
    assert (b, ref, bss) == (0.0, 0.25, 1.0)  # perfect forecast
    b, ref, bss = er.brier_skill_score(y, np.full(4, 0.5), 0.5)
    assert (b, ref, bss) == (0.25, 0.25, 0.0)  # the reference forecast has no skill
    b, _, bss = er.brier_skill_score(y, 1.0 - y.astype(float), 0.5)
    assert b == 1.0 and bss == -3.0  # perfectly wrong


def test_brier_hand_value():
    # probs (0.9, 0.2, 0.8, ...) vs Y: squared errors .01 .64 .04 .01 .49 .09 .16 -> mean 1.44/7
    b, ref, bss = er.brier_skill_score(Y, P, 3 / 7)
    assert b == pytest.approx(1.44 / 7)
    # reference at rate 3/7: 3 * (4/7)^2 + 4 * (3/7)^2 over 7 = (48 + 36) / 343
    assert ref == pytest.approx(84 / 343)
    assert bss == pytest.approx(1 - (1.44 / 7) / (84 / 343))


def test_brier_rejects_degenerate_reference():
    with pytest.raises(AssertionError):
        er.brier_skill_score(Y, P, 1.0)


def test_stratum_report_known_answer():
    # two strata of minimal pairs: s1 = (0.9 valid, 0.1 invalid), (0.2 valid, 0.7 invalid); s2 = (0.8, 0.3)
    y = np.array([True, False, True, False, True, False])
    probs = np.array([0.9, 0.1, 0.2, 0.7, 0.8, 0.3])
    strata = np.array(["s1", "s1", "s1", "s1", "s2", "s2"])
    r = er.stratum_report(y, probs, 0.5, strata)
    assert (r["n_s1"], r["n_s2"]) == (4, 2)
    # s1: valid {0.9, 0.2} vs invalid {0.1, 0.7}: 0.9 beats both, 0.2 beats 0.1 -> 3/4
    assert r["auroc_s1"] == pytest.approx(3 / 4) and r["auroc_s2"] == 1.0
    assert r["detect_s1"] == 0.5 and r["valid_acc_s1"] == 0.5 and r["accuracy_s1"] == 0.5
    with pytest.raises(AssertionError, match="lacks a class"):
        er.stratum_report(y, probs, 0.5, np.array(["s1", "s1", "s1", "s1", "s2", "s3"]))


def test_doc_bootstrap_auroc():
    y = np.array([False, True] * 4)
    docs = np.repeat(["a", "b", "c", "d"], 2)
    assert er.doc_bootstrap_auroc(y, y.astype(float), docs, 50, 0.95, 0) == (1.0, 1.0)
    probs = np.array([0.1, 0.9, 0.6, 0.4, 0.2, 0.8, 0.3, 0.7])   # doc b's pair is ranked the wrong way
    lo, hi = er.doc_bootstrap_auroc(y, probs, docs, 200, 0.95, 0)
    assert lo < float(er.roc_auc_score(y, probs)) <= hi <= 1.0
    assert (lo, hi) == er.doc_bootstrap_auroc(y, probs, docs, 200, 0.95, 0)


def test_stack_pairs_and_structure_baseline():
    s = er.stack_pairs(np.array([3, 5]), np.array([0, 0]), field=np.array(["f", "g"]))
    assert s["idx"].tolist() == [3, 5, 0, 0] and s["y"].tolist() == [False, False, True, True]
    assert s["field"].tolist() == ["f", "g", "f", "g"]
    p = {**P_LR, "decision_threshold": 0.5}
    cats = np.array(["f", "g", "g", "f", "f", "g", "g", "f"])
    y = np.array([False, False, False, False, True, True, True, True])   # every category 50/50
    m = er.assert_structure_auroc_is_chance("field", y, cats, y, cats, p, 0)
    assert m["test_baseline_field_auroc"] == pytest.approx(0.5, abs=er.EXACT_AUROC_ATOL)
    with pytest.raises(AssertionError, match="pairing is broken"):   # "f" mostly invalid: structure predicts the label
        er.assert_structure_auroc_is_chance("field", y, np.array(["f"] * 3 + ["g"] * 5), y, np.array(["f"] * 3 + ["g"] * 5), p, 0)


# ---------------------------------------------------------------- minimal entity pairs on a main row file
# doc a clean: advertiser A, agency G;   doc b clean: advertiser B, product P   (other fields null)
#   a inter_document k=1: advertiser <- "X"
#   a cross_type k=1:     product (null in a) <- "G"            (null-fill: twin = b's product, the only one)
#   b inter_document k=1: agency (null in b) <- "Y"             (null-fill: twin = a's agency, the only one)
def _main_row(doc, et, k, values, errors=None):
    errors = errors or {}
    return {"document_id": doc, "valid": et is None, "error_type": et, "k": k, "invalid_fields": list(errors),
            **{f: values.get(f) for f in er.MAIN_FIELDS},
            "field_errors": {f: ({"error_type": et, "source": {"null_fill": errors[f]}} if f in errors else None)
                             for f in er.MAIN_FIELDS}}


MAIN_ROWS = [
    _main_row("a", None, 0, {"advertiser": "A", "agency": "G"}),
    _main_row("b", None, 0, {"advertiser": "B", "product": "P"}),
    _main_row("a", "inter_document", 1, {"advertiser": "X", "agency": "G"}, {"advertiser": False}),
    _main_row("a", "cross_type", 1, {"advertiser": "A", "agency": "G", "product": "G"}, {"product": True}),
    _main_row("b", "inter_document", 1, {"advertiser": "B", "agency": "Y", "product": "P"}, {"agency": True}),
]


def _main_split(rows):
    spans = [(r, f) for r in rows for f in er.MAIN_FIELDS if r[f] is not None]
    return {"doc_ids": np.array([r["document_id"] for r, _ in spans]),
            "row_error_type": np.array([r["error_type"] or "" for r, _ in spans]),
            "row_k": np.array([r["k"] for r, _ in spans]), "field": np.array([f for _, f in spans]),
            "value": np.array([r[f] for r, f in spans]),
            "label_valid": np.array([f not in r["invalid_fields"] for r, f in spans])}


def _span(split, doc, et, k, field):
    hit = np.flatnonzero((split["doc_ids"] == doc) & (split["row_error_type"] == et) & (split["row_k"] == k)
                         & (split["field"] == field))
    assert len(hit) == 1
    return int(hit[0])


def test_main_span_pairs_hand_worked():
    s = _main_split(MAIN_ROWS)
    z = er.main_span_pairs(s, MAIN_ROWS, 0)
    # strata in order (cross_type, 1), (inter_document, 1); documents in order within a stratum
    assert z["error_type"].tolist() == ["cross_type", "inter_document", "inter_document"]
    assert z["field"].tolist() == ["product", "advertiser", "agency"] and z["null_fill"].tolist() == [True, False, True]
    assert z["inv"].tolist() == [_span(s, "a", "cross_type", 1, "product"), _span(s, "a", "inter_document", 1, "advertiser"),
                                 _span(s, "b", "inter_document", 1, "agency")]
    assert z["val"].tolist() == [_span(s, "b", "", 0, "product"), _span(s, "a", "", 0, "advertiser"), _span(s, "a", "", 0, "agency")]
    assert z["doc"].tolist() == ["a", "a", "b"]
    # kept spans: a inter agency; a cross advertiser, agency; b inter advertiser, product -> each vs its clean span
    assert len(z["kept"]) == 5 and sorted(z["kept_error_type"].tolist()) == ["cross_type"] * 2 + ["inter_document"] * 3
    assert all(s["field"][i] == s["field"][j] and s["row_error_type"][j] == "" for i, j in zip(z["kept"], z["clean"]))


def test_main_span_pairs_pairs_only_judged_rows_and_refuses_mismatches():
    sub = [r for r in MAIN_ROWS if not (r["document_id"] == "b" and r["error_type"])]
    s = _main_split(sub)
    assert er.main_span_pairs(s, MAIN_ROWS, 0)["field"].tolist() == ["product", "advertiser"]   # b's row was not judged
    s = _main_split(MAIN_ROWS)
    s["value"][_span(s, "a", "inter_document", 1, "advertiser")] = "Z"
    with pytest.raises(AssertionError, match="span value differs"):
        er.main_span_pairs(s, MAIN_ROWS, 0)
    no_clean_a = [r for r in MAIN_ROWS if not (r["document_id"] == "a" and r["valid"])]
    with pytest.raises(AssertionError, match="without a clean row"):
        er.main_span_pairs(_main_split(no_clean_a), MAIN_ROWS, 0)
    s = _main_split(MAIN_ROWS)
    s = {k: np.delete(v, _span(s, "a", "", 0, "agency")) for k, v in s.items()}
    with pytest.raises(AssertionError, match="has no span"):
        er.main_span_pairs(s, MAIN_ROWS, 0)


P_LR = {"lr_C": 1.0, "lr_max_iter": 1000, "lr_tol": 1e-4, "n_shuffles": 30}


def test_shuffled_label_auroc_is_chance_when_x_is_independent_of_y_and_deterministic():
    rng = np.random.default_rng(0)
    x_tr, x_te = rng.standard_normal((300, 5)), rng.standard_normal((300, 5))
    y_tr, y_te = rng.random(300) < 0.4, rng.random(300) < 0.4  # labels independent of x
    m, s = er.shuffled_label_auroc(x_tr, y_tr, x_te, y_te, P_LR, 0)
    assert abs(m - 0.5) < 0.05 and s > 0
    assert (m, s) == er.shuffled_label_auroc(x_tr, y_tr, x_te, y_te, P_LR, 0)


def test_shuffled_label_auroc_requires_n_shuffles():
    x, y = np.zeros((4, 2)), np.array([True, False, True, False])
    with pytest.raises(KeyError):
        er.shuffled_label_auroc(x, y, x, y, {k: v for k, v in P_LR.items() if k != "n_shuffles"}, 0)


def test_onehot():
    m = er.onehot(np.array(["b", "a", "b"]), ["a", "b"])
    np.testing.assert_array_equal(m, [[0, 1], [1, 0], [0, 1]])
    with pytest.raises(AssertionError):
        er.onehot(np.array(["c"]), ["a", "b"])


# ---------------------------------------------------------------- loader on a synthetic run
def write_run(root: Path, run_id: str, task: str = "entity_detection", rep0_constant: bool = True) -> None:
    run_dir = root / run_id
    (run_dir / "artifacts").mkdir(parents=True)
    docs = np.array(["a", "a", "a", "b", "b"])
    valid = np.array([True, False, True, True, False])
    fet = np.where(valid, "", "cross_type")
    rep0 = np.ones((5, 4), dtype=np.float32)
    if not rep0_constant:
        rep0[2, 0] = 2.0
    np.savez_compressed(
        run_dir / "artifacts" / "span_representations.npz",
        layers=np.array([0, 8]), rep_0=rep0, rep_8=np.arange(20, dtype=np.float32).reshape(5, 4),
        doc_ids=docs, row_ordinal=np.zeros(5, dtype=np.int64), field=np.array(["f"] * 5),
        field_position=np.array([1, 2, 3, 1, 2]), label_valid=valid, field_error_type=fet,
        row_error_type=fet, row_k=np.array([1, 1, 1, 1, 1]),
    )
    (run_dir / "run.json").write_text(json.dumps({"status": "success", "task": task}))
    (run_dir / "metrics.json").write_text(json.dumps({
        "n_spans": 5, "n_valid_spans": 3, "n_invalid_spans": 2,
        "n_spans_inter_document": 0, "n_spans_intra_document": 0, "n_spans_cross_type": 2,
    }))


def test_load_split_roundtrip(tmp_path):
    write_run(tmp_path, "r")
    s = er.load_split("r", tmp_path, 8, None, "entity_detection", ())
    np.testing.assert_array_equal(s["X"], np.arange(20, dtype=np.float32).reshape(5, 4))  # raw rep_8, no rep_0 subtraction
    assert s["label_valid"].tolist() == [True, False, True, True, False]


def test_load_split_max_docs(tmp_path):
    write_run(tmp_path, "r")
    s = er.load_split("r", tmp_path, 8, 1, "entity_detection", ())
    assert s["doc_ids"].tolist() == ["a"] * 3 and s["X"].shape == (3, 4)


def test_load_split_rejects_wrong_task_layer_and_nonconstant_layer0(tmp_path):
    write_run(tmp_path, "r")
    with pytest.raises(AssertionError):
        er.load_split("r", tmp_path, 8, None, "relation_detection", ())
    with pytest.raises(AssertionError):
        er.load_split("r", tmp_path, 14, None, "entity_detection", ())
    write_run(tmp_path, "bad", rep0_constant=False)
    with pytest.raises(AssertionError, match="layer-0"):
        er.load_split("bad", tmp_path, 8, None, "entity_detection", ())


def test_load_split_rejects_count_mismatch(tmp_path):
    write_run(tmp_path, "r")
    m = json.loads((tmp_path / "r" / "metrics.json").read_text())
    m["n_valid_spans"] = 4
    (tmp_path / "r" / "metrics.json").write_text(json.dumps(m))
    with pytest.raises(AssertionError):
        er.load_split("r", tmp_path, 8, None, "entity_detection", ())


# --- shared preprocessing and probe fit ---------------------------------------------------

DOCS = np.array(["a"] * 5 + ["b"] * 2)


def test_center_per_document_hand_values():
    x = np.array([[1, 10], [3, 10], [5, 10], [0, 0], [4, 8], [7, 7], [9, 9]], dtype=np.float32)
    er.center_per_document(x, DOCS)
    # doc-a mean = ([1,3,5,0,4], [10,10,10,0,8]) = (2.6, 7.6)
    np.testing.assert_allclose(x[0], [1 - 2.6, 10 - 7.6], atol=1e-5)
    np.testing.assert_allclose(x[6], [1.0, 1.0], atol=1e-5)  # doc-b mean (8,8)
    np.testing.assert_allclose(x[5], [-1.0, -1.0], atol=1e-5)


def test_global_mean_is_zero_after_per_doc_centering():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(7, 4)).astype(np.float32) + 5
    er.center_per_document(x, DOCS)
    assert np.abs(x.mean(axis=0)).max() < 1e-5


def test_fit_logreg_separable_and_shuffle_control():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = rng.random(400) < 0.5
    x = rng.normal(size=(400, 5)).astype(np.float32)
    x[:, 0] += 4 * y
    p = {"lr_C": 1.0, "lr_max_iter": 200, "lr_tol": 1e-4}
    clf = er.fit_logreg(x, y, p, 0)
    assert roc_auc_score(y, clf.predict_proba(x)[:, 1]) > 0.99
    again = er.fit_logreg(x, y, p, 0)
    assert np.array_equal(clf.predict_proba(x), again.predict_proba(x))
