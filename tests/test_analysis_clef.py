"""Unit tests for analysis/clef.py on hand-built Clef run directories (no model involved).

Entity run: the main rows of test_analysis_entity_detection.py (3 minimal pairs: a cross_type null-fill
product, an inter_document advertiser, an inter_document null-fill agency; 5 kept spans). Every invalid
span gets P(true) 0.1, every valid one 0.9 -> on the 6 paired items AUROC 1.0, accuracy 1.0, Brier 0.01,
and with bss_ref_rate = set rate 0.5, brier_ref 0.25 -> BSS = 1 - 0.01 / 0.25.

Samples run: the 8 rows (4 minimal pairs) of the relation_detection_mini line fixture.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import test_analysis_entity_detection as ted

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analysis_clef", ROOT / "analysis" / "clef.py")
ac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac)

P = dict(decision_threshold=0.5, ece_n_boot=20, n_shuffles=50, permutation_auroc_tolerance=0.3,
         hist_n_bins=10, auroc_n_boot=50, ci_level=0.95)
LINE_ROWS = json.loads((ROOT / "tests/fixtures/relation_detection_mini/vrdu/line/train.json").read_text())


def _write_run(root: Path, run_id: str, kind: str, cols: dict, **over) -> None:
    d = root / run_id
    (d / "artifacts").mkdir(parents=True)
    p_true = np.asarray(cols["p_true"], dtype=np.float64)
    cols = {**cols, "logit_true": np.log(p_true), "logit_false": np.log(1 - p_true)}
    np.savez_compressed(d / "artifacts" / "clef_predictions.npz", **cols)
    lab = np.asarray(cols["label_valid"], dtype=bool)
    (d / "metrics.json").write_text(json.dumps({
        "n_predictions": len(lab), "n_valid": int(lab.sum()), "n_excluded_overlong": 0, **over}))
    (d / "run.json").write_text(json.dumps({"status": "success", "task": ac.KINDS[kind]}))
    (d / "config.snapshot.yaml").write_text(yaml.safe_dump({"params": {"context_control": "none"}}))


def _entity_cols(p_invalid=0.1, p_valid=0.9, override=None):
    s = ted._main_split(ted.MAIN_ROWS)
    n = len(s["field"])
    p_true = np.where(s["label_valid"], p_valid, p_invalid)
    for (doc, et, k, f), v in (override or {}).items():
        p_true[ted._span(s, doc, et, k, f)] = v
    return dict(**s, field_error_type=np.where(s["label_valid"], "", s["row_error_type"]),
                row_ordinal=np.zeros(n, dtype=np.int64), p_true=p_true, n_tokens=np.full(n, 100), context_doc=s["doc_ids"])


def _entity(tmp_path, cols, ref):
    _write_run(tmp_path, "r", "entity", cols)
    run = ac.load_run("r", tmp_path, "entity")
    ev = ac.entity_eval_set(run, ted.MAIN_ROWS, 0)
    return ac.analyse_run("e", run, ev, {"bss_ref_rate": ref}, P, tmp_path / "fig", 0)


def test_entity_run_hand_worked_metrics(tmp_path):
    m = _entity(tmp_path, _entity_cols(), ac.REF_SET_RATE)
    assert m["n"] == 6 and m["n_pairs"] == 3 and m["n_null_fill_pairs"] == 2
    assert m["test_auroc"] == 1.0 and m["test_accuracy"] == 1.0 and m["test_auroc_ci_lo"] == 1.0
    assert m["test_brier"] == pytest.approx(0.01)
    assert m["bss_ref_rate"] == 0.5 and m["test_bss"] == pytest.approx(1 - 0.01 / 0.25)
    assert m["test_detect_inter_document"] == 1.0 and m["test_detect_cross_type"] == 1.0
    assert m["test_n_inter_document"] == 4 and m["test_n_cross_type_k1"] == 2
    assert m["test_no_null_fill_auroc"] == 1.0 and m["test_no_null_fill_n_inter_document"] == 2
    # kept spans and their clean twins all at 0.9: no row-context signal
    assert m["test_clean_vs_kept_auroc"] == 0.5 and m["test_kept_valid_acc"] == 1.0
    assert (tmp_path / "fig" / "e" / "calibration.png").is_file()


def test_entity_run_with_a_missed_error_type(tmp_path):
    cols = _entity_cols(override={("a", "cross_type", 1, "product"): 0.8})   # the cross_type span is called valid
    m = _entity(tmp_path, cols, 0.5)
    assert m["test_detect_inter_document"] == 1.0 and m["test_detect_cross_type"] == 0.0
    assert m["test_accuracy"] == pytest.approx(5 / 6)
    # ranking is still perfect (0.8 < every valid 0.9) although the 0.5 threshold misses that item
    assert m["test_auroc_cross_type"] == 1.0 and m["test_auroc"] == 1.0


def test_loader_gates(tmp_path):
    cols = _entity_cols()
    _write_run(tmp_path, "bad_count", "entity", cols, n_predictions=7)
    with pytest.raises(AssertionError, match="metrics n_predictions"):
        ac.load_run("bad_count", tmp_path, "entity")
    _write_run(tmp_path, "wrong_task", "entity", cols)
    with pytest.raises(AssertionError, match="task"):
        ac.load_run("wrong_task", tmp_path, "oneshot")
    _write_run(tmp_path, "bad_logit", "entity", cols)
    d = tmp_path / "bad_logit" / "artifacts" / "clef_predictions.npz"
    z = dict(np.load(d))
    z["logit_true"] = z["logit_true"] + 1.0
    np.savez_compressed(d, **z)
    with pytest.raises(AssertionError, match="softmax"):
        ac.load_run("bad_logit", tmp_path, "entity")


def _sample_cols(rows=LINE_ROWS):
    n = len(rows)
    lab = np.array([r["valid"] for r in rows])
    return dict(sample_id=np.array([r["sample_id"] for r in rows]), doc_ids=np.array([r["document_id"] for r in rows]),
                error_type=np.array([r["error_type"] or "" for r in rows]), pair_id=np.array([r["pair_id"] for r in rows]),
                cell=np.array([r["cell"] for r in rows]), size=np.array([r["m"] for r in rows]),
                fields=np.array(["|".join(r["fields"]) for r in rows]), n_invalid_fields=np.array([r["k"] for r in rows]),
                label_valid=lab, p_true=np.where(lab, 0.8, 0.3), n_tokens=np.full(n, 50),
                context_doc=np.array([r["document_id"] for r in rows]))


def test_samples_run_on_minimal_pairs(tmp_path):
    _write_run(tmp_path, "r", "oneshot", _sample_cols())
    run = ac.load_run("r", tmp_path, "oneshot")
    ev = ac.sample_eval_set(run)
    assert ev["y"].tolist() == [False] * 4 + [True] * 4 and (ev["cell"][:4] == ev["cell"][4:]).all()
    m = ac.analyse_run("o", run, ev, {"bss_ref_rate": 0.5}, P, tmp_path / "fig", 0)
    assert m["test_auroc"] == 1.0 and m["test_baseline_size_auroc"] == 0.5
    assert m["test_n_cross_item"] == 4 and m["test_n_intra_document:m5:k2"] == 2 and m["test_detect_intra_document"] == 1.0


def test_samples_run_refuses_a_broken_pair(tmp_path):
    _write_run(tmp_path, "r", "conditional", _sample_cols(LINE_ROWS[:-1]))
    with pytest.raises(AssertionError, match="not held whole"):
        ac.sample_eval_set(ac.load_run("r", tmp_path, "conditional"))
