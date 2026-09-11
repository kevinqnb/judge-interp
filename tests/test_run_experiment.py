"""Unit tests for scripts/run_experiment.py's pure eval code (compute_metrics)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import run_experiment as run  # noqa: E402


def _result(labels, verdict_true, verdict_recognised, p_true):
    return {
        "labels": np.asarray(labels, dtype=bool),
        "verdict_true": np.asarray(verdict_true, dtype=bool),
        "verdict_recognised": np.asarray(verdict_recognised, dtype=bool),
        "p_true": np.asarray(p_true, dtype=np.float64),
    }


def test_compute_metrics_hand_countable():
    # 2 valid rows (verdict correct, then wrong), 2 invalid rows (wrong, then correct).
    result = _result(
        labels=[True, True, False, False],
        verdict_true=[True, False, True, False],
        verdict_recognised=[True, True, True, True],
        p_true=[0.9, 0.4, 0.8, 0.1],
    )
    m = run.compute_metrics(result)
    assert m["n_rows"] == 4 and m["n_valid"] == 2 and m["n_invalid"] == 2
    assert m["verdict_recognised_rate"] == 1.0
    assert m["judge_accuracy"] == 0.5
    assert m["judge_acc_valid"] == 0.5
    assert m["judge_acc_invalid"] == 0.5
    assert m["mean_p_true_valid"] == 0.65
    assert m["mean_p_true_invalid"] == 0.45


def test_compute_metrics_all_unrecognised_gives_null_not_nan():
    result = _result(
        labels=[True, False],
        verdict_true=[True, False],
        verdict_recognised=[False, False],
        p_true=[0.6, 0.2],
    )
    m = run.compute_metrics(result)
    assert m["verdict_recognised_rate"] == 0.0
    assert m["judge_accuracy"] is None
    assert m["judge_acc_valid"] is None
    assert m["judge_acc_invalid"] is None
    # p_true-based means don't depend on recognition and are still computed.
    assert m["mean_p_true_valid"] == 0.6
    assert m["mean_p_true_invalid"] == 0.2

    # metrics.json must stay valid JSON -- json.dumps(float('nan')) writes bare
    # NaN, which json.loads then rejects. None -> "null" round-trips cleanly.
    text = json.dumps(m)
    assert "NaN" not in text
    assert json.loads(text)["judge_accuracy"] is None


def test_compute_metrics_empty_class_mask():
    # No invalid rows at all -- judge_acc_invalid / mean_p_true_invalid undefined.
    result = _result(
        labels=[True, True],
        verdict_true=[True, True],
        verdict_recognised=[True, True],
        p_true=[0.7, 0.8],
    )
    m = run.compute_metrics(result)
    assert m["n_invalid"] == 0
    assert m["judge_acc_invalid"] is None
    assert m["mean_p_true_invalid"] is None
    assert m["judge_acc_valid"] == 1.0
