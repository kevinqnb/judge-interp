"""Unit tests for analysis/efficiency.py on hand-built runs.

Representation run "rep" (3 prompts, per-prompt seconds):
    prompt  0.1 0.1 0.1   trace 1 2 3   readout 0.01 x3   cleanup 0.5 x3   tokens 100 200 300   spans 2 3 4
    -> per-prompt total 1.61 2.61 3.61 (sum 7.83, mean 2.61); excluding cleanup 1.11 2.11 3.11 (sum 6.33)
    -> cleanup share 1.5 / 7.83; seconds per span 7.83 / 9 (6.33 / 9 without cleanup)
    -> trace = tokens / 100 exactly: slope 10 s per 1k tokens, intercept 0, R^2 1; compute tokens/s = 600 / 6
Clef run "clef" (2 records):
    encode 0.1 0.1   forward 1 3   readout 0.01 x2   tokens 100 300   questions 2 2
    -> per-record total 1.11 3.11 (mean 2.11); seconds per question 4.22 / 4; forward = tokens / 100 as well
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

spec = importlib.util.spec_from_file_location("analysis_efficiency", Path(__file__).resolve().parents[1] / "analysis" / "efficiency.py")
ef = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ef)

GPU = "NVIDIA L40S"
GIB = 2**30


def rep_columns():
    return {"seconds_prompt": np.full(3, 0.1), "seconds_trace": np.array([1.0, 2.0, 3.0]), "seconds_readout": np.full(3, 0.01),
            "seconds_cleanup": np.full(3, 0.5), "prompt_n_tokens": np.array([100, 200, 300]), "n_spans": np.array([2, 3, 4]),
            "peak_alloc_bytes": np.array([10, 20, 30]) * GIB, "peak_reserved_bytes": np.array([11, 21, 31]) * GIB}


def clef_columns():
    return {"seconds_encode": np.full(2, 0.1), "seconds_forward": np.array([1.0, 3.0]), "seconds_readout": np.full(2, 0.01),
            "n_tokens": np.array([100, 300]), "n_questions": np.array([2, 2]),
            "peak_alloc_bytes": np.array([12, 24]) * GIB, "peak_reserved_bytes": np.array([13, 25]) * GIB}


def write_run(root: Path, run_id: str, kind: str, cols: dict, gpu: str = GPU, status: str = "success", bump: dict | None = None):
    """A run directory with the files efficiency.py reads; ``bump`` perturbs a json total to test the cross-check."""
    d = root / run_id
    (d / "artifacts").mkdir(parents=True)
    task = {"rep": "relation_detection", "clef": "clef_entity_row"}[kind]
    (d / "run.json").write_text(json.dumps({"status": status, "task": task}))
    if kind == "rep":
        n = len(cols["seconds_trace"])
        m = {"eff_n_prompts": n, "eff_n_shards_resumed": 0, "eff_seconds_prompt_total": cols["seconds_prompt"].sum(),
             "eff_seconds_trace_total": cols["seconds_trace"].sum(), "eff_seconds_readout_total": cols["seconds_readout"].sum(),
             "eff_seconds_cleanup_total": cols["seconds_cleanup"].sum(),
             "eff_seconds_per_prompt_total": sum(cols[k].sum() for k in ("seconds_prompt", "seconds_trace", "seconds_readout", "seconds_cleanup")),
             "eff_prompt_n_tokens_total": cols["prompt_n_tokens"].sum(), "eff_gpu_peak_alloc_bytes_max": cols["peak_alloc_bytes"].max(),
             "eff_gpu_weights_bytes": 8 * GIB, "eff_gpu_peak_activation_bytes_max": cols["peak_alloc_bytes"].max() - 8 * GIB,
             "eff_rep_bytes_per_span_per_layer": 16, "eff_npz_file_bytes": 4 * GIB}
        metrics = {"n_prompts": n, "n_spans": int(cols["n_spans"].sum())}
        np.savez(d / "artifacts" / "efficiency_rows.npz", **cols)
        extra = {"shard_gpu_name": gpu}
    else:
        n = len(cols["seconds_forward"])
        m = {"eff_n_records": n, "eff_seconds_encode_total": cols["seconds_encode"].sum(), "eff_seconds_forward_total": cols["seconds_forward"].sum(),
             "eff_seconds_readout_total": cols["seconds_readout"].sum(),
             "eff_seconds_per_record_total": sum(cols[k].sum() for k in ("seconds_encode", "seconds_forward", "seconds_readout")),
             "eff_input_tokens_total": cols["n_tokens"].sum(), "eff_n_questions_total": cols["n_questions"].sum(),
             "eff_gpu_peak_alloc_bytes_max": cols["peak_alloc_bytes"].max(), "eff_gpu_weights_bytes": 8 * GIB,
             "eff_gpu_peak_activation_bytes_max": cols["peak_alloc_bytes"].max() - 8 * GIB}
        metrics = {"n_predictions": n}
        np.savez(d / "artifacts" / "efficiency_records.npz", **cols)
        extra = {}
    m = {k: float(v) for k, v in m.items()}
    m.update({k: float(v) for k, v in (bump or {}).items()})
    (d / "artifacts" / "efficiency.json").write_text(json.dumps({"hardware": {"gpu_name": gpu}, "metrics": m, **extra}))
    (d / "metrics.json").write_text(json.dumps(metrics))


@pytest.fixture
def runs(tmp_path):
    write_run(tmp_path, "rep", "rep", rep_columns())
    write_run(tmp_path, "clef", "clef", clef_columns())
    return tmp_path


def test_rep_hand_worked(runs):
    m = ef.analyse_run(ef.load_run("rep", runs, "rep"))
    assert m["n_units"] == 3 and m["seconds_unit_mean"] == pytest.approx(2.61) and m["seconds_unit_median"] == pytest.approx(2.61)
    assert m["seconds_unit_excl_cleanup_mean"] == pytest.approx(2.11)
    assert m["cleanup_share"] == pytest.approx(1.5 / 7.83)
    assert m["seconds_per_item"] == pytest.approx(7.83 / 9) and m["seconds_per_item_excl_cleanup"] == pytest.approx(6.33 / 9)
    assert m["seconds_compute_mean"] == pytest.approx(2.0)
    assert m["tokens_mean"] == 200 and m["compute_tokens_per_second"] == pytest.approx(100.0)
    assert m["fit_seconds_per_1k_tokens"] == pytest.approx(10.0) and m["fit_intercept_seconds"] == pytest.approx(0.0, abs=1e-9)
    assert m["fit_r2"] == pytest.approx(1.0)
    assert m["first_over_median"] == pytest.approx(1.61 / 2.61)
    assert m["peak_alloc_gib_max"] == pytest.approx(30.0) and m["weights_gib"] == pytest.approx(8.0)
    assert m["peak_activation_gib_max"] == pytest.approx(22.0) and m["npz_file_gib"] == pytest.approx(4.0)
    assert m["total_hours"] == pytest.approx(7.83 / 3600)


def test_clef_hand_worked_and_has_no_cleanup(runs):
    m = ef.analyse_run(ef.load_run("clef", runs, "clef"))
    assert m["n_units"] == 2 and m["seconds_unit_mean"] == pytest.approx(2.11)
    assert m["seconds_unit_excl_cleanup_mean"] == m["seconds_unit_mean"] and m["cleanup_share"] == 0.0
    assert m["seconds_per_item"] == pytest.approx(4.22 / 4)
    assert m["fit_seconds_per_1k_tokens"] == pytest.approx(10.0) and m["fit_r2"] == pytest.approx(1.0)
    assert "npz_file_gib" not in m


def test_fit_known_answer_and_degenerate_tokens():
    slope, intercept, r2 = ef.linear_fit(np.array([1.0, 2.0, 3.0]), np.array([3.0, 5.0, 7.0]))
    assert (slope, intercept, r2) == (pytest.approx(2.0), pytest.approx(1.0), pytest.approx(1.0))
    with pytest.raises(AssertionError, match="variation"):
        ef.linear_fit(np.array([5.0, 5.0]), np.array([1.0, 2.0]))


def test_a_json_total_that_disagrees_with_the_unit_arrays_is_refused(tmp_path):
    write_run(tmp_path, "bad", "rep", rep_columns(), bump={"eff_seconds_trace_total": 7.0})
    with pytest.raises(AssertionError, match="eff_seconds_trace_total"):
        ef.load_run("bad", tmp_path, "rep")


def test_a_failed_run_or_wrong_kind_is_refused(tmp_path):
    write_run(tmp_path, "failed", "rep", rep_columns(), status="failed")
    with pytest.raises(AssertionError, match="status"):
        ef.load_run("failed", tmp_path, "rep")
    write_run(tmp_path, "rep", "rep", rep_columns())
    with pytest.raises(AssertionError, match="task"):
        ef.load_run("rep", tmp_path, "clef")


def test_resumed_shards_are_refused(tmp_path):
    write_run(tmp_path, "r", "rep", rep_columns(), bump={"eff_n_shards_resumed": 2})
    with pytest.raises(AssertionError, match="resumed"):
        ef.load_run("r", tmp_path, "rep")


def test_mixed_gpus_are_refused(runs, tmp_path):
    write_run(tmp_path / "x", "a100", "rep", rep_columns(), gpu="NVIDIA A100")
    with pytest.raises(AssertionError, match="GPU"):
        ef.assert_gpu(ef.load_run("a100", tmp_path / "x", "rep"), GPU)
    ef.assert_gpu(ef.load_run("rep", runs, "rep"), GPU)


def test_ratio_of_a_metric_between_two_labelled_runs():
    per = {"a": {"seconds_unit_mean": 6.0}, "b": {"seconds_unit_mean": 2.0}}
    assert ef.ratio(per, {"name": "a_over_b", "numerator": "a", "denominator": "b", "metric": "seconds_unit_mean"}) == ("a_over_b", 3.0)
    with pytest.raises(KeyError):
        ef.ratio(per, {"name": "x", "numerator": "a", "denominator": "c", "metric": "seconds_unit_mean"})


def test_run_end_to_end_writes_flat_scalar_metrics(runs, tmp_path, monkeypatch):
    monkeypatch.setattr(ef, "FIGURES_DIR", tmp_path / "figures")
    cfg = {"id": "eff-test", "seed": 0, "params": {
        "gpu_name": GPU, "runs": [{"label": "r", "kind": "rep", "run_id": "rep"}, {"label": "c", "kind": "clef", "run_id": "clef"}],
        "comparisons": [{"name": "c_over_r", "numerator": "c", "denominator": "r", "metric": "seconds_unit_mean"}]}}
    m = ef.run(cfg, runs)
    assert m["r_seconds_unit_mean"] == pytest.approx(2.61) and m["ratio_c_over_r"] == pytest.approx(2.11 / 2.61)
    assert all(isinstance(v, (int, float)) for v in m.values())
    assert json.loads((runs / "eff-test" / "metrics.json").read_text()) == m
    assert (tmp_path / "figures" / "eff-test" / "phases.png").exists()
