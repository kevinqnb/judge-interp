"""Efficiency measurement: hand-built fixtures on CPU, where the GPU-memory fields must be None."""
from types import SimpleNamespace

import numpy as np
import pytest

from judge_interp import clef, efficiency as eff
from judge_interp import row_detection as rd


def test_meter_on_cpu_has_no_gpu_numbers_and_a_monotonic_clock():
    m = eff.Meter("cpu")
    assert m.peaks() == {"peak_alloc_bytes": None, "peak_reserved_bytes": None}
    assert m.allocated_bytes() is None
    t0 = m.now()
    assert m.now() >= t0


def test_meter_refuses_cuda_when_unavailable():
    import torch
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        eff.Meter("cuda")


def test_column_stats_hand_worked():
    s = eff.column_stats("x", [1.0, 2.0, 3.0, 4.0])
    assert s == {"x_total": 10.0, "x_mean": 2.5, "x_median": 2.5, "x_p95": pytest.approx(3.85)}


def test_peak_stats_none_on_cpu_and_mixed_is_an_error():
    assert eff.peak_stats("p", [None, None]) == {"p_max": None, "p_mean": None, "p_median": None}
    assert eff.peak_stats("p", [1, 3])["p_max"] == 3.0
    with pytest.raises(AssertionError):
        eff.peak_stats("p", [1, None])


def test_npz_member_sizes_reports_float32_array_bytes(tmp_path):
    path = tmp_path / "a.npz"
    np.savez_compressed(path, rep_8=np.zeros((10, 4), np.float32))
    full, comp = eff.npz_member_sizes(path)["rep_8"]
    assert full == 10 * 4 * 4 + 128 and comp < full   # .npy header is 128 bytes


def _rows(n_prompts, spans_each):
    return {
        "seconds_prompt": np.full(n_prompts, 0.1), "seconds_trace": np.full(n_prompts, 0.5),
        "seconds_readout": np.full(n_prompts, 0.2), "seconds_cleanup": np.full(n_prompts, 0.2),
        "prompt_n_tokens": np.full(n_prompts, 1000), "n_spans": np.full(n_prompts, spans_each),
        "peak_alloc_bytes": np.full(n_prompts, -1), "peak_reserved_bytes": np.full(n_prompts, -1),
    }


def test_row_write_efficiency_hand_worked_on_cpu(tmp_path):
    (tmp_path / "artifacts").mkdir()
    n_prompts, spans_each, hidden = 4, 3, 8
    n = n_prompts * spans_each
    result = {
        "layers": np.array([0, 5]),
        "representations": {L: np.zeros((n, hidden), np.float32) for L in (0, 5)},
        "efficiency": {"rows": _rows(n_prompts, spans_each), "gpu_name": "cpu", "n_shards_resumed": 1},
    }
    npz = tmp_path / "artifacts" / "span_representations.npz"
    rd.write_span_representations({k: v for k, v in result.items() if k != "efficiency"}, npz)
    rlm = SimpleNamespace(hidden_size=hidden, weights_bytes=None, shard_write_seconds=[0.5, 0.25], device="cpu", model_name="m")
    m = rd.write_efficiency(tmp_path, rlm, result, npz, dict(
        model_load_seconds=1.0, build_items_seconds=2.0, collect_seconds=3.0, final_write_seconds=4.0))
    assert m["eff_seconds_per_prompt_total"] == pytest.approx(4 * 1.0)
    assert m["eff_seconds_per_span_mean"] == pytest.approx(4.0 / 12)
    assert m["eff_prompt_n_tokens_total"] == 4000 and m["eff_output_tokens_total"] == 0
    assert m["eff_rep_bytes_layer_0"] == n * hidden * 4 == m["eff_rep_bytes_layer_5"]
    assert m["eff_rep_bytes_per_span_per_layer"] == hidden * 4
    assert m["eff_gpu_peak_alloc_bytes_max"] is None and m["eff_gpu_peak_activation_bytes_max"] is None
    assert m["eff_n_shards_resumed"] == 1 and m["eff_n_shards_written"] == 2
    assert m["eff_shard_write_seconds_total"] == 0.75
    assert (tmp_path / "artifacts" / "efficiency.json").is_file()
    assert "efficiency" not in np.load(npz).files


def test_span_representations_npz_drops_efficiency_key(tmp_path):
    rd.write_span_representations(
        {"layers": np.array([0]), "representations": {0: np.zeros((1, 2), np.float32)}, "efficiency": {"rows": {}}, "x": np.arange(1)},
        tmp_path / "s.npz")
    assert sorted(np.load(tmp_path / "s.npz").files) == ["layers", "rep_0", "x"]


def test_clef_write_efficiency_hand_worked_on_cpu(tmp_path):
    (tmp_path / "artifacts").mkdir()
    ef = lambda q, t: {"seconds_encode": 0.1, "seconds_forward": 0.3, "seconds_readout": 0.1, "n_tokens": t,
                       "n_questions": q, "peak_alloc_bytes": None, "peak_reserved_bytes": None}
    results = [{"efficiency": ef(2, 100)}, {"efficiency": ef(3, 300)}]
    units = [{"n_tokens": 100}, {"n_tokens": 300}]
    scorer = SimpleNamespace(meter=eff.Meter("cpu"), weights_bytes=None, device="cpu", model=object())
    m = clef.write_efficiency(tmp_path, scorer, units, results, {"model_load_seconds": 9.0}, True)
    assert m["eff_n_records"] == 2 and m["eff_n_questions_total"] == 5
    assert m["eff_input_tokens_total"] == 400 and m["eff_output_tokens_total"] == 0
    assert m["eff_seconds_per_question_mean"] == pytest.approx(1.0 / 5)
    assert m["eff_input_tokens_per_second_forward"] == pytest.approx(400 / 0.6)
    assert m["eff_gpu_peak_alloc_bytes_max"] is None and m["eff_model_load_seconds"] == 9.0


def test_clef_write_efficiency_refuses_token_count_mismatch(tmp_path):
    (tmp_path / "artifacts").mkdir()
    r = {"efficiency": {"seconds_encode": .1, "seconds_forward": .1, "seconds_readout": .1, "n_tokens": 5,
                        "n_questions": 1, "peak_alloc_bytes": None, "peak_reserved_bytes": None}}
    scorer = SimpleNamespace(meter=eff.Meter("cpu"), weights_bytes=None, device="cpu", model=object())
    with pytest.raises(AssertionError):
        clef.write_efficiency(tmp_path, scorer, [{"n_tokens": 6}], [r], {}, True)
