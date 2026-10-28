"""Unit tests for analysis/efficiency.py on a hand-built fixture of two tasks (entity, relation).

Methods: language models ``qwen`` (layer 2) and ``llama`` (layer 3), judges ``clef`` and ``jev``. Each task has a scale ``k``
(entity 1, relation 10). Every number below is worked by hand.

LM run (2 prompts): prompt 0.1 x2, trace ``[1, 2] * k`` (train) / ``[3, 4] * k`` (test), readout 0.01 x2, cleanup 0.5 x2.
    seconds without cleanup: train 0.2 + 3k + 0.02, test 0.2 + 7k + 0.02; cleanup adds 1.0 per run.
    prompt tokens train 30k, test 70k, of which computed = tokens // 10 (3k, 7k) and reused the rest (27k, 63k).
    GPU peak alloc: train 20 GB, test 24 GB (reserved +2); probed layer: 1 GB per run.
Clef run (2 records): encode 0.1 x2, forward ``[5, 5] * k``, readout 0.01 x2 -> 10k + 0.22 s; tokens 40k; GPU peak alloc 30 GB.
Jev run (2 requests): seconds_request ``[1, 2] * k`` -> 3k s; input tokens 50k.

Totals over k = 1 and k = 10 (sum of k = 11):
    qwen time:  train 0.22*2 + 3*11 = 33.44, test 0.22*2 + 7*11 = 77.44
    tokens:     computed train 33, test 77 (the bars); reused train 297, test 693 (metrics only) (qwen and llama alike)
    memory:     model max(20, 24) = 24 GB, activations 4 x 1 GB = 4 GB; clef 30 GB; jev none
    clef time 10*11 + 0.22*2 = 110.44, jev time 33
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

GB = 10**9
GPU = "NVIDIA L40S"
LAYERS = {"qwen": 2, "llama": 3}
KS = {"entity": 1, "relation": 10}
DATASET = {"entity": "main.py", "relation": "line.py"}
LM_TASK = {"entity": "entity_detection", "relation": "relation_detection"}
JUDGE_PARAMS = {"entity": {"split": "test"}, "relation": {"samples_path": "subsets.json"}}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def write_snapshot(root: Path, run_id: str, task: str, **params) -> None:
    (root / run_id).mkdir(parents=True, exist_ok=True)
    (root / run_id / "config.snapshot.yaml").write_text(yaml.safe_dump({"params": {"dataset_config": DATASET[task], **params}}))


def write_lm_run(root: Path, run_id: str, task: str, split: str, trace: list, tokens: list, alloc_gb: int, layer: int, resumed: int = 0,
                 bump: float = 1.0) -> None:
    cols = {"seconds_prompt": np.full(2, 0.1), "seconds_trace": np.array(trace, float), "seconds_readout": np.full(2, 0.01),
            "seconds_cleanup": np.full(2, 0.5), "prompt_n_tokens": np.array(tokens), "computed_n_tokens": np.array(tokens) // 10,
            "reused_n_tokens": np.array(tokens) - np.array(tokens) // 10, "peak_alloc_bytes": np.full(2, alloc_gb * GB),
            "peak_reserved_bytes": np.full(2, (alloc_gb + 2) * GB)}
    d = root / run_id
    write_json(d / "run.json", {"status": "success", "task": LM_TASK[task]})
    write_json(d / "metrics.json", {"n_prompts": 2})
    write_snapshot(root, run_id, task, split=split)
    em = {"eff_n_prompts": 2, "eff_n_shards_resumed": resumed, "eff_prompt_n_tokens_total": float(cols["prompt_n_tokens"].sum()) * bump,
          "eff_reused_tokens_total": float(cols["reused_n_tokens"].sum()), "eff_computed_tokens_total": float(cols["computed_n_tokens"].sum()),
          "eff_gpu_peak_alloc_bytes_max": alloc_gb * GB, "eff_gpu_peak_reserved_bytes_max": (alloc_gb + 2) * GB,
          "eff_final_write_seconds": 7.0, f"eff_rep_bytes_layer_{layer}": GB}
    em.update({f"eff_{c}_total": float(cols[c].sum()) for c in ("seconds_prompt", "seconds_trace", "seconds_readout", "seconds_cleanup")})
    write_json(d / "artifacts" / "efficiency.json", {"hardware": {"gpu_name": GPU}, "layers": [0, layer], "metrics": em})
    np.savez(d / "artifacts" / "efficiency_rows.npz", **cols)


def write_clef_run(root: Path, run_id: str, task: str, k: int) -> None:
    cols = {"seconds_encode": np.full(2, 0.1), "seconds_forward": np.array([5.0, 5.0]) * k, "seconds_readout": np.full(2, 0.01),
            "n_tokens": np.array([15, 25]) * k, "n_questions": np.array([1, 1]), "peak_alloc_bytes": np.full(2, 30 * GB),
            "peak_reserved_bytes": np.full(2, 32 * GB)}
    d = root / run_id
    write_json(d / "run.json", {"status": "success", "task": f"clef_{task}"})
    write_json(d / "metrics.json", {})
    write_snapshot(root, run_id, task, **JUDGE_PARAMS[task])
    em = {"eff_n_records": 2, "eff_input_tokens_total": float(cols["n_tokens"].sum()), "eff_gpu_peak_alloc_bytes_max": 30 * GB,
          "eff_gpu_peak_reserved_bytes_max": 32 * GB}
    em.update({f"eff_{c}_total": float(cols[c].sum()) for c in ("seconds_encode", "seconds_forward", "seconds_readout")})
    write_json(d / "artifacts" / "efficiency.json", {"hardware": {"gpu_name": GPU}, "metrics": em})
    np.savez(d / "artifacts" / "efficiency_records.npz", **cols)


def write_jev_run(root: Path, run_id: str, task: str, k: int) -> None:
    cols = {"seconds_request": np.array([1.0, 2.0]) * k, "input_tokens": np.array([20, 30]) * k, "output_tokens": np.array([1, 1]),
            "n_questions": np.array([1, 1])}
    d = root / run_id
    write_json(d / "run.json", {"status": "success", "task": f"jev_{task}"})
    write_json(d / "metrics.json", {"eff_n_requests": 2, "eff_seconds_request_total": float(cols["seconds_request"].sum()),
                                    "eff_input_tokens_total": float(cols["input_tokens"].sum())})
    write_snapshot(root, run_id, task, **JUDGE_PARAMS[task])
    (d / "artifacts").mkdir(parents=True, exist_ok=True)
    np.savez(d / "artifacts" / "efficiency_records.npz", **cols)


def make_task(root: Path, task: str) -> dict:
    """Every run of one task, and its entry of ``params.tasks``."""
    k = KS[task]
    lm = {}
    for label, layer in LAYERS.items():
        tr, te = f"{task}-{label}-train", f"{task}-{label}-test"
        write_lm_run(root, tr, task, "train", [1 * k, 2 * k], [10 * k, 20 * k], 20, layer)
        write_lm_run(root, te, task, "test", [3 * k, 4 * k], [30 * k, 40 * k], 24, layer)
        lm[label] = {"train_run_id": tr, "test_run_id": te, "layer": layer}
    write_clef_run(root, f"{task}-clef", task, k)
    write_jev_run(root, f"{task}-jev", task, k)
    return {"name": task, "dataset_config": DATASET[task], "lm_task": LM_TASK[task], "lm": lm, "judge_params": JUDGE_PARAMS[task],
            "judges": {"clef": {"judge": "clef", "task": f"clef_{task}", "run_id": f"{task}-clef"},
                       "jev": {"judge": "jev", "task": f"jev_{task}", "run_id": f"{task}-jev"}}}


def cfg_params(root: Path, **over) -> dict:
    segs = lambda *keys: [{"key": k, "label": k, "hatch": h} for k, h in keys]
    p = {"methods": {"qwen": "Qwen", "llama": "Llama", "clef": "Clef", "jev": "Jev"}, "tasks": [make_task(root, t) for t in KS],
         "gpu_name": GPU, "include_cleanup": False, "include_final_write": False, "gpu_memory": "peak_alloc",
         "figures": {"compute_time": {"ylabel": "Compute time (s)", "divisor": 1, "segments": segs(("train_activations", "////"), ("test_activations", "...."))},
                     "token_usage": {"ylabel": "Input tokens (millions)", "divisor": 10**6, "segments": segs(("train_activations", "////"), ("test_activations", "...."))},
                     "memory_usage": {"ylabel": "Memory (GB)", "divisor": GB, "segments": segs(("model", "////"), ("activations", "...."))}},
         "font": "Nimbus Roman", "white_frac": 0.35, "figsize": [3.0, 2.6], "bar_width": 0.7, "edge_color": "0.25", "edge_linewidth": 0.6,
         "hatch_linewidth": 0.6, "legend_ncol": 3, "legend_fontsize": 8, "legend_hatch_face": "0.9"}
    p.update(over)
    return p


@pytest.fixture
def world(tmp_path):
    return tmp_path / "runs"


def load_all(root: Path, p: dict) -> dict:
    return ef.aggregate({t["name"]: ef.load_task(t, p["methods"], root, p) for t in p["tasks"]}, p["methods"])


def test_totals_by_hand(world):
    agg = load_all(world, cfg_params(world))
    t = agg["compute_time"]["total"]
    assert t["qwen"] == pytest.approx({"train_activations": 33.44, "test_activations": 77.44})
    assert t["llama"] == pytest.approx(t["qwen"])
    assert t["clef"] == pytest.approx({"decisions": 110.44})
    assert t["jev"] == pytest.approx({"decisions": 33.0})
    assert agg["compute_time"]["entity"]["qwen"]["train_activations"] == pytest.approx(3.22)
    assert agg["compute_time"]["relation"]["qwen"]["test_activations"] == pytest.approx(70.22)
    tok = agg["token_usage"]["total"]
    assert tok["qwen"] == {"train_activations": 33.0, "test_activations": 77.0}   # computed = tokens // 10
    assert tok["clef"] == {"decisions": 440.0} and tok["jev"] == {"decisions": 550.0}
    mem = agg["memory_usage"]["total"]
    assert mem["qwen"] == {"model": 24.0 * GB, "activations": 4.0 * GB}
    assert mem["clef"] == {"decisions": 30.0 * GB}
    assert "jev" not in mem


def test_definition_switches_move_the_bars(world):
    p = cfg_params(world)
    base = load_all(world, p)["compute_time"]["total"]["qwen"]
    cleanup = load_all(world, {**p, "include_cleanup": True})["compute_time"]["total"]["qwen"]
    assert cleanup["train_activations"] - base["train_activations"] == pytest.approx(2.0)   # 1.0 s per run, two tasks
    write = load_all(world, {**p, "include_final_write": True})["compute_time"]["total"]["qwen"]
    assert write["test_activations"] - base["test_activations"] == pytest.approx(14.0)
    reserved = load_all(world, {**p, "gpu_memory": "peak_reserved"})["memory_usage"]["total"]
    assert reserved["qwen"]["model"] == 26.0 * GB and reserved["clef"]["decisions"] == 32.0 * GB


def test_gates_fail_loud(world):
    p = cfg_params(world)
    with pytest.raises(AssertionError, match="GPU"):
        load_all(world, {**p, "gpu_name": "NVIDIA A100"})
    write_lm_run(world, "entity-qwen-train", "entity", "train", [1, 2], [10, 20], 20, 2, resumed=1)
    with pytest.raises(AssertionError, match="resumed"):
        load_all(world, p)
    write_lm_run(world, "entity-qwen-train", "entity", "train", [1, 2], [10, 20], 20, 2, bump=1.5)
    with pytest.raises(AssertionError, match="eff_prompt_n_tokens_total"):
        load_all(world, p)
    write_lm_run(world, "entity-qwen-train", "relation", "train", [1, 2], [10, 20], 20, 2)
    with pytest.raises(AssertionError, match="task|dataset_config"):
        load_all(world, p)
    write_lm_run(world, "entity-qwen-train", "entity", "train", [1, 2], [10, 20], 20, 2)
    cols = dict(np.load(world / "entity-qwen-train" / "artifacts" / "efficiency_rows.npz"))
    cols["reused_n_tokens"] = cols["reused_n_tokens"] + 1   # reused + computed no longer the prompt length
    np.savez(world / "entity-qwen-train" / "artifacts" / "efficiency_rows.npz", **cols)
    with pytest.raises(AssertionError, match="reused \\+ computed"):
        load_all(world, p)
    del cols["reused_n_tokens"]
    np.savez(world / "entity-qwen-train" / "artifacts" / "efficiency_rows.npz", **cols)
    with pytest.raises(AssertionError, match="reused_n_tokens"):
        load_all(world, p)


def test_data_gate_refuses_the_wrong_split_or_file(world):
    p = cfg_params(world)
    write_snapshot(world, "entity-qwen-test", "entity", split="train")   # a test run that judged the train split
    with pytest.raises(AssertionError, match="split"):
        load_all(world, p)
    write_snapshot(world, "entity-qwen-test", "entity", split="test")
    write_snapshot(world, "relation-jev", "relation", samples_path="conditional-test-parents.json")
    with pytest.raises(AssertionError, match="samples_path"):
        load_all(world, p)


def test_methods_must_match_the_config(world):
    p = cfg_params(world)
    del p["tasks"][1]["lm"]["llama"]
    with pytest.raises(AssertionError, match="params.methods"):
        load_all(world, p)


def test_run_writes_metrics_and_figures(world, tmp_path, monkeypatch):
    monkeypatch.setattr(ef, "FIGURES_DIR", tmp_path / "figures")
    m = ef.run({"id": "eff", "params": cfg_params(world)}, world)
    figs = tmp_path / "figures" / "eff"
    for name in ef.FIGURES:
        assert (figs / f"{name}.pdf").stat().st_size > 0 and (figs / f"{name}_legend.pdf").stat().st_size > 0
    assert m["total_compute_time_qwen_sum"] == pytest.approx(33.44 + 77.44)
    assert not any("probe" in k for k in m)
    assert m["relation_token_usage_jev_decisions"] == 500.0
    assert m["total_token_usage_qwen_sum"] == 110.0   # computed only
    assert m["total_reused_tokens_qwen_train"] == 297.0 and m["total_reused_tokens_qwen_test"] == 693.0   # computed + reused = the full prompts
    assert m["total_memory_usage_qwen_sum"] == 28.0 * GB
    assert "total_memory_usage_jev_decisions" not in m
    assert json.loads((world / "eff" / "metrics.json").read_text()) == m
