"""Compute cost of running the four methods on the VRDU data: compute time, input tokens and memory, one bar chart each
(``compute_time.pdf``, ``token_usage.pdf``, ``memory_usage.pdf``, each with a ``*_legend.pdf``).

Usage: ``uv run --python 3.12 --extra analysis python analysis/efficiency.py analysis/configs/efficiency/<id>.yaml``
(every value lives in the config's ``params``; a missing key is a KeyError).

Independent of the evaluation analyses (which score stratified subsets): ``params.tasks`` lists, per task, the runs to cost
(the language models' train / test runs, the judges' runs); each whole run is counted, whatever rows an analysis later
scores. Each run's config snapshot is gated against the task's ``dataset_config`` and the split / samples file the
config expects (``lm_task`` + ``judge_params``), so a bar cannot silently cost another dataset.

Per method, summed over the tasks (the per-task split is in ``metrics.json``):

- language models (Qwen, Llama): compute time = the model's passes over the TRAIN and the TEST runs (activation
  collection); input tokens = the tokens the model COMPUTED on the TRAIN and TEST runs (prompt tokens served from the
  prefix cache are not computed; they are reported in ``metrics.json`` as ``reused_tokens_*``); memory = the GPU peak of running
  the model (max over the runs; they ran one after another) plus the activations saved for a probe, train + test (the
  probed layer's float32 ``eff_rep_bytes_layer_<layer>``). Probe training is not counted.
- Clef: the test run only (compute time, input tokens, GPU peak).
- Jev: the test run only (sum of request seconds, API-reported input tokens); an API model, so no memory.

Definitions that move the bars (all explicit in the config): ``include_cleanup`` (the per-row ``empty_cache`` + ``gc``
of the representation runs: harness overhead, ~1/3 of their time), ``include_final_write`` (writing the activations
to disk once the pass is done), ``gpu_memory`` (peak ``alloc`` or ``reserved``). Not counted: model loading.

A run's phases are read from its per-unit arrays and every total is checked against the run's own json (a unit is a
prompt for the representation runs, a record for Clef, a request for Jev). Gates: run status and task, no resumed
shards (timings would span sessions), every GPU run on ``params.gpu_name``, the dataset gate above.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Patch

_here = Path(__file__).parent
_cr_spec = importlib.util.spec_from_file_location("analysis_calibration_report", _here / "calibration_report.py")
cr = importlib.util.module_from_spec(_cr_spec)
_cr_spec.loader.exec_module(cr)

FIGURES_DIR = _here / "figures"
RTOL = 1e-9   # recomputed sums against the json's: float summation order only
GPU_MEMORY = ("peak_alloc", "peak_reserved")
FIGURES = ("compute_time", "token_usage", "memory_usage")
JUDGE_SEGMENT = "decisions"   # a judge's one segment: drawn solid, not in the hatch key


def _check_total(run_id: str, name: str, recomputed: float, json_value: float) -> None:
    assert np.isclose(recomputed, json_value, rtol=RTOL, atol=0.0), (
        f"{run_id}: {name} {json_value!r} != {recomputed!r} recomputed from the per-unit arrays")


def _load_run_files(run_id: str, runs_root: Path, expected_task: str, array_file: str) -> tuple[dict, dict, dict]:
    """``(metrics.json, per-unit columns, efficiency.json or {})`` of a successful run of ``expected_task``."""
    run_dir = runs_root / run_id
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["status"] == "success", f"{run_id}: status {manifest['status']!r}"
    assert manifest["task"] == expected_task, f"{run_id}: task {manifest['task']!r}, expected {expected_task!r}"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    with np.load(run_dir / "artifacts" / array_file, allow_pickle=False) as z:
        cols = {k: z[k] for k in z.files}
    n = len(next(iter(cols.values())))
    assert n > 0 and all(len(v) == n for v in cols.values()), f"{run_id}: empty or ragged columns in {array_file}"
    assert all(np.isfinite(v).all() and (v >= 0).all() for v in cols.values()), f"{run_id}: a negative or non-finite value in {array_file}"
    eff_path = run_dir / "artifacts" / "efficiency.json"
    eff = json.loads(eff_path.read_text()) if eff_path.exists() else {}
    return metrics, cols, eff


def _gpu_peaks(run_id: str, cols: dict, em: dict, gpu_name: str, eff: dict) -> None:
    assert eff["hardware"]["gpu_name"] == gpu_name, f"{run_id}: GPU {eff['hardware']['gpu_name']!r}, expected {gpu_name!r} (timings are not comparable)"
    assert (cols["peak_alloc_bytes"] > 0).all(), f"{run_id}: GPU peaks missing (a CPU run?)"
    for kind in GPU_MEMORY:
        _check_total(run_id, f"eff_gpu_{kind}_bytes_max", cols[f"{kind}_bytes"].max(), em[f"eff_gpu_{kind}_bytes_max"])


def load_lm_run(run_id: str, runs_root: Path, expected_task: str, layer: int, p: dict) -> dict:
    """A representation run: ``seconds`` (collection; see the module docstring for what ``include_*`` add), ``tokens``,
    ``gpu_bytes`` (peak over rows), ``activation_bytes`` (the probed layer, float32)."""
    metrics, cols, eff = _load_run_files(run_id, runs_root, expected_task, "efficiency_rows.npz")
    em = eff["metrics"]
    phases = ("seconds_prompt", "seconds_trace", "seconds_readout", "seconds_cleanup")
    for c in (*phases, "prompt_n_tokens", "reused_n_tokens", "computed_n_tokens", "peak_alloc_bytes", "peak_reserved_bytes"):
        assert c in cols, f"{run_id}: no column {c!r} in efficiency_rows.npz"
    assert len(cols["prompt_n_tokens"]) == int(em["eff_n_prompts"]) == int(metrics["n_prompts"]), f"{run_id}: unit counts disagree"
    assert int(em["eff_n_shards_resumed"]) == 0, f"{run_id}: resumed shards -- timings span sessions"
    for c in phases:
        _check_total(run_id, f"eff_{c}_total", cols[c].sum(), em[f"eff_{c}_total"])
    _check_total(run_id, "eff_prompt_n_tokens_total", cols["prompt_n_tokens"].sum(), em["eff_prompt_n_tokens_total"])
    assert (cols["reused_n_tokens"] + cols["computed_n_tokens"] == cols["prompt_n_tokens"]).all(), f"{run_id}: reused + computed != prompt tokens in some row"
    _check_total(run_id, "eff_reused_tokens_total", cols["reused_n_tokens"].sum(), em["eff_reused_tokens_total"])
    _check_total(run_id, "eff_computed_tokens_total", cols["computed_n_tokens"].sum(), em["eff_computed_tokens_total"])
    _gpu_peaks(run_id, cols, em, p["gpu_name"], eff)
    assert layer in eff["layers"], f"{run_id}: layer {layer} not in the run's layers {eff['layers']}"
    seconds = sum(float(cols[c].sum()) for c in phases if c != "seconds_cleanup" or p["include_cleanup"])
    if p["include_final_write"]:
        seconds += float(em["eff_final_write_seconds"])
    return {"run_id": run_id, "seconds": seconds, "tokens": float(cols["prompt_n_tokens"].sum()),
            "computed_tokens": float(cols["computed_n_tokens"].sum()), "reused_tokens": float(cols["reused_n_tokens"].sum()),
            "gpu_bytes": float(em[f"eff_gpu_{p['gpu_memory']}_bytes_max"]), "activation_bytes": float(em[f"eff_rep_bytes_layer_{layer}"])}


def load_clef_run(run_id: str, runs_root: Path, expected_task: str, p: dict) -> dict:
    """A Clef run: ``seconds`` (encode + forward + readout), ``tokens``, ``gpu_bytes``."""
    metrics, cols, eff = _load_run_files(run_id, runs_root, expected_task, "efficiency_records.npz")
    em = eff["metrics"]
    phases = ("seconds_encode", "seconds_forward", "seconds_readout")
    for c in (*phases, "n_tokens", "peak_alloc_bytes", "peak_reserved_bytes"):
        assert c in cols, f"{run_id}: no column {c!r} in efficiency_records.npz"
    assert len(cols["n_tokens"]) == int(em["eff_n_records"]), f"{run_id}: unit counts disagree"
    for c in phases:
        _check_total(run_id, f"eff_{c}_total", cols[c].sum(), em[f"eff_{c}_total"])
    _check_total(run_id, "eff_input_tokens_total", cols["n_tokens"].sum(), em["eff_input_tokens_total"])
    _gpu_peaks(run_id, cols, em, p["gpu_name"], eff)
    return {"run_id": run_id, "seconds": float(sum(cols[c].sum() for c in phases)), "tokens": float(cols["n_tokens"].sum()),
            "gpu_bytes": float(em[f"eff_gpu_{p['gpu_memory']}_bytes_max"])}


def load_jev_run(run_id: str, runs_root: Path, expected_task: str) -> dict:
    """A Jev run (an API model: no ``efficiency.json``, no GPU): ``seconds`` (sum of request seconds), ``tokens`` (API-reported input tokens)."""
    metrics, cols, _ = _load_run_files(run_id, runs_root, expected_task, "efficiency_records.npz")
    for c in ("seconds_request", "input_tokens"):
        assert c in cols, f"{run_id}: no column {c!r} in efficiency_records.npz"
    assert len(cols["input_tokens"]) == int(metrics["eff_n_requests"]), f"{run_id}: unit counts disagree"
    _check_total(run_id, "eff_seconds_request_total", cols["seconds_request"].sum(), metrics["eff_seconds_request_total"])
    _check_total(run_id, "eff_input_tokens_total", cols["input_tokens"].sum(), metrics["eff_input_tokens_total"])
    return {"run_id": run_id, "seconds": float(cols["seconds_request"].sum()), "tokens": float(cols["input_tokens"].sum())}


def check_data(run_id: str, runs_root: Path, dataset_config: str, expected: dict) -> None:
    """The run's config snapshot is of ``dataset_config`` and has every ``expected`` param (split / samples file)."""
    params = yaml.safe_load((runs_root / run_id / "config.snapshot.yaml").read_text())["params"]
    assert params["dataset_config"] == dataset_config, f"{run_id}: dataset_config {params['dataset_config']!r}, expected {dataset_config!r}"
    for k, v in expected.items():
        assert params[k] == v, f"{run_id}: {k} {params[k]!r}, expected {v!r}"


def load_task(t: dict, methods: dict, runs_root: Path, p: dict) -> dict:
    """One task's costs: ``lm``: label -> ``{train, test}`` runs; ``judge``: label -> ``{run}``. Its methods must be exactly
    ``methods`` (``params.methods``)."""
    lm_cfg, judge_cfg = t["lm"], t["judges"]
    assert set(lm_cfg) | set(judge_cfg) == set(methods) and not set(lm_cfg) & set(judge_cfg), (
        f"{t['name']}: methods {sorted(lm_cfg)} + {sorted(judge_cfg)} != params.methods {sorted(methods)}")
    lm, judge = {}, {}
    for label, m in lm_cfg.items():
        assert m["train_run_id"] != m["test_run_id"], f"{label}: train and test run are the same run"
        lm[label] = {}
        for split in ("train", "test"):
            check_data(m[f"{split}_run_id"], runs_root, t["dataset_config"], {"split": split})
            lm[label][split] = load_lm_run(m[f"{split}_run_id"], runs_root, t["lm_task"], m["layer"], p)
    for label, j in judge_cfg.items():
        check_data(j["run_id"], runs_root, t["dataset_config"], t["judge_params"])
        if j["judge"] == "clef":
            run = load_clef_run(j["run_id"], runs_root, j["task"], p)
        else:
            assert j["judge"] == "jev", j["judge"]
            run = load_jev_run(j["run_id"], runs_root, j["task"])
        judge[label] = {"judge": j["judge"], "run": run}
    return {"lm": lm, "judge": judge}


# ------------------------------------------------------------------------------ aggregation
def aggregate(tasks: dict[str, dict], methods: dict) -> dict:
    """``{figure: {task or "total": {method: {segment: value}}}}`` in seconds / tokens / bytes. Every task has exactly the
    methods ``methods`` (``params.methods``: label -> display name, in colour / legend order). Memory has no entry for a method without GPU memory (Jev)."""
    out: dict = {f: {} for f in FIGURES}
    for name, t in tasks.items():
        time, tokens, memory = {}, {}, {}
        for label in methods:
            if label in t["lm"]:
                r = t["lm"][label]
                time[label] = {"train_activations": r["train"]["seconds"], "test_activations": r["test"]["seconds"]}
                tokens[label] = {"train_activations": r["train"]["computed_tokens"], "test_activations": r["test"]["computed_tokens"]}
                memory[label] = {"model": max(r["train"]["gpu_bytes"], r["test"]["gpu_bytes"]),
                                 "activations": r["train"]["activation_bytes"] + r["test"]["activation_bytes"]}
            else:
                run = t["judge"][label]["run"]
                time[label], tokens[label] = {JUDGE_SEGMENT: run["seconds"]}, {JUDGE_SEGMENT: run["tokens"]}
                if "gpu_bytes" in run:
                    memory[label] = {JUDGE_SEGMENT: run["gpu_bytes"]}
        out["compute_time"][name], out["token_usage"][name], out["memory_usage"][name] = time, tokens, memory
    for fig, per_task in out.items():
        total: dict = {}
        n_tasks = len(per_task)
        for name, by_method in per_task.items():
            for label, segs in by_method.items():
                for seg, v in segs.items():
                    total.setdefault(label, {}).setdefault(seg, [])
                    total[label][seg].append((name, v))
        # memory's model segment is the max over tasks (runs were sequential), everything else adds
        out[fig]["total"] = {label: {seg: (max(v for _, v in vs) if (fig == "memory_usage" and seg in ("model", JUDGE_SEGMENT)) else sum(v for _, v in vs))
                                     for seg, vs in segs.items()} for label, segs in total.items()}
        for label, segs in total.items():
            assert all(len(vs) == n_tasks for vs in segs.values()), f"{fig}/{label}: a segment missing from a task"
    return out


# ------------------------------------------------------------------------------ figures
def method_colors(methods: dict, white_frac: float) -> dict[str, tuple]:
    """Tab10 colour (mixed with white) by position in ``params.methods`` (the calibration figures' order, so the same colours)."""
    assert len(methods) <= len(cr.TAB10), "more methods than tab10 colours"
    return {label: cr.pastel(cr.TAB10[i], white_frac) for i, label in enumerate(methods)}


def plot_bars(by_method: dict, methods: dict, fig_spec: dict, p: dict, out: Path) -> None:
    """One stacked bar per method (label-free x axis); a probe's segments are hatched (``fig_spec.segments``), a judge's is solid."""
    colors = method_colors(methods, p["white_frac"])
    hatch = {s["key"]: s["hatch"] for s in fig_spec["segments"]}
    cr.set_style(p["font"])
    plt.rcParams["hatch.linewidth"] = p["hatch_linewidth"]
    fig, ax = plt.subplots(figsize=tuple(p["figsize"]), constrained_layout=True)
    present = [m for m in methods if m in by_method]
    for i, label in enumerate(present):
        segs = by_method[label]
        keys = [JUDGE_SEGMENT] if JUDGE_SEGMENT in segs else [s["key"] for s in fig_spec["segments"]]
        assert set(segs) == set(keys), f"{label}: segments {sorted(segs)} != {keys}"
        bottom = 0.0
        for k in keys:
            h = segs[k] / fig_spec["divisor"]
            ax.bar(i, h, bottom=bottom, width=p["bar_width"], color=colors[label], edgecolor=p["edge_color"], linewidth=p["edge_linewidth"],
                   hatch=hatch.get(k, ""))
            bottom += h
    ax.set_xticks([])
    ax.set_xlim(-0.6, len(present) - 0.4)
    ax.set_ylabel(fig_spec["ylabel"])
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(out)
    plt.close(fig)


def plot_legend(present: list[str], methods: dict, fig_spec: dict, p: dict, out: Path) -> None:
    """The legend alone: the methods' colours, then (if any method is hatched) the hatch key."""
    colors = method_colors(methods, p["white_frac"])
    cr.set_style(p["font"])
    plt.rcParams["hatch.linewidth"] = p["hatch_linewidth"]
    handles = [Patch(facecolor=colors[m], edgecolor=p["edge_color"], linewidth=p["edge_linewidth"], label=methods[m]) for m in present]
    handles += [Patch(facecolor=p["legend_hatch_face"], edgecolor=p["edge_color"], linewidth=p["edge_linewidth"], hatch=s["hatch"], label=s["label"])
                for s in fig_spec["segments"]]
    fig = plt.figure(figsize=(1, 1))
    leg = fig.legend(handles=handles, loc="center", ncol=p["legend_ncol"], frameon=False, fontsize=p["legend_fontsize"])
    fig.canvas.draw()
    fig.savefig(out, bbox_inches=leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).padded(0.02))
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def run(cfg: dict, runs_root: Path) -> dict:
    p = cfg["params"]
    assert p["gpu_memory"] in GPU_MEMORY, p["gpu_memory"]
    assert set(p["figures"]) == set(FIGURES), sorted(p["figures"])
    out = runs_root / cfg["id"]
    fig_dir = FIGURES_DIR / cfg["id"]
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    names = [t["name"] for t in p["tasks"]]
    assert len(set(names)) == len(names) and names, names
    methods = p["methods"]
    tasks = {t["name"]: load_task(t, methods, runs_root, p) for t in p["tasks"]}
    agg = aggregate(tasks, methods)

    metrics: dict[str, float] = {}
    for fig_name, per_task in agg.items():
        spec = p["figures"][fig_name]
        for scope, by_method in per_task.items():
            for label, segs in by_method.items():
                for seg, v in segs.items():
                    metrics[f"{scope}_{fig_name}_{label}_{seg}"] = float(v)
                metrics[f"{scope}_{fig_name}_{label}_sum"] = float(sum(segs.values()))
        total = per_task["total"]
        plot_bars(total, methods, spec, p, fig_dir / f"{fig_name}.pdf")
        plot_legend([m for m in methods if m in total], methods, spec, p, fig_dir / f"{fig_name}_legend.pdf")
    for label in p["methods"]:
        if label in tasks[names[0]]["lm"]:
            for split in ("train", "test"):
                per_task = {n: tasks[n]["lm"][label][split]["reused_tokens"] for n in names}
                metrics.update({f"{n}_reused_tokens_{label}_{split}": v for n, v in per_task.items()})
                metrics[f"total_reused_tokens_{label}_{split}"] = sum(per_task.values())
    metrics_text = json.dumps(metrics, indent=2) + "\n"
    (out / "metrics.json").write_text(metrics_text)
    (fig_dir / "metrics.json").write_text(metrics_text)  # copy beside the figures
    print(metrics_text)
    return metrics


def main() -> None:
    cfg_path = Path(sys.argv[1])
    cfg = yaml.safe_load(cfg_path.read_text())
    runs_root = Path(os.environ["RUNS_ROOT"])
    out = runs_root / cfg["id"]
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cfg_path, out / "config.snapshot.yaml")
    run(cfg, runs_root)


if __name__ == "__main__":
    main()
