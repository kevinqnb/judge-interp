"""Computational performance of the judges: wall time, tokens, GPU memory and storage per judged row,
for the representation runs (entity / relation detection) and the Clef runs.

Usage: ``uv run --python 3.12 --extra analysis python analysis/efficiency.py analysis/configs/efficiency/<id>.yaml``
(every value lives in the config's ``params``; a missing key is a KeyError).

Reads each run named in ``params.runs`` (``artifacts/efficiency.json`` plus the per-unit
``efficiency_rows.npz`` / ``efficiency_records.npz`` that ``src/judge_interp/efficiency.py`` wrote) and
recomputes every statistic from the per-unit arrays. A unit is one judged row or sample: a prompt for
the representation runs, a record for Clef. Per unit the phases are

- representation: ``prompt`` (tokenise), ``trace`` (forward pass; the compute), ``readout`` (device to
  host), ``cleanup`` (``empty_cache`` + ``gc`` after every row: harness overhead, reported separately);
- Clef: ``encode``, ``forward`` (the compute), ``readout``.

A "judged item" is a span (representation) or a question (Clef), so ``seconds_per_item`` is the
amortised cost of one verdict. Unit sizes differ across tasks, so compare units only within a dataset.
Not included: the probe's own fit and inference (the representation judge's verdicts need a trained
probe; Clef's do not), and anything before the first unit (``model_load_seconds`` is in the run).

Gates: run status and task, unit counts and every ``eff_*_total`` / token / peak equal their recomputed
value, no resumed shards (timings would span sessions), every run on the GPU named in ``params.gpu_name``.

Figures go to ``analysis/figures/<id>/`` (``phases.png``, ``scaling.png``, ``memory.png``);
``metrics.json`` (flat, keys prefixed by the run's ``label``) and ``config.snapshot.yaml`` to ``$RUNS_ROOT/<id>/``.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "Times New Roman", "DejaVu Serif"],
})

FIGURES_DIR = Path(__file__).parent / "figures"
GIB = 2**30
PERCENTILE = 95
RTOL = 1e-9   # recomputed sums against the json's: float summation order only

# Per kind: the task the run must have, the phase columns (name -> role) and the other per-unit columns.
KINDS = {
    "rep": {
        "tasks": ("entity_detection", "relation_detection"), "file": "efficiency_rows.npz",
        "phases": {"seconds_prompt": "setup", "seconds_trace": "compute", "seconds_readout": "readout", "seconds_cleanup": "cleanup"},
        "tokens": "prompt_n_tokens", "items": "n_spans", "unit_total": "eff_seconds_per_prompt_total", "n_units": "eff_n_prompts",
        "tokens_total": "eff_prompt_n_tokens_total",
    },
    "clef": {
        "tasks": ("clef_entity_row", "clef_relation_conditional", "clef_record_oneshot"), "file": "efficiency_records.npz",
        "phases": {"seconds_encode": "setup", "seconds_forward": "compute", "seconds_readout": "readout"},
        "tokens": "n_tokens", "items": "n_questions", "unit_total": "eff_seconds_per_record_total", "n_units": "eff_n_records",
        "tokens_total": "eff_input_tokens_total",
    },
}
ROLES = ("setup", "compute", "readout", "cleanup")
ROLE_COLORS = {"setup": "#56B4E9", "compute": "#0072B2", "readout": "#009E73", "cleanup": "#E69F00"}


def _role_column(kind: str, role: str) -> str | None:
    cols = [c for c, r in KINDS[kind]["phases"].items() if r == role]
    assert len(cols) <= 1, (kind, role)
    return cols[0] if cols else None


def _check_total(run_id: str, name: str, recomputed: float, json_value: float) -> None:
    assert np.isclose(recomputed, json_value, rtol=RTOL, atol=0.0), (
        f"{run_id}: {name} {json_value!r} != {recomputed!r} recomputed from the per-unit arrays")


def load_run(run_id: str, runs_root: Path, kind: str) -> dict:
    """One run's per-unit columns and its ``efficiency.json``, after the gates in the module docstring."""
    assert kind in KINDS, kind
    spec = KINDS[kind]
    run_dir = runs_root / run_id
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["status"] == "success", f"{run_id}: status {manifest['status']!r}"
    assert manifest["task"] in spec["tasks"], f"{run_id}: task {manifest['task']!r} is not a {kind} task {spec['tasks']}"
    eff = json.loads((run_dir / "artifacts" / "efficiency.json").read_text())
    em = eff["metrics"]
    with np.load(run_dir / "artifacts" / spec["file"], allow_pickle=False) as z:
        cols = {k: z[k] for k in z.files}
    for c in (*spec["phases"], spec["tokens"], spec["items"], "peak_alloc_bytes", "peak_reserved_bytes"):
        assert c in cols, f"{run_id}: no column {c!r} in {spec['file']}"
    n = len(cols[spec["tokens"]])
    assert n > 0 and all(len(v) == n for v in cols.values()), f"{run_id}: columns differ in length"
    assert n == int(em[spec["n_units"]]), f"{run_id}: {n} units != {spec['n_units']} {em[spec['n_units']]}"
    assert all(np.isfinite(cols[c]).all() and (cols[c] >= 0).all() for c in spec["phases"]), f"{run_id}: a negative or non-finite phase time"
    assert (cols["peak_alloc_bytes"] > 0).all(), f"{run_id}: GPU peaks missing (a CPU run?)"
    assert int(em.get("eff_n_shards_resumed", 0)) == 0, f"{run_id}: resumed shards -- timings span sessions"

    for c in spec["phases"]:
        _check_total(run_id, f"eff_{c}_total", cols[c].sum(), em[f"eff_{c}_total"])
    unit = sum(cols[c] for c in spec["phases"])
    _check_total(run_id, spec["unit_total"], unit.sum(), em[spec["unit_total"]])
    _check_total(run_id, spec["tokens_total"], cols[spec["tokens"]].sum(), em[spec["tokens_total"]])
    _check_total(run_id, "eff_gpu_peak_alloc_bytes_max", cols["peak_alloc_bytes"].max(), em["eff_gpu_peak_alloc_bytes_max"])
    run_metrics = json.loads((run_dir / "metrics.json").read_text())
    n_items = "n_spans" if kind == "rep" else "eff_n_questions_total"
    _check_total(run_id, n_items, cols[spec["items"]].sum(), (run_metrics if kind == "rep" else em)[n_items])
    return {"run_id": run_id, "kind": kind, "cols": cols, "unit": unit, "eff": eff, "gpu_name": eff["hardware"]["gpu_name"]}


def assert_gpu(run: dict, gpu_name: str) -> None:
    assert run["gpu_name"] == gpu_name, f"{run['run_id']}: GPU {run['gpu_name']!r}, expected {gpu_name!r} (timings are not comparable)"


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """``(slope, intercept, r2)`` of the least-squares line ``y = intercept + slope * x``."""
    assert x.shape == y.shape and len(x) >= 2 and x.std() > 0, "tokens need variation to fit a slope"
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    return float(slope), float(intercept), float(1.0 - resid.var() / y.var())


def analyse_run(run: dict) -> dict:
    """Flat metrics of one run (see the module docstring for the definitions)."""
    kind, cols, unit, em = run["kind"], run["cols"], run["unit"], run["eff"]["metrics"]
    spec = KINDS[kind]
    cleanup = cols[_role_column(kind, "cleanup")] if _role_column(kind, "cleanup") else np.zeros_like(unit)
    excl = unit - cleanup
    compute = cols[_role_column(kind, "compute")]
    tokens = cols[spec["tokens"]].astype(np.float64)
    n_items = float(cols[spec["items"]].sum())
    slope, intercept, r2 = linear_fit(tokens, compute)
    m = {
        "n_units": len(unit), "n_items": n_items, "total_hours": float(unit.sum() / 3600),
        "seconds_unit_mean": float(unit.mean()), "seconds_unit_median": float(np.median(unit)),
        "seconds_unit_p95": float(np.percentile(unit, PERCENTILE)),
        "seconds_unit_excl_cleanup_mean": float(excl.mean()), "cleanup_share": float(cleanup.sum() / unit.sum()),
        "seconds_per_item": float(unit.sum() / n_items), "seconds_per_item_excl_cleanup": float(excl.sum() / n_items),
        "seconds_compute_mean": float(compute.mean()),
        "tokens_mean": float(tokens.mean()), "tokens_p95": float(np.percentile(tokens, PERCENTILE)),
        "compute_tokens_per_second": float(tokens.sum() / compute.sum()),
        "fit_seconds_per_1k_tokens": slope * 1000, "fit_intercept_seconds": intercept, "fit_r2": r2,
        "first_over_median": float(unit[0] / np.median(unit)),   # >> 1 means warmup is in the timings
        "peak_alloc_gib_max": float(cols["peak_alloc_bytes"].max() / GIB), "weights_gib": float(em["eff_gpu_weights_bytes"] / GIB),
        "peak_activation_gib_max": float(em["eff_gpu_peak_activation_bytes_max"] / GIB),
    }
    for c, role in spec["phases"].items():
        m[f"seconds_{role}_mean"] = float(cols[c].mean())
    if kind == "rep":
        m["rep_bytes_per_span_per_layer"] = float(em["eff_rep_bytes_per_span_per_layer"])
        m["npz_file_gib"] = float(em["eff_npz_file_bytes"] / GIB)
    return m


def ratio(per_run: dict[str, dict], spec: dict) -> tuple[str, float]:
    """``(name, per_run[numerator][metric] / per_run[denominator][metric])`` for one configured comparison."""
    num, den = per_run[spec["numerator"]][spec["metric"]], per_run[spec["denominator"]][spec["metric"]]
    assert den > 0, spec
    return spec["name"], float(num / den)


# ------------------------------------------------------------------------------ figures
def plot_phases(per_run: dict[str, dict], out_dir: Path) -> None:
    """Mean seconds per unit by phase role, one stacked bar per run."""
    labels = list(per_run)
    fig, ax = plt.subplots(figsize=(max(4.5, 0.9 * len(labels) + 1.5), 4.2), constrained_layout=True)
    bottom = np.zeros(len(labels))
    for role in ROLES:
        vals = np.array([per_run[lb].get(f"seconds_{role}_mean", 0.0) for lb in labels])
        ax.bar(labels, vals, bottom=bottom, color=ROLE_COLORS[role], label=role)
        bottom += vals
    ax.set(ylabel="mean seconds per judged row", ylim=(0, 1.1 * bottom.max()))
    ax.tick_params(axis="x", rotation=60)
    for t in ax.get_xticklabels():
        t.set_ha("right")
    ax.legend(fontsize=9, framealpha=0.9)
    fig.savefig(out_dir / "phases.png", dpi=200)
    plt.close(fig)


def plot_scaling(runs: dict[str, dict], per_run: dict[str, dict], out_dir: Path) -> None:
    """Compute seconds against input tokens per run, with the least-squares line."""
    labels = list(runs)
    ncol = min(3, len(labels))
    nrow = -(-len(labels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), constrained_layout=True, squeeze=False)
    for ax in axes.flat[len(labels):]:
        ax.set_visible(False)
    for ax, lb in zip(axes.flat, labels):
        run, m = runs[lb], per_run[lb]
        spec = KINDS[run["kind"]]
        x, y = run["cols"][spec["tokens"]], run["cols"][_role_column(run["kind"], "compute")]
        ax.scatter(x, y, s=3, alpha=0.3, color=ROLE_COLORS["compute"], linewidths=0)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, m["fit_intercept_seconds"] + m["fit_seconds_per_1k_tokens"] / 1000 * xs, color="k", lw=1)
        ax.set(title=f"{lb}\n{m['fit_seconds_per_1k_tokens']:.2f} s / 1k tokens, $R^2$ = {m['fit_r2']:.2f}", xlabel="input tokens",
               ylabel="compute seconds")
        ax.title.set_fontsize(8)
        ax.grid(alpha=0.25, linewidth=0.4)
    fig.savefig(out_dir / "scaling.png", dpi=200)
    plt.close(fig)


def plot_memory(per_run: dict[str, dict], out_dir: Path) -> None:
    """Weights and peak activation memory (GiB) per run; their sum is the peak allocation."""
    labels = list(per_run)
    fig, ax = plt.subplots(figsize=(max(4.5, 0.9 * len(labels) + 1.5), 4.2), constrained_layout=True)
    w = np.array([per_run[lb]["weights_gib"] for lb in labels])
    a = np.array([per_run[lb]["peak_activation_gib_max"] for lb in labels])
    ax.bar(labels, w, color="#999999", label="weights")
    ax.bar(labels, a, bottom=w, color=ROLE_COLORS["compute"], label="peak activations (max over rows)")
    ax.set(ylabel="GiB")
    ax.tick_params(axis="x", rotation=60)
    for t in ax.get_xticklabels():
        t.set_ha("right")
    ax.legend(fontsize=9, framealpha=0.9)
    fig.savefig(out_dir / "memory.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def run(cfg: dict, runs_root: Path) -> dict:
    p = cfg["params"]
    out = runs_root / cfg["id"]
    fig_dir = FIGURES_DIR / cfg["id"]
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = [s["label"] for s in p["runs"]]
    assert len(set(labels)) == len(labels), f"duplicate run labels {labels}"
    runs, per_run = {}, {}
    for spec in p["runs"]:
        r = load_run(spec["run_id"], runs_root, spec["kind"])
        assert_gpu(r, p["gpu_name"])
        runs[spec["label"]], per_run[spec["label"]] = r, analyse_run(r)
        print(f"{spec['label']}: {r['run_id']} units={per_run[spec['label']]['n_units']} "
              f"s/unit={per_run[spec['label']]['seconds_unit_mean']:.3f} cleanup_share={per_run[spec['label']]['cleanup_share']:.2f}")

    metrics = {f"{lb}_{k}": v for lb, m in per_run.items() for k, v in m.items()}
    for spec in p["comparisons"]:
        name, value = ratio(per_run, spec)
        metrics[f"ratio_{name}"] = value
    plot_phases(per_run, fig_dir)
    plot_scaling(runs, per_run, fig_dir)
    plot_memory(per_run, fig_dir)
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
