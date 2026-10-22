"""Clef-flash judge analysis: calibration, discrimination and per-error-type accuracy of its
direct ``P(true)`` on the entity-detection and relation-detection tasks.

Usage: ``uv run --python 3.12 --extra analysis python analysis/clef.py analysis/configs/<type>/<id>.yaml``
(every value lives in the config's ``params``; a missing key is a KeyError).

Reads ``clef_entity_row`` / ``clef_relation_row`` runs (``src/judge_interp/clef.py``; one
``artifacts/clef_predictions.npz`` each) named in ``params.runs`` and, per run, reports with the
metric code of ``analysis/entity_detection.py`` *unchanged* (loaded, not copied): smECE
(+ bootstrap CI half-width) and its smooth reliability diagram, Brier score, Brier skill score,
AUROC, accuracy, and the per-error-type report (``detect_<T>`` = accuracy on invalid items of
type T, valid-item accuracy in clean rows / in T-corrupted rows, AUROC of T-invalid vs all valid).
Positive class = valid; ``P(valid)`` = Clef's ``P(true)``; predicted valid iff ``> decision_threshold``.

Nothing is trained, so there is no train valid rate to anchor the Brier skill score: each run
states its reference explicitly, ``bss_ref_rate`` = a float in (0, 1) (e.g. 0.5 for the balanced
relation set) or the string ``"evaluation_set_rate"`` (constant forecast at the evaluated set's own
valid rate -- a harder reference than a train rate). There is no shuffled-label *fit* control (no
fit); instead the metrics' own permutation null (mean AUROC over ``n_shuffles`` label
permutations ~ 0.5, asserted) and, for relation runs, parameter-free structure baselines that the
score must beat to mean anything: AUROC of ``-subset size`` (a larger subset is likelier to contain
an invalid field, so size alone carries label information) and AUROC within each subset size.

Figures go to ``analysis/figures/<id>/<label>/`` (``metrics.json`` copy in ``analysis/figures/<id>/``); ``metrics.json`` and ``config.snapshot.yaml`` to
``$RUNS_ROOT/<id>/``. Metric keys are prefixed with the run's ``label``.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

_spec = importlib.util.spec_from_file_location("analysis_entity_detection", Path(__file__).parent / "entity_detection.py")
er = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(er)  # reused unchanged: probe_metrics, error_type_report, plot_calibration (and, through it, er.smooth_ece)

FIGURES_DIR = Path(__file__).parent / "figures"
KINDS = {"entity": "clef_entity_row", "relation": "clef_relation_row"}
REF_SET_RATE = "evaluation_set_rate"
COMMON = ("doc_ids", "label_valid", "row_error_type", "p_true", "logit_true", "logit_false", "n_tokens", "context_doc")
ENTITY_COLS = COMMON + ("field", "field_error_type", "row_ordinal", "row_k")
RELATION_COLS = COMMON + ("size", "fields", "n_invalid_fields", "rows")


def load_run(run_id: str, runs_root: Path, kind: str) -> dict:
    """Load one Clef run's predictions; gates: run.json status/task, counts equal ``metrics.json``,
    ``p_true`` equals the softmax of the stored logits, labels consistent with error types."""
    run_dir = runs_root / run_id
    manifest = json.loads((run_dir / "run.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert manifest["status"] == "success", f"{run_id}: status {manifest['status']!r}"
    assert manifest["task"] == KINDS[kind], f"{run_id}: task {manifest['task']!r}, expected {KINDS[kind]!r}"
    with np.load(run_dir / "artifacts" / "clef_predictions.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in (ENTITY_COLS if kind == "entity" else RELATION_COLS)}
    n = len(d["p_true"])
    assert all(len(v) == n for v in d.values()), run_id
    assert n == metrics["n_predictions"], f"{run_id}: {n} rows != metrics n_predictions {metrics['n_predictions']}"
    d["label_valid"] = d["label_valid"].astype(bool)
    assert int(d["label_valid"].sum()) == metrics["n_valid"], run_id
    p = d["p_true"].astype(np.float64)
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all(), f"{run_id}: p_true outside [0, 1]"
    z64 = np.stack([d["logit_true"], d["logit_false"]], axis=1).astype(np.float64)
    assert np.allclose(p, np.exp(z64[:, 0] - np.logaddexp(z64[:, 0], z64[:, 1])), atol=1e-6), (
        f"{run_id}: p_true is not the softmax of the stored logits"
    )
    d["p_true"] = p
    if kind == "entity":
        assert ((d["field_error_type"] == "") == d["label_valid"]).all(), f"{run_id}: field_error_type empty iff valid violated"
        d["invalid_type"] = d["field_error_type"]
    else:
        assert ((d["n_invalid_fields"] == 0) == d["label_valid"]).all(), f"{run_id}: n_invalid_fields vs label"
        assert not ((d["row_error_type"] == "") & ~d["label_valid"]).any(), f"{run_id}: invalid subset from a clean row"
        d["invalid_type"] = d["row_error_type"]
    assert set(np.unique(d["invalid_type"][~d["label_valid"]]).tolist()) <= set(er.ALL_ERROR_TYPES), run_id
    cfg_snapshot = yaml.safe_load((run_dir / "config.snapshot.yaml").read_text())
    return {"run_id": run_id, "kind": kind, "params": cfg_snapshot["params"], "n_excluded": metrics["n_excluded_overlong"], **d}


def permutation_null_auroc(y: np.ndarray, probs: np.ndarray, n: int, seed: int) -> tuple[float, float]:
    """``(mean, std)`` AUROC of ``probs`` against ``n`` random permutations of ``y`` (should be ~0.5)."""
    rng = np.random.default_rng(seed)
    aucs = [float(roc_auc_score(rng.permutation(y), probs)) for _ in range(n)]
    return float(np.mean(aucs)), float(np.std(aucs))


def analyse_run(label: str, run: dict, spec: dict, p: dict, fig_dir: Path, seed: int) -> dict:
    y, probs, thr = run["label_valid"], run["p_true"], p["decision_threshold"]
    assert y.any() and (~y).any(), f"{label}: one class missing"
    ref = spec["bss_ref_rate"]
    ref_rate = float(y.mean()) if ref == REF_SET_RATE else float(ref)
    assert ref == REF_SET_RATE or isinstance(ref, float), f"{label}: bss_ref_rate must be a float or {REF_SET_RATE!r}"

    m: dict[str, float] = {
        "n": int(len(y)), "n_valid": int(y.sum()), "valid_rate": float(y.mean()), "bss_ref_rate": ref_rate,
        "n_docs": len(set(run["doc_ids"].tolist())), "n_excluded_overlong": int(run["n_excluded"]),
        "max_n_tokens": int(run["n_tokens"].max()), "mean_p_true": float(probs.mean()),
    }
    m.update(er.probe_metrics("test", y, probs, ref_rate, thr))
    d = er.smooth_ece(probs, y, p["ece_n_boot"], seed)
    m["test_smece"], m["test_smece_ci_halfwidth"] = float(d["ce"]), float(d["ce_ci_width"])
    (fig_dir / label).mkdir(parents=True, exist_ok=True)
    er.plot_calibration(d, fig_dir / label)
    er.plot_prob_histogram(probs, p["hist_n_bins"], fig_dir / label)

    types = sorted(set(run["invalid_type"][~y].tolist()))
    for k, v in er.error_type_report(y, probs, thr, run["invalid_type"], run["row_error_type"], types).items():
        m[f"test_{k}"] = v

    if run["kind"] == "entity":
        for f in sorted(set(run["field"].tolist())):
            sel = run["field"] == f
            m[f"test_acc_field_{f}"] = float(((probs[sel] > thr) == y[sel]).mean())
    else:
        # Parameter-free structure baselines: size alone leaks the label.
        size = run["size"]
        m["test_baseline_size_auroc"] = float(roc_auc_score(y, -size))
        for sz in sorted(set(size.tolist())):
            sel = size == sz
            m[f"test_n_size{sz}"] = int(sel.sum())
            if y[sel].any() and (~y[sel]).any():
                m[f"test_auroc_size{sz}"] = float(roc_auc_score(y[sel], probs[sel]))

    mean, std = permutation_null_auroc(y, probs, p["n_shuffles"], seed)
    m["control_permutation_auroc"], m["control_permutation_auroc_std"] = mean, std
    assert abs(mean - 0.5) <= p["permutation_auroc_tolerance"], (
        f"{label}: mean permuted-label AUROC {mean:.4f} (std {std:.4f}, n={p['n_shuffles']}) is not ~0.5"
    )
    return m


def run(cfg: dict, runs_root: Path) -> dict:
    p, seed = cfg["params"], cfg["seed"]
    out = runs_root / cfg["id"]
    fig_dir = FIGURES_DIR / cfg["id"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = [s["label"] for s in p["runs"]]
    assert len(set(labels)) == len(labels), f"duplicate run labels {labels}"
    metrics: dict[str, float] = {}
    for spec in p["runs"]:
        assert spec["kind"] in KINDS, spec
        r = load_run(spec["run_id"], runs_root, spec["kind"])
        print(f"{spec['label']}: {r['run_id']} n={len(r['p_true'])} context_control={r['params']['context_control']}")
        metrics[f"{spec['label']}_context_is_other_document"] = int(r["params"]["context_control"] == "other_document")
        for k, v in analyse_run(spec["label"], r, spec, p, fig_dir, seed).items():
            metrics[f"{spec['label']}_{k}"] = v
    metrics_text = json.dumps(metrics, indent=2) + "\n"
    (out / "metrics.json").write_text(metrics_text)
    (fig_dir / "metrics.json").write_text(metrics_text)  # copy beside the figures
    print(json.dumps(metrics, indent=2))
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
