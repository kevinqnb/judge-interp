"""Clef-flash judge analysis: calibration, discrimination and per-stratum accuracy of its direct
``P(true)`` on minimal pairs, for entity detection (main) and relation detection (line samples).

Usage: ``uv run --python 3.12 --extra analysis python analysis/clef.py analysis/configs/<type>/<id>.yaml``
(every value lives in the config's ``params``; a missing key is a KeyError).

Reads ``clef_entity_row`` / ``clef_relation_conditional`` / ``clef_record_oneshot`` runs
(``src/judge_interp/clef.py``; one ``artifacts/clef_predictions.npz`` each) named in ``params.runs``
and evaluates each on minimal pairs, with the metric code of ``analysis/entity_detection.py``
*unchanged* (loaded, not copied):

- entity: each corrupted span with its clean twin span (``main_span_pairs`` on the main row file the
  run judged; null-fill spans are twinned with another document's span). Reported overall, per
  error type, per (type, k), without null-fill pairs, and the row-context diagnostic (uncorrupted
  spans of corrupted rows vs the clean row), plus per-field accuracy.
- conditional / oneshot: each invalid sample with its twin (same ``pair_id``); a run must hold
  every pair whole. Reported overall, per error type and per cell. Subset size is identical within
  a pair, so the parameter-free size baseline (AUROC of ``-size``) is exactly 0.5 (asserted).

Metrics: smECE (+ bootstrap CI half-width) and its smooth reliability diagram, Brier score, Brier skill
score, AUROC (with a document-bootstrap interval), accuracy, per-stratum detection / valid accuracy.
Positive class = valid; ``P(valid)`` = Clef's ``P(true)``; predicted valid iff ``> decision_threshold``.
Nothing is trained, so each run states its Brier reference, ``bss_ref_rate`` = a float in (0, 1) or
``"evaluation_set_rate"`` (0.5 on minimal pairs). Control: the metrics' own permutation null (mean
AUROC over ``n_shuffles`` label permutations ~ 0.5, asserted).

Figures go to ``analysis/figures/<id>/<label>/`` (``metrics.json`` copy in ``analysis/figures/<id>/``); ``metrics.json`` and ``config.snapshot.yaml`` to
``$RUNS_ROOT/<id>/``. Metric keys are prefixed with the run's ``label``.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

_spec = importlib.util.spec_from_file_location("analysis_entity_detection", Path(__file__).parent / "entity_detection.py")
er = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(er)  # reused unchanged: probe_metrics, stratum_report, pairing, plots

FIGURES_DIR = Path(__file__).parent / "figures"
KINDS = {"entity": "clef_entity_row", "conditional": "clef_relation_conditional", "oneshot": "clef_record_oneshot"}
REF_SET_RATE = "evaluation_set_rate"
COMMON = ("doc_ids", "label_valid", "p_true", "logit_true", "logit_false", "n_tokens", "context_doc")
ENTITY_COLS = COMMON + ("field", "field_error_type", "row_error_type", "row_ordinal", "row_k", "value")
SAMPLE_COLS = COMMON + ("sample_id", "error_type", "pair_id", "cell", "size", "fields", "n_invalid_fields")


def load_run(run_id: str, runs_root: Path, kind: str) -> dict:
    """Load one Clef run's predictions; gates: run.json status/task, counts equal ``metrics.json``,
    ``p_true`` equals the softmax of the stored logits, labels consistent with error types."""
    run_dir = runs_root / run_id
    manifest = json.loads((run_dir / "run.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert manifest["status"] == "success", f"{run_id}: status {manifest['status']!r}"
    assert manifest["task"] == KINDS[kind], f"{run_id}: task {manifest['task']!r}, expected {KINDS[kind]!r}"
    with np.load(run_dir / "artifacts" / "clef_predictions.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in (ENTITY_COLS if kind == "entity" else SAMPLE_COLS)}
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
    invalid_type = d["field_error_type"] if kind == "entity" else d["error_type"]
    assert ((invalid_type == "") == d["label_valid"]).all(), f"{run_id}: error type empty iff valid violated"
    if kind != "entity":
        assert ((d["n_invalid_fields"] == 0) == d["label_valid"]).all(), f"{run_id}: n_invalid_fields vs label"
    assert set(np.unique(invalid_type[~d["label_valid"]]).tolist()) <= set(er.ALL_ERROR_TYPES), run_id
    cfg_snapshot = yaml.safe_load((run_dir / "config.snapshot.yaml").read_text())
    return {"run_id": run_id, "kind": kind, "params": cfg_snapshot["params"], "n_excluded": metrics["n_excluded_overlong"], **d}


# ------------------------------------------------------------------------ evaluation sets
def entity_eval_set(run: dict, rows: list[dict], seed: int) -> dict:
    """Minimal entity pairs of an entity run (``er.main_span_pairs``), both members stacked, with
    ``probs`` and the row-context diagnostic's ``p_kept`` / ``p_clean``."""
    z = er.main_span_pairs(run, rows, seed)
    s = er.stack_pairs(z["inv"], z["val"], error_type=z["error_type"], k=z["k"], field=z["field"],
                       null_fill=z["null_fill"], doc=z["doc"])
    s["type_k"] = np.char.add(np.char.add(s["error_type"], "_k"), s["k"].astype(str))
    s["probs"] = run["p_true"][s["idx"]]
    s["p_kept"], s["p_clean"] = run["p_true"][z["kept"]], run["p_true"][z["clean"]]
    return s


def sample_eval_set(run: dict) -> dict:
    """The minimal pairs of a samples run (by ``pair_id``), invalid members then their twins, with
    ``probs``, ``doc``, ``cell``, ``error_type`` (the cell's) and ``size`` for both members.

    Raises:
        AssertionError: a pair not held whole (e.g. one member excluded as overlong) or whose members differ in cell or size.
    """
    members: dict[str, list[int]] = defaultdict(list)
    for i, pid in enumerate(run["pair_id"].tolist()):
        members[pid].append(i)
    broken = [pid for pid, m in members.items() if sorted(run["label_valid"][m].tolist()) != [False, True]]
    assert not broken, (f"{len(broken)} pairs not held whole (one invalid + one valid), e.g. {broken[:2]}; "
                        f"{run['n_excluded']} units were excluded as overlong")
    pids = sorted(members)
    inv = np.array([next(i for i in members[pid] if not run["label_valid"][i]) for pid in pids])
    val = np.array([next(i for i in members[pid] if run["label_valid"][i]) for pid in pids])
    for c in ("cell", "size", "doc_ids"):
        assert (run[c][inv] == run[c][val]).all(), f"pair members differ in {c}"
    cell = run["cell"][inv]
    s = er.stack_pairs(inv, val, doc=run["doc_ids"][inv], cell=cell, size=run["size"][inv],
                       error_type=np.array([c.split(":")[0] for c in cell.tolist()]))
    assert (s["error_type"][: len(inv)] == run["error_type"][inv]).all(), "an invalid sample's error type is not its cell's"
    s["probs"] = run["p_true"][s["idx"]]
    return s


# ------------------------------------------------------------------------ metrics
def analyse_run(label: str, run: dict, ev: dict, spec: dict, p: dict, fig_dir: Path | None, seed: int) -> dict:
    y, probs, thr = ev["y"], ev["probs"], p["decision_threshold"]
    assert y.any() and (~y).any(), f"{label}: one class missing"
    ref = spec["bss_ref_rate"]
    ref_rate = float(y.mean()) if ref == REF_SET_RATE else float(ref)
    assert ref == REF_SET_RATE or isinstance(ref, float), f"{label}: bss_ref_rate must be a float or {REF_SET_RATE!r}"

    m: dict[str, float] = {
        "n": int(len(y)), "n_pairs": int((~y).sum()), "valid_rate": float(y.mean()), "bss_ref_rate": ref_rate,
        "n_docs": len(set(ev["doc"].tolist())), "n_excluded_overlong": int(run["n_excluded"]),
        "max_n_tokens": int(run["n_tokens"].max()), "mean_p_true": float(probs.mean()),
    }
    m.update(er.probe_metrics("test", y, probs, ref_rate, thr))
    m["test_auroc_ci_lo"], m["test_auroc_ci_hi"] = er.doc_bootstrap_auroc(y, probs, ev["doc"], p["auroc_n_boot"], p["ci_level"], seed)
    d = er.smooth_ece(probs, y, p["ece_n_boot"], seed)
    m["test_smece"], m["test_smece_ci_halfwidth"] = float(d["ce"]), float(d["ce_ci_width"])
    if fig_dir is not None:   # None: the caller draws its own figures (entity_detection.py)
        (fig_dir / label).mkdir(parents=True, exist_ok=True)
        er.plot_calibration(d, fig_dir / label)
        er.plot_prob_histogram(probs, p["hist_n_bins"], fig_dir / label)

    if run["kind"] == "entity":
        for col in ("error_type", "type_k"):
            for k, v in er.stratum_report(y, probs, thr, ev[col]).items():
                m[f"test_{k}"] = v
        no_fill = ~ev["null_fill"]
        m["n_null_fill_pairs"] = int(ev["null_fill"][~y].sum())
        m.update(er.probe_metrics("test_no_null_fill", y[no_fill], probs[no_fill], ref_rate, thr))
        for k, v in er.stratum_report(y[no_fill], probs[no_fill], thr, ev["error_type"][no_fill]).items():
            m[f"test_no_null_fill_{k}"] = v
        n_kept = len(ev["p_kept"])
        m["test_kept_mean_p_valid"], m["test_kept_twin_mean_p_valid"] = float(ev["p_kept"].mean()), float(ev["p_clean"].mean())
        m["test_kept_valid_acc"] = float((ev["p_kept"] > thr).mean())
        m["test_clean_vs_kept_auroc"] = float(roc_auc_score(
            np.concatenate([np.zeros(n_kept, bool), np.ones(n_kept, bool)]), np.concatenate([ev["p_kept"], ev["p_clean"]])))
        for f in sorted(set(ev["field"].tolist())):
            sel = ev["field"] == f
            m[f"test_acc_field_{f}"] = float(((probs[sel] > thr) == y[sel]).mean())
    else:
        for col in ("error_type", "cell"):
            for k, v in er.stratum_report(y, probs, thr, ev[col]).items():
                m[f"test_{k}"] = v
        # Parameter-free structure baseline: subset size is identical within a pair, so exactly chance.
        m["test_baseline_size_auroc"] = float(roc_auc_score(y, -run["size"][ev["idx"]]))   # each member's own size
        assert abs(m["test_baseline_size_auroc"] - 0.5) <= er.EXACT_AUROC_ATOL, f"{label}: size predicts the label"

    mean, std = permutation_null_auroc(y, probs, p["n_shuffles"], seed)
    m["control_permutation_auroc"], m["control_permutation_auroc_std"] = mean, std
    assert abs(mean - 0.5) <= p["permutation_auroc_tolerance"], (
        f"{label}: mean permuted-label AUROC {mean:.4f} (std {std:.4f}, n={p['n_shuffles']}) is not ~0.5"
    )
    return m


def permutation_null_auroc(y: np.ndarray, probs: np.ndarray, n: int, seed: int) -> tuple[float, float]:
    """``(mean, std)`` AUROC of ``probs`` against ``n`` random permutations of ``y`` (should be ~0.5)."""
    rng = np.random.default_rng(seed)
    aucs = [float(roc_auc_score(rng.permutation(y), probs)) for _ in range(n)]
    return float(np.mean(aucs)), float(np.std(aucs))


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
        if spec["kind"] == "entity":
            ev = entity_eval_set(r, er.load_run_rows(spec["run_id"], runs_root, er.REPO_ROOT, "main"), p["twin_seed"])
        else:
            ev = sample_eval_set(r)
        metrics[f"{spec['label']}_context_is_other_document"] = int(r["params"]["context_control"] == "other_document")
        for k, v in analyse_run(spec["label"], r, ev, spec, p, fig_dir, seed).items():
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
