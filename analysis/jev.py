"""Jev judge analysis: calibration, discrimination and per-stratum accuracy of its direct ``P(true)`` on
minimal pairs, for entity detection (main) and relation detection (line samples).

Usage: ``uv run --python 3.12 --extra analysis python analysis/jev.py analysis/configs/<type>/<id>.yaml``
(every value lives in the config's ``params``; a missing key is a KeyError).

Mirror of ``analysis/clef.py``, which it loads and reuses *unchanged* (so the metric code is that of
``analysis/entity_detection.py``, loaded not copied): the minimal-pair evaluation sets (``entity_eval_set``,
``sample_eval_set``) and ``analyse_run`` (smECE + CI and reliability diagram, Brier, Brier skill score, AUROC with a
document-bootstrap interval, accuracy, per-stratum detection, permutation-null control). Only the loader differs.

Reads ``jev_entity_row`` / ``jev_relation_conditional`` / ``jev_record_oneshot`` runs (``src/judge_interp/jev.py``; one
``artifacts/jev_predictions.npz`` each) named in ``params.runs``. Positive class = valid; ``P(valid)`` = Jev's ``P(true)``;
predicted valid iff ``> decision_threshold``. ``kind``, ``bss_ref_rate`` and the output locations are as in ``analysis/clef.py``.

Differences from the Clef analysis, all from the model:
- Jev returns only ``P(true)`` (no logits), so there is no logit/softmax consistency gate; instead ``P(true)`` must lie in
  [0, 1] and be a multiple of 0.01 (the API's resolution; ``p_grid``), which catches a wrong column or a rescaled one.
- ``P(true)`` takes at most 101 values, so AUROC has many ties (counted as half, as ``roc_auc_score`` does); the metrics
  ``{label}_n_distinct_p_true`` and ``{label}_frac_p_saturated`` (``P <= 0.01`` or ``>= 0.99``) report how much.
- The API is nondeterministic for an identical request; ``{label}_determinism_max_abs_diff`` is the run's own
  rescoring drift (``metrics.json``). Each item was scored once, so the metrics include that noise.
- A Jev run excludes nothing (it raises on an over-length request), so ``{label}_n_excluded_overlong`` is 0 by
  construction and every pair is held whole.
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

_spec = importlib.util.spec_from_file_location("analysis_clef", Path(__file__).parent / "clef.py")
ac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ac)  # reused unchanged: eval-set builders, analyse_run (and, through it, entity_detection)
er = ac.er

FIGURES_DIR = ac.FIGURES_DIR
KINDS = {"entity": "jev_entity_row", "conditional": "jev_relation_conditional", "oneshot": "jev_record_oneshot"}
COMMON = ("doc_ids", "label_valid", "p_true", "n_tokens", "context_doc")
ENTITY_COLS = COMMON + ("field", "field_error_type", "row_error_type", "row_ordinal", "row_k", "value")
SAMPLE_COLS = COMMON + ("sample_id", "error_type", "pair_id", "cell", "size", "fields", "n_invalid_fields")
P_GRID = 0.01       # resolution of the API's P(true)
P_GRID_ATOL = 1e-6
SATURATED = 0.01    # P <= SATURATED or >= 1 - SATURATED


def load_run(run_id: str, runs_root: Path, kind: str) -> dict:
    """Load one Jev run's predictions; gates: run.json status/task, counts equal ``metrics.json``,
    ``p_true`` in [0, 1] on the API's 0.01 grid, labels consistent with error types.

    Returns:
        The columns of ``ENTITY_COLS`` / ``SAMPLE_COLS`` plus ``run_id``, ``kind``, the run's ``params``,
        ``n_excluded`` (0: a Jev run excludes nothing) and ``determinism_max_abs_diff``.
    """
    run_dir = runs_root / run_id
    manifest = json.loads((run_dir / "run.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert manifest["status"] == "success", f"{run_id}: status {manifest['status']!r}"
    assert manifest["task"] == KINDS[kind], f"{run_id}: task {manifest['task']!r}, expected {KINDS[kind]!r}"
    assert "n_excluded_overlong" not in metrics, f"{run_id}: a Jev run excludes nothing; is this a Clef run?"
    with np.load(run_dir / "artifacts" / "jev_predictions.npz", allow_pickle=False) as z:
        assert not {"logit_true", "logit_false"} & set(z.files), f"{run_id}: Jev stores no logits"
        d = {k: z[k] for k in (ENTITY_COLS if kind == "entity" else SAMPLE_COLS)}
    n = len(d["p_true"])
    assert all(len(v) == n for v in d.values()), run_id
    assert n == metrics["n_predictions"], f"{run_id}: {n} rows != metrics n_predictions {metrics['n_predictions']}"
    d["label_valid"] = d["label_valid"].astype(bool)
    assert int(d["label_valid"].sum()) == metrics["n_valid"], run_id
    p = d["p_true"].astype(np.float64)
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all(), f"{run_id}: p_true outside [0, 1]"
    assert np.allclose(p / P_GRID, np.round(p / P_GRID), atol=P_GRID_ATOL / P_GRID), f"{run_id}: p_true is not on the {P_GRID} grid"
    d["p_true"] = p
    invalid_type = d["field_error_type"] if kind == "entity" else d["error_type"]
    assert ((invalid_type == "") == d["label_valid"]).all(), f"{run_id}: error type empty iff valid violated"
    if kind != "entity":
        assert ((d["n_invalid_fields"] == 0) == d["label_valid"]).all(), f"{run_id}: n_invalid_fields vs label"
    assert set(np.unique(invalid_type[~d["label_valid"]]).tolist()) <= set(er.ALL_ERROR_TYPES), run_id
    cfg_snapshot = yaml.safe_load((run_dir / "config.snapshot.yaml").read_text())
    return {"run_id": run_id, "kind": kind, "params": cfg_snapshot["params"], "n_excluded": 0,
            "determinism_max_abs_diff": float(metrics["determinism_max_abs_diff"]), **d}


def resolution_metrics(run: dict) -> dict:
    """How coarse and saturated the run's ``P(true)`` is, and the run's own rescoring drift."""
    p = run["p_true"]
    return {
        "n_distinct_p_true": int(len(np.unique(np.round(p / P_GRID)))),
        "frac_p_saturated": float(((p <= SATURATED + P_GRID_ATOL) | (p >= 1 - SATURATED - P_GRID_ATOL)).mean()),
        "determinism_max_abs_diff": run["determinism_max_abs_diff"],
    }


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
            ev = ac.entity_eval_set(r, er.load_run_rows(spec["run_id"], runs_root, er.REPO_ROOT, "main"), p["twin_seed"])
        else:
            ev = ac.sample_eval_set(r)
        metrics[f"{spec['label']}_context_is_other_document"] = int(r["params"]["context_control"] == "other_document")
        for k, v in {**ac.analyse_run(spec["label"], r, ev, spec, p, fig_dir, seed), **resolution_metrics(r)}.items():
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
