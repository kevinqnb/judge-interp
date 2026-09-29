"""Mixed entity-detection representations: PCA of the token-normalised set X, then a
logistic-regression validity probe with calibration.

Usage: ``uv run --python 3.12 python analysis/entity_detection.py configs/<id>.yaml``
(every value lives in the config's ``params``; a missing key is a KeyError).

Reads the ``entity_detection_mixed`` train and test runs named in the config
(``src/judge_interp/entity_detection.py``; layers ``[0, last]``, one row per list
entry, read at the entry's content span) and does, in order:

1. **Preprocess** each split into ``X``:
   (a) ``X = rep_last - rep_0`` -- layer 0 is the token-embedding output, so this
       removes the token-identity component ("token-normalised");
   (b) subtract each *document's* mean row (per ``doc_id`` -- all of a document's
       prompts/samples pooled, not per prompt; per-prompt centering would erase the
       between-prompt k/m variation that figure 2 colours by, and every k=0 or k=m
       prompt would be forced to a single label);
   (b2) if ``subtract_entity_type_mean``: subtract each entity type's mean row, with
       the means fit on the *train* split (after (b)) and applied to both splits.
       Per-document centering leaves type differences (dates vs amounts vs names,
       ``main`` vs ``line`` fields) in place, and they dominate the top PCs;
   (c) subtract the *train* split's global mean from both splits. Without (b2),
       every row belongs to a zero-mean document, so the global mean is zero up to
       float error: (c) is then a numerical no-op, implemented and asserted, not a
       step that changes anything. With (b2) it is exactly zero on train;
   (d) if ``standardize``: divide each dimension by its *train* std (applied to
       both splits), so a handful of outlier dimensions cannot set the PCs.
   Both flags are explicit config keys (no default): False reproduces the plain
   (a)-(c) pipeline used by ``-01``/``-02``.
2. **PCA** fit on the train ``X`` only (all rows), two figures with the same
   projection and the same scatter subsample: coloured by ``label_valid``, and by
   the prompt's error rate ``k/m`` (m = list length, k = number of decoy slots).
3. **Logistic regression** trained on train ``X``, evaluated on test ``X``.
   **Positive class = valid** (precision/recall are for "valid"). Accuracy,
   precision, recall, F1, AUROC, plus smooth ECE (relplot, as in scholarlm's
   ``analysis/calibration_updated.py``) and a smoothed reliability diagram.
   A **probe-axis plot** (supervised, test split): x = the probe's logit (decision
   function; 0 is the 0.5 threshold), y = PC1 of the train ``X`` after removing the
   probe direction ``w`` (``X - (X.u)u``, ``u = w/|w|``), fit on train, applied to
   test; one panel coloured by validity, one by k/m.
4. **Controls** (``run_controls``): a shuffled-train-label fit must give test AUROC
   ~ 0.5, and a repeated fit with the same seed must give identical test
   probabilities.

Rows with ``exact_end_alignment == False`` (~7%) or ``decoy_in_ocr == True`` (~2.4%)
are *kept*; their rates are printed and written to ``metrics.json``.

Figures go to ``analysis/figures/<id>/``; ``metrics.json``, ``config.snapshot.yaml``
and ``artifacts/test_predictions.npz`` go to ``$RUNS_ROOT/<id>/``.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import relplot
import yaml
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "Times New Roman", "DejaVu Serif"],
})

# Okabe-Ito colourblind-safe pair, fixed order in every panel.
INVALID_COLOR = "#E69F00"
VALID_COLOR = "#0072B2"
# Perceptually uniform, CVD-safe sequential map for k/m (saturated ends; cividis washed out under alpha).
ERROR_RATE_CMAP = "viridis"

# Numerical tolerance for "this mean is zero" assertions, relative to the std of X.
# Not an experimental parameter: it only bounds float32 summation error.
ZERO_MEAN_RTOL = 1e-3
# Floor on density-weighted curve alpha (same idea as calibration_updated.py).
CURVE_ALPHA_FLOOR = 0.15

FIGURES_DIR = Path(__file__).parent / "figures"

REQUIRED_META = (
    "doc_ids", "sample_index", "m", "k", "label_valid", "exact_end_alignment", "decoy_in_ocr",
)


# --------------------------------------------------------------------------- loading
def load_split(run_id: str, runs_root: Path, max_docs: int | None) -> dict:
    """Load one run into ``X = rep_last - rep_0`` (float32) plus metadata arrays.

    Gates: run.json status/task; layers == [0, last]; when the full split is loaded
    (``max_docs is None``) the row and valid/invalid counts equal ``metrics.json``.
    """
    run_dir = runs_root / run_id
    manifest = json.loads((run_dir / "run.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert manifest["status"] == "success", f"{run_id}: status {manifest['status']!r}"
    assert manifest["task"] == "entity_detection_mixed", f"{run_id}: task {manifest['task']!r}"

    with np.load(run_dir / "artifacts" / "span_representations.npz", allow_pickle=False) as z:
        layers = z["layers"].tolist()
        assert len(layers) == 2 and layers[0] == 0, f"{run_id}: layers {layers}, expected [0, last]"
        last = layers[1]
        meta = {name: z[name] for name in REQUIRED_META}
        meta["entity_type"] = z["entity_type"]
        meta["source_dataset"] = z["source_dataset"]
        n_all = meta["doc_ids"].shape[0]
        keep = np.ones(n_all, dtype=bool)
        if max_docs is not None:
            first_docs = list(dict.fromkeys(meta["doc_ids"].tolist()))[:max_docs]
            keep = np.isin(meta["doc_ids"], first_docs)
        for name in list(meta):
            assert meta[name].shape[0] == n_all, (run_id, name, meta[name].shape)
            meta[name] = meta[name][keep]
        rep0 = z["rep_0"][keep]
        x = z[f"rep_{last}"][keep]
        assert rep0.shape == x.shape and x.dtype == np.float32 and rep0.dtype == np.float32
        x -= rep0
        del rep0

    n = x.shape[0]
    meta["label_valid"] = meta["label_valid"].astype(bool)
    meta["exact_end_alignment"] = meta["exact_end_alignment"].astype(bool)
    meta["decoy_in_ocr"] = meta["decoy_in_ocr"].astype(bool)
    if max_docs is None:
        assert n == metrics["n_spans"], f"{run_id}: {n} rows != metrics n_spans {metrics['n_spans']}"
        assert int(meta["label_valid"].sum()) == metrics["n_valid_spans"], run_id
        assert int((~meta["label_valid"]).sum()) == metrics["n_invalid_spans"], run_id
    assert np.isfinite(x).all(), f"{run_id}: non-finite values in X"

    print(
        f"{run_id}: layers=[0,{last}] n={n} valid={int(meta['label_valid'].sum())} "
        f"docs={len(set(meta['doc_ids'].tolist()))} hidden={x.shape[1]} | "
        f"exact_end_alignment=False rate={float((~meta['exact_end_alignment']).mean()):.4f}, "
        f"decoy_in_ocr rate={float(meta['decoy_in_ocr'].mean()):.4f} (kept, not filtered)"
    )
    return {"run_id": run_id, "X": x, "last_layer": last, **meta}


def error_rate_from_counts(doc_ids, sample_index, m, k, label_valid) -> np.ndarray:
    """Per-row ``k/m`` after verifying the per-prompt bookkeeping.

    Known-answer check: within every ``(doc, sample)`` prompt, ``m`` and ``k`` are
    constant, the row count equals ``m``, and the number of invalid rows equals ``k``.
    """
    df = pd.DataFrame({"doc": doc_ids, "s": sample_index, "m": m, "k": k, "invalid": ~label_valid})
    g = df.groupby(["doc", "s"], sort=False)
    agg = g.agg(m_n=("m", "nunique"), k_n=("k", "nunique"), m=("m", "first"), k=("k", "first"),
                rows=("m", "size"), n_invalid=("invalid", "sum"))
    assert (agg["m_n"] == 1).all() and (agg["k_n"] == 1).all(), "m or k varies within a prompt"
    assert (agg["rows"] == agg["m"]).all(), "prompt row count != m"
    assert (agg["n_invalid"] == agg["k"]).all(), "prompt invalid-row count != k"
    assert (agg["k"] <= agg["m"]).all() and (agg["m"] > 0).all()
    rate = np.asarray(k, dtype=np.float64) / np.asarray(m, dtype=np.float64)
    assert rate.shape == (len(doc_ids),) and ((0.0 <= rate) & (rate <= 1.0)).all()
    return rate


# ------------------------------------------------------------------- preprocessing
def center_per_document(x: np.ndarray, doc_ids: np.ndarray) -> None:
    """Subtract each document's own mean row, in place. Asserts every doc mean is ~0."""
    scale = float(x.std())
    for doc in np.unique(doc_ids):
        rows = np.flatnonzero(doc_ids == doc)
        x[rows] -= x[rows].mean(axis=0, keepdims=True)
    for doc in np.unique(doc_ids):
        rows = np.flatnonzero(doc_ids == doc)
        dev = float(np.abs(x[rows].mean(axis=0)).max())
        assert dev <= ZERO_MEAN_RTOL * scale, f"per-document centering failed for {doc!r}: {dev}"


def fit_type_means(x: np.ndarray, types: np.ndarray) -> dict[str, np.ndarray]:
    """Per-entity-type mean rows (float64 accumulation)."""
    return {t: x[types == t].mean(axis=0, dtype=np.float64).astype(np.float32) for t in np.unique(types)}


def subtract_type_means(x: np.ndarray, types: np.ndarray, means: dict[str, np.ndarray]) -> None:
    """Subtract each row's entity-type mean in place. Every type must have a fitted mean."""
    unseen = set(np.unique(types).tolist()) - set(means)
    assert not unseen, f"entity types with no train mean: {sorted(unseen)}"
    for t, mu in means.items():
        x[types == t] -= mu


def fit_dim_std(x: np.ndarray) -> np.ndarray:
    """Per-dimension std (float64 accumulation); every dimension must vary."""
    sd = x.std(axis=0, dtype=np.float64).astype(np.float32)
    assert (sd > 0).all(), "constant dimension(s): cannot standardize"
    return sd


def center_global(x: np.ndarray, mean: np.ndarray) -> None:
    """Subtract ``mean`` in place (the train split's global mean, for both splits)."""
    assert mean.shape == (x.shape[1],)
    x -= mean


# ------------------------------------------------------------------------- metrics
def classification_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """Metrics with valid (True) as the positive class."""
    assert y_true.dtype == bool and probs.shape == y_true.shape
    assert ((0.0 <= probs) & (probs <= 1.0)).all()
    preds = probs > threshold
    assert preds.any() and y_true.any(), "no predicted or no true positives: precision/recall undefined"
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds)),
        "recall": float(recall_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds)),
        "auroc": float(roc_auc_score(y_true, probs)),
    }


def smooth_ece(probs: np.ndarray, y_true: np.ndarray, n_boot: int, seed: int) -> dict:
    """Smooth ECE + relplot bootstrap CI half-width, and the diagram for plotting.

    relplot's bootstraps draw from the global numpy RNG, so reseed immediately before.
    """
    np.random.seed(seed)
    d = relplot.prepare_rel_diagram(probs, y_true, num_bootstrap=n_boot)
    return d


def plot_reliability(ax, d: dict, color: str) -> None:
    """Density-weighted smoothed reliability curve with bootstrap band."""
    mesh, mu, density = d["mesh"], d["mu"], d["density"]
    dens = density / density.max()
    alpha = CURVE_ALPHA_FLOOR + (1 - CURVE_ALPHA_FLOOR) * dens
    pts = np.array([mesh, mu]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cols = np.tile(to_rgba(color), (len(segs), 1))
    cols[:, 3] = (alpha[:-1] + alpha[1:]) / 2
    ax.add_collection(LineCollection(segs, colors=cols, lw=2.5, capstyle="round", zorder=3))
    ax.fill_between(mesh, d["lower"], d["upper"], color=color, alpha=0.20, linewidth=0, zorder=1)
    ax.plot([0, 1], [0, 1], "k:", lw=1.0, alpha=0.5, zorder=2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Predicted probability (valid)")
    ax.set_ylabel("Observed frequency")
    ax.set_box_aspect(1)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)


def project_out(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Copy of ``x`` with the unit direction ``u`` removed: ``x - (x.u) u``."""
    assert abs(float(np.linalg.norm(u)) - 1.0) < 1e-4, "u must be a unit vector"
    return x - (x @ u)[:, None] * u[None, :]


# ---------------------------------------------------------------------- probe fits
def fit_logreg(x: np.ndarray, y: np.ndarray, p: dict, seed: int) -> LogisticRegression:
    """Fit LR; a ConvergenceWarning is an error (fail loud, never a half-fit model)."""
    clf = LogisticRegression(C=p["lr_C"], max_iter=p["lr_max_iter"], tol=p["lr_tol"], random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        clf.fit(x, y)
    assert list(clf.classes_) == [False, True], clf.classes_  # positive class = valid
    return clf


# ---------------------------------------------------------------------------- plots
def scatter_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    """Random scatter subsample (order also random, so neither class is drawn on top)."""
    rng = np.random.default_rng(seed)
    if n <= max_points:
        return rng.permutation(n)
    return rng.choice(n, size=max_points, replace=False)


def plot_pca(scores, idx, label_valid, error_rate, evr, out_dir: Path, title: str) -> None:
    s = scores[idx]
    xlab, ylab = f"PC1 ({evr[0]:.1%})", f"PC2 ({evr[1]:.1%})"

    fig, ax = plt.subplots(figsize=(5.5, 5.5), constrained_layout=True)
    colors = np.where(label_valid[idx], VALID_COLOR, INVALID_COLOR)
    ax.scatter(s[:, 0], s[:, 1], c=colors, s=5, alpha=0.5, linewidths=0)
    for name, c in (("invalid", INVALID_COLOR), ("valid", VALID_COLOR)):
        ax.scatter([], [], c=c, s=25, label=name)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.set(xlabel=xlab, ylabel=ylab, title=f"{title}: validity")
    ax.set_box_aspect(1)
    fig.savefig(out_dir / "pca_validity.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.3, 5.5), constrained_layout=True)
    sc = ax.scatter(s[:, 0], s[:, 1], c=error_rate[idx], cmap=ERROR_RATE_CMAP, vmin=0.0, vmax=1.0,
                    s=5, alpha=0.85, linewidths=0)
    fig.colorbar(sc, ax=ax, label="error rate k/m")
    ax.set(xlabel=xlab, ylabel=ylab, title=f"{title}: error rate k/m")
    ax.set_box_aspect(1)
    fig.savefig(out_dir / "pca_error_rate.png", dpi=150)
    plt.close(fig)


def plot_probe_axis(logit, orth_pc, label_valid, error_rate, orth_evr, idx, out_dir: Path, title: str) -> None:
    """Test points on (probe logit, PC1 orthogonal to the probe): validity | k/m."""
    lg, pc = logit[idx], orth_pc[idx]
    xlab = "Probe logit (valid); 0 = 0.5 threshold"
    ylab = f"PC1 orthogonal to probe ({orth_evr:.1%} of orthogonal variance)"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.2), sharex=True, sharey=True, constrained_layout=True)
    colors = np.where(label_valid[idx], VALID_COLOR, INVALID_COLOR)
    ax1.scatter(lg, pc, c=colors, s=5, alpha=0.5, linewidths=0)
    for name, c in (("invalid", INVALID_COLOR), ("valid", VALID_COLOR)):
        ax1.scatter([], [], c=c, s=25, label=name)
    ax1.legend(loc="best", fontsize=9, framealpha=0.9)
    ax1.set(xlabel=xlab, ylabel=ylab, title="validity")
    sc = ax2.scatter(lg, pc, c=error_rate[idx], cmap=ERROR_RATE_CMAP, vmin=0.0, vmax=1.0,
                     s=5, alpha=0.85, linewidths=0)
    fig.colorbar(sc, ax=ax2, label="error rate k/m")
    ax2.set(xlabel=xlab, title="error rate k/m")
    for ax in (ax1, ax2):
        ax.axvline(0.0, color="k", lw=0.8, ls=":", alpha=0.6)
    fig.suptitle(title)
    fig.savefig(out_dir / "probe_axis.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def run(cfg: dict, runs_root: Path) -> dict:
    p, seed = cfg["params"], cfg["seed"]
    out = runs_root / cfg["id"]
    fig_dir = FIGURES_DIR / cfg["id"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)

    tr = load_split(p["train_run_id"], runs_root, p["max_docs"])
    te = load_split(p["test_run_id"], runs_root, p["max_docs"])
    assert tr["last_layer"] == te["last_layer"]
    assert not (set(tr["doc_ids"].tolist()) & set(te["doc_ids"].tolist())), "train/test documents overlap"

    rate_tr = error_rate_from_counts(tr["doc_ids"], tr["sample_index"], tr["m"], tr["k"], tr["label_valid"])
    rate_te = error_rate_from_counts(te["doc_ids"], te["sample_index"], te["m"], te["k"], te["label_valid"])

    # (b) per-document centering, (c) global centering with the train mean.
    center_per_document(tr["X"], tr["doc_ids"])
    center_per_document(te["X"], te["doc_ids"])
    # (b2) per-entity-type centering, train means applied to both splits.
    if p["subtract_entity_type_mean"]:
        type_means = fit_type_means(tr["X"], tr["entity_type"])
        subtract_type_means(tr["X"], tr["entity_type"], type_means)
        subtract_type_means(te["X"], te["entity_type"], type_means)
        scale = float(tr["X"].std())
        for t in type_means:
            dev = float(np.abs(tr["X"][tr["entity_type"] == t].mean(axis=0)).max())
            assert dev <= ZERO_MEAN_RTOL * scale, f"type {t!r} mean not ~0 after (b2): {dev}"
        print(f"per-entity-type centering: {len(type_means)} types {sorted(type_means)}")
    g_mean = tr["X"].mean(axis=0, dtype=np.float64).astype(np.float32)
    assert float(np.abs(g_mean).max()) <= ZERO_MEAN_RTOL * float(tr["X"].std()), "global mean not ~0 before (c)"
    center_global(tr["X"], g_mean)
    center_global(te["X"], g_mean)
    # (d) per-dimension standardization with train std.
    if p["standardize"]:
        sd = fit_dim_std(tr["X"])
        tr["X"] /= sd
        te["X"] /= sd
        assert abs(float(tr["X"].std(axis=0).mean()) - 1.0) < 1e-3, "standardization failed"
    x_tr, x_te = tr["X"], te["X"]
    y_tr, y_te = tr["label_valid"], te["label_valid"]

    metrics: dict[str, float] = {
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "train_valid_rate": float(y_tr.mean()), "test_valid_rate": float(y_te.mean()),
        "train_exact_end_false_rate": float((~tr["exact_end_alignment"]).mean()),
        "test_exact_end_false_rate": float((~te["exact_end_alignment"]).mean()),
        "train_decoy_in_ocr_rate": float(tr["decoy_in_ocr"].mean()),
        "test_decoy_in_ocr_rate": float(te["decoy_in_ocr"].mean()),
    }

    # PCA on train, two figures.
    pca = PCA(n_components=p["pca_n_components"], svd_solver=p["pca_svd_solver"], random_state=seed)
    scores = pca.fit_transform(x_tr)
    idx = scatter_indices(len(y_tr), p["max_scatter_points"], seed)
    plot_pca(scores, idx, y_tr, rate_tr, pca.explained_variance_ratio_, fig_dir, "Qwen2.5-7B train X")
    metrics["pca_evr_pc1"] = float(pca.explained_variance_ratio_[0])
    metrics["pca_evr_pc2"] = float(pca.explained_variance_ratio_[1])
    del scores

    # Logistic regression: train -> test.
    clf = fit_logreg(x_tr, y_tr, p, seed)
    probs = clf.predict_proba(x_te)[:, 1]
    assert probs.shape == y_te.shape
    for k_, v in classification_metrics(y_te, probs, p["decision_threshold"]).items():
        metrics[f"test_{k_}"] = v
    metrics["train_auroc"] = float(roc_auc_score(y_tr, clf.predict_proba(x_tr)[:, 1]))

    # Probe-axis plot: logit on x, PC1 of train X with the probe direction removed on y.
    w = clf.coef_[0].astype(np.float32)
    u = w / np.linalg.norm(w)
    logit = clf.decision_function(x_te)
    assert np.allclose(logit, x_te @ w + clf.intercept_[0], rtol=1e-3, atol=1e-3 * float(np.abs(logit).max()))
    assert np.array_equal(logit > 0, probs > 0.5), "logit sign disagrees with predict_proba > 0.5"
    orth_tr = project_out(x_tr, u)
    assert float(np.abs(orth_tr @ u).max()) <= ZERO_MEAN_RTOL * float(np.abs(x_tr @ u).max()), "probe direction not removed"
    orth_pca = PCA(n_components=1, svd_solver=p["pca_svd_solver"], random_state=seed).fit(orth_tr)
    del orth_tr
    orth_pc = orth_pca.transform(project_out(x_te, u))[:, 0]
    plot_probe_axis(logit, orth_pc, y_te, rate_te, float(orth_pca.explained_variance_ratio_[0]),
                    scatter_indices(len(y_te), p["max_scatter_points"], seed), fig_dir, "Qwen2.5-7B test X")
    metrics["probe_orth_pc1_evr"] = float(orth_pca.explained_variance_ratio_[0])

    d = smooth_ece(probs, y_te, p["ece_n_boot"], seed)
    metrics["test_smece"] = float(d["ce"])
    metrics["test_smece_ci_halfwidth"] = float(d["ce_ci_width"])
    fig, ax = plt.subplots(figsize=(4.4, 4.4), constrained_layout=True)
    plot_reliability(ax, d, VALID_COLOR)
    ax.set_title(f"Test reliability (smECE = {d['ce']:.3f} ± {d['ce_ci_width']:.3f})", fontsize=11)
    fig.savefig(fig_dir / "calibration.png", dpi=200)
    plt.close(fig)

    if p["run_controls"]:
        # Shuffled train labels -> test AUROC ~ 0.5 (leakage check).
        y_perm = np.random.default_rng(seed).permutation(y_tr)
        auc_shuf = float(roc_auc_score(y_te, fit_logreg(x_tr, y_perm, p, seed).predict_proba(x_te)[:, 1]))
        metrics["control_shuffled_auroc"] = auc_shuf
        assert abs(auc_shuf - 0.5) <= p["shuffled_label_auroc_tolerance"], (
            f"shuffled-label AUROC {auc_shuf:.4f} is not ~0.5 -- something leaks"
        )
        # Same seed twice -> identical test probabilities.
        probs2 = fit_logreg(x_tr, y_tr, p, seed).predict_proba(x_te)[:, 1]
        assert np.array_equal(probs, probs2), "same-seed refit gave different probabilities"
        metrics["control_seed_determinism"] = 1.0

    np.savez_compressed(
        out / "artifacts" / "test_predictions.npz",
        prob_valid=probs, label_valid=y_te, doc_ids=te["doc_ids"], sample_index=te["sample_index"],
        error_rate=rate_te,
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
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
