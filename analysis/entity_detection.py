"""Row-based entity detection: PCA of per-entity activations, then a single-entity logistic
validity probe with calibration and per-error-type accuracy.

Usage: ``uv run --python 3.12 --extra analysis python analysis/entity_detection.py analysis/configs/<type>/<id>.yaml``
(every value lives in the config's ``params``; a missing key is a KeyError).

Reads the ``entity_detection`` train and test runs named in the config
(``src/judge_interp/entity_detection.py``: one prompt per ``(document, row)``, one
activation per non-null field, read at the field's closing quote) and does, in order:

1. **Load** one layer (``layer``, an int in the run's ``layers``) as ``X``. Each sample is
   ``(activation, source row, row field, error type if invalid)``: ``X[i]`` with
   ``(doc_id, row_ordinal)``, ``field``, and ``field_error_type`` (empty if valid).
   The layer-0 activation is the token embedding of the closing quote, identical for every
   span (asserted), so unlike ``analysis/entity_detection.py`` there is no ``rep_L - rep_0``:
   it would be a constant shift that step 2 removes anyway.
2. **Preprocess**: subtract each *document's* mean row -- nothing else. No field-type
   normalisation, no global mean, no standardisation. (Field priors therefore stay in ``X``;
   the field-identity baseline in step 4 measures how much of the probe is just that.)
3. **PCA** fit on the train ``X``, scatter coloured by validity.
4. **Logistic regression** (train -> test) on a *single* entity's activation. Positive class
   = valid. ``lr_C`` is sklearn's inverse regularisation strength: a small value is a strong
   penalty. Reports smECE (+ bootstrap CI half-width), Brier score, Brier skill score
   (reference = constant forecast at the *train* valid rate, scored on test), AUROC, accuracy,
   and the smoothed reliability diagram.
5. **Per-error-type report** (test): for each error type T present, the detection rate on
   invalid spans of type T (fraction predicted invalid), accuracy on valid spans in clean rows
   and on valid spans inside corrupted rows of type T, and AUROC of T-invalid vs all valid.
6. **Controls**: field-identity baseline (one-hot ``field`` -> LR); mean test AUROC over
   ``n_shuffles`` shuffled-train-label fits ~ 0.5 (a single permutation is too noisy to assert,
   see ``shuffled_label_auroc``); same-seed refit gives identical probabilities.

Figures go to ``analysis/figures/<id>/`` (with a copy of ``metrics.json``); ``metrics.json``, ``config.snapshot.yaml`` and
``artifacts/test_predictions.npz`` go to ``$RUNS_ROOT/<id>/``.
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
import relplot
import yaml
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "Times New Roman", "DejaVu Serif"],
})


# ------------------------------------------------------------ shared preprocessing, probe fit, plots
# Okabe-Ito colourblind-safe pair, fixed order in every panel.
INVALID_COLOR = "#E69F00"
VALID_COLOR = "#0072B2"

# Numerical tolerance for "this mean is zero" assertions, relative to the std of X.
# Not an experimental parameter: it only bounds float32 summation error.
ZERO_MEAN_RTOL = 1e-3
# Floor on density-weighted curve alpha.
CURVE_ALPHA_FLOOR = 0.15


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


def fit_logreg(x: np.ndarray, y: np.ndarray, p: dict, seed: int) -> LogisticRegression:
    """Fit LR; a ConvergenceWarning is an error (fail loud, never a half-fit model)."""
    clf = LogisticRegression(C=p["lr_C"], max_iter=p["lr_max_iter"], tol=p["lr_tol"], random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        clf.fit(x, y)
    assert list(clf.classes_) == [False, True], clf.classes_  # positive class = valid
    return clf


def scatter_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    """Random scatter subsample (order also random, so neither class is drawn on top)."""
    rng = np.random.default_rng(seed)
    if n <= max_points:
        return rng.permutation(n)
    return rng.choice(n, size=max_points, replace=False)

FIGURES_DIR = Path(__file__).parent / "figures"
# Same set as src/judge_interp/entity_detection.ALL_ERROR_TYPES (not imported: that module needs torch).
ALL_ERROR_TYPES = ("inter_document", "intra_document", "cross_type")
REQUIRED_META = (
    "row_ordinal", "field", "field_position", "label_valid", "field_error_type", "row_error_type", "row_k",
)


# --------------------------------------------------------------------------- loading
def load_split(run_id: str, runs_root: Path, layer: int, max_docs: int | None, expected_task: str) -> dict:
    """Load one row-task run: ``X = rep_<layer>`` (float32, uncentred) plus per-span metadata.

    Gates: run.json status/task; ``layer`` is one of the run's layers; ``rep_0`` is identical for
    every span (the producer's null-control claim); when the full split is loaded
    (``max_docs is None``) the span and valid/invalid/per-error-type counts equal ``metrics.json``.
    """
    run_dir = runs_root / run_id
    manifest = json.loads((run_dir / "run.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert manifest["status"] == "success", f"{run_id}: status {manifest['status']!r}"
    assert manifest["task"] == expected_task, f"{run_id}: task {manifest['task']!r}, expected {expected_task!r}"

    with np.load(run_dir / "artifacts" / "span_representations.npz", allow_pickle=False) as z:
        layers = z["layers"].tolist()
        assert 0 in layers, f"{run_id}: layers {layers} has no layer 0 to check the null control against"
        assert layer in layers, f"{run_id}: layer {layer} not in {layers}"
        doc_ids = z["doc_ids"]
        n_all = doc_ids.shape[0]
        rep0 = z["rep_0"]
        assert rep0.shape[0] == n_all and (rep0 == rep0[:1]).all(), (
            f"{run_id}: layer-0 rows are not all identical -- the closing-quote read is leaking entity text"
        )
        del rep0
        keep = np.ones(n_all, dtype=bool)
        if max_docs is not None:
            keep = np.isin(doc_ids, list(dict.fromkeys(doc_ids.tolist()))[:max_docs])
        meta = {"doc_ids": doc_ids[keep]}
        for name in REQUIRED_META:
            col = z[name]
            assert col.shape[0] == n_all, (run_id, name, col.shape)
            meta[name] = col[keep]
        x = z[f"rep_{layer}"][keep]
    assert x.dtype == np.float32 and x.ndim == 2 and x.shape[0] == len(meta["doc_ids"]), (run_id, x.shape, x.dtype)
    assert np.isfinite(x).all(), f"{run_id}: non-finite values in X"

    meta["label_valid"] = meta["label_valid"].astype(bool)
    valid = meta["label_valid"]
    assert ((meta["field_error_type"] == "") == valid).all(), f"{run_id}: field_error_type empty iff valid violated"
    assert set(np.unique(meta["field_error_type"][~valid]).tolist()) <= set(ALL_ERROR_TYPES), run_id
    if max_docs is None:
        assert len(valid) == metrics["n_spans"], f"{run_id}: {len(valid)} rows != metrics n_spans {metrics['n_spans']}"
        assert int(valid.sum()) == metrics["n_valid_spans"], run_id
        assert int((~valid).sum()) == metrics["n_invalid_spans"], run_id
        for et in ALL_ERROR_TYPES:
            assert int((meta["field_error_type"] == et).sum()) == metrics[f"n_spans_{et}"], (run_id, et)
    print(f"{run_id}: layer={layer} n={len(valid)} valid={int(valid.sum())} docs={len(set(meta['doc_ids'].tolist()))} "
          f"hidden={x.shape[1]}")
    return {"run_id": run_id, "X": x, **meta}


# ------------------------------------------------------------------------- metrics
def brier_skill_score(y_true: np.ndarray, probs: np.ndarray, ref_rate: float) -> tuple[float, float, float]:
    """``(brier, brier_ref, bss)``: ``bss = 1 - brier / brier_ref``, ``brier_ref`` = constant forecast ``ref_rate``."""
    assert y_true.dtype == bool and probs.shape == y_true.shape and 0.0 < ref_rate < 1.0
    assert ((0.0 <= probs) & (probs <= 1.0)).all()
    y = y_true.astype(np.float64)
    brier = float(np.mean((probs - y) ** 2))
    brier_ref = float(np.mean((ref_rate - y) ** 2))
    assert brier_ref > 0
    return brier, brier_ref, 1.0 - brier / brier_ref


def error_type_report(y_valid: np.ndarray, probs: np.ndarray, threshold: float, invalid_type: np.ndarray,
                      row_type: np.ndarray, types: list[str]) -> dict:
    """Flat per-error-type metrics (positive class = valid; "detected" = predicted invalid).

    ``invalid_type[i]``: error type of item ``i`` if it is invalid (ignored where valid);
    ``row_type[i]``: error type of the source row, ``""`` for a clean row. Keys:

    - ``valid_acc_clean_rows`` / ``n_valid_clean_rows``: valid items from clean rows predicted valid;
    - per T: ``detect_<T>`` / ``n_invalid_<T>`` (fraction of T-invalid predicted invalid),
      ``valid_acc_in_<T>_rows`` / ``n_valid_in_<T>_rows`` (valid items inside T-corrupted rows
      predicted valid; omitted if there are none), ``auroc_<T>_vs_valid`` (T-invalid vs all valid).
    """
    assert y_valid.dtype == bool and probs.shape == y_valid.shape == invalid_type.shape == row_type.shape
    pred_valid = probs > threshold
    out: dict[str, float] = {}
    clean = y_valid & (row_type == "")
    assert clean.any(), "no valid items from clean rows"
    out["valid_acc_clean_rows"] = float(pred_valid[clean].mean())
    out["n_valid_clean_rows"] = int(clean.sum())
    assert set(np.unique(invalid_type[~y_valid]).tolist()) <= set(types), "invalid item with an unlisted error type"
    for t in types:
        inv_t = ~y_valid & (invalid_type == t)
        assert inv_t.any(), f"error type {t!r} has no invalid items"
        out[f"detect_{t}"] = float((~pred_valid[inv_t]).mean())
        out[f"n_invalid_{t}"] = int(inv_t.sum())
        sel = inv_t | y_valid
        out[f"auroc_{t}_vs_valid"] = float(roc_auc_score(y_valid[sel], probs[sel]))
        val_in_t = y_valid & (row_type == t)
        if val_in_t.any():
            out[f"valid_acc_in_{t}_rows"] = float(pred_valid[val_in_t].mean())
            out[f"n_valid_in_{t}_rows"] = int(val_in_t.sum())
    return out


def shuffled_label_auroc(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray, p: dict,
                         seed: int) -> tuple[float, float]:
    """``(mean, std)`` of test AUROC over ``p["n_shuffles"]`` fits on permuted train labels.

    A probe fit on permuted labels is a random direction; its chance overlap with a strong planted
    or real signal moves a *single* permutation's test AUROC well off 0.5 with a random sign (std
    ~0.1 in 64-d), so one permutation cannot be asserted against a tight tolerance. The mean over
    many permutations is ~0.5 unless something leaks.
    """
    rng = np.random.default_rng(seed)
    aucs = [float(roc_auc_score(y_te, fit_logreg(x_tr, rng.permutation(y_tr), p, seed).predict_proba(x_te)[:, 1]))
            for _ in range(p["n_shuffles"])]
    return float(np.mean(aucs)), float(np.std(aucs))


def onehot(values: np.ndarray, categories: list[str]) -> np.ndarray:
    """One-hot rows over ``categories``; every value must be a listed category."""
    unseen = set(np.unique(values).tolist()) - set(categories)
    assert not unseen, f"values with no category: {sorted(unseen)}"
    return (values[:, None] == np.asarray(categories)[None, :]).astype(np.float64)


def probe_metrics(prefix: str, y_te: np.ndarray, probs: np.ndarray, ref_rate: float, threshold: float) -> dict:
    """AUROC, Brier, BSS, accuracy of one probe on the test set."""
    brier, brier_ref, bss = brier_skill_score(y_te, probs, ref_rate)
    return {
        f"{prefix}_auroc": float(roc_auc_score(y_te, probs)),
        f"{prefix}_brier": brier,
        f"{prefix}_brier_ref": brier_ref,
        f"{prefix}_bss": bss,
        f"{prefix}_accuracy": float(accuracy_score(y_te, probs > threshold)),
    }


# ---------------------------------------------------------------------------- plots
def plot_pca_validity(scores, idx, label_valid, evr, out_dir: Path, title: str) -> None:
    s = scores[idx]
    fig, ax = plt.subplots(figsize=(5.5, 5.5), constrained_layout=True)
    colors = np.where(label_valid[idx], VALID_COLOR, INVALID_COLOR)
    ax.scatter(s[:, 0], s[:, 1], c=colors, s=5, alpha=0.5, linewidths=0)
    for name, c in (("invalid", INVALID_COLOR), ("valid", VALID_COLOR)):
        ax.scatter([], [], c=c, s=25, label=name)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.set(xlabel=f"PC1 ({evr[0]:.1%})", ylabel=f"PC2 ({evr[1]:.1%})", title=f"{title}: validity")
    ax.set_box_aspect(1)
    fig.savefig(out_dir / "pca_validity.png", dpi=150)
    plt.close(fig)


def plot_calibration(d: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 4.4), constrained_layout=True)
    plot_reliability(ax, d, VALID_COLOR)
    ax.set_title(f"Test reliability (smECE = {d['ce']:.3f} ± {d['ce_ci_width']:.3f})", fontsize=11)
    fig.savefig(out_dir / "calibration.png", dpi=200)
    plt.close(fig)


def plot_prob_histogram(probs: np.ndarray, n_bins: int, out_dir: Path) -> None:
    """Histogram of ``P(valid)`` over [0, 1] (``n_bins`` equal-width bins) -> ``prob_histogram.png``."""
    assert probs.ndim == 1 and len(probs) > 0 and np.isfinite(probs).all() and ((0.0 <= probs) & (probs <= 1.0)).all()
    assert isinstance(n_bins, int) and n_bins > 0, n_bins
    fig, ax = plt.subplots(figsize=(4.4, 3.2), constrained_layout=True)
    ax.hist(probs, bins=n_bins, range=(0.0, 1.0), color=VALID_COLOR)
    ax.set(xlabel="P(valid)", ylabel="count", xlim=(0.0, 1.0), title=f"Test probabilities (n = {len(probs)})")
    fig.savefig(out_dir / "prob_histogram.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def run(cfg: dict, runs_root: Path) -> dict:
    p, seed = cfg["params"], cfg["seed"]
    out = runs_root / cfg["id"]
    fig_dir = FIGURES_DIR / cfg["id"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    layer, thr = p["layer"], p["decision_threshold"]
    assert isinstance(layer, int) and layer > 0, "layer must be a positive int (layer 0 is the constant null control)"

    tr = load_split(p["train_run_id"], runs_root, layer, p["max_docs"], "entity_detection")
    te = load_split(p["test_run_id"], runs_root, layer, p["max_docs"], "entity_detection")
    assert not (set(tr["doc_ids"].tolist()) & set(te["doc_ids"].tolist())), "train/test documents overlap"
    types = sorted(set(np.unique(tr["field_error_type"][~tr["label_valid"]]).tolist()))
    assert types == sorted(set(np.unique(te["field_error_type"][~te["label_valid"]]).tolist())), (
        "train and test have different error types"
    )

    # The only preprocessing: per-document mean subtraction (asserted ~0 per document).
    center_per_document(tr["X"], tr["doc_ids"])
    center_per_document(te["X"], te["doc_ids"])
    x_tr, x_te, y_tr, y_te = tr["X"], te["X"], tr["label_valid"], te["label_valid"]
    for y in (y_tr, y_te):
        assert y.any() and (~y).any(), "one class missing"
    ref_rate = float(y_tr.mean())

    metrics: dict[str, float] = {
        "layer": layer, "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "train_valid_rate": ref_rate, "test_valid_rate": float(y_te.mean()),
        "n_train_docs": len(set(tr["doc_ids"].tolist())), "n_test_docs": len(set(te["doc_ids"].tolist())),
    }

    pca = PCA(n_components=p["pca_n_components"], svd_solver=p["pca_svd_solver"], random_state=seed)
    scores = pca.fit_transform(x_tr)
    plot_pca_validity(scores, scatter_indices(len(y_tr), p["max_scatter_points"], seed), y_tr,
                      pca.explained_variance_ratio_, fig_dir, f"train X (layer {layer})")
    metrics["pca_evr_pc1"], metrics["pca_evr_pc2"] = (float(v) for v in pca.explained_variance_ratio_[:2])
    del scores

    clf = fit_logreg(x_tr, y_tr, p, seed)
    probs = clf.predict_proba(x_te)[:, 1]
    assert probs.shape == y_te.shape
    metrics.update(probe_metrics("test", y_te, probs, ref_rate, thr))
    metrics["train_auroc"] = float(roc_auc_score(y_tr, clf.predict_proba(x_tr)[:, 1]))

    d = smooth_ece(probs, y_te, p["ece_n_boot"], seed)
    metrics["test_smece"], metrics["test_smece_ci_halfwidth"] = float(d["ce"]), float(d["ce_ci_width"])
    plot_calibration(d, fig_dir)
    plot_prob_histogram(probs, p["hist_n_bins"], fig_dir)

    for k_, v in error_type_report(y_te, probs, thr, te["field_error_type"], te["row_error_type"], types).items():
        metrics[f"test_{k_}"] = v

    # Field-identity baseline: what a probe that sees only which field this is can do.
    fields = sorted(set(tr["field"].tolist()))
    base = fit_logreg(onehot(tr["field"], fields), y_tr, p, seed)
    base_probs = base.predict_proba(onehot(te["field"], fields))[:, 1]
    metrics.update(probe_metrics("test_baseline_field", y_te, base_probs, ref_rate, thr))

    if p["run_controls"]:
        auc_shuf, auc_shuf_std = shuffled_label_auroc(x_tr, y_tr, x_te, y_te, p, seed)
        metrics["control_shuffled_auroc"], metrics["control_shuffled_auroc_std"] = auc_shuf, auc_shuf_std
        assert abs(auc_shuf - 0.5) <= p["shuffled_label_auroc_tolerance"], (
            f"mean shuffled-label AUROC {auc_shuf:.4f} (std {auc_shuf_std:.4f}, n={p['n_shuffles']}) is not ~0.5 -- something leaks"
        )
        assert np.array_equal(probs, fit_logreg(x_tr, y_tr, p, seed).predict_proba(x_te)[:, 1]), (
            "same-seed refit gave different probabilities"
        )
        metrics["control_seed_determinism"] = 1.0

    np.savez_compressed(
        out / "artifacts" / "test_predictions.npz",
        prob_valid=probs, label_valid=y_te, doc_ids=te["doc_ids"], row_ordinal=te["row_ordinal"],
        field=te["field"], field_error_type=te["field_error_type"], row_error_type=te["row_error_type"],
    )
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
