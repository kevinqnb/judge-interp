"""Row-based entity detection on the main dataset: PCA of per-entity activations and a single-entity
logistic validity probe, scored on minimal pairs.

Usage: ``uv run --python 3.12 --extra analysis python analysis/entity_detection.py analysis/configs/<type>/<id>.yaml``

Reads the ``entity_detection`` train and test runs named in the config, then:

1. Load one layer and subtract each document's mean activation (nothing else).
2. Pair every corrupted span with its clean twin span (``data/vrdu/main_samples.main_entity_pairs``
   on the row file the run judged): the same field in the document's clean row, or, for a null-fill
   span, in another document's clean row. Train and test sets are these pairs, so each (error type,
   k, field) stratum is exactly 50/50.
3. PCA on train, coloured by validity.
4. Logistic regression on one entity's activation, train -> test: smECE, Brier (skill), AUROC (with a
   document-bootstrap interval), accuracy, reliability diagram; per error type, per (type, k), and
   with / without null-fill pairs.
5. Row-context diagnostic: the uncorrupted spans of corrupted rows against the same spans in the clean
   row (``kept_span_pairs``; both valid). AUROC ~ 0.5 means the probe does not just sense "this row
   has a corruption somewhere".
6. Controls: field-identity baseline (exactly 0.5 AUROC on pairs, asserted), shuffled-label AUROC
   ~ 0.5, seed determinism.

Outputs go to ``$RUNS_ROOT/<id>/`` and ``analysis/figures/<id>/``.
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

from judge_interp.dataset_spec import load_dataset_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "data" / "vrdu"))
from main_samples import kept_span_pairs, main_entity_pairs  # noqa: E402
from vrdu_common import MAIN_FIELDS, check_build  # noqa: E402

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
# Every error type of the VRDU datasets (main: inter_document, cross_type; line: intra_document, cross_item).
ALL_ERROR_TYPES = ("inter_document", "intra_document", "cross_type", "cross_item")
# Tolerance for "this AUROC is exactly 0.5" (a structure-only baseline on minimal pairs): float rounding only.
EXACT_AUROC_ATOL = 1e-9
REQUIRED_META = (
    "row_ordinal", "field", "field_position", "label_valid", "field_error_type", "row_error_type", "row_k",
)


# --------------------------------------------------------------------------- loading
def load_split(run_id: str, runs_root: Path, layer: int, max_docs: int | None, expected_task: str,
               extra_meta: tuple[str, ...]) -> dict:
    """Load one row-task run: ``X = rep_<layer>`` plus per-span metadata (``extra_meta``: extra columns).

    Checks run status/task, that ``rep_0`` is constant, and (full split) counts against ``metrics.json``.
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
        assert not set(extra_meta) & set(REQUIRED_META), extra_meta
        for name in (*REQUIRED_META, *extra_meta):
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
        run_types = [et for et in ALL_ERROR_TYPES if f"n_spans_{et}" in metrics]   # the run's spec.error_types
        assert sum(metrics[f"n_spans_{et}"] for et in run_types) == metrics["n_invalid_spans"], (run_id, run_types)
        for et in run_types:
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


def stratum_report(y_valid: np.ndarray, probs: np.ndarray, threshold: float, strata: np.ndarray) -> dict:
    """Per-stratum metrics of a minimal-pair set (both members of a pair share a stratum).

    For each stratum value ``s``: ``n_<s>``, ``auroc_<s>``, ``accuracy_<s>``, ``detect_<s>`` (invalid
    items called invalid) and ``valid_acc_<s>`` (valid items called valid).

    Raises:
        AssertionError: a stratum without both classes.
    """
    assert y_valid.dtype == bool and probs.shape == y_valid.shape == strata.shape
    pred_valid = probs > threshold
    out: dict[str, float] = {}
    for s in sorted(set(strata.tolist())):
        sel = strata == s
        y = y_valid[sel]
        assert y.any() and (~y).any(), f"stratum {s!r} lacks a class"
        out[f"n_{s}"] = int(sel.sum())
        out[f"auroc_{s}"] = float(roc_auc_score(y, probs[sel]))
        out[f"accuracy_{s}"] = float((pred_valid[sel] == y).mean())
        out[f"detect_{s}"] = float((~pred_valid[sel][~y]).mean())
        out[f"valid_acc_{s}"] = float(pred_valid[sel][y].mean())
    return out


def doc_bootstrap_auroc(y_valid: np.ndarray, probs: np.ndarray, docs: np.ndarray, n_boot: int, ci_level: float,
                        seed: int) -> tuple[float, float]:
    """Percentile interval of AUROC over ``n_boot`` resamples of documents, with replacement.

    ``docs``: the document each item is resampled with (for a minimal pair, both members carry the
    invalid member's document, so a resample keeps pairs whole).
    """
    assert isinstance(n_boot, int) and n_boot > 0 and 0.0 < ci_level < 1.0, (n_boot, ci_level)
    assert y_valid.shape == probs.shape == docs.shape
    uniq = np.unique(docs)
    members = [np.flatnonzero(docs == d) for d in uniq]
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate([members[j] for j in rng.integers(len(uniq), size=len(uniq))])
        assert y_valid[idx].any() and (~y_valid[idx]).any(), "a document resample lacks a class"
        aucs.append(roc_auc_score(y_valid[idx], probs[idx]))
    lo, hi = np.quantile(aucs, [(1 - ci_level) / 2, (1 + ci_level) / 2])
    return float(lo), float(hi)


# --------------------------------------------------------------------- minimal pairs (main)
def load_run_rows(run_id: str, runs_root: Path, repo_root: Path, dataset: str) -> list[dict]:
    """The row file a run judged (named by its ``config.snapshot.yaml``); refuses a stale file (``check_build``)."""
    params = yaml.safe_load((runs_root / run_id / "config.snapshot.yaml").read_text())["params"]
    spec = load_dataset_spec(repo_root / params["dataset_config"])
    assert spec.name == dataset, f"{run_id}: dataset {spec.name!r}, expected {dataset!r}"
    path = repo_root / params["data_root"] / spec.split_path.format(split=params["split"])
    check_build(path.parent.parent, f"{path.parent.name}/{path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def main_span_pairs(split: dict, rows: list[dict], seed: int) -> dict:
    """Minimal entity pairs of a main run, as span indices.

    Only the rows the run judged are paired (a run may draw a subset of rows); every shown field of a
    judged row must be a span with the row file's value.

    Args:
        split: per-span ``doc_ids``, ``row_error_type``, ``row_k``, ``field``, ``value``, ``label_valid``.
        rows: the main row file the run judged (``load_run_rows``).
        seed: seeds the other-document twins of null-fill spans (``main_entity_pairs``).

    Returns:
        Arrays ``inv`` and ``val`` (the invalid span and its clean twin), ``error_type``, ``k``,
        ``field``, ``null_fill`` and ``doc`` (the invalid span's document), one entry per pair; and
        ``kept``, ``clean``, ``kept_error_type``, ``kept_doc`` for the row-context diagnostic.

    Raises:
        AssertionError: a span of no row, a row value the span disagrees with, a judged corrupted row
            whose clean row was not judged, or a label that disagrees with the pairing.
    """
    keys = zip(split["doc_ids"].tolist(), split["row_error_type"].tolist(), split["row_k"].tolist(), split["field"].tolist())
    index: dict[tuple, int] = {}
    for i, key in enumerate(keys):
        assert key not in index, f"span {key} occurs twice"
        index[key] = i
    by_row = {(r["document_id"], r["error_type"] or "", r["k"]): r for r in rows}
    assert len(by_row) == len(rows), "row keys are not unique"
    judged = sorted({key[:3] for key in index})
    assert set(judged) <= set(by_row), f"spans of rows not in the row file, e.g. {sorted(set(judged) - set(by_row))[:3]}"
    n_shown = 0
    for rk in judged:
        r = by_row[rk]
        for f in (f for f in MAIN_FIELDS if r[f] is not None):
            assert (*rk, f) in index, f"row {rk}: shown field {f!r} has no span"
            assert split["value"][index[(*rk, f)]] == r[f], f"row {rk}, {f!r}: span value differs from the row file"
            n_shown += 1
    assert n_shown == len(index), "a span of a field the row does not show"

    sub = [by_row[rk] for rk in judged]
    pairs = main_entity_pairs(sub, seed)
    out = {
        "inv": np.array([index[(p["invalid"]["document_id"], p["error_type"], p["k"], p["field"])] for p in pairs]),
        "val": np.array([index[(p["valid"]["document_id"], "", 0, p["field"])] for p in pairs]),
        "error_type": np.array([p["error_type"] for p in pairs]), "k": np.array([p["k"] for p in pairs]),
        "field": np.array([p["field"] for p in pairs]), "null_fill": np.array([p["null_fill"] for p in pairs]),
        "doc": np.array([p["invalid"]["document_id"] for p in pairs]),
    }
    kept = kept_span_pairs(sub)
    out["kept"] = np.array([index[(p["kept"]["document_id"], p["error_type"], p["k"], p["field"])] for p in kept], dtype=np.int64)
    out["clean"] = np.array([index[(p["clean"]["document_id"], "", 0, p["field"])] for p in kept], dtype=np.int64)
    out["kept_error_type"] = np.array([p["error_type"] for p in kept])
    out["kept_doc"] = np.array([p["kept"]["document_id"] for p in kept])
    valid = split["label_valid"]
    assert not valid[out["inv"]].any() and valid[out["val"]].all(), "a pair's labels disagree with the span labels"
    assert valid[out["kept"]].all() and valid[out["clean"]].all(), "a kept / clean span is not valid"
    assert len(out["inv"]) == int((~valid).sum()), "not every invalid span is paired"
    return out


def stack_pairs(inv: np.ndarray, val: np.ndarray, **cols: np.ndarray) -> dict:
    """Both members of every pair as one evaluation set: ``idx`` (invalid members, then their twins),
    ``y`` (valid), and each of ``cols`` (per pair) repeated for both members."""
    assert inv.shape == val.shape and all(c.shape == inv.shape for c in cols.values())
    n = len(inv)
    return {"idx": np.concatenate([inv, val]), "y": np.concatenate([np.zeros(n, bool), np.ones(n, bool)]),
            **{k: np.concatenate([c, c]) for k, c in cols.items()}}


def assert_structure_auroc_is_chance(name: str, y_tr: np.ndarray, cats_tr: np.ndarray, y_te: np.ndarray,
                                     cats_te: np.ndarray, p: dict, seed: int) -> dict:
    """Known-answer control: a probe on one-hot structure categories (each exactly 50/50 in a minimal-pair
    set) has test AUROC exactly 0.5; anything else means the pairing is broken. Returns its metrics."""
    cats = sorted(set(cats_tr.tolist()) | set(cats_te.tolist()))
    base = fit_logreg(onehot(cats_tr, cats), y_tr, p, seed)
    probs = base.predict_proba(onehot(cats_te, cats))[:, 1]
    m = probe_metrics(f"test_baseline_{name}", y_te, probs, float(y_tr.mean()), p["decision_threshold"])
    assert abs(m[f"test_baseline_{name}_auroc"] - 0.5) <= EXACT_AUROC_ATOL, (
        f"{name} baseline AUROC {m[f'test_baseline_{name}_auroc']} != 0.5 on minimal pairs -- the pairing is broken")
    return m


def shuffled_label_auroc(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray, p: dict,
                         seed: int) -> tuple[float, float]:
    """``(mean, std)`` test AUROC over ``n_shuffles`` fits on permuted train labels (a single one is too noisy)."""
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
def pair_set(split: dict, rows: list[dict], seed: int) -> tuple[dict, dict]:
    """``(z, s)``: ``main_span_pairs`` and both members of its pairs stacked (``stack_pairs``)."""
    z = main_span_pairs(split, rows, seed)
    s = stack_pairs(z["inv"], z["val"], error_type=z["error_type"], k=z["k"], field=z["field"],
                    null_fill=z["null_fill"], doc=z["doc"])
    s["type_k"] = np.char.add(np.char.add(s["error_type"], "_k"), s["k"].astype(str))
    return z, s


def run(cfg: dict, runs_root: Path) -> dict:
    p, seed = cfg["params"], cfg["seed"]
    out = runs_root / cfg["id"]
    fig_dir = FIGURES_DIR / cfg["id"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    layer, thr = p["layer"], p["decision_threshold"]
    assert isinstance(layer, int) and layer > 0, "layer must be a positive int (layer 0 is the constant null control)"

    tr = load_split(p["train_run_id"], runs_root, layer, p["max_docs"], "entity_detection", ("value",))
    te = load_split(p["test_run_id"], runs_root, layer, p["max_docs"], "entity_detection", ("value",))
    assert not (set(tr["doc_ids"].tolist()) & set(te["doc_ids"].tolist())), "train/test documents overlap"

    # The only preprocessing: per-document mean subtraction (asserted ~0 per document).
    center_per_document(tr["X"], tr["doc_ids"])
    center_per_document(te["X"], te["doc_ids"])
    z_tr, s_tr = pair_set(tr, load_run_rows(p["train_run_id"], runs_root, REPO_ROOT, "main"), p["twin_seed"])
    z_te, s_te = pair_set(te, load_run_rows(p["test_run_id"], runs_root, REPO_ROOT, "main"), p["twin_seed"])
    x_tr, x_te, y_tr, y_te = tr["X"][s_tr["idx"]], te["X"][s_te["idx"]], s_tr["y"], s_te["y"]
    types = sorted(set(z_tr["error_type"].tolist()))
    assert types == sorted(set(z_te["error_type"].tolist())), "train and test have different error types"
    ref_rate = float(y_tr.mean())
    assert ref_rate == 0.5 == float(y_te.mean()), "minimal-pair sets are not 50/50"

    metrics: dict[str, float] = {
        "layer": layer, "n_train_pairs": int(len(z_tr["inv"])), "n_test_pairs": int(len(z_te["inv"])),
        "n_train_null_fill_pairs": int(z_tr["null_fill"].sum()), "n_test_null_fill_pairs": int(z_te["null_fill"].sum()),
        # one clean span is the twin in every (type, k) stratum, so far fewer valid spans than pairs
        "n_train_unique_valid_spans": len(set(z_tr["val"].tolist())), "n_test_unique_valid_spans": len(set(z_te["val"].tolist())),
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
    metrics["test_auroc_ci_lo"], metrics["test_auroc_ci_hi"] = doc_bootstrap_auroc(
        y_te, probs, s_te["doc"], p["auroc_n_boot"], p["ci_level"], seed)
    no_fill = ~s_te["null_fill"]
    metrics.update(probe_metrics("test_no_null_fill", y_te[no_fill], probs[no_fill], ref_rate, thr))
    metrics["test_no_null_fill_auroc_ci_lo"], metrics["test_no_null_fill_auroc_ci_hi"] = doc_bootstrap_auroc(
        y_te[no_fill], probs[no_fill], s_te["doc"][no_fill], p["auroc_n_boot"], p["ci_level"], seed)
    if s_te["null_fill"].any():
        fill = s_te["null_fill"]
        metrics["test_null_fill_only_auroc"] = float(roc_auc_score(y_te[fill], probs[fill]))

    d = smooth_ece(probs, y_te, p["ece_n_boot"], seed)
    metrics["test_smece"], metrics["test_smece_ci_halfwidth"] = float(d["ce"]), float(d["ce_ci_width"])
    plot_calibration(d, fig_dir)
    plot_prob_histogram(probs, p["hist_n_bins"], fig_dir)

    for col in ("error_type", "type_k"):
        for k_, v in stratum_report(y_te, probs, thr, s_te[col]).items():
            metrics[f"test_{k_}"] = v
    for k_, v in stratum_report(y_te[no_fill], probs[no_fill], thr, s_te["error_type"][no_fill]).items():
        metrics[f"test_no_null_fill_{k_}"] = v

    # Row-context diagnostic: uncorrupted spans of corrupted rows vs the same spans in the clean row.
    p_kept, p_clean = clf.predict_proba(te["X"][z_te["kept"]])[:, 1], clf.predict_proba(te["X"][z_te["clean"]])[:, 1]
    metrics["test_kept_mean_p_valid"], metrics["test_kept_twin_mean_p_valid"] = float(p_kept.mean()), float(p_clean.mean())
    metrics["test_kept_valid_acc"] = float((p_kept > thr).mean())
    n_kept = len(p_kept)
    metrics["test_clean_vs_kept_auroc"] = float(roc_auc_score(
        np.concatenate([np.zeros(n_kept, bool), np.ones(n_kept, bool)]), np.concatenate([p_kept, p_clean])))

    # Field-identity baseline: on minimal pairs every field is 50/50, so exactly chance (asserted).
    metrics.update(assert_structure_auroc_is_chance("field", y_tr, s_tr["field"], y_te, s_te["field"], p, seed))

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
        prob_valid=probs, label_valid=y_te, span=s_te["idx"], pair_doc=s_te["doc"], doc_ids=te["doc_ids"][s_te["idx"]],
        field=s_te["field"], error_type=s_te["error_type"], k=s_te["k"], null_fill=s_te["null_fill"],
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
