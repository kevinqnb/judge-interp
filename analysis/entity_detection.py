"""Entity detection on the main dataset: per-model probes and the direct judges (Clef, Jev), all scored on the
same minimal pairs of ``params.test_rows`` (``data/vrdu/main/test.json``).

Usage: ``uv run --python 3.12 --extra analysis python analysis/entity_detection.py analysis/configs/<type>/<id>.yaml``

Methods (``params.models``: a probe per entry; ``params.judges``: a Clef / Jev run per entry, nothing trained).
Probes train on the ``entity_detection`` train runs, which must have judged ``params.train_rows``
(``data/vrdu/main/train.json``); every method's test run must have judged ``params.test_rows``. Metric keys and
figure directories are prefixed with the method's ``label``.

1. Load one layer per model and subtract each document's mean activation (nothing else), over the model's whole run.
2. The common set: the test rows every method scored (a judge excludes over-length documents; the LMs have a
   context limit). Each method's spans are cut to them and each method's dropped rows are reported. Pairs are built
   per method from these rows (``data/vrdu/main_samples.main_entity_pairs``, same twin seed) and must be
   identical across methods (asserted): every corrupted span with its clean twin span (the same field in the
   document's clean row, or, for a null-fill span, in another document's clean row). Each (error type, k,
   field) stratum is exactly 50/50.
3. Probes: PCA on train, coloured by validity; logistic regression on one entity's activation, train -> test.
   Judges: ``analysis/clef.py``'s metrics on the same pairs. Both: smECE, Brier (skill), AUROC (with a
   document-bootstrap interval), accuracy; per error type, per (type, k), and with / without null-fill pairs.
4. Row-context diagnostic: the uncorrupted spans of corrupted rows against the same spans in the clean
   row (``kept_span_pairs``; both valid). AUROC ~ 0.5 means the method does not just sense "this row
   has a corruption somewhere".
5. Controls: probes -- field-identity baseline (exactly 0.5 AUROC on pairs, asserted), shuffled-label AUROC
   ~ 0.5, seed determinism; judges -- permutation null of the metrics.

6. Report (``analysis/calibration_report.py``, ``params.report``): the only figures. Per error type, a calibration
   diagram of all methods and a smECE / BSS / AUROC table, every interval over ``report.n_boot`` resamples of documents
   (a null-fill pair is resampled with its invalid member's document, as in the AUROC interval above). The
   pooled values are asserted equal to the metrics above.

Outputs go to ``$RUNS_ROOT/<id>/`` and ``analysis/figures/<id>/``.
"""
from __future__ import annotations

import importlib.util
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

_cr_spec = importlib.util.spec_from_file_location("analysis_calibration_report", Path(__file__).parent / "calibration_report.py")
cr = importlib.util.module_from_spec(_cr_spec)
_cr_spec.loader.exec_module(cr)

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
def run_rows_path(run_id: str, runs_root: Path, repo_root: Path, dataset: str) -> Path:
    """Path of the row file a run judged (named by its ``config.snapshot.yaml``)."""
    params = yaml.safe_load((runs_root / run_id / "config.snapshot.yaml").read_text())["params"]
    spec = load_dataset_spec(repo_root / params["dataset_config"])
    assert spec.name == dataset, f"{run_id}: dataset {spec.name!r}, expected {dataset!r}"
    return repo_root / params["data_root"] / spec.split_path.format(split=params["split"])


def load_run_rows(run_id: str, runs_root: Path, repo_root: Path, dataset: str) -> list[dict]:
    """The row file a run judged (``run_rows_path``); refuses a stale file (``check_build``)."""
    path = run_rows_path(run_id, runs_root, repo_root, dataset)
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


def line_span_pairs(split: dict, rows: list[dict]) -> dict:
    """Minimal entity pairs of a line run, as span indices.

    A corrupted span is paired with the same field of its clean twin sample (same ``pair_id``, original
    values). Spans left uncorrupted in an invalid sample are paired with the twin's span too, for the
    context diagnostic. Only the samples the run judged are paired; a pair must be judged whole.

    Args:
        split: per-span ``doc_ids``, ``pair_id``, ``cell``, ``row_error_type``, ``row_k``, ``field``,
            ``value``, ``label_valid``, ``field_error_type``.
        rows: the line row file the run judged (``load_run_rows``).

    Returns:
        The arrays of ``main_span_pairs`` (``inv``, ``val``, ``error_type``, ``k``, ``field``, ``null_fill``
        -- always False on line --, ``doc``, ``kept``, ``clean``, ``kept_error_type``, ``kept_doc``) and
        ``cell``, ``kept_cell``.

    Raises:
        AssertionError: a span of no sample, a value or label the row file disagrees with, a sample whose
            twin was not judged, or a corruption that left the value unchanged.
    """
    keys = zip(split["pair_id"].tolist(), (split["row_error_type"] == "").tolist(), split["field"].tolist())
    index: dict[tuple, int] = {}
    for i, key in enumerate(keys):
        assert key not in index, f"span {key} occurs twice"
        index[key] = i
    by_sample = {(r["pair_id"], r["valid"]): r for r in rows}
    assert len(by_sample) == len(rows), "(pair_id, valid) is not unique in the row file"
    judged = sorted({key[:2] for key in index})
    assert set(judged) <= set(by_sample), f"spans of samples not in the row file, e.g. {sorted(set(judged) - set(by_sample))[:3]}"
    pair_ids = sorted({pid for pid, _ in judged})
    judged_set = set(judged)
    missing = [(pid, v) for pid in pair_ids for v in (True, False) if (pid, v) not in judged_set]
    assert not missing, f"{len(missing)} samples whose twin was not judged, e.g. {missing[:2]}"

    n_shown = 0
    for key in judged:
        r = by_sample[key]
        for f in r["fields"]:
            assert (*key, f) in index, f"sample {key}: shown field {f!r} has no span"
            i = index[(*key, f)]
            assert split["value"][i] == r[f], f"sample {key}, {f!r}: span value differs from the row file"
            assert bool(split["label_valid"][i]) == (r["field_errors"][f] is None), f"sample {key}, {f!r}: span label differs from the row file"
            n_shown += 1
    assert n_shown == len(index), "a span of a field the sample does not show"

    inv, val, kept, clean = [], [], [], []
    for pid in pair_ids:
        for f in by_sample[(pid, False)]["fields"]:
            i, j = index[(pid, False, f)], index[(pid, True, f)]
            if split["label_valid"][i]:
                assert split["value"][i] == split["value"][j], f"{pid}, {f!r}: an uncorrupted span differs from its twin"
                kept.append(i)
                clean.append(j)
            else:
                assert split["value"][i] != split["value"][j], f"{pid}, {f!r}: the corruption left the value unchanged"
                inv.append(i)
                val.append(j)
    inv, val, kept, clean = (np.array(a, dtype=np.int64) for a in (inv, val, kept, clean))
    valid = split["label_valid"]
    assert not valid[inv].any() and valid[val].all() and valid[kept].all() and valid[clean].all(), "a pair's labels disagree with the span labels"
    assert len(inv) == int((~valid).sum()), "not every invalid span is paired"
    return {
        "inv": inv, "val": val, "error_type": split["field_error_type"][inv], "k": split["row_k"][inv],
        "field": split["field"][inv], "null_fill": np.zeros(len(inv), dtype=bool), "doc": split["doc_ids"][inv],
        "cell": split["cell"][inv], "kept": kept, "clean": clean, "kept_error_type": split["row_error_type"][kept],
        "kept_doc": split["doc_ids"][kept], "kept_cell": split["cell"][kept],
    }


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
DATASETS = ("main", "line")


def pair_set(split: dict, rows: list[dict], seed: int | None, dataset: str) -> tuple[dict, dict]:
    """``(z, s)``: the dataset's span pairs (``main_span_pairs`` / ``line_span_pairs``) and both members
    stacked (``stack_pairs``). ``s["stratum"]`` is the finer stratification: (type, k) on main, the cell on line."""
    assert dataset in DATASETS, dataset
    z = main_span_pairs(split, rows, seed) if dataset == "main" else line_span_pairs(split, rows)
    s = stack_pairs(z["inv"], z["val"], error_type=z["error_type"], k=z["k"], field=z["field"],
                    null_fill=z["null_fill"], doc=z["doc"])
    if dataset == "main":
        s["stratum"] = np.char.add(np.char.add(s["error_type"], "_k"), s["k"].astype(str))
    else:
        s["stratum"] = np.concatenate([z["cell"]] * 2)
    return z, s


JUDGE_KINDS = ("clef", "jev")


def load_sibling(name: str):
    """Load ``analysis/<name>.py`` as a module. Imported here, not at the top: ``clef.py`` and ``jev.py`` load this file."""
    spec = importlib.util.spec_from_file_location(f"analysis_{name}", Path(__file__).parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def row_keys(split: dict) -> set[tuple]:
    """The ``(document, row error type, k)`` rows a run has spans of."""
    return set(zip(split["doc_ids"].tolist(), split["row_error_type"].tolist(), split["row_k"].tolist()))


def restrict_to_rows(split: dict, keys: set[tuple]) -> dict:
    """``split`` cut to the spans of rows ``keys`` (a subset of its rows): every per-span array (leading
    dimension = number of spans) is cut, everything else is kept. Per-document centring must come first --
    it is a mean over the document's spans in the run, not over the common rows.

    Raises:
        AssertionError: a key the run has no span of.
    """
    n = len(split["doc_ids"])
    keep = np.array([k in keys for k in zip(split["doc_ids"].tolist(), split["row_error_type"].tolist(), split["row_k"].tolist())], dtype=bool)
    out = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v) for k, v in split.items()}
    assert row_keys(out) == keys, f"{split.get('run_id')}: rows {sorted(keys - row_keys(out))[:3]} not in the run"
    return out


def pair_signature(split: dict, z: dict) -> list[tuple]:
    """Each pair as ``(invalid span, clean twin span, null_fill)``, a span being ``(document, row error type, k,
    field)``: identical across methods iff they are scored on the same pairs."""
    def span(i: int) -> tuple:
        return (str(split["doc_ids"][i]), str(split["row_error_type"][i]), int(split["row_k"][i]), str(split["field"][i]))
    return [(span(i), span(j), bool(nf)) for i, j, nf in zip(z["inv"], z["val"], z["null_fill"])]


def run_lm(label: str, p: dict, seed: int, spec: dict, tr: dict, te: dict, z_tr: dict, s_tr: dict, z_te: dict, s_te: dict,
           art_dir: Path) -> tuple[dict, np.ndarray]:
    """One model's probe: PCA of train, logistic probe train -> test, the test metrics, row-context diagnostic
    and controls. ``tr`` is the model's full train split, ``te`` its test split cut to the common rows; both centred.
    Returns ``(metrics (unprefixed), P(valid) of the test pairs' members)`` and writes the predictions to ``art_dir``."""
    layer, thr = spec["layer"], p["decision_threshold"]
    x_tr, x_te, y_tr, y_te = tr["X"][s_tr["idx"]], te["X"][s_te["idx"]], s_tr["y"], s_te["y"]
    types = sorted(set(z_tr["error_type"].tolist()))
    assert types == sorted(set(z_te["error_type"].tolist())), f"{label}: train and test have different error types"
    ref_rate = float(y_tr.mean())
    assert ref_rate == 0.5 == float(y_te.mean()), f"{label}: minimal-pair sets are not 50/50"

    metrics: dict[str, float] = {
        "layer": layer, "n_train_pairs": int(len(z_tr["inv"])),
        # one clean span is the twin in every (type, k) stratum, so far fewer valid spans than pairs
        "n_train_unique_valid_spans": len(set(z_tr["val"].tolist())), "n_test_unique_valid_spans": len(set(z_te["val"].tolist())),
        "n_train_docs": len(set(tr["doc_ids"].tolist())), "n_test_docs": len(set(te["doc_ids"].tolist())),
        "n_train_null_fill_pairs": int(z_tr["null_fill"].sum()),
    }

    pca = PCA(n_components=p["pca_n_components"], svd_solver=p["pca_svd_solver"], random_state=seed)
    pca.fit(x_tr)
    metrics["pca_evr_pc1"], metrics["pca_evr_pc2"] = (float(v) for v in pca.explained_variance_ratio_[:2])

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

    for col in ("error_type", "stratum"):
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

    # Field-identity baseline, from each member's own span metadata (not the pair's): on minimal pairs every
    # field is 50/50, so exactly chance (asserted).
    metrics.update(assert_structure_auroc_is_chance("field", y_tr, tr["field"][s_tr["idx"]], y_te, te["field"][s_te["idx"]], p, seed))

    if p["run_controls"]:
        auc_shuf, auc_shuf_std = shuffled_label_auroc(x_tr, y_tr, x_te, y_te, p, seed)
        metrics["control_shuffled_auroc"], metrics["control_shuffled_auroc_std"] = auc_shuf, auc_shuf_std
        assert abs(auc_shuf - 0.5) <= p["shuffled_label_auroc_tolerance"], (
            f"{label}: mean shuffled-label AUROC {auc_shuf:.4f} (std {auc_shuf_std:.4f}, n={p['n_shuffles']}) is not ~0.5 -- something leaks"
        )
        assert np.array_equal(probs, fit_logreg(x_tr, y_tr, p, seed).predict_proba(x_te)[:, 1]), (
            f"{label}: same-seed refit gave different probabilities"
        )
        metrics["control_seed_determinism"] = 1.0

    np.savez_compressed(
        art_dir / f"{label}_test_predictions.npz",
        prob_valid=probs, label_valid=y_te, span=s_te["idx"], pair_doc=s_te["doc"], doc_ids=te["doc_ids"][s_te["idx"]],
        field=s_te["field"], error_type=s_te["error_type"], k=s_te["k"], null_fill=s_te["null_fill"],
    )
    return metrics, probs


def run_judge(label: str, spec: dict, p: dict, seed: int, mod, judge: dict, rows_te: list[dict]) -> tuple[dict, dict]:
    """One direct judge (Clef / Jev) on the common pairs: the metric code of ``analysis/clef.py``, with Jev's
    resolution metrics added. ``judge`` is its run cut to the common rows. Returns ``(metrics, eval set)``."""
    ac = getattr(mod, "ac", mod)   # jev.py reuses clef.py's metric code
    ev = ac.entity_eval_set(judge, rows_te, p["twin_seed"])
    m = ac.analyse_run(label, judge, ev, spec, p, None, seed)
    if spec["kind"] == "jev":
        m.update(mod.resolution_metrics(judge))
    return m, ev


def run(cfg: dict, runs_root: Path) -> dict:
    """Every model's probe and every direct judge on the same test pairs. Probes train on ``params.train_rows``'s
    runs only; everything is evaluated on ``params.test_rows``, restricted to the rows every method scored
    (a judge may have excluded over-length documents); each method's dropped rows are reported."""
    p, seed = cfg["params"], cfg["seed"]
    out = runs_root / cfg["id"]
    fig_dir = FIGURES_DIR / cfg["id"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    assert p["dataset"] == "main", f"entity detection is evaluated on main only, got {p['dataset']!r}"
    labels = [s["label"] for s in (*p["models"], *p["judges"])]
    assert len(set(labels)) == len(labels) and labels, f"duplicate or no method labels {labels}"
    train_path, test_path = REPO_ROOT / p["train_rows"], REPO_ROOT / p["test_rows"]
    assert train_path != test_path, "train and test row files are the same file"

    def check_rows_file(run_id: str, expected: Path) -> list[dict]:
        assert run_rows_path(run_id, runs_root, REPO_ROOT, "main") == expected, f"{run_id}: judged another row file than {expected}"
        return load_run_rows(run_id, runs_root, REPO_ROOT, "main")

    rows_te = None
    lms: dict[str, dict] = {}
    for spec in p["models"]:
        layer = spec["layer"]
        assert isinstance(layer, int) and layer > 0, "layer must be a positive int (layer 0 is the constant null control)"
        tr = load_split(spec["train_run_id"], runs_root, layer, p["max_docs"], "entity_detection", ("value",))
        te = load_split(spec["test_run_id"], runs_root, layer, p["max_docs"], "entity_detection", ("value",))
        assert not (set(tr["doc_ids"].tolist()) & set(te["doc_ids"].tolist())), f"{spec['label']}: train/test documents overlap"
        # The only preprocessing: per-document mean subtraction (asserted ~0 per document), on the model's whole run.
        center_per_document(tr["X"], tr["doc_ids"])
        center_per_document(te["X"], te["doc_ids"])
        rows_tr = check_rows_file(spec["train_run_id"], train_path)
        rows_te = check_rows_file(spec["test_run_id"], test_path)
        lms[spec["label"]] = {"tr": tr, "te": te, "rows_tr": rows_tr}
    judges: dict[str, dict] = {}
    for spec in p["judges"]:
        assert spec["kind"] in JUDGE_KINDS, spec
        mod = load_sibling(spec["kind"])
        judge = mod.load_run(spec["run_id"], runs_root, "entity")
        rows_te = check_rows_file(spec["run_id"], test_path)
        judges[spec["label"]] = {"mod": mod, "run": judge}

    # The common set: rows every method scored. Pairs are then built per method from identical rows with the
    # same twin seed, and must come out identical (asserted).
    scored = {**{k: v["te"] for k, v in lms.items()}, **{k: v["run"] for k, v in judges.items()}}
    common = set.intersection(*(row_keys(s) for s in scored.values()))
    assert common, "no test row was scored by every method"
    metrics: dict[str, float] = {"n_common_test_rows": len(common)}
    for label, s in scored.items():
        metrics[f"{label}_n_test_rows"], metrics[f"{label}_n_dropped_rows"] = len(row_keys(s)), len(row_keys(s)) - len(common)
    signature = None
    for label, s in scored.items():
        cut = restrict_to_rows(s, common)
        z = main_span_pairs(cut, rows_te, p["twin_seed"])
        sig = pair_signature(cut, z)
        if signature is None:
            signature = sig
            metrics["n_test_pairs"], metrics["n_test_null_fill_pairs"] = len(sig), int(z["null_fill"].sum())
        assert sig == signature, f"{label}: scored on other pairs than {next(iter(scored))}"
        if label in lms:
            lms[label]["te"] = cut
        else:
            judges[label]["run"] = cut

    art_dir = out / "artifacts"
    y_ref = None
    probs_by_method: dict[str, np.ndarray] = {}
    ref_set = None   # the first method's (docs, error types): every method must be scored on the same pairs, in the same order
    for spec in p["models"]:
        label, m = spec["label"], lms[spec["label"]]
        z_tr, s_tr = pair_set(m["tr"], m["rows_tr"], p["twin_seed"], "main")
        z_te, s_te = pair_set(m["te"], rows_te, p["twin_seed"], "main")
        assert y_ref is None or np.array_equal(y_ref, s_te["y"]), f"{label}: labels differ from the other methods'"
        y_ref = s_te["y"]
        m_metrics, lm_probs = run_lm(label, p, seed, spec, m["tr"], m["te"], z_tr, s_tr, z_te, s_te, art_dir)
        probs_by_method[label] = lm_probs.astype(np.float64)
        for k_, v in m_metrics.items():
            metrics[f"{label}_{k_}"] = v
        ref_set = ref_set or (s_te["doc"], s_te["error_type"])
        assert np.array_equal(ref_set[0], s_te["doc"]) and np.array_equal(ref_set[1], s_te["error_type"]), f"{label}: pairs differ from the other methods'"
    for spec in p["judges"]:
        label, j = spec["label"], judges[spec["label"]]
        m, ev = run_judge(label, spec, p, seed, j["mod"], j["run"], rows_te)
        assert y_ref is None or np.array_equal(y_ref, ev["y"]), f"{label}: labels differ from the other methods'"
        y_ref = ev["y"]
        probs_by_method[label] = ev["probs"].astype(np.float64)
        ref_set = ref_set or (ev["doc"], ev["error_type"])
        assert np.array_equal(ref_set[0], ev["doc"]) and np.array_equal(ref_set[1], ev["error_type"]), f"{label}: pairs differ from the other methods'"
        for k_, v in m.items():
            metrics[f"{label}_{k_}"] = v
    evset = {"y": y_ref, "docs": ref_set[0], "error_type": ref_set[1], "probs": probs_by_method}
    cr.save_evset(art_dir / "calibration_predictions.npz", evset)
    cr.assert_matches_existing(evset, metrics)
    metrics.update(cr.write_report(evset, p["report"], list(p["report"]["error_type_names"]), p["ci_level"], seed, fig_dir, out,
                                   "calibration", "Evaluation: minimal pairs of the main test split."))
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
