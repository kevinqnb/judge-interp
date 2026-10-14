"""Row-based relation detection: a validity probe on pairwise distances between the
activations of a random subset of one line-item row's fields.

Usage: ``uv run --python 3.12 --extra analysis python analysis/relation_detection_row.py configs/<id>.yaml``
(every value lives in the config's ``params``; a missing key is a KeyError).

Reads the ``relation_detection_row`` train and test runs named in the config
(``src/judge_interp/relation_detection_row.py``: one prompt per ``(document, line row)``, one
activation per non-null field, read at the field's closing quote) and does, in order:

1. **Load** one layer (``layer``) as ``X`` and subtract each document's mean row -- nothing
   else (no field-type normalisation). Per-document centring is a common shift for every span
   of a document, so it cannot change a within-sample Euclidean distance (unit-tested); it is
   applied as specified.
2. **Samples.** A *source row* is a ``(doc_id, row_ordinal)`` prompt. For each row with at least
   ``min_subset_size`` non-null fields, ``samples_per_row`` distinct subsets (of size >=
   ``min_subset_size``) are drawn uniformly without replacement from all such subsets of its
   fields. A sample is **invalid iff any field in the subset is an invalid entry** (labels are per
   entity), valid otherwise. A sample inherits the source row's error type.
3. **Features.** One slot per unordered pair of the 5 line fields (``C(5,2) = 10``,
   ``PAIRS``): the Euclidean distance between the two fields' activations; pairs not both in the
   subset hold ``absent_fill`` (a config key). If ``standardize_pair_distances``, each pair's
   distances are then standardized with that pair's train mean and std over the samples where it
   is *present* (``fit_pair_stats``; applied to both splits), and absent slots are set to
   ``absent_fill`` *after* standardizing (0.0 = "an average distance"), so an absent pair no longer
   sits on a different scale from a present one. This is a departure from leaving absent entries as
   a raw 0; run it on and off to compare. The raw features stay in ``f_raw``.
4. **Label rates.** Each split is cut to exactly ``n_per_class_<split>`` valid and as many invalid
   samples (random, seeded), so train and test have the same valid rate (asserted); a pool too
   small for the requested count is a hard error.
5. **Logistic regression** (train -> test), positive class = valid: smECE (+ bootstrap CI),
   Brier score, Brier skill score (reference = constant forecast at the train valid rate),
   AUROC, accuracy, smoothed reliability diagram, and the per-error-type report of
   ``analysis/entity_detection_row.py`` (error type = the source row's).
6. **Controls.** Mean test AUROC over ``n_shuffles`` shuffled-train-label fits ~ 0.5 (asserted;
   one permutation is too noisy, see ``entity_detection_row.shuffled_label_auroc``); same-seed
   resample + refit gives identical output (asserted). Reported, NOT asserted, because this design lets labels leak
   through structure: a presence-only baseline (the 10 slots as 0/1), a noise-``X`` probe (same
   subsets, Gaussian activations), and AUROC per subset size. A larger subset is more likely to
   contain an invalid entry, and corruption is field-specific, so the set of filled slots carries
   label information with no representation in it; the probe is only informative to the extent
   it beats these.

Figures go to ``analysis/figures/<id>/``; ``metrics.json``, ``config.snapshot.yaml`` and
``artifacts/{train,test}_samples.npz`` go to ``$RUNS_ROOT/<id>/``.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from judge_interp.prompts import LINE_FIELDS

_spec = importlib.util.spec_from_file_location("analysis_entity_detection_row", Path(__file__).parent / "entity_detection_row.py")
er = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(er)  # reused unchanged: loader, metrics, error-type report, plots (and, through it, analysis/entity_detection.py)
ed = er.ed

FIGURES_DIR = Path(__file__).parent / "figures"
MAX_FIELDS = len(LINE_FIELDS)
# Field pairs, unordered, in LINE_FIELDS order: one feature slot each.
PAIRS = list(itertools.combinations(LINE_FIELDS, 2))
PAIR_INDEX = {pr: i for i, pr in enumerate(PAIRS)}


# ------------------------------------------------------------------------ source rows
def group_rows(split: dict) -> list[dict]:
    """One entry per ``(doc_id, row_ordinal)`` prompt: ``{doc, ordinal, rows, fields, error_type}``.

    ``rows`` are indices into the split's arrays (``field_position`` order). Asserted: fields
    distinct and line fields, the row error type constant within a row. (Labels are per entity, so
    only the error type is row-level.)
    """
    by_key: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, (d, o) in enumerate(zip(split["doc_ids"].tolist(), split["row_ordinal"].tolist())):
        by_key[(d, o)].append(i)
    groups = []
    for (d, o), rows in by_key.items():
        rows = sorted(rows, key=lambda i: int(split["field_position"][i]))
        fields = [str(split["field"][i]) for i in rows]
        assert len(set(fields)) == len(fields) and set(fields) <= set(LINE_FIELDS), (d, o, fields)
        ets = {str(split["row_error_type"][i]) for i in rows}
        assert len(ets) == 1, f"{d}/{o}: row_error_type varies within a row: {ets}"
        groups.append({"doc": d, "ordinal": o, "rows": rows, "fields": fields, "error_type": ets.pop()})
    return groups


def draw_samples(groups: list[dict], label_valid: np.ndarray, samples_per_row: int, min_subset_size: int,
                 rng: np.random.Generator) -> dict:
    """Random subsets of each row's spans, one sample each (module docstring, step 2).

    Returns arrays, one entry per sample: ``rows`` (``[n, MAX_FIELDS]`` span indices, ``-1`` padded,
    in ``field_position`` order), ``label`` (valid iff every span in the subset is valid),
    ``size``, ``doc``, ``ordinal``, ``error_type`` (the source row's), plus ``stats``.
    """
    assert min_subset_size >= 2, "a pair feature needs at least two fields"
    out: dict[str, list] = defaultdict(list)
    stats = {"n_rows": len(groups), "n_rows_too_small": 0, "n_rows_fewer_subsets_than_requested": 0}
    for g in groups:
        n = len(g["rows"])
        if n < min_subset_size:
            stats["n_rows_too_small"] += 1
            continue
        subsets = [c for r in range(min_subset_size, n + 1) for c in itertools.combinations(range(n), r)]
        take = min(samples_per_row, len(subsets))
        if take < samples_per_row:
            stats["n_rows_fewer_subsets_than_requested"] += 1
        for j in rng.choice(len(subsets), size=take, replace=False):
            rows = [g["rows"][k] for k in subsets[int(j)]]
            out["rows"].append(rows + [-1] * (MAX_FIELDS - len(rows)))
            out["label"].append(bool(label_valid[rows].all()))
            out["size"].append(len(rows))
            out["doc"].append(g["doc"])
            out["ordinal"].append(g["ordinal"])
            out["error_type"].append(g["error_type"])
    assert out["rows"], "no samples drawn"
    res = {k: np.array(v) for k, v in out.items()}
    res["stats"] = stats
    assert res["rows"].shape[1] == MAX_FIELDS and (res["rows"] >= 0).sum(axis=1).tolist() == res["size"].tolist()
    # An invalid sample must come from a corrupted row; a clean row only yields valid samples.
    assert not ((res["error_type"] == "") & ~res["label"]).any(), "invalid sample from a clean row"
    return res


def balance(samples: dict, n_per_class: int, rng: np.random.Generator) -> dict:
    """Exactly ``n_per_class`` valid and ``n_per_class`` invalid samples (random); order kept."""
    assert isinstance(n_per_class, int) and n_per_class >= 1
    label = samples["label"]
    keep = []
    for lab, name in ((True, "valid"), (False, "invalid")):
        pool = np.flatnonzero(label == lab)
        if len(pool) < n_per_class:
            raise ValueError(f"only {len(pool)} {name} samples available, {n_per_class} requested")
        keep.append(rng.choice(pool, size=n_per_class, replace=False))
    keep = np.sort(np.concatenate(keep))
    res = {k: v[keep] for k, v in samples.items() if k != "stats"}
    res["stats"] = samples["stats"]
    assert int(res["label"].sum()) == n_per_class and int((~res["label"]).sum()) == n_per_class
    return res


def pair_features(x: np.ndarray, fields: np.ndarray, sample_rows: np.ndarray, absent_fill: float) -> tuple[np.ndarray, np.ndarray]:
    """``(f, present)``, both ``[n, len(PAIRS)]`` (float64, bool).

    ``f[i, PAIR_INDEX[(a, b)]]`` = Euclidean distance between the activations of fields ``a`` and
    ``b`` in sample ``i`` if both are in its subset, else ``absent_fill``.
    """
    f = np.full((len(sample_rows), len(PAIRS)), absent_fill, dtype=np.float64)
    present = np.zeros(f.shape, dtype=bool)
    for i, rows in enumerate(sample_rows):
        rows = [int(r) for r in rows if r >= 0]
        assert len(rows) >= 2
        for a, b in itertools.combinations(rows, 2):
            pr = tuple(sorted((str(fields[a]), str(fields[b])), key=LINE_FIELDS.index))
            assert pr[0] != pr[1], "two spans of one field in a sample"
            j = PAIR_INDEX[pr]
            assert not present[i, j], "a field pair filled twice"
            present[i, j] = True
            f[i, j] = np.linalg.norm(x[a].astype(np.float64) - x[b].astype(np.float64))
        assert present[i].sum() == len(rows) * (len(rows) - 1) // 2
    assert np.isfinite(f).all()
    return f, present


def fit_pair_stats(f: np.ndarray, present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair ``(mean, std)`` over the entries where the pair is present (float64). Every pair
    needs >= 2 present entries and a non-zero std."""
    assert f.shape == present.shape and f.shape[1] == len(PAIRS)
    n = present.sum(axis=0)
    assert (n >= 2).all(), f"pairs with < 2 present train entries: {[PAIRS[j] for j in np.flatnonzero(n < 2)]}"
    mean = np.array([f[present[:, j], j].mean() for j in range(f.shape[1])])
    std = np.array([f[present[:, j], j].std() for j in range(f.shape[1])])
    assert (std > 0).all(), "a pair has constant distances: cannot standardize"
    return mean, std


def apply_pair_stats(f: np.ndarray, present: np.ndarray, mean: np.ndarray, std: np.ndarray, fill: float) -> np.ndarray:
    """Standardize present entries; absent entries become ``fill``."""
    return np.where(present, (f - mean) / std, fill)


def make_features(f_tr, present_tr, f_te, present_te, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Probe features: raw distances, or per-pair standardized (train stats) per the config flag."""
    if not p["standardize_pair_distances"]:
        return f_tr, f_te
    mean, std = fit_pair_stats(f_tr, present_tr)
    g_tr = apply_pair_stats(f_tr, present_tr, mean, std, p["absent_fill"])
    g_te = apply_pair_stats(f_te, present_te, mean, std, p["absent_fill"])
    for j in range(len(PAIRS)):
        assert abs(g_tr[present_tr[:, j], j].mean()) < 1e-9 and abs(g_tr[present_tr[:, j], j].std() - 1) < 1e-9
    return g_tr, g_te


# ----------------------------------------------------------------------------- main
def build_dataset(split: dict, groups: list[dict], n_per_class: int, p: dict, stream: int, seed: int,
                  x: np.ndarray | None = None) -> dict:
    """Samples + balancing + features for one split. ``x`` overrides the activations (noise control)."""
    rng = np.random.default_rng([seed, stream])
    s = draw_samples(groups, split["label_valid"], p["samples_per_row"], p["min_subset_size"], rng)
    s = balance(s, n_per_class, rng)
    s["f"], s["present"] = pair_features(split["X"] if x is None else x, split["field"], s["rows"], p["absent_fill"])
    return s


def run(cfg: dict, runs_root: Path) -> dict:
    p, seed = cfg["params"], cfg["seed"]
    out = runs_root / cfg["id"]
    fig_dir = FIGURES_DIR / cfg["id"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    layer, thr = p["layer"], p["decision_threshold"]
    assert isinstance(layer, int) and layer > 0, "layer must be a positive int (layer 0 is the constant null control)"

    tr = er.load_split(p["train_run_id"], runs_root, layer, p["max_docs"], "relation_detection_row")
    te = er.load_split(p["test_run_id"], runs_root, layer, p["max_docs"], "relation_detection_row")
    assert not (set(tr["doc_ids"].tolist()) & set(te["doc_ids"].tolist())), "train/test documents overlap"
    assert set(np.unique(tr["field"]).tolist()) <= set(LINE_FIELDS), "non-line field in the relation run"
    ed.center_per_document(tr["X"], tr["doc_ids"])
    ed.center_per_document(te["X"], te["doc_ids"])
    g_tr, g_te = group_rows(tr), group_rows(te)

    z_tr = build_dataset(tr, g_tr, p["n_per_class_train"], p, 0, seed)
    z_te = build_dataset(te, g_te, p["n_per_class_test"], p, 1, seed)
    for z in (z_tr, z_te):
        z["f_raw"] = z["f"]
    z_tr["f"], z_te["f"] = make_features(z_tr["f_raw"], z_tr["present"], z_te["f_raw"], z_te["present"], p)
    y_tr, y_te = z_tr["label"], z_te["label"]
    assert float(y_tr.mean()) == float(y_te.mean()), "train and test valid rates differ"
    ref_rate = float(y_tr.mean())
    types = sorted(set(z_tr["error_type"][~y_tr].tolist()))
    assert types == sorted(set(z_te["error_type"][~y_te].tolist())), "train and test have different error types"
    assert set(types) <= set(er.ALL_ERROR_TYPES) and "" not in types

    metrics: dict[str, float] = {
        "layer": layer, "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "train_valid_rate": ref_rate, "test_valid_rate": float(y_te.mean()),
        "n_train_docs": len(set(z_tr["doc"].tolist())), "n_test_docs": len(set(z_te["doc"].tolist())),
    }
    for name, z in (("train", z_tr), ("test", z_te)):
        for k_, v in z["stats"].items():
            metrics[f"{name}_{k_}"] = int(v)
        for sz in range(p["min_subset_size"], MAX_FIELDS + 1):
            metrics[f"{name}_n_size{sz}"] = int((z["size"] == sz).sum())
            if (z["size"] == sz).any():
                metrics[f"{name}_valid_rate_size{sz}"] = float(z["label"][z["size"] == sz].mean())

    clf = ed.fit_logreg(z_tr["f"], y_tr, p, seed)
    probs = clf.predict_proba(z_te["f"])[:, 1]
    metrics.update(er.probe_metrics("test", y_te, probs, ref_rate, thr))
    metrics["train_auroc"] = float(roc_auc_score(y_tr, clf.predict_proba(z_tr["f"])[:, 1]))
    metrics["lr_intercept"] = float(clf.intercept_[0])
    for j, (a, b) in enumerate(PAIRS):
        metrics[f"lr_coef_{a}__{b}"] = float(clf.coef_[0, j])

    d = ed.smooth_ece(probs, y_te, p["ece_n_boot"], seed)
    metrics["test_smece"], metrics["test_smece_ci_halfwidth"] = float(d["ce"]), float(d["ce_ci_width"])
    er.plot_calibration(d, fig_dir)

    for k_, v in er.error_type_report(y_te, probs, thr, z_te["error_type"], z_te["error_type"], types).items():
        metrics[f"test_{k_}"] = v

    # Structure-only baselines and per-size AUROC (reported, not asserted -- see module docstring).
    pres = ed.fit_logreg(z_tr["present"].astype(float), y_tr, p, seed)
    pres_probs = pres.predict_proba(z_te["present"].astype(float))[:, 1]
    metrics.update(er.probe_metrics("test_baseline_presence", y_te, pres_probs, ref_rate, thr))
    for sz in range(p["min_subset_size"], MAX_FIELDS + 1):
        sel = z_te["size"] == sz
        if sel.any() and y_te[sel].any() and (~y_te[sel]).any():
            metrics[f"test_auroc_size{sz}"] = float(roc_auc_score(y_te[sel], probs[sel]))

    if p["run_controls"]:
        rng_n = np.random.default_rng([seed, 2])
        xn_tr = rng_n.standard_normal(tr["X"].shape, dtype=np.float32)
        xn_te = rng_n.standard_normal(te["X"].shape, dtype=np.float32)
        fn_tr, _ = pair_features(xn_tr, tr["field"], z_tr["rows"], p["absent_fill"])
        fn_te, _ = pair_features(xn_te, te["field"], z_te["rows"], p["absent_fill"])
        fn_tr, fn_te = make_features(fn_tr, z_tr["present"], fn_te, z_te["present"], p)
        noise = ed.fit_logreg(fn_tr, y_tr, p, seed)
        metrics["control_noise_x_auroc"] = float(roc_auc_score(y_te, noise.predict_proba(fn_te)[:, 1]))

        auc_shuf, auc_shuf_std = er.shuffled_label_auroc(z_tr["f"], y_tr, z_te["f"], y_te, p, seed)
        metrics["control_shuffled_auroc"], metrics["control_shuffled_auroc_std"] = auc_shuf, auc_shuf_std
        assert abs(auc_shuf - 0.5) <= p["shuffled_label_auroc_tolerance"], (
            f"mean shuffled-label AUROC {auc_shuf:.4f} (std {auc_shuf_std:.4f}, n={p['n_shuffles']}) is not ~0.5 -- something leaks"
        )
        z_te2 = build_dataset(te, g_te, p["n_per_class_test"], p, 1, seed)
        assert np.array_equal(z_te2["rows"], z_te["rows"]) and np.array_equal(z_te2["f"], z_te["f_raw"]), (
            "same-seed resample differs"
        )
        assert np.array_equal(probs, ed.fit_logreg(z_tr["f"], y_tr, p, seed).predict_proba(z_te["f"])[:, 1]), (
            "same-seed refit gave different probabilities"
        )
        metrics["control_seed_determinism"] = 1.0

    for name, z in (("train", z_tr), ("test", z_te)):
        np.savez_compressed(out / "artifacts" / f"{name}_samples.npz", **{k_: v for k_, v in z.items() if k_ != "stats"})
    np.savez_compressed(out / "artifacts" / "test_predictions.npz", prob_valid=probs, label_valid=y_te,
                        error_type=z_te["error_type"], size=z_te["size"], f=z_te["f"])
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
