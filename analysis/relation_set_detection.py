"""Sample-level relation detection on the line dataset: a validity probe on the combined activations of
all shown fields of one sample, scored on minimal pairs of samples.

Usage: ``uv run --python 3.12 --extra analysis python analysis/relation_set_detection.py analysis/configs/<type>/<id>.yaml``

Reads the ``relation_detection`` train and test runs named in the config, then:

1. Load one layer and subtract each document's mean activation.
2. Join every judged sample to its row in the line file (``relation_detection.match_samples``); the
   label is the sample's, checked against the label rule on the whole shown subset. Train and test
   sets are the minimal pairs of samples (invalid sample + clean twin, same fields in the same shown
   order), so each (cell, shown fields) stratum is exactly 50/50.
3. Feature per sample with shown spans ``f_1..f_k`` (``params.feature``, no default):
   ``sum`` = ``sum_i X[f_i]`` (joint); ``cond`` = ``[X[f_k], sum_{i<k} X[f_i]]`` (conditional; ``f_k`` is
   the last-shown field). ``last`` = ``X[f_k]`` and ``context`` = ``sum_{i<k} X[f_i]`` are the halves of
   ``cond``, for ablation (``params.ablations``). The span activation is the value's last token in a
   causal model, so ``X[f_k]`` has already attended to ``f_1..f_{k-1}``: compare ``cond`` with ``last``.
4. Logistic regression, train -> test, with the metrics of ``analysis/entity_detection.py``: overall
   (with a document-bootstrap interval), per error type, per cell and by whether the last-shown value is
   the (or a) corrupted one (separates entity-level signal in ``X[f_k]`` from relation); detection on invalid samples with
   no invalid field pair (invisible to the pair probe of ``relation_detection.py``).
5. Ablations: the same probe on each feature in ``params.ablations``.
6. Controls: structure baseline (cell x shown fields one-hot; exactly 0.5 AUROC, asserted),
   shuffled-label AUROC ~ 0.5 (asserted), seed determinism (asserted).

Outputs go to ``$RUNS_ROOT/<id>/`` and ``analysis/figures/<id>/``.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

_here = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("analysis_relation_detection", _here / "relation_detection.py")
rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rr)  # sample matching and label rule
er = rr.er  # shared loader, metrics and plots

FIGURES_DIR = _here / "figures"
FEATURES = ("sum", "cond", "last", "context")


# ------------------------------------------------------------------------ samples
def build_samples(split: dict, groups: list[dict], data: dict) -> dict:
    """Both members of every judged minimal pair of samples, invalid members first then their twins.

    Returns ``rows`` (list of span-index arrays, shown order), ``y`` (valid), ``fields`` (shown fields
    joined by ``,``), ``cell``, ``error_type``, ``doc``, ``pair_id`` (arrays), ``no_invalid_pair`` (bool:
    an invalid member all of whose field pairs match some item; False for twins), ``last_corrupted`` (bool:
    the invalid member's last-shown value is corrupted; repeated for its twin, so each stratum is 50/50)
    and ``n_samples``.

    Raises:
        AssertionError: as ``relation_detection.match_samples``, a twin with other fields or order, a
            sample of fewer than two fields, or a sample label the label rule disagrees with.
    """
    by_pair = rr.match_samples(split, groups, data)
    members: dict[bool, list[dict]] = {False: [], True: []}
    no_invalid_pair = []
    for pid in sorted(by_pair):
        bad, good = by_pair[pid][False], by_pair[pid][True]
        assert bad["fields"] == good["fields"] and bad["doc"] == good["doc"] and bad["cell"] == good["cell"], pid
        assert len(bad["fields"]) >= 2, f"{pid}: a sample of {len(bad['fields'])} field(s)"
        keys = data["item_keys"][bad["doc"]]
        values = {v: {f: str(split["value"][i]) for f, i in zip(g["fields"], g["rows"])} for v, g in ((False, bad), (True, good))}
        assert rr.is_item_pair(values[True], keys, data["eq"]), (pid, "a twin that is no item's")
        assert not rr.is_item_pair(values[False], keys, data["eq"]), (pid, "an invalid sample that is an item's")
        no_invalid_pair.append(all(rr.is_item_pair({f: values[False][f] for f in ab}, keys, data["eq"])
                                   for ab in itertools.combinations(bad["fields"], 2)))
        members[False].append(bad)
        members[True].append(good)
    gs = members[False] + members[True]
    n = len(members[False])
    out = {"rows": [np.array(g["rows"]) for g in gs], "y": np.repeat([False, True], n),
           "fields": np.array([",".join(g["fields"]) for g in gs]), "cell": np.array([g["cell"] for g in gs]),
           "error_type": np.array([g["cell"].split(":")[0] for g in gs]), "doc": np.array([g["doc"] for g in gs]),
           "pair_id": np.array([g["pair_id"] for g in gs]),
           "no_invalid_pair": np.concatenate([np.array(no_invalid_pair, bool), np.zeros(n, bool)]), "n_samples": len(groups)}
    last_bad = ~split["label_valid"][[g["rows"][-1] for g in members[False]]].astype(bool)
    out["last_corrupted"] = np.concatenate([last_bad, last_bad])
    assert n > 0 and 2 * n == len(groups), (n, len(groups))
    assert (out["pair_id"][:n] == out["pair_id"][n:]).all() and (out["fields"][:n] == out["fields"][n:]).all()
    return out


def sample_features(x: np.ndarray, rows: list[np.ndarray], feature: str) -> np.ndarray:
    """One row per sample (spans ``rows[s]`` in shown order, ``f_k`` the last): see ``FEATURES``."""
    assert feature in FEATURES, f"unknown feature {feature!r}; one of {FEATURES}"
    combine = {"sum": lambda r: x[r].sum(axis=0),
               "cond": lambda r: np.concatenate([x[r[-1]], x[r[:-1]].sum(axis=0)]),
               "last": lambda r: x[r[-1]],
               "context": lambda r: x[r[:-1]].sum(axis=0)}[feature]
    f = np.stack([combine(r) for r in rows])
    width = 2 * x.shape[1] if feature == "cond" else x.shape[1]
    assert f.shape == (len(rows), width) and np.isfinite(f).all(), (feature, f.shape)
    return f


# ----------------------------------------------------------------------------- main
def run(cfg: dict, runs_root: Path) -> dict:
    p, seed = cfg["params"], cfg["seed"]
    out = runs_root / cfg["id"]
    fig_dir = FIGURES_DIR / cfg["id"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    layer, thr, feature, ablations = p["layer"], p["decision_threshold"], p["feature"], p["ablations"]
    assert isinstance(layer, int) and layer > 0, "layer must be a positive int (layer 0 is the constant null control)"
    assert feature in FEATURES and isinstance(ablations, list) and set(ablations) <= set(FEATURES) - {feature}, (feature, ablations)

    extra = ("line_index", "value", "pair_id", "cell")
    tr = er.load_split(p["train_run_id"], runs_root, layer, p["max_docs"], "relation_detection", extra)
    te = er.load_split(p["test_run_id"], runs_root, layer, p["max_docs"], "relation_detection", extra)
    assert not (set(tr["doc_ids"].tolist()) & set(te["doc_ids"].tolist())), "train/test documents overlap"
    for s in (tr, te):
        assert set(np.unique(s["field"]).tolist()) <= set(rr.LINE_FIELDS), "non-line field in the relation run"
    er.center_per_document(tr["X"], tr["doc_ids"])
    er.center_per_document(te["X"], te["doc_ids"])

    s_tr = build_samples(tr, rr.group_rows(tr), rr.load_line_data(p["train_run_id"], runs_root, rr.REPO_ROOT))
    s_te = build_samples(te, rr.group_rows(te), rr.load_line_data(p["test_run_id"], runs_root, rr.REPO_ROOT))
    f_tr, f_te = sample_features(tr["X"], s_tr["rows"], feature), sample_features(te["X"], s_te["rows"], feature)
    y_tr, y_te = s_tr["y"], s_te["y"]
    ref_rate = float(y_tr.mean())
    assert ref_rate == 0.5 == float(y_te.mean()), "matched sample sets are not 50/50"
    types = sorted(set(s_tr["error_type"].tolist()))
    assert types == sorted(set(s_te["error_type"].tolist())), "train and test have different error types"
    assert set(types) <= set(er.ALL_ERROR_TYPES), types

    metrics: dict[str, float] = {
        "layer": layer, "feature": feature,
        "n_train_samples": s_tr["n_samples"], "n_test_samples": s_te["n_samples"],
        "n_train_invalid_samples_without_invalid_pair": int(s_tr["no_invalid_pair"].sum()),
        "n_test_invalid_samples_without_invalid_pair": int(s_te["no_invalid_pair"].sum()),
        "n_train_docs": len(set(tr["doc_ids"].tolist())), "n_test_docs": len(set(te["doc_ids"].tolist())),
    }

    clf = er.fit_logreg(f_tr, y_tr, p, seed)
    probs = clf.predict_proba(f_te)[:, 1]
    assert probs.shape == y_te.shape
    metrics.update(er.probe_metrics("test", y_te, probs, ref_rate, thr))
    metrics["train_auroc"] = float(roc_auc_score(y_tr, clf.predict_proba(f_tr)[:, 1]))
    metrics["test_auroc_ci_lo"], metrics["test_auroc_ci_hi"] = er.doc_bootstrap_auroc(
        y_te, probs, s_te["doc"], p["auroc_n_boot"], p["ci_level"], seed)

    d = er.smooth_ece(probs, y_te, p["ece_n_boot"], seed)
    metrics["test_smece"], metrics["test_smece_ci_halfwidth"] = float(d["ce"]), float(d["ce_ci_width"])
    er.plot_calibration(d, fig_dir)
    er.plot_prob_histogram(probs, p["hist_n_bins"], fig_dir)

    last_stratum = np.where(s_te["last_corrupted"], "last_corrupted", "last_clean")
    for strata in (s_te["error_type"], s_te["cell"], last_stratum):
        for k_, v in er.stratum_report(y_te, probs, thr, strata).items():
            metrics[f"test_{k_}"] = v
    if s_te["no_invalid_pair"].any():
        metrics["test_detect_no_invalid_pair"] = float((probs[s_te["no_invalid_pair"]] <= thr).mean())

    for a in ablations:
        a_tr, a_te = sample_features(tr["X"], s_tr["rows"], a), sample_features(te["X"], s_te["rows"], a)
        a_probs = er.fit_logreg(a_tr, y_tr, p, seed).predict_proba(a_te)[:, 1]
        metrics.update(er.probe_metrics(f"test_ablation_{a}", y_te, a_probs, ref_rate, thr))

    # Structure baseline: cell x shown fields (in order); each is exactly 50/50 on matched samples -> AUROC 0.5.
    struct_tr, struct_te = (np.char.add(np.char.add(s["cell"], "|"), s["fields"]) for s in (s_tr, s_te))
    metrics.update(er.assert_structure_auroc_is_chance("cell_fields", y_tr, struct_tr, y_te, struct_te, p, seed))

    if p["run_controls"]:
        auc_shuf, auc_shuf_std = er.shuffled_label_auroc(f_tr, y_tr, f_te, y_te, p, seed)
        metrics["control_shuffled_auroc"], metrics["control_shuffled_auroc_std"] = auc_shuf, auc_shuf_std
        assert abs(auc_shuf - 0.5) <= p["shuffled_label_auroc_tolerance"], (
            f"mean shuffled-label AUROC {auc_shuf:.4f} (std {auc_shuf_std:.4f}, n={p['n_shuffles']}) is not ~0.5 -- something leaks"
        )
        assert np.array_equal(probs, er.fit_logreg(f_tr, y_tr, p, seed).predict_proba(f_te)[:, 1]), (
            "same-seed refit gave different probabilities"
        )
        metrics["control_seed_determinism"] = 1.0

    np.savez_compressed(
        out / "artifacts" / "test_predictions.npz",
        prob_valid=probs, label_valid=y_te, doc=s_te["doc"], pair_id=s_te["pair_id"], cell=s_te["cell"],
        error_type=s_te["error_type"], fields=s_te["fields"], no_invalid_pair=s_te["no_invalid_pair"],
        last_corrupted=s_te["last_corrupted"],
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
