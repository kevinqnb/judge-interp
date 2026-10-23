"""Row-based relation detection: a validity probe on the concatenated activations of two fields
of one line-item row.

Usage: ``uv run --python 3.12 --extra analysis python analysis/relation_detection.py analysis/configs/<type>/<id>.yaml``

Reads the ``relation_detection`` train and test runs named in the config, then:

1. Load one layer and subtract each document's mean activation.
2. Every unordered pair of non-null fields of a row is a sample; feature ``[X[a], X[b]]``.
3. Label (valid = positive): the data's label rule (``build_vrdu_invalids.matches_item``) on the two
   shown values, checked against the row file the run judged.
4. Logistic regression, train -> test, with the metrics of ``analysis/entity_detection.py``.
5. Controls: field-pair baseline, shuffled-label AUROC ~ 0.5 (asserted), seed determinism (asserted).

Outputs go to ``$RUNS_ROOT/<id>/`` and ``analysis/figures/<id>/``.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from judge_interp.dataset_spec import load_dataset_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "data" / "vrdu"))
from build_vrdu_invalids import matches_item  # noqa: E402
from subsets import equivalence_from_invalids, items_by_document, make_sample_eq  # noqa: E402

LINE_FIELDS = list(
    load_dataset_spec(Path(__file__).resolve().parents[1] / "experiments" / "dataset-configs" / "vrdu_line.py").fields
)

_spec = importlib.util.spec_from_file_location("analysis_entity_detection", Path(__file__).parent / "entity_detection.py")
er = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(er)  # shared loader, metrics and plots

FIGURES_DIR = Path(__file__).parent / "figures"
# Unordered field pairs (the baseline's categories).
PAIRS = list(itertools.combinations(LINE_FIELDS, 2))
PAIR_INDEX = {pr: i for i, pr in enumerate(PAIRS)}


# ------------------------------------------------------------------------ source rows
def group_rows(split: dict) -> list[dict]:
    """One entry per ``(doc_id, row_ordinal)`` prompt: ``{doc, ordinal, rows, fields, error_type}``."""
    by_key: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, (d, o) in enumerate(zip(split["doc_ids"].tolist(), split["row_ordinal"].tolist())):
        by_key[(d, o)].append(i)
    groups = []
    for (d, o), rows in by_key.items():
        rows = sorted(rows, key=lambda i: LINE_FIELDS.index(str(split["field"][i])))
        fields = [str(split["field"][i]) for i in rows]
        assert len(set(fields)) == len(fields) and set(fields) <= set(LINE_FIELDS), (d, o, fields)
        ets = {str(split["row_error_type"][i]) for i in rows}
        assert len(ets) == 1, f"{d}/{o}: row_error_type varies within a row: {ets}"
        groups.append({"doc": d, "ordinal": o, "rows": rows, "fields": fields, "error_type": ets.pop()})
    return groups


def load_line_data(run_id: str, runs_root: Path, repo_root: Path) -> dict:
    """The row file a run judged (via its ``config.snapshot.yaml``): ``{items, rows, eq}``."""
    params = yaml.safe_load((runs_root / run_id / "config.snapshot.yaml").read_text())["params"]
    spec = load_dataset_spec(repo_root / params["dataset_config"])
    assert spec.name == "line" and tuple(spec.fields) == tuple(LINE_FIELDS), (run_id, spec.name)
    path = repo_root / params["data_root"] / spec.split_path.format(split=params["split"])
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_key = {(r["document_id"], r["line_index"], r["error_type"] or "", r["k"]): r for r in rows}
    assert len(by_key) == len(rows), f"{path}: row keys are not unique"
    return {"items": items_by_document(rows), "rows": by_key,
            "eq": make_sample_eq(equivalence_from_invalids(path.parent.parent))}


def build_pairs(split: dict, groups: list[dict], data: dict) -> dict:
    """Every unordered field pair of every row, labelled by ``matches_item``.

    Arrays ``i, j, label, pair, error_type, doc, ordinal``, plus ``relabelled``: counts of pairs whose
    label differs from the old per-entity rule.
    """
    for i in itertools.chain.from_iterable(g["rows"] for g in groups):   # spans must match the row file
        key = (str(split["doc_ids"][i]), int(split["line_index"][i]), str(split["row_error_type"][i]), int(split["row_k"][i]))
        assert key in data["rows"], f"span {i}: no row {key} in the row file (a run on other data?)"
        assert data["rows"][key][str(split["field"][i])] == str(split["value"][i]), (
            f"span {i}: value {split['value'][i]!r} != row file's {data['rows'][key][str(split['field'][i])]!r} for {key}")
    out: dict[str, list] = defaultdict(list)
    relabelled: Counter = Counter()
    valid, eq = split["label_valid"], data["eq"]
    for g in groups:
        items = data["items"][g["doc"]]
        for i, j in itertools.combinations(g["rows"], 2):
            fi, fj = str(split["field"][i]), str(split["field"][j])
            assert LINE_FIELDS.index(fi) < LINE_FIELDS.index(fj), (g["doc"], g["ordinal"], fi, fj)
            label = matches_item({fi: str(split["value"][i]), fj: str(split["value"][j])}, items, eq)
            if valid[i] and valid[j]:
                assert label, (g["doc"], g["ordinal"], fi, fj, "two correct entities that are no item's")
            old = "valid" if valid[i] and valid[j] else "invalid" if valid[i] or valid[j] else "dropped"
            if old != ("valid" if label else "invalid"):
                relabelled[f"{g['error_type']}_{old}_to_{'valid' if label else 'invalid'}"] += 1
            out["i"].append(i)
            out["j"].append(j)
            out["label"].append(label)
            out["pair"].append(PAIR_INDEX[(fi, fj)])
            out["error_type"].append(g["error_type"])
            out["doc"].append(g["doc"])
            out["ordinal"].append(g["ordinal"])
    assert out["i"], "no pairs"
    res = {k: np.array(v) for k, v in out.items()}
    res["relabelled"] = dict(sorted(relabelled.items()))
    assert len(res["i"]) == sum(len(g["rows"]) * (len(g["rows"]) - 1) // 2 for g in groups)
    assert len(set(zip(res["i"].tolist(), res["j"].tolist()))) == len(res["i"]), "a pair listed twice"
    assert (split["doc_ids"][res["i"]] == split["doc_ids"][res["j"]]).all()
    assert (split["row_ordinal"][res["i"]] == split["row_ordinal"][res["j"]]).all()
    # An invalid pair must come from a corrupted row; a clean row only yields valid pairs.
    assert not ((res["error_type"] == "") & ~res["label"]).any(), "invalid pair from a clean row"
    return res


def pair_features(x: np.ndarray, pairs: dict) -> np.ndarray:
    """``[n_pairs, 2 * hidden]``: ``[X[i], X[j]]`` per pair."""
    f = np.concatenate([x[pairs["i"]], x[pairs["j"]]], axis=1)
    assert f.shape == (len(pairs["i"]), 2 * x.shape[1]) and np.isfinite(f).all()
    return f


# ----------------------------------------------------------------------------- main
def run(cfg: dict, runs_root: Path) -> dict:
    p, seed = cfg["params"], cfg["seed"]
    out = runs_root / cfg["id"]
    fig_dir = FIGURES_DIR / cfg["id"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    layer, thr = p["layer"], p["decision_threshold"]
    assert isinstance(layer, int) and layer > 0, "layer must be a positive int (layer 0 is the constant null control)"

    tr = er.load_split(p["train_run_id"], runs_root, layer, p["max_docs"], "relation_detection", ("line_index", "value"))
    te = er.load_split(p["test_run_id"], runs_root, layer, p["max_docs"], "relation_detection", ("line_index", "value"))
    assert not (set(tr["doc_ids"].tolist()) & set(te["doc_ids"].tolist())), "train/test documents overlap"
    for s in (tr, te):
        assert set(np.unique(s["field"]).tolist()) <= set(LINE_FIELDS), "non-line field in the relation run"
    er.center_per_document(tr["X"], tr["doc_ids"])
    er.center_per_document(te["X"], te["doc_ids"])

    z_tr = build_pairs(tr, group_rows(tr), load_line_data(p["train_run_id"], runs_root, REPO_ROOT))
    z_te = build_pairs(te, group_rows(te), load_line_data(p["test_run_id"], runs_root, REPO_ROOT))
    f_tr, f_te = pair_features(tr["X"], z_tr), pair_features(te["X"], z_te)
    y_tr, y_te = z_tr["label"], z_te["label"]
    for y in (y_tr, y_te):
        assert y.any() and (~y).any(), "one class missing"
    ref_rate = float(y_tr.mean())
    types = sorted(set(z_tr["error_type"][~y_tr].tolist()))
    assert types == sorted(set(z_te["error_type"][~y_te].tolist())), "train and test have different error types"
    assert set(types) <= set(er.ALL_ERROR_TYPES) and "" not in types

    metrics: dict[str, float] = {
        "layer": layer, "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "train_valid_rate": ref_rate, "test_valid_rate": float(y_te.mean()),
        "n_train_docs": len(set(z_tr["doc"].tolist())), "n_test_docs": len(set(z_te["doc"].tolist())),
        **{f"n_train_pairs_relabelled_{k}": v for k, v in z_tr["relabelled"].items()},
        **{f"n_test_pairs_relabelled_{k}": v for k, v in z_te["relabelled"].items()},
    }

    clf = er.fit_logreg(f_tr, y_tr, p, seed)
    probs = clf.predict_proba(f_te)[:, 1]
    assert probs.shape == y_te.shape
    metrics.update(er.probe_metrics("test", y_te, probs, ref_rate, thr))
    metrics["train_auroc"] = float(roc_auc_score(y_tr, clf.predict_proba(f_tr)[:, 1]))

    d = er.smooth_ece(probs, y_te, p["ece_n_boot"], seed)
    metrics["test_smece"], metrics["test_smece_ci_halfwidth"] = float(d["ce"]), float(d["ce_ci_width"])
    er.plot_calibration(d, fig_dir)
    er.plot_prob_histogram(probs, p["hist_n_bins"], fig_dir)

    for k_, v in er.error_type_report(y_te, probs, thr, z_te["error_type"], z_te["error_type"], types).items():
        metrics[f"test_{k_}"] = v

    # Field-pair-identity baseline: what a probe that sees only which two fields these are can do.
    pair_ids = np.array([f"{a}__{b}" for a, b in PAIRS])
    base = er.fit_logreg(er.onehot(pair_ids[z_tr["pair"]], list(pair_ids)), y_tr, p, seed)
    base_probs = base.predict_proba(er.onehot(pair_ids[z_te["pair"]], list(pair_ids)))[:, 1]
    metrics.update(er.probe_metrics("test_baseline_field_pair", y_te, base_probs, ref_rate, thr))

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
        prob_valid=probs, label_valid=y_te, doc_ids=z_te["doc"], row_ordinal=z_te["ordinal"],
        pair=pair_ids[z_te["pair"]], error_type=z_te["error_type"], span_i=z_te["i"], span_j=z_te["j"],
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
