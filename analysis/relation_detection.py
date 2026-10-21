"""Row-based relation detection: a validity probe on the concatenated activations of a pair of
fields from one line-item row.

Usage: ``uv run --python 3.12 --extra analysis python analysis/relation_detection.py configs/<id>.yaml``
(every value lives in the config's ``params``; a missing key is a KeyError).

Reads the ``relation_detection`` train and test runs named in the config
(``src/judge_interp/relation_detection.py``: one prompt per ``(document, line row)``, one
activation per non-null field, read at the field's closing quote) and does, in order:

1. **Load** one layer (``layer``) as ``X`` and subtract each document's mean row -- nothing
   else (as in ``analysis/entity_detection.py``).
2. **Pairs.** A *source row* is a ``(doc_id, row_ordinal)`` prompt. Every unordered pair of
   non-null fields of a source row is one sample (no subsampling, no balancing). Its feature is
   ``[X[a], X[b]]``, the two activations concatenated with the fields in ``LINE_FIELDS`` order.
3. **Pair label** (positive class = valid). Both fields valid -> valid; exactly one invalid ->
   invalid. A pair with *both* fields invalid is **dropped** (its label is ambiguous: two values
   copied from one donor row do co-occur there). The dropped count is reported
   (``n_*_pairs_dropped_both_invalid``).
4. **Logistic regression** (train -> test) on the concatenated activations, exactly as
   ``entity_detection``: smECE (+ bootstrap CI), Brier score, Brier skill score (reference =
   constant forecast at the train valid rate), AUROC, accuracy, smoothed reliability diagram, and
   the per-error-type report (error type = the source row's). ``lr_C`` is sklearn's inverse
   regularisation strength: a small value is a strong penalty.
5. **Controls.** A field-pair-identity baseline (one-hot over the 10 field pairs -> LR, reported);
   mean test AUROC over ``n_shuffles`` shuffled-train-label fits ~ 0.5 (asserted); same-seed refit
   gives identical probabilities (asserted).

Figures go to ``analysis/figures/<id>/`` (with a copy of ``metrics.json``); ``metrics.json``, ``config.snapshot.yaml`` and
``artifacts/test_predictions.npz`` go to ``$RUNS_ROOT/<id>/``.
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

_spec = importlib.util.spec_from_file_location("analysis_entity_detection", Path(__file__).parent / "entity_detection.py")
er = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(er)  # reused unchanged: loader, metrics, error-type report, plots

FIGURES_DIR = Path(__file__).parent / "figures"
# Field pairs, unordered, in LINE_FIELDS order: the baseline's categories.
PAIRS = list(itertools.combinations(LINE_FIELDS, 2))
PAIR_INDEX = {pr: i for i, pr in enumerate(PAIRS)}


# ------------------------------------------------------------------------ source rows
def group_rows(split: dict) -> list[dict]:
    """One entry per ``(doc_id, row_ordinal)`` prompt: ``{doc, ordinal, rows, fields, error_type}``.

    ``rows`` are indices into the split's arrays, in ``LINE_FIELDS`` order. Asserted: fields
    distinct and line fields, the row error type constant within a row. (Labels are per entity, so
    only the error type is row-level.)
    """
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


def build_pairs(split: dict, groups: list[dict]) -> dict:
    """All unordered field pairs of every source row. Arrays, one entry per pair: ``i``, ``j``
    (span indices, ``field[i]`` before ``field[j]`` in ``LINE_FIELDS``), ``label`` (valid),
    ``pair`` (index into ``PAIRS``), ``error_type`` (the source row's), ``doc``, ``ordinal``;
    plus ``n_dropped`` (both-invalid pairs, not included)."""
    out: dict[str, list] = defaultdict(list)
    n_dropped = 0
    valid = split["label_valid"]
    for g in groups:
        for i, j in itertools.combinations(g["rows"], 2):
            fi, fj = str(split["field"][i]), str(split["field"][j])
            assert LINE_FIELDS.index(fi) < LINE_FIELDS.index(fj), (g["doc"], g["ordinal"], fi, fj)
            if not valid[i] and not valid[j]:
                n_dropped += 1
                continue
            out["i"].append(i)
            out["j"].append(j)
            out["label"].append(bool(valid[i] and valid[j]))
            out["pair"].append(PAIR_INDEX[(fi, fj)])
            out["error_type"].append(g["error_type"])
            out["doc"].append(g["doc"])
            out["ordinal"].append(g["ordinal"])
    assert out["i"], "no pairs"
    res = {k: np.array(v) for k, v in out.items()}
    res["n_dropped"] = n_dropped
    assert len(res["i"]) + n_dropped == sum(len(g["rows"]) * (len(g["rows"]) - 1) // 2 for g in groups)
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

    tr = er.load_split(p["train_run_id"], runs_root, layer, p["max_docs"], "relation_detection")
    te = er.load_split(p["test_run_id"], runs_root, layer, p["max_docs"], "relation_detection")
    assert not (set(tr["doc_ids"].tolist()) & set(te["doc_ids"].tolist())), "train/test documents overlap"
    for s in (tr, te):
        assert set(np.unique(s["field"]).tolist()) <= set(LINE_FIELDS), "non-line field in the relation run"
    er.center_per_document(tr["X"], tr["doc_ids"])
    er.center_per_document(te["X"], te["doc_ids"])

    z_tr, z_te = build_pairs(tr, group_rows(tr)), build_pairs(te, group_rows(te))
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
        "n_train_pairs_dropped_both_invalid": z_tr["n_dropped"], "n_test_pairs_dropped_both_invalid": z_te["n_dropped"],
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
