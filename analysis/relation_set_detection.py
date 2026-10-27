"""Subset-level relation detection on the line dataset: a validity probe on the summed activations of field
subsets drawn from full (m = 5) records, scored on minimal pairs of subsets.

Usage: ``uv run --python 3.12 --extra analysis python analysis/relation_set_detection.py analysis/configs/<type>/<id>.yaml``

Reads the ``relation_detection`` train and test runs named in the config, then:

1. Load one layer and subtract each document's mean activation.
2. Join every judged sample to its row in the line file (``relation_detection.match_samples``) and keep the
   minimal pairs of full records (all five fields shown; invalid record + clean twin, same shown order).
3. Every field subset of each size in ``params.subset_sizes`` is taken from both records of a pair, reusing
   the record's activations (one span per field, from the m = 5 prompt). The label rule
   (``vrdu_common.matches_keys`` against the split's line items) labels each; every twin subset must be
   valid. A subset invalid in the invalid record, with the twin's same subset, is a matched pair: train and
   test sets are these, so each (cell, subset) stratum is exactly 50/50. A subset still valid in the
   invalid record is a context pair.
4. Feature ``sum_i X[f_i]`` over the subset. Ablations (``params.ablations``): ``last`` = ``X[f_k]`` (the
   member shown last in the prompt) and ``context`` = the sum of the other members. The span activation is
   the value's last token in a causal model, so a member has attended to every field shown before it in
   the m = 5 prompt -- including fields outside the subset.
5. Logistic regression, train -> test, with the metrics of ``analysis/entity_detection.py``: overall (with
   a document-bootstrap interval), per error type, cell, subset size, number of corrupted members and
   whether the last member is corrupted; the same for each ablation, with a paired document-bootstrap
   interval of AUROC(sum) - AUROC(ablation).
6. Context diagnostic, by kind: ``hidden`` (no corrupted member, every corrupted field shown after all
   members: the members' prompt prefixes equal the twin's, so the features must too -- asserted to
   ``params.prefix_match_rtol``), ``visible`` (no corrupted member, a corrupted field shown before one) and
   ``corrupted_member`` (valid by the rule despite a corrupted member).
7. Controls: structure baseline (cell x subset one-hot; exactly 0.5 AUROC, asserted), shuffled-label
   AUROC ~ 0.5 (asserted), seed determinism (asserted).

Outputs go to ``$RUNS_ROOT/<id>/`` and ``analysis/figures/<id>/``.
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

_here = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("analysis_relation_detection", _here / "relation_detection.py")
rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rr)  # sample matching and label rule
er = rr.er  # shared loader, metrics and plots

FIGURES_DIR = _here / "figures"
FEATURES = ("sum", "last", "context")
CONTEXT_KINDS = ("hidden", "visible", "corrupted_member")
N_FIELDS = len(rr.LINE_FIELDS)
# Per-subset columns of build_subsets (besides the span-index lists).
COLUMNS = ("size", "cell", "error_type", "doc", "pair_id", "fields", "n_corrupted", "last_corrupted", "context_kind")


# ------------------------------------------------------------------------ subsets
def build_subsets(split: dict, groups: list[dict], data: dict, sizes: list[int]) -> dict:
    """Matched and context subset pairs of every judged minimal pair of full records.

    Returns, for matched pairs (no prefix) and context pairs (``ctx_``): ``inv_rows`` / ``val_rows``
    (lists of span-index arrays, members in shown order, from the invalid record / the twin) and arrays
    ``size``, ``cell``, ``error_type``, ``doc``, ``pair_id``, ``fields`` (members joined by ``,``),
    ``n_corrupted`` (corrupted members), ``last_corrupted`` (the last member is corrupted) and
    ``context_kind`` (``""`` for matched pairs, else one of ``CONTEXT_KINDS``); and ``n_records`` (full-record
    pairs) and ``n_samples`` (all judged samples).

    Raises:
        AssertionError: as ``relation_detection.match_samples``, a full record that is not five fields, a
            twin with other fields or order, a full record the label rule disagrees with, or a twin subset
            that is no item's.
    """
    assert sizes and all(isinstance(s, int) and 2 <= s <= N_FIELDS for s in sizes) and len(set(sizes)) == len(sizes), sizes
    by_pair = rr.match_samples(split, groups, data)
    out: dict[str, list] = defaultdict(list)
    n_records = 0
    for pid in sorted(by_pair):
        bad, good = by_pair[pid][False], by_pair[pid][True]
        assert bad["fields"] == good["fields"] and bad["doc"] == good["doc"] and bad["cell"] == good["cell"], pid
        m = int(bad["cell"].split(":")[1].removeprefix("m"))
        assert m == len(bad["fields"]), (pid, m, bad["fields"])
        if m != N_FIELDS:
            continue
        n_records += 1
        keys, eq = data["item_keys"][bad["doc"]], data["eq"]
        val = {v: [str(split["value"][i]) for i in g["rows"]] for v, g in ((False, bad), (True, good))}
        corrupt = [not bool(split["label_valid"][i]) for i in bad["rows"]]
        assert all(bool(split["label_valid"][i]) for i in good["rows"]), (pid, "a corrupted twin span")
        assert rr.is_item_pair(dict(zip(good["fields"], val[True])), keys, eq), (pid, "a twin that is no item's")
        assert not rr.is_item_pair(dict(zip(bad["fields"], val[False])), keys, eq), (pid, "an invalid record that is an item's")
        for s in sizes:
            for pos in itertools.combinations(range(N_FIELDS), s):
                fs = [bad["fields"][p] for p in pos]
                assert rr.is_item_pair({f: val[True][p] for f, p in zip(fs, pos)}, keys, eq), (pid, fs, "a twin subset that is no item's")
                valid = rr.is_item_pair({f: val[False][p] for f, p in zip(fs, pos)}, keys, eq)
                n_corr = sum(corrupt[p] for p in pos)
                if not valid:
                    pre, kind = "", ""
                else:
                    pre = "ctx_"
                    kind = ("corrupted_member" if n_corr else
                            "visible" if any(c and q < pos[-1] for q, c in enumerate(corrupt)) else "hidden")
                out[f"{pre}inv_rows"].append(np.array([bad["rows"][p] for p in pos]))
                out[f"{pre}val_rows"].append(np.array([good["rows"][p] for p in pos]))
                for c, v in zip(COLUMNS, (s, bad["cell"], bad["cell"].split(":")[0], bad["doc"], pid, ",".join(fs),
                                          n_corr, corrupt[pos[-1]], kind)):
                    out[f"{pre}{c}"].append(v)
    assert out["inv_rows"], "no matched subset pairs"
    res = {k: (v if k.endswith("_rows") else np.array(v)) for k, v in out.items()}
    res["n_records"], res["n_samples"] = n_records, len(groups)
    for pre in ("", "ctx_"):
        if f"{pre}inv_rows" in res:
            assert len(res[f"{pre}inv_rows"]) == len(res[f"{pre}val_rows"]) == len(res[f"{pre}size"])
    assert (res["n_corrupted"] > 0).all(), "a matched invalid subset with no corrupted member"
    return res


def stack(z: dict, pre: str = "") -> dict:
    """Both members of every subset pair (``pre`` = ``""`` or ``"ctx_"``): ``rows`` (invalid-record subsets,
    then the twins'), ``y`` (valid: the twin's subset), and every column of ``COLUMNS`` repeated for both."""
    n = len(z[f"{pre}inv_rows"])
    return {"rows": z[f"{pre}inv_rows"] + z[f"{pre}val_rows"], "y": np.repeat([False, True], n),
            **{c: np.concatenate([z[f"{pre}{c}"]] * 2) for c in COLUMNS}}


def subset_features(x: np.ndarray, rows: list[np.ndarray], feature: str) -> np.ndarray:
    """One row per subset (spans ``rows[s]`` in shown order, the last shown last): see ``FEATURES``."""
    assert feature in FEATURES, f"unknown feature {feature!r}; one of {FEATURES}"
    assert all(len(r) >= 2 for r in rows), "a subset of fewer than two members"
    combine = {"sum": lambda r: x[r].sum(axis=0),
               "last": lambda r: x[r[-1]],
               "context": lambda r: x[r[:-1]].sum(axis=0)}[feature]
    f = np.stack([combine(r) for r in rows])
    assert f.shape == (len(rows), x.shape[1]) and np.isfinite(f).all(), (feature, f.shape)
    return f


def prefix_match_dev(x: np.ndarray, z: dict) -> tuple[float, float]:
    """``(largest ||sum(inv) - sum(twin)|| / ||sum(twin)||, share exactly equal)`` over ``hidden`` context
    pairs (``(0, 1)`` if there are none). Their members' prompt prefixes are identical, so a difference is
    bf16 run-to-run noise (small) or a misaligned span (large)."""
    sel = np.flatnonzero(z["ctx_context_kind"] == "hidden") if "ctx_context_kind" in z else np.array([], int)
    if not len(sel):
        return 0.0, 1.0
    a = subset_features(x, [z["ctx_inv_rows"][i] for i in sel], "sum")
    b = subset_features(x, [z["ctx_val_rows"][i] for i in sel], "sum")
    rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)
    return float(rel.max()), float((rel == 0).mean())


def doc_bootstrap_auroc_diff(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, docs: np.ndarray, n_boot: int,
                             ci_level: float, seed: int) -> tuple[float, float]:
    """Percentile interval of ``AUROC(pa) - AUROC(pb)`` over ``n_boot`` document resamples shared by both."""
    assert isinstance(n_boot, int) and n_boot > 0 and 0.0 < ci_level < 1.0, (n_boot, ci_level)
    assert y.shape == pa.shape == pb.shape == docs.shape
    uniq = np.unique(docs)
    members = [np.flatnonzero(docs == d) for d in uniq]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([members[j] for j in rng.integers(len(uniq), size=len(uniq))])
        assert y[idx].any() and (~y[idx]).any(), "a document resample lacks a class"
        diffs.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    lo, hi = np.quantile(diffs, [(1 - ci_level) / 2, (1 + ci_level) / 2])
    return float(lo), float(hi)


def strata_metrics(prefix: str, y: np.ndarray, probs: np.ndarray, thr: float, s: dict) -> dict:
    """``stratum_report`` by error type, cell, subset size, corrupted members and last-member corruption."""
    out = {}
    for strata in (s["error_type"], s["cell"], np.char.add("size", s["size"].astype(str)),
                   np.char.add("ncorr", s["n_corrupted"].astype(str)),
                   np.where(s["last_corrupted"], "last_corrupted", "last_clean")):
        out.update({f"{prefix}_{k}": v for k, v in er.stratum_report(y, probs, thr, strata).items()})
    return out


# ----------------------------------------------------------------------------- main
def run(cfg: dict, runs_root: Path) -> dict:
    p, seed = cfg["params"], cfg["seed"]
    out = runs_root / cfg["id"]
    fig_dir = FIGURES_DIR / cfg["id"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    layer, thr, sizes, ablations = p["layer"], p["decision_threshold"], p["subset_sizes"], p["ablations"]
    assert isinstance(layer, int) and layer > 0, "layer must be a positive int (layer 0 is the constant null control)"
    assert isinstance(ablations, list) and set(ablations) <= set(FEATURES) - {"sum"}, ablations

    extra = ("line_index", "value", "pair_id", "cell")
    tr = er.load_split(p["train_run_id"], runs_root, layer, p["max_docs"], "relation_detection", extra)
    te = er.load_split(p["test_run_id"], runs_root, layer, p["max_docs"], "relation_detection", extra)
    assert not (set(tr["doc_ids"].tolist()) & set(te["doc_ids"].tolist())), "train/test documents overlap"
    for s in (tr, te):
        assert set(np.unique(s["field"]).tolist()) <= set(rr.LINE_FIELDS), "non-line field in the relation run"
    er.center_per_document(tr["X"], tr["doc_ids"])
    er.center_per_document(te["X"], te["doc_ids"])

    z_tr = build_subsets(tr, rr.group_rows(tr), rr.load_line_data(p["train_run_id"], runs_root, rr.REPO_ROOT), sizes)
    z_te = build_subsets(te, rr.group_rows(te), rr.load_line_data(p["test_run_id"], runs_root, rr.REPO_ROOT), sizes)

    # Known-answer: hidden context pairs have identical prompt prefixes, so identical features up to bf16 noise.
    (dev_tr, eq_tr), (dev_te, eq_te) = prefix_match_dev(tr["X"], z_tr), prefix_match_dev(te["X"], z_te)
    dev = max(dev_tr, dev_te)
    assert dev <= p["prefix_match_rtol"], f"hidden context pairs differ by {dev:.3g} (relative norm) -- misaligned spans?"

    s_tr, s_te = stack(z_tr), stack(z_te)
    f_tr, f_te = subset_features(tr["X"], s_tr["rows"], "sum"), subset_features(te["X"], s_te["rows"], "sum")
    y_tr, y_te = s_tr["y"], s_te["y"]
    ref_rate = float(y_tr.mean())
    assert ref_rate == 0.5 == float(y_te.mean()), "matched subset sets are not 50/50"
    types = sorted(set(z_tr["error_type"].tolist()))
    assert types == sorted(set(z_te["error_type"].tolist())), "train and test have different error types"
    assert set(types) <= set(er.ALL_ERROR_TYPES), types

    metrics: dict[str, float] = {
        "layer": layer, "subset_sizes": sizes, "prefix_match_dev": dev,
        "train_prefix_match_exact_share": eq_tr, "test_prefix_match_exact_share": eq_te,
        "n_train_records": z_tr["n_records"], "n_test_records": z_te["n_records"],
        "n_train_pairs": len(z_tr["inv_rows"]), "n_test_pairs": len(z_te["inv_rows"]),
        "n_train_docs": len(set(z_tr["doc"].tolist())), "n_test_docs": len(set(z_te["doc"].tolist())),
    }
    for split_name, z in (("train", z_tr), ("test", z_te)):
        for kind in CONTEXT_KINDS:
            metrics[f"n_{split_name}_context_{kind}"] = int((z["ctx_context_kind"] == kind).sum()) if "ctx_context_kind" in z else 0

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
    metrics.update(strata_metrics("test", y_te, probs, thr, s_te))

    abl_probs, abl_clf = {}, {}
    for a in ablations:
        a_tr, a_te = subset_features(tr["X"], s_tr["rows"], a), subset_features(te["X"], s_te["rows"], a)
        abl_clf[a] = er.fit_logreg(a_tr, y_tr, p, seed)
        abl_probs[a] = abl_clf[a].predict_proba(a_te)[:, 1]
        metrics.update(er.probe_metrics(f"test_ablation_{a}", y_te, abl_probs[a], ref_rate, thr))
        metrics.update(strata_metrics(f"test_ablation_{a}", y_te, abl_probs[a], thr, s_te))
        metrics[f"test_auroc_minus_{a}_ci_lo"], metrics[f"test_auroc_minus_{a}_ci_hi"] = doc_bootstrap_auroc_diff(
            y_te, probs, abl_probs[a], s_te["doc"], p["auroc_n_boot"], p["ci_level"], seed)

    # Context diagnostic: still-valid subsets of invalid records vs the twin's same subsets, by kind.
    if "ctx_inv_rows" in z_te:
        c = stack(z_te, "ctx_")
        p_ctx = clf.predict_proba(subset_features(te["X"], c["rows"], "sum"))[:, 1]
        for kind in CONTEXT_KINDS:
            sel = c["context_kind"] == kind
            if not sel.any():
                continue
            inv, twin = sel & ~c["y"], sel & c["y"]
            metrics[f"test_context_{kind}_mean_p_valid"] = float(p_ctx[inv].mean())
            metrics[f"test_context_{kind}_twin_mean_p_valid"] = float(p_ctx[twin].mean())
            metrics[f"test_context_{kind}_valid_acc"] = float((p_ctx[inv] > thr).mean())
            metrics[f"test_twin_vs_context_{kind}_auroc"] = float(roc_auc_score(c["y"][sel], p_ctx[sel]))
        np.savez_compressed(
            out / "artifacts" / "test_context_predictions.npz",
            prob_valid=p_ctx, label_valid=c["y"], **{c_: c[c_] for c_ in COLUMNS},
            **{f"prob_valid_{a}": abl_clf[a].predict_proba(subset_features(te["X"], c["rows"], a))[:, 1] for a in ablations},
        )

    # Structure baseline: cell x subset (members in shown order); each exactly 50/50 -> AUROC 0.5.
    struct_tr, struct_te = (np.char.add(np.char.add(s["cell"], "|"), s["fields"]) for s in (s_tr, s_te))
    metrics.update(er.assert_structure_auroc_is_chance("cell_subset", y_tr, struct_tr, y_te, struct_te, p, seed))

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
        prob_valid=probs, label_valid=y_te, **{f"prob_valid_{a}": v for a, v in abl_probs.items()},
        **{c: s_te[c] for c in COLUMNS if c != "context_kind"},
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
