"""Relation detection on the line subsets: per-model probes and the direct judges (Clef, Jev), all scored on the
same rows of ``params.test_subsets`` (``data/vrdu/line/test_subsets.json``).

Usage: ``uv run --python 3.12 --extra analysis python analysis/relation_set_detection.py analysis/configs/<type>/<id>.yaml``

The unit is a subset row: 2 to m of a full-length sample's fields (m = 2-5), labelled by the row's own ``valid``
(the label rule, ``vrdu_common.matches_keys`` against the document's line items). Rows are NOT minimal pairs and
not balanced: a clean parent and its invalid twin share subsets, a subset of an invalid parent can still be valid,
and rows repeat by content. So nothing here assumes 50/50: the Brier reference is the evaluation set's own valid
rate, AUROC is bootstrapped by document, and the structure baseline is reported, not asserted.

Methods (``params.models``: a probe per entry; ``params.judges``: a Clef / Jev ``subsets`` run per entry,
nothing trained). Metric keys and figure directories are prefixed with the method's ``label``.

1. Probes train on the rows of ``params.train_subsets`` (``data/vrdu/line/train_subsets.json``) whose parent
   sample is in the model's train run; every method is scored on ``params.test_subsets``. Each judged run must
   hold subsets of the test split only.
2. Probe feature ``sum_i X[f_i]`` over the subset's members, the activations of the value's last token in the
   PARENT's prompt (one span per shown field, per-document mean-centred over the run). Ablations
   (``params.ablations``): ``last`` = ``X[f_k]`` (the member shown last) and ``context`` = the sum of the other
   members. Asymmetry to keep in mind when comparing to a judge: a member has attended to every field shown before
   it in the parent, including fields outside the subset, whereas a judge sees only the subset.
3. Every span value is checked against the subsets file, and every row's label is recomputed by the label rule
   from the items and must equal ``valid``.
4. The common set: the test rows every method scored (a judge excludes over-length documents; a probe needs the
   parent in its run). Each method's scored and dropped rows are reported, and every method's labels are the file's.
5. Metrics (one code path for every method): smECE, Brier skill, AUROC (document-bootstrap interval), accuracy,
   permutation-null AUROC (asserted ~0.5); per error type, cell, size, corrupted members, parent validity and row
   kind (``invalid``; ``clean_parent``; ``corrupted_member`` = valid despite a corrupted member; ``hidden`` /
   ``visible`` = valid, no corrupted member, every corrupted parent field shown after all members / before one).
   A stratum with one class reports no AUROC. Context diagnostic: each valid row of an invalid parent against the
   same subset of the clean parent (its twin).
6. Probes only: sum vs ablation (paired document-bootstrap AUROC difference), shuffled-label AUROC ~ 0.5
   (asserted), seed determinism (asserted), and the ``hidden`` known answer: the members' prompt prefixes equal the
   twin's, so their features must too (relative difference <= ``params.prefix_match_rtol``, asserted).
   All methods: the structure baseline (a probe on which fields are shown, trained on the train subsets).

7. Report (``analysis/calibration_report.py``, ``params.report``): the only figures. Per error type (intra_document,
   cross_item), a calibration diagram of all methods and a smECE / BSS / AUROC table, every interval over
   ``report.n_boot`` resamples of documents; the pooled values are asserted equal to the metrics above. A second, CONTROL
   report (``control_calibration_*``) scores the same predictions on the rows that are either full-length (a corrupted
   parent and its clean twin) or a strict subset of a corrupted parent, dropping strict subsets of clean parents:
   a clean parent's strict subsets are all valid, so without the control "clean parent" and "short" are confounded
   with the label. Nothing is refit; ``control_*`` metrics are new keys.

Outputs go to ``$RUNS_ROOT/<id>/`` and ``analysis/figures/<id>/``.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

_here = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("analysis_relation_detection", _here / "relation_detection.py")
rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rr)  # sample matching and label rule
er = rr.er  # shared loader, metrics and plots
_cr_spec = importlib.util.spec_from_file_location("analysis_calibration_report", _here / "calibration_report.py")
cr = importlib.util.module_from_spec(_cr_spec)
_cr_spec.loader.exec_module(cr)
REPO_ROOT = rr.REPO_ROOT

FIGURES_DIR = _here / "figures"
FEATURES = ("sum", "last", "context")
ROW_KINDS = ("invalid", "clean_parent", "corrupted_member", "visible", "hidden")
CONTEXT_KINDS = ("corrupted_member", "visible", "hidden")   # valid rows of an invalid parent
JUDGES = ("clef", "jev")
JUDGE_TASKS = ("conditional", "oneshot")
# Per-row columns of a subsets frame (besides the python lists ``members``, ``rows``).
SUBSET_COLS = ("subset_id", "parent_id", "pair_id", "doc", "cell", "error_type", "parent_valid", "size", "parent_size", "fields",
               "n_corrupted", "y", "kind")


@lru_cache(maxsize=None)
def sibling(name: str):
    """``analysis/<name>.py`` as a module, loaded once."""
    return er.load_sibling(name)


# ------------------------------------------------------------------------ subsets frame
def load_subsets(subsets_path: Path, parents_path: Path) -> dict:
    """The rows of a ``line/<split>_subsets.json`` as a frame, checked against the parent samples file.

    Returns arrays ``SUBSET_COLS`` (``parent_size``: the parent's number of fields; ``fields``: members joined by ``,``; ``y``: the row's ``valid``; ``kind``:
    one of ``ROW_KINDS``), the lists ``members`` (field names in shown order) and ``rows`` (the raw rows),
    and ``index`` (``{subset_id: position}``).

    Raises:
        AssertionError: a stale file (``check_build``), a duplicate id, a parent that is missing or disagrees
            (validity, pair, document, cell), members not a subset of the parent's fields in shown order, corrupted
            fields that are not the parent's shown ones, or an invalid row with no corrupted member.
    """
    for path in (subsets_path, parents_path):
        rr.check_build(path.parent.parent, f"{path.parent.name}/{path.name}")
    rows = json.loads(subsets_path.read_text(encoding="utf-8"))
    parent_rows = json.loads(parents_path.read_text(encoding="utf-8"))
    parents = {r["sample_id"]: r for r in parent_rows}
    assert len(parents) == len(parent_rows), f"{parents_path}: sample_id is not unique"
    out: dict[str, list] = defaultdict(list)
    for r in rows:
        sid = r["subset_id"]
        assert r["parent_id"] in parents, f"{sid}: parent {r['parent_id']!r} not in {parents_path.name}"
        par = parents[r["parent_id"]]
        assert (par["valid"], par["pair_id"], par["document_id"], par["cell"]) == (r["parent_valid"], r["pair_id"], r["document_id"], r["cell"]), (
            f"{sid}: disagrees with its parent")
        members = list(r["fields"])
        assert r["size"] == len(members) >= 2 and len(set(members)) == len(members), (sid, members)
        assert set(members) <= set(par["fields"]), (sid, "a member the parent does not show")
        pos = [par["fields"].index(f) for f in members]
        assert pos == sorted(pos), (sid, "members not in the parent's shown order")
        assert set(r["invalid_fields"]) == set(par["invalid_fields"]) & set(members), (sid, "corrupted fields are not the parent's shown ones")
        n_corr = len(r["invalid_fields"])
        assert n_corr == r["num_invalid_fields"], sid
        assert r["valid"] or n_corr, (sid, "an invalid subset with no corrupted member")
        assert not (r["parent_valid"] and (n_corr or not r["valid"])), (sid, "a clean parent's subset that is corrupted or invalid")
        if not r["valid"]:
            kind = "invalid"
        elif r["parent_valid"]:
            kind = "clean_parent"
        elif n_corr:
            kind = "corrupted_member"
        else:
            first_corrupt = min(par["fields"].index(f) for f in par["invalid_fields"])   # an invalid parent has one
            kind = "hidden" if first_corrupt > pos[-1] else "visible"
        for c, v in zip(SUBSET_COLS, (sid, r["parent_id"], r["pair_id"], r["document_id"], r["cell"], r["cell"].split(":")[0],
                                      r["parent_valid"], r["size"], len(par["fields"]), ",".join(members), n_corr, r["valid"], kind)):
            out[c].append(v)
        out["members"].append(members)
        out["rows"].append(r)
    res = {c: np.array(out[c]) for c in SUBSET_COLS}
    res["y"] = res["y"].astype(bool)
    res["parent_valid"] = res["parent_valid"].astype(bool)
    res["members"], res["rows"] = out["members"], out["rows"]
    res["index"] = {sid: i for i, sid in enumerate(res["subset_id"].tolist())}
    assert len(res["index"]) == len(rows), f"{subsets_path}: subset_id is not unique"
    assert set(res["kind"].tolist()) <= set(ROW_KINDS), set(res["kind"].tolist())
    return res


def cut(sub: dict, keep: np.ndarray) -> dict:
    """The rows ``keep`` (positions, kept in the order given) of a frame, with ``index`` rebuilt and ``twin``
    added: for each valid row of an invalid parent, the position of the same subset (``pair_id``, ``fields``) of the
    clean parent (-1 for every other row).

    Raises:
        AssertionError: a context row whose twin is not among ``keep`` or is not valid.
    """
    keep = np.asarray(keep, dtype=np.int64)
    out = {c: sub[c][keep] for c in SUBSET_COLS}
    out["members"], out["rows"] = [sub["members"][i] for i in keep], [sub["rows"][i] for i in keep]
    out["index"] = {sid: i for i, sid in enumerate(out["subset_id"].tolist())}
    assert len(out["index"]) == len(keep), "duplicate rows in the cut"
    clean = {(pid, f): i for i, (pid, f, pv) in enumerate(zip(out["pair_id"].tolist(), out["fields"].tolist(), out["parent_valid"].tolist())) if pv}
    out["twin"] = np.full(len(keep), -1, dtype=np.int64)
    for i in np.flatnonzero(np.isin(out["kind"], CONTEXT_KINDS)):
        key = (str(out["pair_id"][i]), str(out["fields"][i]))
        assert key in clean, f"{out['subset_id'][i]}: its clean-parent twin is not among the rows"
        assert out["y"][clean[key]], f"{out['subset_id'][i]}: its clean-parent twin is not valid"
        out["twin"][i] = clean[key]
    return out


# ------------------------------------------------------------------------ probe features
def lm_member_spans(sub: dict, split: dict, data: dict, require_all: bool) -> tuple[np.ndarray, list[np.ndarray]]:
    """Each row's member spans in a model's run, from the parent's prompt: ``(positions of the rows whose
    parent is in the run, one span-index array per such row in shown order)``.

    Every span value must equal the row's, a span's corrupted flag must be the row's, and the label rule applied
    to the row's values must give the row's ``valid``.

    Raises:
        AssertionError: as ``relation_detection.match_samples`` (a span the row file disagrees with, a pair
            not judged whole), a disagreement above, or (``require_all``) a row whose parent is not in the run.
    """
    by_pair = rr.match_samples(split, rr.group_rows(split), data)
    idx, spans, missing = [], [], 0
    for i in range(len(sub["subset_id"])):
        g = by_pair.get(str(sub["pair_id"][i]), {}).get(bool(sub["parent_valid"][i]))
        if g is None:
            missing += 1
            continue
        r = sub["rows"][i]
        s = np.array([g["rows"][g["fields"].index(f)] for f in sub["members"][i]])
        assert [str(split["value"][j]) for j in s] == [r[f] for f in sub["members"][i]], f"{r['subset_id']}: span values differ from the row"
        assert [not bool(split["label_valid"][j]) for j in s] == [f in r["invalid_fields"] for f in sub["members"][i]], (
            f"{r['subset_id']}: a span's corrupted flag differs from the row's")
        assert rr.is_item_pair({f: r[f] for f in sub["members"][i]}, data["item_keys"][r["document_id"]], data["eq"]) == bool(r["valid"]), (
            f"{r['subset_id']}: the label rule disagrees with valid={r['valid']}")
        idx.append(i)
        spans.append(s)
    assert not (require_all and missing), f"{missing} rows whose parent is not in the run"
    assert idx, "no row has its parent in the run"
    return np.array(idx, dtype=np.int64), spans


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


def prefix_match_dev(f: np.ndarray, ev: dict) -> tuple[float, float]:
    """``(largest ||f(row) - f(twin)|| / ||f(twin)||, share exactly equal)`` over ``hidden`` rows, ``f`` being the
    ``sum`` features of the eval rows. A hidden row's members' prompt prefixes equal its twin's, so a difference is
    bf16 run-to-run noise (small) or a misaligned span (large).

    Raises:
        AssertionError: no hidden row (the known answer would be vacuous).
    """
    hid = np.flatnonzero(ev["kind"] == "hidden")
    assert len(hid), "no hidden row: the prefix known answer cannot be checked"
    a, b = f[hid], f[ev["twin"][hid]]
    rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)
    return float(rel.max()), float((rel == 0).mean())


# ------------------------------------------------------------------------ direct judges
def load_judge(spec: dict, runs_root: Path, sub_te: dict) -> dict:
    """One Clef / Jev run on the test subsets: ``idx`` (positions in ``sub_te``) and ``probs`` (``P(true)``).

    Gates: run.json status / task, counts equal ``metrics.json``, the run is of a ``*-subsets.json`` samples file,
    unique ids that all are test subsets, and each row's stored label, document, cell and size equal the
    subsets file's; ``P(true)`` in [0, 1] and, for Clef, the softmax of its stored logits, for Jev on the
    API's 0.01 grid.

    Returns:
        ``{idx, probs, n_excluded, extra}`` where ``extra`` is Jev's resolution metrics (empty for Clef).
    """
    judge, kind, run_id = spec["judge"], spec["kind"], spec["run_id"]
    assert judge in JUDGES and kind in JUDGE_TASKS, spec
    mod = sibling(judge)
    run_dir = runs_root / run_id
    manifest = json.loads((run_dir / "run.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert manifest["status"] == "success", f"{run_id}: status {manifest['status']!r}"
    assert manifest["task"] == mod.KINDS[kind], f"{run_id}: task {manifest['task']!r}, expected {mod.KINDS[kind]!r}"
    params = yaml.safe_load((run_dir / "config.snapshot.yaml").read_text())["params"]
    assert params["samples_path"].endswith("-subsets.json"), f"{run_id}: not a subsets run ({params['samples_path']})"
    cols = ["p_true", "label_valid", "sample_id", "doc_ids", "cell", "size"] + (["logit_true", "logit_false"] if judge == "clef" else [])
    with np.load(run_dir / "artifacts" / f"{judge}_predictions.npz", allow_pickle=False) as z:
        d = {c: z[c] for c in cols}
    n = len(d["p_true"])
    assert all(len(v) == n for v in d.values()), run_id
    assert n == metrics["n_predictions"], f"{run_id}: {n} rows != metrics n_predictions {metrics['n_predictions']}"
    d["label_valid"] = d["label_valid"].astype(bool)
    assert int(d["label_valid"].sum()) == metrics["n_valid"], run_id
    ids = d["sample_id"].tolist()
    assert len(set(ids)) == n, f"{run_id}: duplicate sample ids"
    unknown = [s for s in ids if s not in sub_te["index"]]
    assert not unknown, f"{run_id}: {len(unknown)} ids that are no test subset's, e.g. {unknown[:2]}"
    idx = np.array([sub_te["index"][s] for s in ids], dtype=np.int64)
    for name, got, want in (("label", d["label_valid"], sub_te["y"][idx]), ("document", d["doc_ids"], sub_te["doc"][idx]),
                            ("cell", d["cell"], sub_te["cell"][idx]), ("size", d["size"], sub_te["size"][idx])):
        assert (got == want).all(), f"{run_id}: stored {name} differs from the subsets file's"
    p = d["p_true"].astype(np.float64)
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all(), f"{run_id}: p_true outside [0, 1]"
    if judge == "clef":
        z64 = np.stack([d["logit_true"], d["logit_false"]], axis=1).astype(np.float64)
        assert np.allclose(p, np.exp(z64[:, 0] - np.logaddexp(z64[:, 0], z64[:, 1])), atol=1e-6), f"{run_id}: p_true is not the softmax of the stored logits"
    else:
        assert np.allclose(p / mod.P_GRID, np.round(p / mod.P_GRID), atol=mod.P_GRID_ATOL / mod.P_GRID), f"{run_id}: p_true is not on the {mod.P_GRID} grid"
    return {"idx": idx, "probs": p, "n_excluded": 0 if judge == "jev" else metrics["n_excluded_overlong"],
            "extra": mod.resolution_metrics({"p_true": p, "determinism_max_abs_diff": float(metrics["determinism_max_abs_diff"])}) if judge == "jev" else {}}


# ------------------------------------------------------------------------ metrics
def row_stratum_report(y: np.ndarray, probs: np.ndarray, threshold: float, strata: np.ndarray) -> dict:
    """Per-stratum metrics of an unbalanced row set. For each stratum value ``s``: ``n_<s>``, ``valid_rate_<s>``,
    ``accuracy_<s>``, ``mean_p_valid_<s>``; ``auroc_<s>`` if it has both classes, ``detect_<s>`` (invalid called
    invalid) if it has an invalid row, ``valid_acc_<s>`` (valid called valid) if it has a valid one."""
    assert y.dtype == bool and probs.shape == y.shape == strata.shape
    pred = probs > threshold
    out: dict[str, float] = {}
    for s in sorted(set(strata.tolist())):
        sel = strata == s
        ys, ps = y[sel], pred[sel]
        out[f"n_{s}"], out[f"valid_rate_{s}"] = int(sel.sum()), float(ys.mean())
        out[f"accuracy_{s}"], out[f"mean_p_valid_{s}"] = float((ps == ys).mean()), float(probs[sel].mean())
        if ys.any() and (~ys).any():
            out[f"auroc_{s}"] = float(roc_auc_score(ys, probs[sel]))
        if (~ys).any():
            out[f"detect_{s}"] = float((~ps[~ys]).mean())
        if ys.any():
            out[f"valid_acc_{s}"] = float(ps[ys].mean())
    return out


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


def score_method(label: str, ev: dict, probs: np.ndarray, p: dict, seed: int) -> dict:
    """The metrics every method gets, on the common rows ``ev`` (``probs``: its ``P(valid)`` per row).

    Raises:
        AssertionError: one class missing, or the permutation-null AUROC not ~0.5.
    """
    y, thr = ev["y"], p["decision_threshold"]
    assert probs.shape == y.shape and y.any() and (~y).any(), f"{label}: bad probabilities or one class missing"
    ref_rate = float(y.mean())
    m: dict[str, float] = {"n": int(len(y)), "n_valid": int(y.sum()), "valid_rate": ref_rate, "n_docs": len(set(ev["doc"].tolist())),
                           "mean_p_valid": float(probs.mean())}
    m.update(er.probe_metrics("test", y, probs, ref_rate, thr))
    m["test_auroc_ci_lo"], m["test_auroc_ci_hi"] = er.doc_bootstrap_auroc(y, probs, ev["doc"], p["auroc_n_boot"], p["ci_level"], seed)
    d = er.smooth_ece(probs, y, p["ece_n_boot"], seed)
    m["test_smece"], m["test_smece_ci_halfwidth"] = float(d["ce"]), float(d["ce_ci_width"])
    for strata in (ev["error_type"], ev["cell"], np.char.add("size", ev["size"].astype(str)),
                   np.char.add("ncorr", ev["n_corrupted"].astype(str)),
                   np.where(ev["parent_valid"], "parent_clean", "parent_invalid"), ev["kind"]):
        m.update({f"test_{k}": v for k, v in row_stratum_report(y, probs, thr, strata).items()})
    # Context diagnostic: valid rows of an invalid parent against the same subset of the clean parent.
    for kind in CONTEXT_KINDS:
        sel = np.flatnonzero(ev["kind"] == kind)
        if not len(sel):
            continue
        twin = ev["twin"][sel]
        m[f"test_context_{kind}_mean_p_valid"], m[f"test_context_{kind}_twin_mean_p_valid"] = float(probs[sel].mean()), float(probs[twin].mean())
        m[f"test_context_{kind}_valid_acc"] = float((probs[sel] > thr).mean())
        m[f"test_twin_vs_context_{kind}_auroc"] = float(roc_auc_score(
            np.concatenate([np.zeros(len(sel), bool), np.ones(len(sel), bool)]), np.concatenate([probs[sel], probs[twin]])))
    mean, std = sibling("clef").permutation_null_auroc(y, probs, p["n_permutations"], seed)   # as the judges' analysis
    m["control_permutation_auroc"], m["control_permutation_auroc_std"] = mean, std
    assert abs(mean - 0.5) <= p["permutation_auroc_tolerance"], (
        f"{label}: mean permuted-label AUROC {mean:.4f} (std {std:.4f}, n={p['n_permutations']}) is not ~0.5")
    return m


def structure_baseline(sub_tr: dict, ev: dict, p: dict, seed: int) -> dict:
    """A probe on which fields are shown (one-hot of ``fields``), trained on all train subsets, on the common rows:
    what structure alone predicts. Reported, not asserted: on unbalanced subsets it is above chance."""
    cats = sorted(set(sub_tr["fields"].tolist()) | set(ev["fields"].tolist()))
    clf = er.fit_logreg(er.onehot(sub_tr["fields"], cats), sub_tr["y"], p, seed)
    probs = clf.predict_proba(er.onehot(ev["fields"], cats))[:, 1]
    return er.probe_metrics("test_baseline_fields", ev["y"], probs, float(ev["y"].mean()), p["decision_threshold"])


def check_pairs(ev: dict) -> None:
    """The rows split cleanly by error type and pair: each ``pair_id`` has one error type and rows of both its invalid
    parent and its clean twin.

    Raises:
        AssertionError: a pair with two error types or with rows of only one of its parents.
    """
    for pid in sorted(set(ev["pair_id"].tolist())):
        sel = ev["pair_id"] == pid
        assert len(set(ev["error_type"][sel].tolist())) == 1, f"{pid}: more than one error type"
        assert set(ev["parent_valid"][sel].tolist()) == {True, False}, f"{pid}: rows of only one of the pair's parents"


def control_mask(ev: dict) -> np.ndarray:
    """Rows of the control: every row of an invalid parent (the full-length row and its strict subsets, valid or
    not) and the full-length rows of clean parents; i.e. all but the strict subsets of clean parents.

    Raises:
        AssertionError: a pair whose full-length clean row and full-length corrupted row are not both present (the
            control's matched counterparts), or a full-length corrupted row that is not invalid.
    """
    full = ev["size"] == ev["parent_size"]
    clean_full, bad_full = ev["parent_valid"] & full, ~ev["parent_valid"] & full
    a, b = set(ev["pair_id"][clean_full].tolist()), set(ev["pair_id"][bad_full].tolist())
    assert a == b, (f"full-length rows are not matched by pair: {len(a - b)} pairs with only the clean one, {len(b - a)} with only the "
                    f"corrupted one (an over-length exclusion?), e.g. {sorted(a ^ b)[:2]}")
    assert not ev["y"][bad_full].any(), "a full-length row of a corrupted parent is valid"
    assert ev["y"][clean_full].all(), "a full-length row of a clean parent is invalid"
    return ~ev["parent_valid"] | full


def run_lm(label: str, layer: int, p: dict, seed: int, tr: dict, rows_tr: list[np.ndarray], y_tr: np.ndarray, te: dict,
           rows_ev: list[np.ndarray], ev: dict, art_dir: Path) -> tuple[dict, np.ndarray]:
    """One model's probe: logistic regression on the train subsets' ``sum`` features, scored on the common rows, the
    ablations and the probe controls. ``tr`` / ``te``: the model's centred train / test splits; ``rows_tr`` /
    ``rows_ev``: the member spans of each train subset / eval row in them. Returns ``(metrics, P(valid) per eval row)``."""
    thr, y_te = p["decision_threshold"], ev["y"]
    ref_rate = float(y_te.mean())
    assert y_tr.any() and (~y_tr).any(), f"{label}: one class missing in the train subsets"
    assert len(rows_tr) == len(y_tr)
    t0 = time.perf_counter()
    f_tr = subset_features(tr["X"], rows_tr, "sum")
    feature_seconds = time.perf_counter() - t0
    f_te = subset_features(te["X"], rows_ev, "sum")
    dev, exact = prefix_match_dev(f_te, ev)
    assert dev <= p["prefix_match_rtol"], f"{label}: hidden rows differ from their twins by {dev:.3g} (relative norm) -- misaligned spans?"
    metrics: dict[str, float] = {"layer": layer, "prefix_match_dev": dev, "prefix_match_exact_share": exact,
                                 "n_train": int(len(y_tr)), "n_train_valid": int(y_tr.sum())}
    clf, fit_seconds = er.timed_fit_logreg(f_tr, y_tr, p, seed)
    metrics["probe_train_time"] = {"feature_seconds": feature_seconds, "fit_seconds": fit_seconds,
                                   "n_train": int(f_tr.shape[0]), "n_features": int(f_tr.shape[1])}
    probs = clf.predict_proba(f_te)[:, 1]
    metrics["train_auroc"] = float(roc_auc_score(y_tr, clf.predict_proba(f_tr)[:, 1]))
    metrics.update(score_method(label, ev, probs, p, seed))
    abl_probs = {}
    for a in p["ablations"]:
        a_tr, a_te = subset_features(tr["X"], rows_tr, a), subset_features(te["X"], rows_ev, a)
        abl = er.fit_logreg(a_tr, y_tr, p, seed)
        abl_probs[a] = abl.predict_proba(a_te)[:, 1]
        metrics.update({f"ablation_{a}_{k}": v for k, v in er.probe_metrics("test", y_te, abl_probs[a], ref_rate, thr).items()})
        metrics[f"test_auroc_minus_{a}_ci_lo"], metrics[f"test_auroc_minus_{a}_ci_hi"] = doc_bootstrap_auroc_diff(
            y_te, probs, abl_probs[a], ev["doc"], p["auroc_n_boot"], p["ci_level"], seed)
    if p["run_controls"]:
        auc_shuf, auc_shuf_std = er.shuffled_label_auroc(f_tr, y_tr, f_te, y_te, p, seed)
        metrics["control_shuffled_auroc"], metrics["control_shuffled_auroc_std"] = auc_shuf, auc_shuf_std
        assert abs(auc_shuf - 0.5) <= p["shuffled_label_auroc_tolerance"], (
            f"{label}: mean shuffled-label AUROC {auc_shuf:.4f} (std {auc_shuf_std:.4f}, n={p['n_shuffles']}) is not ~0.5 -- something leaks")
        assert np.array_equal(probs, er.fit_logreg(f_tr, y_tr, p, seed).predict_proba(f_te)[:, 1]), f"{label}: same-seed refit gave different probabilities"
        metrics["control_seed_determinism"] = 1.0
    np.savez_compressed(art_dir / f"{label}_test_predictions.npz", subset_id=ev["subset_id"], prob_valid=probs, label_valid=y_te,
                        **{f"prob_valid_{a}": v for a, v in abl_probs.items()})
    return metrics, probs


# ----------------------------------------------------------------------------- report
def report(ev: dict, probs_by_method: dict[str, np.ndarray], metrics: dict, p: dict, seed: int, fig_dir: Path, out: Path) -> dict:
    """The calibration report on the common rows, then the control report on ``control_mask`` of them (see the module
    docstring). Returns the new ``metrics`` entries: split counts per error type, and the control's row counts."""
    rep_cfg, ets = p["report"], list(p["report"]["error_type_names"])
    evset = {"y": ev["y"], "docs": ev["doc"], "error_type": ev["error_type"], "probs": probs_by_method}
    cr.check_evset(evset, list(rep_cfg["methods"]))
    cr.assert_matches_existing(evset, metrics)
    check_pairs(ev)
    cr.save_evset(out / "artifacts" / "calibration_predictions.npz", evset, subset_id=ev["subset_id"], parent_valid=ev["parent_valid"],
                  size=ev["size"], parent_size=ev["parent_size"], pair_id=ev["pair_id"])
    m = cr.write_report(evset, rep_cfg, ets, p["ci_level"], seed, fig_dir, out, "calibration", "Evaluation: all common test subset rows.")
    keep = control_mask(ev)   # after the main report: a control that cannot be formed must not cost the main figures
    n_drop = int((~keep).sum())
    assert n_drop == int((ev["parent_valid"] & (ev["size"] < ev["parent_size"])).sum())
    m["control_n_rows"], m["control_n_dropped_strict_clean_subsets"] = int(keep.sum()), n_drop
    m.update(cr.write_report(cr.subset_evset(evset, keep), rep_cfg, ets, p["ci_level"], seed, fig_dir, out, "control_calibration",
                             "Control: full-length rows (corrupted parent and clean twin) and strict subsets of corrupted parents only; "
                             "strict subsets of clean parents are dropped."))
    return m


# ----------------------------------------------------------------------------- main
def run(cfg: dict, runs_root: Path) -> dict:
    """Every model's probe and every direct judge on the same test subsets (see the module docstring)."""
    p, seed = cfg["params"], cfg["seed"]
    out = runs_root / cfg["id"]
    fig_dir = FIGURES_DIR / cfg["id"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    assert isinstance(p["ablations"], list) and set(p["ablations"]) <= set(FEATURES) - {"sum"}, p["ablations"]
    labels = [s["label"] for s in (*p["models"], *p["judges"])]
    assert len(set(labels)) == len(labels) and labels, f"duplicate or no method labels {labels}"
    paths = {k: REPO_ROOT / p[k] for k in ("train_subsets", "test_subsets", "train_parents", "test_parents")}
    assert len(set(paths.values())) == 4, "train and test files overlap"
    sub_tr, sub_te = load_subsets(paths["train_subsets"], paths["train_parents"]), load_subsets(paths["test_subsets"], paths["test_parents"])
    assert not (set(sub_tr["doc"].tolist()) & set(sub_te["doc"].tolist())), "train/test documents overlap"

    lms: dict[str, dict] = {}
    for spec in p["models"]:
        layer = spec["layer"]
        assert isinstance(layer, int) and layer > 0, "layer must be a positive int (layer 0 is the constant null control)"
        extra = ("line_index", "value", "pair_id", "cell")
        tr = er.load_split(spec["train_run_id"], runs_root, layer, p["max_docs"], "relation_detection", extra)
        te = er.load_split(spec["test_run_id"], runs_root, layer, p["max_docs"], "relation_detection", extra)
        assert not (set(tr["doc_ids"].tolist()) & set(te["doc_ids"].tolist())), f"{spec['label']}: train/test documents overlap"
        for s in (tr, te):
            assert set(np.unique(s["field"]).tolist()) <= set(rr.LINE_FIELDS), "non-line field in the relation run"
        for run_id, parents in ((spec["train_run_id"], paths["train_parents"]), (spec["test_run_id"], paths["test_parents"])):
            assert er.run_rows_path(run_id, runs_root, rr.REPO_ROOT, "line") == parents, f"{run_id}: judged another file than {parents}"
        er.center_per_document(tr["X"], tr["doc_ids"])
        er.center_per_document(te["X"], te["doc_ids"])
        idx_tr, rows_tr = lm_member_spans(sub_tr, tr, rr.load_line_data(spec["train_run_id"], runs_root, rr.REPO_ROOT), require_all=p["max_docs"] is None)
        idx_te, rows_te = lm_member_spans(sub_te, te, rr.load_line_data(spec["test_run_id"], runs_root, rr.REPO_ROOT), require_all=False)
        lms[spec["label"]] = {"tr": tr, "te": te, "idx_tr": idx_tr, "rows_tr": rows_tr, "idx_te": idx_te, "rows_te": dict(zip(idx_te.tolist(), rows_te))}
    judges = {spec["label"]: load_judge(spec, runs_root, sub_te) for spec in p["judges"]}

    # The common set: test rows every method scored; the labels are the file's for all of them.
    scored = {**{k: v["idx_te"] for k, v in lms.items()}, **{k: v["idx"] for k, v in judges.items()}}
    common = sorted(set.intersection(*(set(i.tolist()) for i in scored.values())))
    assert common, "no test row was scored by every method"
    ev = cut(sub_te, np.array(common))
    metrics: dict[str, float] = {"n_test_subsets_file": len(sub_te["subset_id"]), "n_common_test_rows": len(common)}
    for label, i in scored.items():
        metrics[f"{label}_n_test_rows"], metrics[f"{label}_n_dropped_rows"] = int(len(i)), int(len(i)) - len(common)
    metrics.update(structure_baseline(sub_tr, ev, p, seed))

    art_dir = out / "artifacts"
    probs_by_method: dict[str, np.ndarray] = {}
    probe_times: dict[str, dict] = {}
    for spec in p["models"]:
        label, m = spec["label"], lms[spec["label"]]
        rows_ev = [m["rows_te"][i] for i in common]
        m_metrics, lm_probs = run_lm(label, spec["layer"], p, seed, m["tr"], m["rows_tr"], sub_tr["y"][m["idx_tr"]], m["te"], rows_ev, ev, art_dir)
        probs_by_method[label] = lm_probs.astype(np.float64)
        probe_times[label] = m_metrics.pop("probe_train_time")
        for k_, v in m_metrics.items():
            metrics[f"{label}_{k_}"] = v
    er.write_probe_train_times(cfg["id"], probe_times)
    for spec in p["judges"]:
        label, j = spec["label"], judges[spec["label"]]
        pos = {int(i): n for n, i in enumerate(j["idx"].tolist())}
        probs = j["probs"][[pos[i] for i in common]]
        np.savez_compressed(art_dir / f"{label}_test_predictions.npz", subset_id=ev["subset_id"], prob_valid=probs, label_valid=ev["y"])
        probs_by_method[label] = probs.astype(np.float64)
        for k_, v in {**score_method(label, ev, probs, p, seed), "n_excluded_overlong": j["n_excluded"], **j["extra"]}.items():
            metrics[f"{label}_{k_}"] = v
    metrics.update(report(ev, probs_by_method, metrics, p, seed, fig_dir, out))
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
