"""Relation detection on the line subsets: per-model probes and the direct judges (Clef, Jev), scored on the same
rows of ``params.test_subsets`` (``data/vrdu/line/test_subsets.json``).

Usage: ``uv run --python 3.12 --extra analysis python analysis/relation_set_detection.py analysis/configs/<type>/<id>.yaml``

Data. The unit is a subset row: 2 to m of a full-length sample's fields, labelled by its own ``valid`` (the label rule
against the document's line items). Rows are not balanced and not minimal pairs: a clean parent and its corrupted
twin share subsets, a subset of a corrupted parent can still be valid, and rows repeat by content. So the Brier
reference is the evaluation set's own valid rate and intervals are bootstrapped by document.

Row selection (``params.selection``, required; applied to the train and the test subsets alike):
  null                every row.
  ``corrupted_parents``  every row of a corrupted parent (kinds ``invalid``, ``corrupted_member``, ``hidden``,
                      ``visible``) and no row of a clean parent, so every row has a corrupted prompt as context. Without
                      clean-parent rows there is no twin: the twin diagnostics and the control report are skipped.

Methods: a probe per ``params.models`` entry, a Clef / Jev ``subsets`` run per ``params.judges`` entry (nothing
trained). Metric keys and figure directories are prefixed with the method's ``label``.

1. Probe: logistic regression on the sum of the subset members' activations (last token of the value, in the PARENT's
   prompt, per-document mean-centred), trained on the selected ``params.train_subsets`` rows. Ablations
   (``params.ablations``): ``last`` = the member shown last, ``context`` = the sum of the others. A member has attended
   to every field shown before it in the parent, a judge sees only the subset.
2. Common set: the selected test rows every method scored. Each method's scored and dropped rows are reported.
3. Metrics, one code path for every method: smECE, Brier skill, AUROC, accuracy, overall and per error type, cell,
   size, corrupted members, parent validity and row kind (``invalid``; ``clean_parent``; ``corrupted_member`` = valid
   despite a corrupted member; ``hidden`` / ``visible`` = valid, no corrupted member, every corrupted parent field
   shown after all members / before one). A stratum with one class has no AUROC. With ``selection: null``, a
   context diagnostic compares each valid row of a corrupted parent with the same subset of the clean parent.
4. Controls. Structure baseline, all methods: a probe on the field set, trained on the same train rows, reported. The
   corrupted count is not a feature (it is derived from the label and no method sees it); it is a stratum.
   Probes only: the TRAIN-label shuffle (AUROC ~ 0.5 asserted), seed determinism (asserted), the ``hidden`` prefix
   known answer (a hidden row's features equal its twin's up to ``params.prefix_match_rtol``, asserted, run on all of
   the model's test rows), and sum vs ablation (document-bootstrap AUROC difference).
5. Report (``analysis/calibration_report.py``, ``params.report``): the only figures. Per error type, a calibration
   diagram and a smECE / BSS / AUROC table of all methods, intervals over ``report.n_boot`` document resamples; the
   pooled values are asserted equal to the metrics above. With ``selection: null`` a second, CONTROL report
   (``control_calibration_*``) drops the strict subsets of clean parents, which are all valid, so that "clean parent"
   and "short" are not confounded with the label.

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
CORRUPTED_PARENTS = "corrupted_parents"
SELECTIONS = (None, CORRUPTED_PARENTS)   # the values of ``params.selection``
JUDGE_TASKS = ("conditional", "oneshot")
# Per-row columns of a subsets frame (besides the python lists ``members``, ``rows``).
SUBSET_COLS = ("subset_id", "parent_id", "pair_id", "doc", "cell", "error_type", "parent_valid", "size", "parent_size", "fields",
               "n_corrupted", "y", "kind")


@lru_cache(maxsize=None)
def sibling(name: str):
    """Load ``analysis/<name>.py`` as a module (once).

    Args:
        name: module file name without ``.py`` (``clef`` or ``jev``).

    Returns:
        The loaded module.
    """
    return er.load_sibling(name)


# ------------------------------------------------------------------------ subsets frame
def load_subsets(subsets_path: Path, parents_path: Path) -> dict:
    """The rows of a ``line/<split>_subsets.json`` as a frame, checked against the parent samples file.

    Args:
        subsets_path: the subsets file (``data/vrdu/line/<split>_subsets.json``).
        parents_path: the full-length parent samples file of the same split (``data/vrdu/line/<split>.json``).

    Returns:
        The frame, a dict with one numpy array per ``SUBSET_COLS`` entry (``parent_size``: the parent's number of
        fields; ``fields``: the members joined by ``,``; ``y``: the row's ``valid``; ``kind``: one of ``ROW_KINDS``),
        the lists ``members`` (field names in shown order) and ``rows`` (the raw rows), and ``index``
        (``{subset_id: position}``).

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


def cut(sub: dict, keep: np.ndarray, link_twins: bool = True) -> dict:
    """A frame cut to some rows, with ``index`` rebuilt and ``twin`` added.

    Args:
        sub: a frame (``load_subsets`` or ``cut``).
        keep: positions in ``sub`` of the rows to keep; the cut has them in the order given.
        link_twins: True for a set that holds the clean parents' rows; False for one that does not (then ``twin`` is
            None and the twin diagnostics are not computed, never indexed with a missing twin).

    Returns:
        A frame with the same keys as ``sub`` plus ``twin``: for each valid row of a corrupted parent, the position (in
        the cut) of the same subset (``pair_id``, ``fields``) of the clean parent; -1 for every other row.

    Raises:
        AssertionError: duplicate rows, or (``link_twins``) a context row whose twin is not among ``keep`` or is not valid.
    """
    keep = np.asarray(keep, dtype=np.int64)
    out = {c: sub[c][keep] for c in SUBSET_COLS}
    out["members"], out["rows"] = [sub["members"][i] for i in keep], [sub["rows"][i] for i in keep]
    out["index"] = {sid: i for i, sid in enumerate(out["subset_id"].tolist())}
    assert len(out["index"]) == len(keep), "duplicate rows in the cut"
    clean = {(pid, f): i for i, (pid, f, pv) in enumerate(zip(out["pair_id"].tolist(), out["fields"].tolist(), out["parent_valid"].tolist())) if pv}
    if not link_twins:
        out["twin"] = None
        return out
    out["twin"] = np.full(len(keep), -1, dtype=np.int64)
    for i in np.flatnonzero(np.isin(out["kind"], CONTEXT_KINDS)):
        key = (str(out["pair_id"][i]), str(out["fields"][i]))
        assert key in clean, f"{out['subset_id'][i]}: its clean-parent twin is not among the rows"
        assert out["y"][clean[key]], f"{out['subset_id'][i]}: its clean-parent twin is not valid"
        out["twin"][i] = clean[key]
    return out


# ------------------------------------------------------------------------ probe features
def select_rows(sub: dict, selection: str | None) -> tuple[np.ndarray, dict]:
    """The rows of a frame that make up a train / test set.

    Args:
        sub: a frame (``load_subsets``).
        selection: ``params.selection``; None keeps every row, ``corrupted_parents`` keeps the rows of corrupted parents.

    Returns:
        ``(positions, counts)``: the selected rows' ascending positions in ``sub``; and, for ``corrupted_parents``,
        ``n_corrupted_parent_rows`` and ``n_clean_parent_rows_dropped`` (empty for None).

    Raises:
        AssertionError: ``selection`` is neither value, or ``corrupted_parents`` selects no row.
    """
    assert selection in SELECTIONS, f"selection must be null or {CORRUPTED_PARENTS!r}, got {selection!r}"
    if selection is None:
        return np.arange(len(sub["y"])), {}
    pos = np.flatnonzero(~sub["parent_valid"])
    assert len(pos), "no row of a corrupted parent"
    return pos, {"n_corrupted_parent_rows": int(len(pos)), "n_clean_parent_rows_dropped": int(sub["parent_valid"].sum())}


def lm_member_spans(sub: dict, split: dict, data: dict, require_all: bool) -> tuple[np.ndarray, list[np.ndarray]]:
    """Each row's member spans in a model's run, from the parent's prompt.

    Every span value must equal the row's, a span's corrupted flag must be the row's, and the label rule applied
    to the row's values must give the row's ``valid``.

    Args:
        sub: a frame (``load_subsets`` or ``cut``).
        split: the model's loaded run (``entity_detection.load_split``), per-document centred.
        data: the line data of the run (``relation_detection.load_line_data``).
        require_all: True if every row's parent must be in the run (a run restricted to some documents: False).

    Returns:
        ``(positions, spans)``: the positions in ``sub`` of the rows whose parent is in the run, and for each such row
        the array of its member spans' indices into ``split``, in shown order.

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
    """One feature vector per subset.

    Args:
        x: ``(n_spans, hidden)`` activations of a run.
        rows: for each subset, the indices of its member spans in ``x`` in shown order (at least two).
        feature: ``sum`` (all members), ``last`` (the member shown last) or ``context`` (all members but the last).

    Returns:
        ``(len(rows), hidden)`` features.

    Raises:
        AssertionError: an unknown feature, a subset of fewer than two members, or non-finite features.
    """
    assert feature in FEATURES, f"unknown feature {feature!r}; one of {FEATURES}"
    assert all(len(r) >= 2 for r in rows), "a subset of fewer than two members"
    combine = {"sum": lambda r: x[r].sum(axis=0),
               "last": lambda r: x[r[-1]],
               "context": lambda r: x[r[:-1]].sum(axis=0)}[feature]
    f = np.stack([combine(r) for r in rows])
    assert f.shape == (len(rows), x.shape[1]) and np.isfinite(f).all(), (feature, f.shape)
    return f


def prefix_match_dev(f: np.ndarray, ev: dict) -> tuple[float, float]:
    """The prefix known answer: a ``hidden`` row's members' prompt prefixes equal its twin's, so their features must too.
    A difference is bf16 run-to-run noise (small) or a misaligned span (large).

    Args:
        f: ``sum`` features (``subset_features``) of the rows of ``ev``.
        ev: a frame with twins (``cut(..., link_twins=True)``).

    Returns:
        ``(largest relative difference ||f(row) - f(twin)|| / ||f(twin)||, share of rows exactly equal)`` over the
        ``hidden`` rows.

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
    """One Clef / Jev run on the test subsets.

    Args:
        spec: the ``params.judges`` entry (``judge``: clef | jev, ``kind``: conditional | oneshot, ``run_id``).
        runs_root: ``$RUNS_ROOT``.
        sub_te: the whole test frame (``load_subsets``).

    Returns:
        ``{idx, probs, n_excluded, extra}``: the positions in ``sub_te`` of the rows it scored, its ``P(true)`` for
        each, the number of over-length rows it excluded, and Jev's resolution metrics (empty for Clef).

    Raises:
        AssertionError: run.json status or task is wrong, counts differ from ``metrics.json``, the run is not of a
            ``*-subsets.json`` samples file, ids are duplicated or not test subsets', a stored label / document /
            cell / size differs from the subsets file's, or ``P(true)`` is outside [0, 1], is not the softmax of
            Clef's stored logits, or is not on Jev's 0.01 grid.
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
    """Per-stratum metrics of an unbalanced row set.

    Args:
        y: True where the row is valid.
        probs: ``P(valid)`` per row.
        threshold: ``P(valid)`` above this is called valid.
        strata: each row's stratum name.

    Returns:
        For each stratum value ``s``: ``n_<s>``, ``valid_rate_<s>``, ``accuracy_<s>``, ``mean_p_valid_<s>``;
        ``auroc_<s>`` if it has both classes, ``detect_<s>`` (invalid called invalid) if it has an invalid row,
        ``valid_acc_<s>`` (valid called valid) if it has a valid one.
    """
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
    """Percentile interval of ``AUROC(pa) - AUROC(pb)`` over ``n_boot`` resamples of documents, shared by both.

    Args:
        y: True where the row is valid.
        pa, pb: the two methods' ``P(valid)`` per row.
        docs: each row's document (a resample draws documents with replacement and keeps their rows whole).
        n_boot: number of resamples.
        ci_level: interval mass, in (0, 1).
        seed: seeds the resampling.

    Returns:
        ``(lower, upper)`` of the difference.

    Raises:
        AssertionError: a bad ``n_boot`` / ``ci_level``, mismatched shapes, or a resample with one class.
    """
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
    """The metrics every method gets, on the common rows.

    Args:
        label: the method's label (error messages only).
        ev: the common rows as a frame (``cut``). Its ``twin`` is None for a set without clean-parent rows, which gets
            no context diagnostic.
        probs: the method's ``P(valid)`` per row.
        p: the config's ``params`` (``decision_threshold``).
        seed: unused (kept so every method scorer has the same signature).

    Returns:
        ``n``, ``n_valid``, ``valid_rate``, ``n_docs``, ``mean_p_valid``; ``test_`` AUROC / Brier / BSS / accuracy / smECE;
        the ``test_`` strata of ``row_stratum_report`` by error type, cell, size, corrupted members, parent validity and
        kind; and, with twins, ``test_context_<kind>_*`` and ``test_twin_vs_context_<kind>_auroc`` per context kind.

    Raises:
        AssertionError: bad probabilities, or one class missing.
    """
    y, thr = ev["y"], p["decision_threshold"]
    assert probs.shape == y.shape and y.any() and (~y).any(), f"{label}: bad probabilities or one class missing"
    ref_rate = float(y.mean())
    m: dict[str, float] = {"n": int(len(y)), "n_valid": int(y.sum()), "valid_rate": ref_rate, "n_docs": len(set(ev["doc"].tolist())),
                           "mean_p_valid": float(probs.mean())}
    m.update(er.probe_metrics("test", y, probs, ref_rate, thr))
    m["test_smece"] = er.smece(probs, y)
    for strata in (ev["error_type"], ev["cell"], np.char.add("size", ev["size"].astype(str)),
                   np.char.add("ncorr", ev["n_corrupted"].astype(str)),
                   np.where(ev["parent_valid"], "parent_clean", "parent_invalid"), ev["kind"]):
        m.update({f"test_{k}": v for k, v in row_stratum_report(y, probs, thr, strata).items()})
    # Context diagnostic: valid rows of an invalid parent against the same subset of the clean parent (a set with no
    # clean-parent rows, ``twin`` None, has none).
    for kind in CONTEXT_KINDS if ev["twin"] is not None else ():
        sel = np.flatnonzero(ev["kind"] == kind)
        if not len(sel):
            continue
        twin = ev["twin"][sel]
        m[f"test_context_{kind}_mean_p_valid"], m[f"test_context_{kind}_twin_mean_p_valid"] = float(probs[sel].mean()), float(probs[twin].mean())
        m[f"test_context_{kind}_valid_acc"] = float((probs[sel] > thr).mean())
        m[f"test_twin_vs_context_{kind}_auroc"] = float(roc_auc_score(
            np.concatenate([np.zeros(len(sel), bool), np.ones(len(sel), bool)]), np.concatenate([probs[sel], probs[twin]])))
    return m


def structure_baseline(sub_tr: dict, ev: dict, p: dict, seed: int) -> dict:
    """What structure alone predicts: a logistic regression on the one-hot field set, trained on the train rows.
    Reported, not asserted (the field set predicts the label on these unbalanced sets).

    Args:
        sub_tr: the (selected) train frame.
        ev: the common test rows as a frame.
        p: the config's ``params`` (``lr_*``, ``decision_threshold``).
        seed: seeds the fit.

    Returns:
        ``test_baseline_fields_`` AUROC, Brier, Brier reference, BSS and accuracy on ``ev``.
    """
    cats = sorted(set(sub_tr["fields"].tolist()) | set(ev["fields"].tolist()))
    clf = er.fit_logreg(er.onehot(sub_tr["fields"], cats), sub_tr["y"], p, seed)
    probs = clf.predict_proba(er.onehot(ev["fields"], cats))[:, 1]
    return er.probe_metrics("test_baseline_fields", ev["y"], probs, float(ev["y"].mean()), p["decision_threshold"])


def check_pairs(ev: dict) -> None:
    """Each ``pair_id`` has one error type and rows of both its corrupted parent and its clean twin.

    Args:
        ev: the common rows as a frame (``selection: null``).

    Raises:
        AssertionError: a pair with two error types or with rows of only one of its parents.
    """
    for pid in sorted(set(ev["pair_id"].tolist())):
        sel = ev["pair_id"] == pid
        assert len(set(ev["error_type"][sel].tolist())) == 1, f"{pid}: more than one error type"
        assert set(ev["parent_valid"][sel].tolist()) == {True, False}, f"{pid}: rows of only one of the pair's parents"


def control_mask(ev: dict) -> np.ndarray:
    """Rows of the control: every row of a corrupted parent (the full-length row and its strict subsets, valid or
    not) and the full-length rows of clean parents; i.e. all but the strict subsets of clean parents.

    Args:
        ev: the common rows as a frame (``selection: null``).

    Returns:
        A boolean mask over ``ev``'s rows.

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
           rows_ev: list[np.ndarray], ev: dict, prefix: tuple[float, float], art_dir: Path) -> tuple[dict, np.ndarray]:
    """One model's probe: logistic regression on the train subsets' ``sum`` features, scored on the common rows, with the
    ablations and the probe controls. Writes ``<label>_test_predictions.npz`` to ``art_dir``.

    Args:
        label: the model's label.
        layer: the layer the activations come from.
        p: the config's ``params``.
        seed: seeds the fits and the shuffles.
        tr: the model's centred train split (``entity_detection.load_split``).
        rows_tr: for each (selected) train subset, its member spans' indices in ``tr``.
        y_tr: each train subset's label (valid).
        te: the model's centred test split.
        rows_ev: for each common test row, its member spans' indices in ``te``.
        ev: the common test rows as a frame.
        prefix: ``prefix_match_dev`` over ALL the model's test rows (not only ``ev``'s, which may hold no hidden row).
        art_dir: directory of the run's artifacts.

    Returns:
        ``(metrics, probs)``: metrics (unprefixed; ``probe_train_time`` is a dict for the caller to pop) and the
        probe's ``P(valid)`` per row of ``ev``.

    Raises:
        AssertionError: a train set with one class, hidden rows further from their twins than ``prefix_match_rtol``,
            the train-label shuffle AUROC not ~0.5, or a same-seed refit that differs.
    """
    thr, y_te = p["decision_threshold"], ev["y"]
    ref_rate = float(y_te.mean())
    assert y_tr.any() and (~y_tr).any(), f"{label}: one class missing in the train subsets"
    assert len(rows_tr) == len(y_tr)
    t0 = time.perf_counter()
    f_tr = subset_features(tr["X"], rows_tr, "sum")
    feature_seconds = time.perf_counter() - t0
    f_te = subset_features(te["X"], rows_ev, "sum")
    dev, exact = prefix
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
            y_te, probs, abl_probs[a], ev["doc"], p["report"]["n_boot"], p["ci_level"], seed)
    if p["run_controls"]:
        auc_shuf, auc_shuf_std = er.train_label_shuffle_auroc(f_tr, y_tr, f_te, y_te, p, seed, p["n_train_label_shuffles"])
        metrics["control_train_label_shuffle_auroc"], metrics["control_train_label_shuffle_auroc_std"] = auc_shuf, auc_shuf_std
        assert abs(auc_shuf - 0.5) <= p["train_label_shuffle_auroc_tolerance"], (
            f"{label}: mean test AUROC of probes trained on shuffled TRAIN labels {auc_shuf:.4f} (std {auc_shuf_std:.4f}, "
            f"n={p['n_train_label_shuffles']}) is not ~0.5 -- something leaks")
        assert np.array_equal(probs, er.fit_logreg(f_tr, y_tr, p, seed).predict_proba(f_te)[:, 1]), f"{label}: same-seed refit gave different probabilities"
        metrics["control_seed_determinism"] = 1.0
    np.savez_compressed(art_dir / f"{label}_test_predictions.npz", subset_id=ev["subset_id"], prob_valid=probs, label_valid=y_te,
                        **{f"prob_valid_{a}": v for a, v in abl_probs.items()})
    return metrics, probs


# ----------------------------------------------------------------------------- report
def report(ev: dict, probs_by_method: dict[str, np.ndarray], metrics: dict, p: dict, seed: int, fig_dir: Path, out: Path) -> dict:
    """The calibration report on the common rows, then (``selection: null``) the control report on ``control_mask`` of them.

    Args:
        ev: the common rows as a frame.
        probs_by_method: each method's ``P(valid)`` per row of ``ev``.
        metrics: the metrics so far (the report's pooled values must equal them).
        p: the config's ``params`` (``selection``, ``report``, ``ci_level``).
        seed: seeds the resampling.
        fig_dir: directory of the figures.
        out: the run directory (tables and ``artifacts/calibration_predictions.npz``).

    Returns:
        The new ``metrics`` entries: the report's counts per error type and, for ``selection: null``, the control's row
        counts; for ``corrupted_parents``, ``control_report_skipped_corrupted_parents`` = 1 (no control report).
    """
    rep_cfg, ets = p["report"], list(p["report"]["error_type_names"])
    evset = {"y": ev["y"], "docs": ev["doc"], "error_type": ev["error_type"], "probs": probs_by_method}
    cr.check_evset(evset, list(rep_cfg["methods"]))
    cr.assert_matches_existing(evset, metrics)
    selection = p["selection"]
    if selection is None:   # a corrupted-parents set holds no clean parent, so no pair has both
        check_pairs(ev)
    cr.save_evset(out / "artifacts" / "calibration_predictions.npz", evset, subset_id=ev["subset_id"], parent_valid=ev["parent_valid"],
                  size=ev["size"], parent_size=ev["parent_size"], pair_id=ev["pair_id"])
    caption = {None: "Evaluation: all common test subset rows.",
               CORRUPTED_PARENTS: "Evaluation: all common test subset rows of corrupted parents (clean-parent rows excluded)."}[selection]
    m = cr.write_report(evset, rep_cfg, ets, p["ci_level"], seed, fig_dir, out, "calibration", caption)
    if selection is not None:
        m[f"control_report_skipped_{selection}"] = 1
        return m
    keep = control_mask(ev)   # after the main report: a control that cannot be formed must not cost the main figures
    n_drop = int((~keep).sum())
    assert n_drop == int((ev["parent_valid"] & (ev["size"] < ev["parent_size"])).sum())
    m["control_n_rows"], m["control_n_dropped_strict_clean_subsets"] = int(keep.sum()), n_drop
    cr.save_evset(out / "artifacts" / "calibration_predictions.npz", evset, subset_id=ev["subset_id"], parent_valid=ev["parent_valid"],
                  size=ev["size"], parent_size=ev["parent_size"], pair_id=ev["pair_id"], control=keep)   # + the control mask, for `calibration_report.py --control`
    m.update(cr.write_report(cr.subset_evset(evset, keep), rep_cfg, ets, p["ci_level"], seed, fig_dir, out, "control_calibration",
                             "Control: full-length rows (corrupted parent and clean twin) and strict subsets of corrupted parents only; "
                             "strict subsets of clean parents are dropped."))
    return m


# ----------------------------------------------------------------------------- main
def run(cfg: dict, runs_root: Path) -> dict:
    """Every model's probe and every direct judge on the same test subsets (see the module docstring).

    Args:
        cfg: the config (``id``, ``seed``, ``params``).
        runs_root: ``$RUNS_ROOT``; the model and judge runs are read from it and ``<id>/`` is written to it.

    Returns:
        The run's metrics (also written to ``metrics.json`` in the run directory and the figures directory).

    Raises:
        AssertionError: a failed gate (bad config, run, labels or control; see the helpers).
    """
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
    selection = p["selection"]
    pos_tr, counts_tr = select_rows(sub_tr, selection)
    pos_te, counts_te = select_rows(sub_te, selection)   # sub_te stays whole: the judges' ids are positions in it, and the prefix check reads all its rows
    if selection is not None:
        sub_tr = cut(sub_tr, pos_tr, link_twins=False)

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
        # the prefix known answer on every test row the model scored, selected or not (it needs the clean twins)
        prefix = prefix_match_dev(subset_features(te["X"], rows_te, "sum"), cut(sub_te, idx_te))
        lms[spec["label"]] = {"tr": tr, "te": te, "idx_tr": idx_tr, "rows_tr": rows_tr, "idx_te": idx_te, "rows_te": dict(zip(idx_te.tolist(), rows_te)),
                              "prefix": prefix}
    judges = {spec["label"]: load_judge(spec, runs_root, sub_te) for spec in p["judges"]}

    # The common set: test rows every method scored; the labels are the file's for all of them.
    selected = set(pos_te.tolist())
    scored = {k: sorted(set(i.tolist()) & selected)
              for k, i in {**{k: v["idx_te"] for k, v in lms.items()}, **{k: v["idx"] for k, v in judges.items()}}.items()}
    common = sorted(set.intersection(*(set(i) for i in scored.values())))
    assert common, "no test row was scored by every method"
    ev = cut(sub_te, np.array(common), link_twins=selection is None)
    metrics: dict[str, float] = {"n_test_subsets_file": len(sub_te["subset_id"]), "n_common_test_rows": len(common),
                                 "n_selected_train_rows": int(len(pos_tr)), "n_selected_test_rows": int(len(pos_te))}
    metrics.update({f"train_{k}": v for k, v in counts_tr.items()})
    metrics.update({f"test_{k}": v for k, v in counts_te.items()})
    for label, i in scored.items():
        metrics[f"{label}_n_test_rows"], metrics[f"{label}_n_dropped_rows"] = len(i), len(i) - len(common)
    metrics.update(structure_baseline(sub_tr, ev, p, seed))

    art_dir = out / "artifacts"
    probs_by_method: dict[str, np.ndarray] = {}
    probe_times: dict[str, dict] = {}
    for spec in p["models"]:
        label, m = spec["label"], lms[spec["label"]]
        rows_ev = [m["rows_te"][i] for i in common]
        m_metrics, lm_probs = run_lm(label, spec["layer"], p, seed, m["tr"], m["rows_tr"], sub_tr["y"][m["idx_tr"]], m["te"], rows_ev, ev, m["prefix"], art_dir)
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
    """Run the config at ``sys.argv[1]`` (``$RUNS_ROOT`` from the environment); its snapshot is copied to the run directory."""
    cfg_path = Path(sys.argv[1])
    cfg = yaml.safe_load(cfg_path.read_text())
    runs_root = Path(os.environ["RUNS_ROOT"])
    out = runs_root / cfg["id"]
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cfg_path, out / "config.snapshot.yaml")
    run(cfg, runs_root)


if __name__ == "__main__":
    main()
