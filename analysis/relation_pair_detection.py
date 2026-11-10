"""Relation pair detection: a per-model linear probe on each entity pair's read-token activation, and the direct judges
(Clef, Jev), scored on the same balanced pairs of corrupted line items.

Usage: ``uv run --python 3.12 --extra analysis python analysis/relation_pair_detection.py analysis/configs/<type>/<id>.yaml``

Data. The unit is a pair: two of a full-length line sample's fields, i.e. a size-2 row of ``params.{train,test}_subsets``
(``data/vrdu/line/<split>_subsets.json``), identified by its ``subset_id`` and labelled by its own ``valid`` (the label
rule against the document's line items; a pair with a corrupted member can be valid). The LM runs
(``relation_pair_detection`` task) show every pair of a row in one prompt and read one activation per pair; the judge
runs (Clef / Jev ``conditional`` on ``experiments/clef_samples.py --source line_pairs``) see one pair per prompt, oriented
as the LM saw it (asserted).

The prompt variant is per model spec (``task``): ``relation_pair_detection`` (field names, three criteria) or
``relation_pair_values`` (values only, relatedness only; same rows, pair order and orientation). Two variants of one
model are two methods of the same analysis, scored on the same pairs; ``params.comparisons`` adds a document-bootstrap
interval of their AUROC difference.

1. Selection, train and test: pairs of CORRUPTED parents only (every invalid pair comes from a corrupted prompt, so
   clean-parent pairs would let a "this prompt is corrupted" detector pass as a pair detector), of the error types
   ``params.error_types``; with ``params.drop_field_free_mismatch``, minus the pairs whose label would differ if field
   assignment were ignored (``field_free_valid``: the two values are one item's under two other fields), so the label
   is the same for a prompt that shows field names and one that does not. Then randomly subsampled
   (the config's ``seed``) to as many valid as invalid pairs in every (error type, unordered field pair) cell
   (``relation_set_detection.balance_cells``; a one-class cell is dropped). The test side is drawn from the pairs every
   method scored, so every method is scored on exactly the same balanced pairs.
2. Probe: logistic regression on the pair's activation at ``layer`` (per-document mean-centred over the whole run).
3. Metrics, one code path for every method: smECE, Brier skill (reference = the set's valid rate, 0.5), AUROC,
   accuracy; overall and per error type, cell, field pair, corrupted members (0 / 1 / 2), pair position, parent size
   and ``after`` (a pair with a corrupted member was shown earlier in the LM prompt). A stratum with one class has no
   AUROC.
4. Controls. Structure baseline: a probe on the field-pair one-hot, AUROC exactly 0.5 (asserted: the set is balanced
   per field pair). Probes only: TRAIN-label shuffle (AUROC ~ 0.5, asserted), seed determinism (asserted). Layer 0 is
   constant over all pairs (asserted at load; the null control). Prefix known answer: a corrupted prompt's pairs shown
   before its first pair with a corrupted member have the same prompt prefix as the clean twin's pairs at the same
   positions (twins share the pair order), so their activations must match up to ``params.prefix_match_rtol``
   (asserted on every test run).
5. Report (``analysis/calibration_report.py``, ``params.report``): per error type, a calibration diagram and a table of
   all methods with document-bootstrap intervals; pooled values asserted equal to the metrics.

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
_spec = importlib.util.spec_from_file_location("analysis_relation_set_detection", _here / "relation_set_detection.py")
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)  # balancing, strata, the judge modules
er, cr, rr = rs.er, rs.cr, rs.rr

REPO_ROOT = rr.REPO_ROOT
FIGURES_DIR = _here / "figures"
TASKS = ("relation_pair_detection", "relation_pair_values")
JUDGES = ("clef", "jev")
PAIR_COLS = ("subset_id", "parent_id", "pair_id", "doc", "cell", "error_type", "parent_valid", "fields", "n_corrupted", "y")
RUN_META = ("row_ordinal", "sample_id", "pair_id", "cell", "row_error_type", "parent_valid", "parent_m", "subset_id",
            "pair_position", "field_a", "field_b", "corrupted_a", "corrupted_b", "n_corrupted", "label_valid")
LAYOUT_COLS = ("field_a", "field_b", "pair_position", "parent_m", "after")


# ------------------------------------------------------------------------ pairs frame
def pair_frame(subset_rows: list[dict], parent_rows: list[dict]) -> dict:
    """The size-2 subset rows as a frame, checked against their parents.

    Args:
        subset_rows: rows of a ``line/<split>_subsets.json``.
        parent_rows: rows of the same split's ``line/<split>.json``.

    Returns:
        One numpy array per ``PAIR_COLS`` entry (``fields``: the two fields in ``LINE_FIELDS`` order joined by ``+``;
        ``error_type``: the parent cell's error type, also for a clean parent; ``y``: the pair's ``valid``), plus
        ``index`` (``{subset_id: position}``).

    Raises:
        AssertionError: a parent without exactly C(m, 2) pairs, a pair that disagrees with its parent, a duplicate id,
            an invalid pair with no corrupted member, or a clean parent's pair that is corrupted or invalid.
    """
    parents = {r["sample_id"]: r for r in parent_rows}
    assert len(parents) == len(parent_rows), "parent sample_id is not unique"
    out: dict[str, list] = defaultdict(list)
    n_per_parent: dict[str, int] = defaultdict(int)
    for r in subset_rows:
        if r["size"] != 2:
            continue
        sid, par = r["subset_id"], parents[r["parent_id"]]
        assert (par["valid"], par["pair_id"], par["document_id"], par["cell"]) == (r["parent_valid"], r["pair_id"], r["document_id"], r["cell"]), (
            f"{sid}: disagrees with its parent")
        assert len(r["fields"]) == 2 and set(r["fields"]) <= set(par["fields"]) and all(r[f] == par[f] for f in r["fields"]), sid
        corrupted = set(par["invalid_fields"]) & set(r["fields"])
        assert set(r["invalid_fields"]) == corrupted, (sid, "corrupted members are not the parent's")
        assert r["valid"] or corrupted, (sid, "an invalid pair with no corrupted member")
        assert not (r["parent_valid"] and (corrupted or not r["valid"])), (sid, "a clean parent's pair is corrupted or invalid")
        n_per_parent[r["parent_id"]] += 1
        for c, v in zip(PAIR_COLS, (sid, r["parent_id"], r["pair_id"], r["document_id"], r["cell"], r["cell"].split(":")[0],
                                    r["parent_valid"], "+".join(f for f in rr.LINE_FIELDS if f in r["fields"]), len(corrupted), r["valid"])):
            out[c].append(v)
    for sid, par in parents.items():
        assert n_per_parent[sid] == par["m"] * (par["m"] - 1) // 2, f"{sid}: {n_per_parent[sid]} pairs for m = {par['m']}"
    res = {c: np.array(out[c]) for c in PAIR_COLS}
    res["y"], res["parent_valid"] = res["y"].astype(bool), res["parent_valid"].astype(bool)
    res["index"] = {s: i for i, s in enumerate(res["subset_id"].tolist())}
    assert len(res["index"]) == len(res["y"]), "subset_id is not unique"
    return res


def load_pairs(subsets_path: Path, parents_path: Path) -> dict:
    """``pair_frame`` of two data files, each refused if stale (``check_build``)."""
    for path in (subsets_path, parents_path):
        rr.check_build(path.parent.parent, f"{path.parent.name}/{path.name}")
    return pair_frame(json.loads(subsets_path.read_text(encoding="utf-8")), json.loads(parents_path.read_text(encoding="utf-8")))


def field_free_valid(fr: dict, subsets_path: Path, parents_path: Path) -> np.ndarray:
    """Per frame pair: would it be valid if field assignment were ignored?

    True iff some line item of the pair's document has its two values under any two distinct fields (by the label
    rule's ``equivalence_key``, each value keyed as the field it is compared with). Computed for corrupted parents;
    a clean parent's pairs are valid either way.

    Raises:
        AssertionError: a stale build, a pair of the file missing from the frame, or a valid pair that is not
            field-free valid (the field-aware rule must be the stricter one).
    """
    build = rr.check_build(parents_path.parent.parent, f"{parents_path.parent.name}/{parents_path.name}")
    bp = build["config_snapshot"]["params"]
    eq = rr.make_eq(rr.field_kinds(bp["cross_type"]["kinds"]), bp["dates"]["year_pivot"])
    items: dict[str, list[dict]] = defaultdict(list)
    for it in rr.read_base(parents_path.parent / f"{parents_path.stem}_base.json", rr.LINE_FIELDS, ["document_id", "line_index"]):
        items[it["document_id"]].append(it)
    keys = {d: rr.item_keys(its, rr.LINE_FIELDS, eq) for d, its in items.items()}
    out = fr["parent_valid"].copy()
    done = fr["parent_valid"].copy()
    for r in json.loads(subsets_path.read_text(encoding="utf-8")):
        if r["size"] != 2 or r["parent_valid"]:
            continue
        i = fr["index"][r["subset_id"]]
        a, b = (r[f] for f in r["fields"])
        out[i] = any(rr.matches_keys({g: eq(g, a), h: eq(h, b)}, keys[r["document_id"]])
                     for g, h in itertools.permutations(rr.LINE_FIELDS, 2))
        done[i] = True
    assert done.all(), f"{subsets_path}: {int((~done).sum())} frame pairs not in the file"
    assert not (fr["y"] & ~out).any(), "a valid pair is not field-free valid"
    return out


def cut(fr: dict, keep: np.ndarray) -> dict:
    """The rows ``keep`` (positions, in the order given) of a frame, with ``index`` rebuilt."""
    keep = np.asarray(keep, dtype=np.int64)
    out = {c: fr[c][keep] for c in fr if c != "index"}
    out["index"] = {s: i for i, s in enumerate(out["subset_id"].tolist())}
    assert len(out["index"]) == len(keep), "duplicate rows in the cut"
    return out


# ------------------------------------------------------------------------ LM runs
def after_corruption(run: dict) -> np.ndarray:
    """Per pair: True if a pair with a corrupted member was shown earlier in the same prompt."""
    key = np.char.add(np.char.add(run["doc_ids"], "|"), run["row_ordinal"].astype(str))
    out = np.zeros(len(key), dtype=bool)
    for k in np.unique(key):
        rows = np.flatnonzero(key == k)
        rows = rows[np.argsort(run["pair_position"][rows])]
        corrupt = run["n_corrupted"][rows] > 0
        out[rows] = np.r_[False, np.cumsum(corrupt)[:-1] > 0]
    return out


def load_pair_run(run_id: str, runs_root: Path, layer: int, max_docs: int | None, task: str) -> dict:
    """One pair run of ``task`` (one of ``TASKS``): ``X`` = the activations at ``layer`` (float32, NOT centred), the
    ``RUN_META`` columns, ``doc_ids``, ``after``, and the run's ``subsets_path`` / parents file.

    Raises:
        AssertionError: a failed run, another task, a layer not collected, layer 0 not constant (the read token
            leaks text), or (whole run) counts that differ from ``metrics.json``.
    """
    assert task in TASKS, f"task {task!r} is not one of {TASKS}"
    run_dir = runs_root / run_id
    manifest = json.loads((run_dir / "run.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert manifest["status"] == "success", f"{run_id}: status {manifest['status']!r}"
    assert manifest["task"] == task, f"{run_id}: task {manifest['task']!r}, expected {task!r}"
    params = yaml.safe_load((run_dir / "config.snapshot.yaml").read_text())["params"]
    with np.load(run_dir / "artifacts" / "span_representations.npz", allow_pickle=False) as z:
        layers = z["layers"].tolist()
        assert 0 in layers and layer in layers, f"{run_id}: layers {layers} lack 0 or {layer}"
        doc_ids = z["doc_ids"]
        rep0 = z["rep_0"]
        assert (rep0 == rep0[:1]).all(), f"{run_id}: layer-0 rows are not all identical -- the read token leaks pair text"
        del rep0
        keep = np.ones(len(doc_ids), dtype=bool)
        if max_docs is not None:
            keep = np.isin(doc_ids, list(dict.fromkeys(doc_ids.tolist()))[:max_docs])
        out = {"run_id": run_id, "doc_ids": doc_ids[keep], **{k: z[k][keep] for k in RUN_META}}
        out["X"] = z[f"rep_{layer}"][keep]
    assert out["X"].dtype == np.float32 and out["X"].shape[0] == len(out["doc_ids"]) and np.isfinite(out["X"]).all(), run_id
    for k in ("label_valid", "parent_valid", "corrupted_a", "corrupted_b"):
        out[k] = out[k].astype(bool)
    if max_docs is None:
        assert len(out["doc_ids"]) == metrics["n_spans"] and int(out["label_valid"].sum()) == metrics["n_valid_spans"], run_id
    assert len(np.unique(out["subset_id"])) == len(out["subset_id"]), f"{run_id}: a pair is read twice"
    out["after"] = after_corruption(out)
    spec = rr.load_dataset_spec(REPO_ROOT / params["dataset_config"])
    out["parents_path"] = REPO_ROOT / params["data_root"] / spec.split_path.format(split=params["split"])
    out["subsets_path"] = REPO_ROOT / params["data_root"] / params["subsets_path"].format(split=params["split"])
    out["seed"] = manifest_seed(run_dir)
    print(f"{run_id}: layer={layer} n={len(out['doc_ids'])} valid={int(out['label_valid'].sum())} "
          f"docs={len(set(out['doc_ids'].tolist()))} hidden={out['X'].shape[1]}")
    return out


def manifest_seed(run_dir: Path) -> int:
    """The seed a run was made with (``config.snapshot.yaml``): it fixed the pair order."""
    return int(yaml.safe_load((run_dir / "config.snapshot.yaml").read_text())["seed"])


def match_run(fr: dict, run: dict, required: np.ndarray) -> np.ndarray:
    """Each frame row's span index in ``run`` (-1 if the run has no such pair), with the run's labels checked.

    Args:
        required: bool per frame row: the run must hold it (e.g. every pair of the error types it collected).

    Raises:
        AssertionError: a run pair not in the frame, a disagreeing label / document / corrupted count / pair id, or
            a ``required`` frame pair the run lacks.
    """
    assert required.dtype == bool and required.shape == fr["y"].shape
    pos = np.full(len(fr["y"]), -1, dtype=np.int64)
    for j, sid in enumerate(run["subset_id"].tolist()):
        assert sid in fr["index"], f"{run['run_id']}: pair {sid!r} is not in the subsets file"
        pos[fr["index"][sid]] = j
    hit = pos >= 0
    for name, a, b in (("label", fr["y"], run["label_valid"]), ("document", fr["doc"], run["doc_ids"]),
                       ("corrupted count", fr["n_corrupted"], run["n_corrupted"]), ("pair id", fr["pair_id"], run["pair_id"]),
                       ("parent validity", fr["parent_valid"], run["parent_valid"])):
        assert (a[hit] == b[pos[hit]]).all(), f"{run['run_id']}: {name} differs from the subsets file"
    assert hit[required].all(), f"{run['run_id']}: {int((required & ~hit).sum())} required pairs of the file are not in the run"
    return pos


def prefix_known_answer(run: dict) -> tuple[float, int]:
    """Largest relative difference between a corrupted prompt's pre-corruption pairs and its clean twin's same positions.

    Twins share the pair order, so up to (excluding) a corrupted prompt's first pair with a corrupted member, both
    prompts are token-identical: the activations differ only by run-to-run noise. Raw (uncentred) activations.

    Returns:
        ``(max ||x - x_twin|| / ||x_twin||, number of pairs compared)``.

    Raises:
        AssertionError: twins whose pair layout differs, or no pair to compare.
    """
    by_prompt: dict[tuple[str, bool], np.ndarray] = {}
    key = list(zip(run["pair_id"].tolist(), run["parent_valid"].tolist()))
    groups: dict[tuple[str, bool], list[int]] = defaultdict(list)
    for i, k in enumerate(key):
        groups[k].append(i)
    for k, rows in groups.items():
        rows = np.asarray(rows)
        by_prompt[k] = rows[np.argsort(run["pair_position"][rows])]
    devs = []
    for (pid, valid), bad in by_prompt.items():
        if valid or (pid, True) not in by_prompt:
            continue
        good = by_prompt[(pid, True)]
        assert len(good) == len(bad), pid
        for c in ("field_a", "field_b"):
            assert (run[c][bad] == run[c][good]).all(), f"{pid}: twins show the pairs in different orders"
        first = int(np.argmax(run["n_corrupted"][bad] > 0))
        a, b = run["X"][bad[:first]], run["X"][good[:first]]
        devs += (np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)).tolist()
    assert devs, f"{run['run_id']}: no pre-corruption pair to compare (the known answer would be vacuous)"
    return float(max(devs)), len(devs)


# ------------------------------------------------------------------------ judges
def load_judge(spec: dict, runs_root: Path, fr: dict) -> dict:
    """One Clef / Jev ``conditional`` run on a ``line_pairs`` samples file.

    Returns:
        ``{idx, probs, primary, condition, n_excluded, extra}``: the frame positions of the pairs it scored, ``P(true)``
        for each, the field shown as the primary / the condition, the over-length exclusions, Jev's resolution metrics.

    Raises:
        AssertionError: as ``relation_set_detection.load_judge``, plus a run not of a ``-pairs.json`` file or a
            sample with more than one condition.
    """
    judge, run_id = spec["judge"], spec["run_id"]
    assert judge in JUDGES, spec
    mod = rs.sibling(judge)
    run_dir = runs_root / run_id
    manifest = json.loads((run_dir / "run.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert manifest["status"] == "success", f"{run_id}: status {manifest['status']!r}"
    assert manifest["task"] == mod.KINDS["conditional"], f"{run_id}: task {manifest['task']!r}"
    params = yaml.safe_load((run_dir / "config.snapshot.yaml").read_text())["params"]
    assert params["samples_path"].endswith("-pairs.json"), f"{run_id}: not a line_pairs run ({params['samples_path']})"
    cols = ["p_true", "label_valid", "sample_id", "doc_ids", "cell", "primary", "conditions"] + (
        ["logit_true", "logit_false"] if judge == "clef" else [])
    with np.load(run_dir / "artifacts" / f"{judge}_predictions.npz", allow_pickle=False) as z:
        d = {c: z[c] for c in cols}
    n = len(d["p_true"])
    assert all(len(v) == n for v in d.values()) and n == metrics["n_predictions"], run_id
    assert not any("|" in c for c in d["conditions"].tolist()), f"{run_id}: a sample with more than one condition"
    ids = d["sample_id"].tolist()
    assert len(set(ids)) == n, f"{run_id}: duplicate sample ids"
    unknown = [s for s in ids if s not in fr["index"]]
    assert not unknown, f"{run_id}: {len(unknown)} ids that are no pair of the file, e.g. {unknown[:2]}"
    idx = np.array([fr["index"][s] for s in ids], dtype=np.int64)
    for name, got, want in (("label", d["label_valid"].astype(bool), fr["y"][idx]), ("document", d["doc_ids"], fr["doc"][idx]),
                            ("cell", d["cell"], fr["cell"][idx])):
        assert (got == want).all(), f"{run_id}: stored {name} differs from the subsets file's"
    p = d["p_true"].astype(np.float64)
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all(), f"{run_id}: p_true outside [0, 1]"
    if judge == "clef":
        z64 = np.stack([d["logit_true"], d["logit_false"]], axis=1).astype(np.float64)
        assert np.allclose(p, np.exp(z64[:, 0] - np.logaddexp(z64[:, 0], z64[:, 1])), atol=1e-6), f"{run_id}: p_true is not the softmax of the stored logits"
    else:
        assert np.allclose(p / mod.P_GRID, np.round(p / mod.P_GRID), atol=mod.P_GRID_ATOL / mod.P_GRID), f"{run_id}: p_true is not on the {mod.P_GRID} grid"
    return {"idx": idx, "probs": p, "primary": d["primary"], "condition": d["conditions"],
            "n_excluded": 0 if judge == "jev" else metrics["n_excluded_overlong"],
            "extra": mod.resolution_metrics({"p_true": p, "determinism_max_abs_diff": float(metrics["determinism_max_abs_diff"])}) if judge == "jev" else {}}


def check_judge_orientation(label: str, layout: dict, judge: dict) -> None:
    """The judge saw every pair the LM also read in the LM's orientation: primary = ``field_a``, condition = ``field_b``."""
    shown = layout["pair_position"][judge["idx"]] >= 0
    assert shown.any(), f"{label}: no pair in common with the LM runs"
    assert (layout["field_a"][judge["idx"]][shown] == judge["primary"][shown]).all(), f"{label}: primary is not the LM's first field"
    assert (layout["field_b"][judge["idx"]][shown] == judge["condition"][shown]).all(), f"{label}: condition is not the LM's second field"


# ------------------------------------------------------------------------ scoring
def strata(ev: dict) -> list[np.ndarray]:
    """The per-pair strata every method is reported on."""
    return [ev["error_type"], ev["cell"], np.char.add("pair_", ev["fields"]), np.char.add("ncorr", ev["n_corrupted"].astype(str)),
            np.char.add("pos", ev["pair_position"].astype(str)), np.char.add("m", ev["parent_m"].astype(str)),
            np.where(ev["after"], "after_corruption", "before_corruption")]


def score_method(label: str, ev: dict, probs: np.ndarray, p: dict) -> dict:
    """The metrics every method gets, on the evaluated pairs (overall and per ``strata``)."""
    y, thr = ev["y"], p["decision_threshold"]
    assert probs.shape == y.shape and y.any() and (~y).any(), f"{label}: bad probabilities or one class missing"
    ref_rate = float(y.mean())
    m: dict[str, float] = {"n": int(len(y)), "n_valid": int(y.sum()), "valid_rate": ref_rate,
                           "n_docs": len(set(ev["doc"].tolist())), "mean_p_valid": float(probs.mean())}
    m.update(er.probe_metrics("test", y, probs, ref_rate, thr))
    m["test_smece"] = er.smece(probs, y)
    for s in strata(ev):
        m.update({f"test_{k}": v for k, v in rs.row_stratum_report(y, probs, thr, s).items()})
    return m


def structure_baseline(tr: dict, ev: dict, p: dict, seed: int) -> dict:
    """A logistic regression on the field-pair one-hot; AUROC exactly 0.5 on a set balanced per field pair (asserted)."""
    cats = sorted(set(tr["fields"].tolist()) | set(ev["fields"].tolist()))
    clf = er.fit_logreg(er.onehot(tr["fields"], cats), tr["y"], p, seed)
    probs = clf.predict_proba(er.onehot(ev["fields"], cats))[:, 1]
    m = er.probe_metrics("test_baseline_fields", ev["y"], probs, float(ev["y"].mean()), p["decision_threshold"])
    assert abs(m["test_baseline_fields_auroc"] - 0.5) <= er.EXACT_AUROC_ATOL, (
        f"structure baseline AUROC {m['test_baseline_fields_auroc']} != 0.5 on a set balanced per field pair -- the balancing is broken")
    return m


def check_balanced(fr: dict) -> None:
    """Exactly as many valid as invalid pairs in every (error type, field pair) cell."""
    for et, f in set(zip(fr["error_type"].tolist(), fr["fields"].tolist())):
        sel = (fr["error_type"] == et) & (fr["fields"] == f)
        assert 2 * int(fr["y"][sel].sum()) == int(sel.sum()), f"cell ({et}, {f}) is not balanced"


def run_lm(label: str, layer: int, p: dict, seed: int, x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, ev: dict,
           prefix: tuple[float, int], art_dir: Path) -> tuple[dict, np.ndarray]:
    """One model's probe on the pair activations, scored on ``ev``, with the probe controls.

    Raises:
        AssertionError: one train class, the prefix known answer above ``prefix_match_rtol``, the train-label shuffle
            AUROC not ~0.5, or a same-seed refit that differs.
    """
    y_te, thr = ev["y"], p["decision_threshold"]
    assert y_tr.any() and (~y_tr).any(), f"{label}: one class missing in the train pairs"
    dev, n_dev = prefix
    assert dev <= p["prefix_match_rtol"], f"{label}: pre-corruption pairs differ from their twins by {dev:.3g} (relative norm)"
    metrics: dict = {"layer": layer, "prefix_match_dev": dev, "prefix_match_n": n_dev, "n_train": int(len(y_tr)),
                     "n_train_valid": int(y_tr.sum())}
    clf, fit_seconds = er.timed_fit_logreg(x_tr, y_tr, p, seed)
    metrics["probe_train_time"] = {"feature_seconds": 0.0, "fit_seconds": fit_seconds, "n_train": int(x_tr.shape[0]),
                                   "n_features": int(x_tr.shape[1])}
    probs = clf.predict_proba(x_te)[:, 1]
    metrics["train_auroc"] = float(roc_auc_score(y_tr, clf.predict_proba(x_tr)[:, 1]))
    metrics.update(score_method(label, ev, probs, p))
    if p["run_controls"]:
        auc, std = er.train_label_shuffle_auroc(x_tr, y_tr, x_te, y_te, p, seed, p["n_train_label_shuffles"])
        metrics["control_train_label_shuffle_auroc"], metrics["control_train_label_shuffle_auroc_std"] = auc, std
        assert abs(auc - 0.5) <= p["train_label_shuffle_auroc_tolerance"], (
            f"{label}: mean test AUROC of probes trained on shuffled TRAIN labels {auc:.4f} (std {std:.4f}) is not ~0.5 -- something leaks")
        assert np.array_equal(probs, er.fit_logreg(x_tr, y_tr, p, seed).predict_proba(x_te)[:, 1]), f"{label}: same-seed refit differs"
        metrics["control_seed_determinism"] = 1.0
    np.savez_compressed(art_dir / f"{label}_test_predictions.npz", subset_id=ev["subset_id"], prob_valid=probs, label_valid=y_te)
    return metrics, probs


# ----------------------------------------------------------------------------- main
def run(cfg: dict, runs_root: Path) -> dict:
    """Every model's probe and every judge on the same balanced test pairs (see the module docstring)."""
    p, seed = cfg["params"], cfg["seed"]
    out, fig_dir = runs_root / cfg["id"], FIGURES_DIR / cfg["id"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    art_dir = out / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    labels = [s["label"] for s in (*p["models"], *p["judges"])]
    assert labels and len(set(labels)) == len(labels), f"duplicate or no method labels {labels}"
    assert p["models"], "no model: the pair layout (position, orientation) strata need an LM run"
    paths = {k: REPO_ROOT / p[k] for k in ("train_subsets", "test_subsets", "train_parents", "test_parents")}
    assert len(set(paths.values())) == 4, "train and test files overlap"
    fr_tr, fr_te = load_pairs(paths["train_subsets"], paths["train_parents"]), load_pairs(paths["test_subsets"], paths["test_parents"])
    assert not (set(fr_tr["doc"].tolist()) & set(fr_te["doc"].tolist())), "train/test documents overlap"
    error_types = list(p["error_types"])
    assert error_types and len(set(error_types)) == len(error_types), error_types
    assert set(error_types) <= set(fr_tr["error_type"].tolist()), f"error types {error_types} not all in the data"
    assert set(error_types) == set(p["report"]["error_type_names"]), "report error_type_names must be exactly params.error_types"
    assert isinstance(p["drop_field_free_mismatch"], bool)

    def candidates(fr: dict, sub: str, par: str) -> tuple[np.ndarray, int]:
        """Corrupted-parent pairs of ``error_types`` (minus the field-free mismatches if dropped), and the mismatch count."""
        cand = ~fr["parent_valid"] & np.isin(fr["error_type"], error_types)
        mismatch = cand & (fr["y"] != field_free_valid(fr, paths[sub], paths[par]))
        return (cand & ~mismatch if p["drop_field_free_mismatch"] else cand), int(mismatch.sum())

    cand_tr, n_mis_tr = candidates(fr_tr, "train_subsets", "train_parents")
    cand_te, n_mis_te = candidates(fr_te, "test_subsets", "test_parents")

    # Train side: corrupted parents, balanced per cell; the LM train runs must hold every selected pair.
    pos_tr, bal_tr = rs.balance_cells(fr_tr, np.flatnonzero(cand_tr), seed)
    tr = cut(fr_tr, pos_tr)
    check_balanced(tr)

    lms, layout, seeds = {}, None, set()
    for spec in p["models"]:
        layer = spec["layer"]
        assert isinstance(layer, int) and layer > 0, "layer must be a positive int (layer 0 is the constant null control)"
        runs = {}
        for side, fr, key in (("tr", fr_tr, "train"), ("te", fr_te, "test")):
            r = load_pair_run(spec[f"{key}_run_id"], runs_root, layer, p["max_docs"], spec["task"])
            assert (r["parents_path"], r["subsets_path"]) == (paths[f"{key}_parents"], paths[f"{key}_subsets"]), (
                f"{r['run_id']}: judged other files than this config's {key} files")
            seeds.add(r["seed"])
            required = np.isin(fr["error_type"], error_types) & (p["max_docs"] is None)
            r["pos"] = match_run(fr, r, required)
            runs[side] = r
        prefix = prefix_known_answer(runs["te"])   # on raw activations, before centring
        for r in runs.values():
            er.center_per_document(r["X"], r["doc_ids"])
        te_pos = runs["te"]["pos"]
        lay = {c: np.where(te_pos >= 0, runs["te"][c][np.maximum(te_pos, 0)], -1 if c in ("pair_position", "parent_m") else "")
               for c in LAYOUT_COLS if c != "after"}
        lay["after"] = np.where(te_pos >= 0, runs["te"]["after"][np.maximum(te_pos, 0)], False)
        if layout is None:
            layout = lay
        else:   # every LM saw the same prompts
            both = (layout["pair_position"] >= 0) & (lay["pair_position"] >= 0)
            for c in LAYOUT_COLS:
                assert (layout[c][both] == lay[c][both]).all(), f"{spec['label']}: pair layout {c} differs from the first model's"
        lms[spec["label"]] = {"runs": runs, "prefix": prefix}
    assert len(seeds) == 1, f"LM runs made with different seeds (pair orders): {sorted(seeds)}"
    judges = {spec["label"]: load_judge(spec, runs_root, fr_te) for spec in p["judges"]}
    for label, j in judges.items():
        check_judge_orientation(label, layout, j)

    # Common set: corrupted-parent test pairs every method scored; then balanced.
    selected = set(np.flatnonzero(cand_te).tolist())
    scored = {**{k: set(np.flatnonzero(v["runs"]["te"]["pos"] >= 0).tolist()) & selected for k, v in lms.items()},
              **{k: set(v["idx"].tolist()) & selected for k, v in judges.items()}}
    common = sorted(set.intersection(*scored.values()))
    assert common, "no test pair was scored by every method"
    evaluated, bal_te = rs.balance_cells(fr_te, np.array(common), seed)
    ev = cut(fr_te, evaluated)
    for c in LAYOUT_COLS:
        ev[c] = layout[c][evaluated]
    check_balanced(ev)

    metrics: dict = {"n_test_pairs_file": len(fr_te["y"]), "n_selected_test_pairs": len(selected), "n_common_test_pairs": len(common),
                     "n_field_free_mismatch_train": n_mis_tr, "n_field_free_mismatch_test": n_mis_te}
    metrics.update({f"train_{k}": v for k, v in bal_tr.items()})
    metrics.update({f"test_{k}": v for k, v in bal_te.items()})
    for label, s in scored.items():
        metrics[f"{label}_n_test_pairs"], metrics[f"{label}_n_dropped_pairs"] = len(s), len(s) - len(common)
    metrics.update(structure_baseline(tr, ev, p, seed))

    probs_by_method: dict[str, np.ndarray] = {}
    probe_times: dict[str, dict] = {}
    for spec in p["models"]:
        label, runs = spec["label"], lms[spec["label"]]["runs"]
        tr_pos = runs["tr"]["pos"][pos_tr]
        assert (tr_pos >= 0).all() or p["max_docs"] is not None
        has = tr_pos >= 0
        m, probs = run_lm(label, spec["layer"], p, seed, runs["tr"]["X"][tr_pos[has]], tr["y"][has],
                          runs["te"]["X"][runs["te"]["pos"][evaluated]], ev, lms[label]["prefix"], art_dir)
        probe_times[label] = m.pop("probe_train_time")
        probs_by_method[label] = probs.astype(np.float64)
        metrics.update({f"{label}_{k}": v for k, v in m.items()})
    er.write_probe_train_times(cfg["id"], probe_times)
    for spec in p["judges"]:
        label, j = spec["label"], judges[spec["label"]]
        at = {int(i): n for n, i in enumerate(j["idx"].tolist())}
        probs = j["probs"][[at[int(i)] for i in evaluated]]
        np.savez_compressed(art_dir / f"{label}_test_predictions.npz", subset_id=ev["subset_id"], prob_valid=probs, label_valid=ev["y"])
        probs_by_method[label] = probs.astype(np.float64)
        metrics.update({f"{label}_{k}": v for k, v in {**score_method(label, ev, probs, p), "n_excluded_overlong": j["n_excluded"], **j["extra"]}.items()})

    # Paired comparisons (e.g. two prompt variants of one model): AUROC(a) - AUROC(b) on the same pairs, with a
    # document-bootstrap interval (report n_boot resamples).
    for a, b in p["comparisons"]:
        assert a in probs_by_method and b in probs_by_method and a != b, (a, b)
        key = f"{a}_minus_{b}_test_auroc"
        metrics[key] = metrics[f"{a}_test_auroc"] - metrics[f"{b}_test_auroc"]
        metrics[f"{key}_ci_lo"], metrics[f"{key}_ci_hi"] = rs.doc_bootstrap_auroc_diff(
            ev["y"], probs_by_method[a], probs_by_method[b], ev["doc"], p["report"]["n_boot"], p["ci_level"], seed)

    rep = p["report"]
    evset ={"y": ev["y"], "docs": ev["doc"], "error_type": ev["error_type"], "probs": probs_by_method}
    cr.check_evset(evset, list(rep["methods"]))
    cr.assert_matches_existing(evset, metrics)
    cr.save_evset(art_dir / "calibration_predictions.npz", evset, subset_id=ev["subset_id"], fields=ev["fields"])
    metrics.update(cr.write_report(evset, rep, list(rep["error_type_names"]), p["ci_level"], seed, fig_dir, out, "calibration",
                                   "Evaluation: entity pairs of corrupted test line items, balanced valid / invalid per error type and field pair."))
    text = json.dumps(metrics, indent=2) + "\n"
    (out / "metrics.json").write_text(text)
    (fig_dir / "metrics.json").write_text(text)
    print(text)
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
