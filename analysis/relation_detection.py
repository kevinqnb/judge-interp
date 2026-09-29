"""Listed relation-detection representations: PCA by source tuple, then a validity probe on
per-relation-type cosine similarities with calibration.

Usage: ``uv run --python 3.12 python analysis/relation_detection.py configs/<id>.yaml``
(every value lives in the config's ``params``; a missing key is a KeyError).

Reads the ``relation_detection_listed`` train and test runs named in the config
(``src/judge_interp/relation_detection.py``; layers ``[0, last]``, one row per list
entry = one row per distinct ``(entity_type, value)`` of a document) and does, in order:

1. **Preprocess** each split into ``X``:
   (a) ``X = rep_last - rep_0`` (layer 0 is the token-embedding output, so this removes
       the token-identity component);
   (b) subtract each document's mean row;
   (c) if ``subtract_entity_type_mean``: subtract each entity type's mean row, fit on the
       *train* split after (b) and applied to both splits (as in analysis/entity_detection.py).
2. **PCA** fit on the train ``X`` (all rows). For a few example train documents, each
   entity is coloured by the ground-truth source tuple it belongs to; an entity that belongs
   to more than one source is grey. Two panels per document: the train-wide PCA projected,
   and a PCA refit on that document alone.
   *Source* = a ``valid: true`` row of ``tuple_membership.json``; valid rows with identical
   ``entity_index`` are collapsed into one source (entity lists show values, never row
   identity, so the model cannot tell them apart). Entities join to sources on
   ``(entity_type, value)`` and the join is asserted complete in both directions.
3. **Datasets** ``Z^Tr``, ``Z^Tst`` (one per split, own RNG): for each of ``s`` samples draw a
   document, an entity ``e`` uniformly from its entities, a source tuple uniformly among
   the sources containing ``e`` (resampled if the tuple has < 2 non-null entities: nothing
   to subsample), ``k ~ U{1..m-1}`` other members to keep -> valid ``z``; then
   ``q ~ U{1..k}`` of the kept others are replaced by a *different-valued* entity of the
   same type from the same document -> ``z'``. ``z'`` is labelled invalid only if **no**
   source contains all its ``(type, value)`` pairs (a swap can land on another valid tuple,
   e.g. a shared channel/date with a sibling program). If no different-valued swap exists, or
   the swap lands on another valid tuple, the *whole sample is dropped* (valid entry too) and
   redrawn, so every emitted sample is one valid + one invalid entry over the same entity types
   (``s`` = number of such pairs). Otherwise decoy availability would depend on the type pattern
   and leak the label through which relation slots are filled. Drop rates are reported.
4. **Feature**: a ``C(5,2)=10``-dimensional vector, one slot per *relation type* (unordered pair
   of the 5 entity types, ``RELATION_TYPES``; the fixed type count, not the per-tuple ``m``, so
   a slot means the same relation in every sample). For each other entity ``o`` in ``z`` the
   cosine similarity of ``X[e]`` and ``X[o]`` goes in the slot of ``(type(e), type(o))``;
   relations not present in ``z`` stay 0. (Only pairs involving ``e`` are filled.) A valid ``z``
   and its ``z'`` have the same types, so the set of filled slots carries no label information.
   Logistic regression on the vector (train -> test), **positive class = valid**.
   Accuracy/precision/recall/F1/AUROC, smooth ECE + reliability diagram (the functions in
   analysis/entity_detection.py, reused unchanged).
5. **Controls**: shuffled-train-label AUROC ~ 0.5; same-seed resample + refit gives identical
   output; replacing X with Gaussian noise gives AUROC ~ 0.5; per-``k`` AUROC is reported
   (the number of filled slots equals ``k``).

Figures go to ``analysis/figures/<id>/``; ``metrics.json``, ``config.snapshot.yaml`` and
``artifacts/{train,test}_pairs.npz`` go to ``$RUNS_ROOT/<id>/``.
"""
from __future__ import annotations

import colorsys
import itertools
import importlib.util
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from judge_interp.prompts import LINE_FIELDS

_spec = importlib.util.spec_from_file_location("analysis_entity_detection", Path(__file__).parent / "entity_detection.py")
ed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ed)  # reused unchanged: preprocessing helpers, metrics, smECE, reliability plot, LR fit

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "Times New Roman", "DejaVu Serif"],
})

FIGURES_DIR = Path(__file__).parent / "figures"
MULTI_SOURCE_COLOR = "#9A9A9A"
MARKER_SIZE = 30
GOLDEN = 0.6180339887498949  # hue step maximising separation for an unknown, large source count
MAX_OTHERS = len(LINE_FIELDS) - 1
# Relation types = unordered pairs of entity types, in LINE_FIELDS order.
RELATION_TYPES = list(itertools.combinations(LINE_FIELDS, 2))
RELATION_INDEX = {rt: i for i, rt in enumerate(RELATION_TYPES)}


# --------------------------------------------------------------------------- loading
def load_split(run_id: str, runs_root: Path, max_docs: int | None) -> dict:
    """Load one run into ``X = rep_last - rep_0`` plus per-row metadata and tuple membership."""
    run_dir = runs_root / run_id
    manifest = json.loads((run_dir / "run.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert manifest["status"] == "success", f"{run_id}: status {manifest['status']!r}"
    assert manifest["task"] == "relation_detection_listed", f"{run_id}: task {manifest['task']!r}"
    tm = json.loads((run_dir / "artifacts" / "tuple_membership.json").read_text())

    with np.load(run_dir / "artifacts" / "span_representations.npz", allow_pickle=False) as z:
        layers = z["layers"].tolist()
        assert len(layers) == 2 and layers[0] == 0, f"{run_id}: layers {layers}, expected [0, last]"
        doc_ids, etype, value = z["doc_ids"], z["entity_type"], z["value"]
        n_all = doc_ids.shape[0]
        assert etype.shape == value.shape == (n_all,)
        keep = np.ones(n_all, dtype=bool)
        if max_docs is not None:
            keep = np.isin(doc_ids, list(dict.fromkeys(doc_ids.tolist()))[:max_docs])
        x = z[f"rep_{layers[1]}"][keep]
        rep0 = z["rep_0"][keep]
        assert x.dtype == np.float32 and rep0.shape == x.shape
        x -= rep0
        del rep0
        doc_ids, etype, value = doc_ids[keep], etype[keep], value[keep]
    if max_docs is None:
        assert len(x) == metrics["n_spans"], f"{run_id}: {len(x)} rows != metrics n_spans"
    assert np.isfinite(x).all(), f"{run_id}: non-finite values in X"
    docs = list(dict.fromkeys(doc_ids.tolist()))
    assert set(docs) <= set(tm), f"{run_id}: npz documents missing from tuple_membership"
    if max_docs is None:
        assert set(docs) == set(tm), f"{run_id}: npz documents != tuple_membership documents"
    tm = {d: tm[d] for d in docs}
    keys = list(zip(doc_ids.tolist(), etype.tolist(), value.tolist()))
    assert len(set(keys)) == len(keys), f"{run_id}: duplicate (doc, entity_type, value) rows"
    print(f"{run_id}: n={len(x)} docs={len(docs)} hidden={x.shape[1]} layers=[0,{layers[1]}]")
    return {"run_id": run_id, "X": x, "doc_ids": doc_ids, "entity_type": etype, "value": value,
            "docs": docs, "tuple_membership": tm}


# ------------------------------------------------------------------ sources / joins
def collapse_valid_sources(tuples: list[dict]) -> list[dict]:
    """One source per distinct ``entity_index`` among ``valid: true`` rows, sorted by smallest
    ``line_index``. Each: ``{"line_indices", "entity_index", "items"}``, ``items`` = frozenset of
    the non-null ``(field, value)`` pairs.
    """
    by_key: dict[tuple, dict] = {}
    for t in tuples:
        if not t["valid"]:
            continue
        key = tuple(sorted(t["entity_index"].items()))
        src = by_key.setdefault(key, {"line_indices": [], "entity_index": t["entity_index"]})
        src["line_indices"].append(t["row_key"][1])
    assert by_key, "document has no valid tuples"
    sources = list(by_key.values())
    for s in sources:
        s["line_indices"].sort()
        s["items"] = frozenset((f, v) for f, v in s["entity_index"].items() if v is not None)
        assert set(s["entity_index"]) <= set(LINE_FIELDS)
    sources.sort(key=lambda s: s["line_indices"][0])
    return sources


def build_doc_info(split: dict) -> dict[str, dict]:
    """Per document: ``rows`` (row index per entity key), ``keys`` (row order), ``sources``,
    ``matches`` (source indices per key), ``by_type`` (type -> values). Join asserted complete
    both ways: every rendered entity is in >= 1 source, every source item was rendered.
    """
    rows_by_doc: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(split["doc_ids"].tolist()):
        rows_by_doc[d].append(i)
    info = {}
    for doc in split["docs"]:
        rows = rows_by_doc[doc]
        keys = [(split["entity_type"][i], split["value"][i]) for i in rows]
        key_row = {k: r for k, r in zip(keys, rows)}
        sources = collapse_valid_sources(split["tuple_membership"][doc])
        matches = [[j for j, s in enumerate(sources) if k in s["items"]] for k in keys]
        for k, m in zip(keys, matches):
            assert m, f"{doc}: rendered entity {k} is in no valid source -- join is broken"
        rendered = set(keys)
        for s in sources:
            assert s["items"] <= rendered, f"{doc}: source {s['line_indices']} has unrendered items -- join is broken"
        by_type: dict[str, list[str]] = defaultdict(list)
        for t, v in keys:
            by_type[t].append(v)
        info[doc] = {"key_row": key_row, "keys": keys, "sources": sources, "matches": matches, "by_type": dict(by_type)}
    return info


# ------------------------------------------------------------------ preprocessing
def preprocess(tr: dict, te: dict, subtract_entity_type_mean: bool) -> None:
    """In place: per-document centering, then (optionally) train-fit per-entity-type centering."""
    ed.center_per_document(tr["X"], tr["doc_ids"])
    ed.center_per_document(te["X"], te["doc_ids"])
    if subtract_entity_type_mean:
        means = ed.fit_type_means(tr["X"], tr["entity_type"])
        ed.subtract_type_means(tr["X"], tr["entity_type"], means)
        ed.subtract_type_means(te["X"], te["entity_type"], means)
        scale = float(tr["X"].std())
        for t in means:
            dev = float(np.abs(tr["X"][tr["entity_type"] == t].mean(axis=0)).max())
            assert dev <= ed.ZERO_MEAN_RTOL * scale, f"type {t!r} mean not ~0 after type centering: {dev}"
        print(f"per-entity-type centering: {sorted(means)}")


# -------------------------------------------------------------------------- sampler
def sample_pairs(info: dict[str, dict], docs: list[str], s: int, rng: np.random.Generator) -> dict:
    """Build ``Z`` (see module docstring, step 3). Returns arrays, one row per emitted z / z'.

    ``e_row`` / ``other_rows`` (padded with -1 to ``MAX_OTHERS``) index into the split's X.
    Asserts every valid z is a subset of some source and every emitted invalid z' of none.
    """
    out: dict[str, list] = defaultdict(list)
    n_pairs = 0
    stats = {"draws": 0, "rejected_m_lt_2": 0, "no_swap_possible": 0, "swap_still_valid": 0}
    while n_pairs < s:
        stats["draws"] += 1
        doc = docs[rng.integers(len(docs))]
        d = info[doc]
        e_idx = int(rng.integers(len(d["keys"])))
        src = d["sources"][d["matches"][e_idx][rng.integers(len(d["matches"][e_idx]))]]
        e_key = d["keys"][e_idx]
        members = [(f, src["entity_index"][f]) for f in LINE_FIELDS if src["entity_index"].get(f) is not None]
        m = len(members)
        if m < 2:
            stats["rejected_m_lt_2"] += 1
            continue
        assert e_key in members
        others = [x for x in members if x != e_key]
        k = int(rng.integers(1, m))  # 1..m-1
        kept = [others[i] for i in sorted(rng.permutation(len(others))[:k])]
        z = [e_key] + kept
        assert any(frozenset(z) <= sx["items"] for sx in d["sources"]), "valid z not inside any source"

        q = int(rng.integers(1, k + 1))  # 1..k
        swap_pos = sorted(rng.permutation(k)[:q])
        z2 = list(kept)
        n_changed = 0
        for pos in swap_pos:
            t, v = kept[pos]
            cands = [c for c in d["by_type"][t] if c != v]
            if cands:
                z2[pos] = (t, cands[rng.integers(len(cands))])
                n_changed += 1

        def emit(others_keys, label, n_ch):
            rows = [d["key_row"][kk] for kk in others_keys]
            out["doc"].append(doc)
            out["e_row"].append(d["key_row"][e_key])
            out["other_rows"].append(rows + [-1] * (MAX_OTHERS - len(rows)))
            out["label"].append(label)
            out["m"].append(m)
            out["k"].append(k)
            out["q"].append(q)
            out["n_changed"].append(n_ch)
            out["pair_id"].append(n_pairs)

        if n_changed == 0:
            stats["no_swap_possible"] += 1
            continue
        if any(frozenset([e_key] + z2) <= sx["items"] for sx in d["sources"]):
            stats["swap_still_valid"] += 1
            continue
        emit(kept, True, 0)
        emit(z2, False, n_changed)
        n_pairs += 1
    res = {k_: np.array(v) for k_, v in out.items()}
    res["label"] = res["label"].astype(bool)
    assert res["other_rows"].shape[1] == MAX_OTHERS
    assert int(res["label"].sum()) == s and int((~res["label"]).sum()) == s, "not one valid + one invalid per pair"
    assert stats["draws"] == s + stats["rejected_m_lt_2"] + stats["no_swap_possible"] + stats["swap_still_valid"]
    res["stats"] = stats
    return res


def relation_features(x: np.ndarray, etype: np.ndarray, e_row: np.ndarray, other_rows: np.ndarray,
                      absent_fill: float) -> np.ndarray:
    """``(f, present)``, both ``(n, len(RELATION_TYPES))`` (float64, bool): slot ``(type(e), type(o))`` = ``cos(X[e], X[o])`` for each
    other entity ``o``; absent relations are ``absent_fill`` (config ``absent_fill``). ``-1`` entries in ``other_rows`` are padding.
    """
    f = np.full((len(e_row), len(RELATION_TYPES)), absent_fill, dtype=np.float64)
    present = np.zeros(f.shape, dtype=bool)
    for i in range(len(e_row)):
        xe = x[e_row[i]].astype(np.float64)
        ne = np.linalg.norm(xe)
        assert ne > 0
        te = str(etype[e_row[i]])
        n_filled = 0
        for r in other_rows[i]:
            if r < 0:
                continue
            xo = x[r].astype(np.float64)
            no = np.linalg.norm(xo)
            assert no > 0
            to = str(etype[r])
            assert to != te
            idx = RELATION_INDEX[tuple(sorted((te, to), key=LINE_FIELDS.index))]
            assert not present[i, idx], "two others share a relation type"
            present[i, idx] = True
            f[i, idx] = xe @ xo / (ne * no)
            n_filled += 1
        assert n_filled == int((other_rows[i] >= 0).sum()) >= 1
    assert np.isfinite(f).all() and (np.abs(f[present]) <= 1 + 1e-9).all()
    return f, present


# ---------------------------------------------------------------------------- plots
def source_colors(n: int) -> list[tuple[float, float, float]]:
    return [colorsys.hsv_to_rgb((i * GOLDEN) % 1.0, 0.75, 0.85) for i in range(n)]


def plot_example_doc(doc: str, d: dict, x_doc: np.ndarray, global_scores: np.ndarray, max_legend: int,
                     seed: int, out_dir: Path) -> None:
    n_src = len(d["sources"])
    cols = source_colors(n_src)
    face = [cols[m[0]] if len(m) == 1 else MULTI_SOURCE_COLOR for m in d["matches"]]
    n_multi = sum(len(m) > 1 for m in d["matches"])
    local = PCA(n_components=2, svd_solver="full", random_state=seed).fit(x_doc)
    panels = [("train-wide PCA", global_scores, None), ("per-document PCA", local.transform(x_doc), local)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), constrained_layout=True)
    for ax, (name, sc, pca) in zip(axes, panels):
        ax.scatter(sc[:, 0], sc[:, 1], c=face, s=MARKER_SIZE, edgecolors="black", linewidths=0.3)
        ev = ("" if pca is None else f" ({pca.explained_variance_ratio_[0]:.1%})", "" if pca is None else f" ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set(xlabel="PC1" + ev[0], ylabel="PC2" + ev[1], title=name)
        ax.set_box_aspect(1)
    if n_src <= max_legend:
        h = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="black", markersize=7,
                        label=f"line {s['line_indices']}") for c, s in zip(cols, d["sources"])]
        h.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=MULTI_SOURCE_COLOR,
                            markeredgecolor="black", markersize=7, label="multiple sources"))
        axes[1].legend(handles=h, loc="best", fontsize=7, framealpha=0.9)
    fig.suptitle(f"doc {doc[:8]} | {len(face)} entities, {n_src} sources, {n_multi} multi-source (grey)")
    fig.savefig(out_dir / f"pca_sources_{doc[:8]}.png", dpi=150)
    plt.close(fig)


def pick_example_docs(info: dict, docs: list[str], n: int, lo: int, hi: int, seed: int) -> list[str]:
    cand = [d for d in docs if lo <= len(info[d]["sources"]) <= hi]
    assert len(cand) >= n, f"only {len(cand)} train docs with {lo}-{hi} sources, need {n}"
    rng = np.random.default_rng(seed)
    return [cand[i] for i in sorted(rng.choice(len(cand), size=n, replace=False))]


# ----------------------------------------------------------------------------- main
def fit_and_score(f_tr, y_tr, f_te, p, seed):
    clf = ed.fit_logreg(f_tr, y_tr, p, seed)
    return clf, clf.predict_proba(f_te)[:, 1]


def run(cfg: dict, runs_root: Path) -> dict:
    p, seed = cfg["params"], cfg["seed"]
    out = runs_root / cfg["id"]
    fig_dir = FIGURES_DIR / cfg["id"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)

    tr = load_split(p["train_run_id"], runs_root, p["max_docs"])
    te = load_split(p["test_run_id"], runs_root, p["max_docs"])
    assert not (set(tr["docs"]) & set(te["docs"])), "train/test documents overlap"
    info_tr, info_te = build_doc_info(tr), build_doc_info(te)
    preprocess(tr, te, p["subtract_entity_type_mean"])

    metrics: dict[str, float] = {}

    # -- PCA by source tuple (train) --
    pca = PCA(n_components=p["pca_n_components"], svd_solver=p["pca_svd_solver"], random_state=seed).fit(tr["X"])
    metrics["pca_evr_pc1"], metrics["pca_evr_pc2"] = (float(v) for v in pca.explained_variance_ratio_[:2])
    rows_by_doc = defaultdict(list)
    for i, dd in enumerate(tr["doc_ids"].tolist()):
        rows_by_doc[dd].append(i)
    ex_docs = pick_example_docs(info_tr, tr["docs"], p["n_example_docs"], p["example_min_sources"],
                                p["example_max_sources"], seed)
    print("example documents:", ex_docs)
    for doc in ex_docs:
        r = np.array(rows_by_doc[doc])
        plot_example_doc(doc, info_tr[doc], tr["X"][r], pca.transform(tr["X"][r]), p["max_legend_sources"], seed, fig_dir)
    del pca

    # -- Z^Tr / Z^Tst --
    def make(split_name, split, info, s, stream):
        z = sample_pairs(info, split["docs"], s, np.random.default_rng([seed, stream]))
        z["f"], z["present"] = relation_features(split["X"], split["entity_type"], z["e_row"], z["other_rows"], p["absent_fill"])
        st = z["stats"]
        for k_, v in st.items():
            metrics[f"{split_name}_{k_}"] = int(v)
        metrics[f"{split_name}_n_valid"] = int(z["label"].sum())
        metrics[f"{split_name}_n_invalid"] = int((~z["label"]).sum())
        metrics[f"{split_name}_no_swap_rate"] = st["no_swap_possible"] / st["draws"]
        metrics[f"{split_name}_swap_still_valid_rate"] = st["swap_still_valid"] / st["draws"]
        metrics[f"{split_name}_m_lt_2_reject_rate"] = st["rejected_m_lt_2"] / st["draws"]
        for j, rt in enumerate(RELATION_TYPES):
            for lab, name in ((True, "valid"), (False, "invalid")):
                mk = (z["label"] == lab) & z["present"][:, j]
                if mk.any():
                    metrics[f"{split_name}_cos_mean_{rt[0]}__{rt[1]}_{name}"] = float(z["f"][mk, j].mean())
        print(f"{split_name}: valid={metrics[f'{split_name}_n_valid']} invalid={metrics[f'{split_name}_n_invalid']} {st}")
        return z

    z_tr = make("train", tr, info_tr, p["s_train"], 0)
    z_te = make("test", te, info_te, p["s_test"], 1)
    y_tr, y_te = z_tr["label"], z_te["label"]
    for y in (y_tr, y_te):
        assert y.any() and (~y).any(), "one class missing"

    # -- logistic regression on d --
    clf, probs = fit_and_score(z_tr["f"], y_tr, z_te["f"], p, seed)
    for k_, v in ed.classification_metrics(y_te, probs, p["decision_threshold"]).items():
        metrics[f"test_{k_}"] = v
    metrics["train_auroc"] = float(roc_auc_score(y_tr, clf.predict_proba(z_tr["f"])[:, 1]))
    metrics["lr_intercept"] = float(clf.intercept_[0])
    for j, rt in enumerate(RELATION_TYPES):
        metrics[f"lr_coef_{rt[0]}__{rt[1]}"] = float(clf.coef_[0, j])
    for kk in range(1, MAX_OTHERS + 1):
        sel = z_te["k"] == kk
        if sel.any() and y_te[sel].any() and (~y_te[sel]).any():
            metrics[f"test_auroc_k{kk}"] = float(roc_auc_score(y_te[sel], probs[sel]))
            metrics[f"test_n_k{kk}"] = int(sel.sum())

    dd = ed.smooth_ece(probs, y_te, p["ece_n_boot"], seed)
    metrics["test_smece"], metrics["test_smece_ci_halfwidth"] = float(dd["ce"]), float(dd["ce_ci_width"])
    fig, ax = plt.subplots(figsize=(4.4, 4.4), constrained_layout=True)
    ed.plot_reliability(ax, dd, ed.VALID_COLOR)
    ax.set_title(f"Test reliability (smECE = {dd['ce']:.3f} ± {dd['ce_ci_width']:.3f})", fontsize=11)
    fig.savefig(fig_dir / "calibration.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    ys = np.arange(len(RELATION_TYPES))
    for lab, name, c, off in ((True, "valid", ed.VALID_COLOR, -0.15), (False, "invalid", ed.INVALID_COLOR, 0.15)):
        means = [np.nan if not ((z_te["label"] == lab) & z_te["present"][:, j]).any()
                 else z_te["f"][(z_te["label"] == lab) & z_te["present"][:, j], j].mean() for j in range(len(RELATION_TYPES))]
        ax.scatter(means, ys + off, color=c, s=30, label=name)
    ax.set_yticks(ys, [f"{a}–{b}" for a, b in RELATION_TYPES], fontsize=8)
    ax.set(xlabel="mean cosine(X[e], X[o]) where relation present (test)")
    ax.legend()
    fig.savefig(fig_dir / "relation_cosine_means.png", dpi=150)
    plt.close(fig)

    # presence-only baseline: the same slots as 0/1 indicators, no cosines. If the cosine probe's AUROC
    # is no better than this, the probe is using which relation types are present, not representations.
    _, pres_probs = fit_and_score(z_tr["present"].astype(float), y_tr, z_te["present"].astype(float), p, seed)
    metrics["test_auroc_presence_only"] = float(roc_auc_score(y_te, pres_probs))
    docs_te = np.unique(z_te["doc"])
    by_doc = {d_: np.flatnonzero(z_te["doc"] == d_) for d_ in docs_te}
    brng = np.random.default_rng([seed, 3])
    boots = {"cos": [], "presence": []}
    for _ in range(p["auroc_n_boot"]):
        sel = np.concatenate([by_doc[d_] for d_ in brng.choice(docs_te, len(docs_te))])
        if y_te[sel].any() and (~y_te[sel]).any():
            boots["cos"].append(roc_auc_score(y_te[sel], probs[sel]))
            boots["presence"].append(roc_auc_score(y_te[sel], pres_probs[sel]))
    for name, v_ in boots.items():
        metrics[f"test_auroc_{name}_ci_lo"], metrics[f"test_auroc_{name}_ci_hi"] = (float(x_) for x_ in np.percentile(v_, [2.5, 97.5]))

    # -- controls --
    if p["run_controls"]:
        tol = p["shuffled_label_auroc_tolerance"]
        y_perm = np.random.default_rng(seed).permutation(y_tr)
        _, pp = fit_and_score(z_tr["f"], y_perm, z_te["f"], p, seed)
        metrics["control_shuffled_auroc"] = float(roc_auc_score(y_te, pp))
        assert abs(metrics["control_shuffled_auroc"] - 0.5) <= tol, "shuffled-label AUROC not ~0.5 -- leakage"
        # same seed -> identical sampling and predictions
        z_te2 = sample_pairs(info_te, te["docs"], p["s_test"], np.random.default_rng([seed, 1]))
        f2, _ = relation_features(te["X"], te["entity_type"], z_te2["e_row"], z_te2["other_rows"], p["absent_fill"])
        assert np.array_equal(f2, z_te["f"]) and np.array_equal(z_te2["label"], y_te), "same-seed resample differs"
        _, probs2 = fit_and_score(z_tr["f"], y_tr, z_te["f"], p, seed)
        assert np.array_equal(probs, probs2), "same-seed refit differs"
        metrics["control_seed_determinism"] = 1.0
        # noise-X ablation
        rng = np.random.default_rng([seed, 2])
        xn_tr = rng.standard_normal(tr["X"].shape, dtype=np.float32)
        xn_te = rng.standard_normal(te["X"].shape, dtype=np.float32)
        dn_tr, _ = relation_features(xn_tr, tr["entity_type"], z_tr["e_row"], z_tr["other_rows"], p["absent_fill"])
        dn_te, _ = relation_features(xn_te, te["entity_type"], z_te["e_row"], z_te["other_rows"], p["absent_fill"])
        _, pn = fit_and_score(dn_tr, y_tr, dn_te, p, seed)
        metrics["control_noise_x_auroc"] = float(roc_auc_score(y_te, pn))
        assert abs(metrics["control_noise_x_auroc"] - 0.5) <= tol, "noise-X AUROC not ~0.5 -- label leaks through structure"

    for name, z in (("train", z_tr), ("test", z_te)):
        np.savez_compressed(out / "artifacts" / f"{name}_pairs.npz", **{k_: v for k_, v in z.items() if k_ != "stats"})
    np.savez_compressed(out / "artifacts" / "test_predictions.npz", prob_valid=probs, label_valid=y_te, d=z_te["f"])
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
