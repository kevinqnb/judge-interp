"""Do the entities of one line item cluster, inside a corrupted prompt? Distances and PCA plots on the `line` relation runs.

Usage: ``uv run --python 3.12 python analysis/relation_detection_clustering.py analysis/configs/relation_detection_clustering/<id>.yaml``

Data. The corrupted ("invalid") full-row prompts of a ``relation_detection`` TEST run (one prompt per row, 2-5 entities, each
read at its closing quote, the same token everywhere). The metric's group label of an entity is its PROVENANCE line item:
``line_index`` for a native entity, ``source_line_index`` for a corrupted one. The figure's edges use VALUE-based membership: an
entity belongs to every line item of its document whose (source) field value equals the shown value under the label rule's
equivalence key (the source field is the shown field for a native entity, the donor field for a corrupted one, so a ``cross_item``
swapped value still belongs to the row's own item). Known answer (asserted for every entity): the provenance line item is among
the value matches.

Preprocessing (``params.preprocess``): ``raw``, or ``field_doc_mean``: from every entity, subtract the mean of the entities of
its (document, field) over ALL rows of the test run (valid and corrupted, unsupervised, no labels). This removes field identity
and the document offset; it does not remove an entity's value content. A group of 2 rows (an invalid row and its twin) would be
left as +-half their difference, so prompts with a field whose (document, field) group has fewer than
``params.min_rows_per_doc_field`` rows are EXCLUDED (counted in the metrics) for every preprocessing, so that raw and centred
results describe the same prompts. Prompts with an entity centred to exactly zero (norm <= ``params.zero_norm_tol``: identical in
every row of its group, mostly the first field) in ANY model are excluded the same way and counted (such a point would sit at the
origin; the sets differ between models, so the union keeps every model on the same prompts). (Subtracting the layer-0 vector would do nothing: every
read token is the same, so layer 0 is one constant.)

Metric (per prompt, per ``params.distances`` entry). Pairs of entities, *within* (same line item) or *between* (different
items). ``win_frac`` = share of (within, between) pair comparisons in which the within pair is closer (ties count 1/2; 0.5 =
no clustering). A prompt is *eligible* iff it has a within and a between pair; in ``intra_document`` rows only the native group
can supply a within pair (distinct donors), and ``cross_item`` rows are never eligible (asserted: every swapped value belongs to
the row's own item). Cells are reported separately with a bootstrap over DOCUMENTS; the eligible count per cell is reported.

Controls.
- Permutation: the provenance labels are shuffled within every prompt (``params.n_permutations``); the
  mean shuffled ``win_frac`` of a prompt is its own chance level (NOT 0.5 in general: with an outlier entity, which pairs are
  within or between changes with the shuffle), so ``win_excess`` = ``win_frac`` - the prompt's permutation mean is reported.
- Random vectors known answer: the same prompts with iid Gaussian vectors in place of the data; the pooled ``win_frac`` must be
  within ``params.random_tolerance`` of 0.5 (asserted when there are at least ``params.random_min_prompts`` prompts).
- Twin: each prompt's clean twin shows the same fields with all values native. ``twin_win_frac`` applies the invalid prompt's
  group labels to the twin's vectors (what field type / position alone give); ``contrast`` = mean over between pairs of
  (d_invalid - d_twin) minus the same over within pairs (> 0: corrupted pairs move apart more than within pairs);
  ``win_vs_twin`` = ``win_frac`` - ``twin_win_frac`` per prompt (paired), with a document-bootstrap interval: the part of the
  clustering that the corruption's binding adds over what the same fields give in the clean row.
- Layer-0 known answer: on the UNCENTRED layer-0 vectors (one constant) every distance ties, so ``win_frac`` is exactly 0.5 (asserted).
- Seed determinism: the permutation control is run twice and must be identical (asserted).

Figures: PCA of the (preprocessed) entity vectors of example prompts, one grid per model: rows = documents, columns = cells;
one PCA basis per document (fit on all its example prompts, so its panels are comparable). Colour = corrupted or not. Marker =
triangle if some corrupted entity is shown strictly earlier (a corruption is visible before it; for a corrupted entity, not
counting itself), circle otherwise. Dashed edges join every pair of entities that belong together in at least one ground-truth
line item (their value-based membership sets intersect): black if neither is corrupted (the clean item's relations), light grey if
one is (incidental relations).
Examples are chosen at random (seeded by the config ``seed``, never on a metric): ``n_example_docs`` documents are drawn from
those that have an invalid prompt with ``example_m`` fields in every ``example_cells`` cell, and per cell one such prompt. The
chosen ``pair_id`` s are written to ``artifacts/examples.json``.

Outputs go to ``$RUNS_ROOT/<id>/`` and ``analysis/figures/<id>/``.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

_here = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("analysis_relation_detection", _here / "relation_detection.py")
rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rr)  # sample matching, label rule, line data
er = rr.er  # shared loader
REPO_ROOT = rr.REPO_ROOT
FIGURES_DIR = _here / "figures"
EXTRA_META = ("line_index", "value", "pair_id", "cell", "source_field", "source_line_index")
DISTANCES = ("euclidean", "cosine")
PREPROCESSES = ("raw", "field_doc_mean")


# ------------------------------------------------------------------------- preprocessing
def center_field_doc(x: np.ndarray, docs: np.ndarray, fields: np.ndarray) -> tuple[np.ndarray, dict[tuple[str, str], int]]:
    """Subtract from every row the mean of the rows with its (document, field).

    Args:
        x: ``[n, d]`` vectors.
        docs, fields: ``[n]`` document id and field of every row.

    Returns:
        A new float64 ``[n, d]`` array, and the row count of every (document, field) group. A group of 2 rows (an invalid
        row and its twin) is left as +-half their difference, so callers must drop prompts of small groups.
    """
    assert x.ndim == 2 and x.shape[0] == len(docs) == len(fields), (x.shape, len(docs), len(fields))
    keys = np.array([f"{d}|{f}" for d, f in zip(docs.tolist(), fields.tolist())])
    uniq, inv = np.unique(keys, return_inverse=True)
    out = x.astype(np.float64)
    counts = {}
    for g in range(len(uniq)):
        idx = np.where(inv == g)[0]
        out[idx] -= out[idx].mean(axis=0)
        d, f = uniq[g].split("|")
        counts[(d, f)] = len(idx)
    return out, counts


def zero_rows(centred: np.ndarray, tol: float) -> np.ndarray:
    """Rows of a centred array with norm <= ``tol``: an entity identical in every row of its (document, field) group (same
    prefix, same value) is centred to exactly zero, which carries no information and would sit at the origin."""
    assert centred.ndim == 2 and tol >= 0
    return np.linalg.norm(centred, axis=1) <= tol


# ------------------------------------------------------------------------- membership
def membership(field: str, source_field: str, value: str, items: dict[int, dict], eq) -> frozenset:
    """The line items of one document whose ``source_field`` value equals ``value`` (equivalence key).

    Args:
        field: the shown field.
        source_field: the donor field of a corrupted entity, ``""`` for a native one (then ``field``).
        value: the shown value.
        items: ``{line_index: base row}`` of the document.
        eq: ``make_eq`` function.

    Returns:
        The matching ``line_index`` values.
    """
    sf = source_field or field
    key = eq(sf, value)
    assert key is not None
    return frozenset(li for li, it in items.items() if eq(sf, it[sf]) == key)


def entity_table(split: dict, groups: list[dict], items: dict[str, dict[int, dict]], eq) -> dict:
    """Per-entity membership of the given prompts.

    Args:
        split: loaded run (``doc_ids``, ``field``, ``field_position``, ``label_valid``, ``value``, ``source_field``,
            ``line_index``, ``source_line_index``).
        groups: ``rr.group_rows`` entries (invalid prompts only).
        items: ``{doc: {line_index: base row}}``.
        eq: ``make_eq`` function.

    Returns:
        ``{span index: {"prov": provenance line_index, "members": frozenset of matching line indices, "corrupted": bool}}``.

    Raises:
        AssertionError: the provenance line item is not among the matches, or a corrupted entity has no source field.
    """
    out = {}
    for g in groups:
        for i in g["rows"]:
            corrupted = not bool(split["label_valid"][i])
            sf = str(split["source_field"][i])
            assert bool(sf) == corrupted, (g["pair_id"], i, "source_field must be set exactly for corrupted entities")
            prov = int(split["source_line_index"][i]) if corrupted else int(split["line_index"][i])
            match = membership(str(split["field"][i]), sf, str(split["value"][i]), items[g["doc"]], eq)
            assert prov in match, f"{g['pair_id']} span {i}: provenance line item {prov} not in value matches {sorted(match)}"
            out[i] = {"prov": prov, "members": match, "corrupted": corrupted}
    return out


def fill_flags(positions: np.ndarray, corrupted: np.ndarray) -> np.ndarray:
    """Filled iff some corrupted entity is shown strictly earlier (``positions`` are shown orders)."""
    assert positions.shape == corrupted.shape and len(set(positions.tolist())) == len(positions)
    return np.array([bool((corrupted & (positions < p)).any()) for p in positions])


def links(members: list[frozenset], corrupted: np.ndarray) -> list[tuple[int, int, bool]]:
    """Ground-truth relations among a prompt's entities: every pair ``(i, j)``, ``i < j``, whose membership sets share a line
    item (value-based membership), with ``both_native`` = neither member is corrupted."""
    assert len(members) == len(corrupted)
    return [(i, j, not (corrupted[i] or corrupted[j])) for i in range(len(members)) for j in range(i + 1, len(members)) if members[i] & members[j]]


def union_exclusions(scans: dict[str, dict]) -> dict:
    """Prompts excluded for every model: those in a small (document, field) group, and those with a zero-centred entity in ANY model.

    Args:
        scans: ``{label: {"all": pair_ids of the invalid prompts, "small": pair_ids, "zero": pair_ids}}``.

    Returns:
        ``{"all", "small", "zero_union", "drop"}`` (sets of pair ids).

    Raises:
        AssertionError: the models do not have the same prompts, or their small-group sets differ (they depend on row counts only).
    """
    first = next(iter(scans.values()))
    for label, sc in scans.items():
        assert sc["all"] == first["all"], f"{label}: judged other prompts than the first model"
        assert sc["small"] == first["small"], f"{label}: small-group set differs (row counts are model independent)"
        assert sc["zero"] <= sc["all"]
    zero_union = set().union(*(sc["zero"] for sc in scans.values()))
    return {"all": first["all"], "small": first["small"], "zero_union": zero_union, "drop": first["small"] | zero_union}


# ------------------------------------------------------------------------- distances and the metric
def dist_matrix(v: np.ndarray, kind: str) -> np.ndarray:
    """All pairwise distances of the rows of ``v`` (float64)."""
    assert kind in DISTANCES, kind
    v = np.asarray(v, dtype=np.float64)
    if kind == "euclidean":
        return np.linalg.norm(v[:, None, :] - v[None, :, :], axis=-1)
    n = np.linalg.norm(v, axis=1)
    assert (n > 0).all(), "cosine distance of a zero vector"
    return 1.0 - (v @ v.T) / (n[:, None] * n[None, :])


def pair_sets(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Within (same label) and between (different label) pairs of the entities: ``(iw, jw, ib, jb)``, ``i < j``;
    None if the prompt has no within pair or no between pair."""
    iu, ju = np.triu_indices(len(lab), k=1)
    same = lab[iu] == lab[ju]
    if not same.any() or same.all():
        return None
    return iu[same], ju[same], iu[~same], ju[~same]


def win_frac(d: np.ndarray, ps: tuple) -> float:
    """Share of (within, between) comparisons with the within pair closer (ties 1/2)."""
    iw, jw, ib, jb = ps
    dw, db = d[iw, jw][:, None], d[ib, jb][None, :]
    return float((np.less(dw, db) + 0.5 * np.equal(dw, db)).mean())


def permuted_win_frac(d: np.ndarray, lab: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    """Mean ``win_frac`` over ``n_perm`` shuffles of the group labels (group sizes kept)."""
    vals = []
    for _ in range(n_perm):
        ps = pair_sets(rng.permutation(lab))
        assert ps is not None   # sizes are kept, so the prompt stays eligible
        vals.append(win_frac(d, ps))
    return float(np.mean(vals))


def prompt_stats(v: np.ndarray, v_twin: np.ndarray, lab: np.ndarray, kind: str, n_perm: int, rng: np.random.Generator) -> dict | None:
    """Metrics of one prompt; None if not eligible.

    Args:
        v, v_twin: ``[n, d]`` vectors of the prompt's entities and of the same fields in its clean twin (same order).
        lab: ``[n]`` group labels (provenance line index).
        kind: distance.
        n_perm: permutations of the control.
        rng: generator for the control.
    """
    ps = pair_sets(lab)
    if ps is None:
        return None
    d, dt = dist_matrix(v, kind), dist_matrix(v_twin, kind)
    iw, jw, ib, jb = ps
    out = {
        "win_frac": win_frac(d, ps), "mean_within": float(d[iw, jw].mean()), "mean_between": float(d[ib, jb].mean()),
        "twin_win_frac": win_frac(dt, ps),
        "contrast": float((d[ib, jb] - dt[ib, jb]).mean() - (d[iw, jw] - dt[iw, jw]).mean()),
        "perm_win_frac": permuted_win_frac(d, lab, n_perm, rng),
    }
    out["win_excess"] = out["win_frac"] - out["perm_win_frac"]
    out["win_vs_twin"] = out["win_frac"] - out["twin_win_frac"]
    return out


def doc_bootstrap_mean(vals: np.ndarray, docs: np.ndarray, n_boot: int, ci_level: float, rng: np.random.Generator) -> tuple[float, float, float]:
    """Mean of per-prompt ``vals`` and its percentile interval over resamples of whole documents."""
    assert len(vals) == len(docs) > 0 and np.isfinite(vals).all()
    ud, inv = np.unique(docs, return_inverse=True)
    s, n = np.bincount(inv, weights=vals, minlength=len(ud)), np.bincount(inv, minlength=len(ud))
    draws = rng.integers(0, len(ud), size=(n_boot, len(ud)))
    boot = s[draws].sum(1) / n[draws].sum(1)
    a = (1 - ci_level) / 2
    return float(vals.mean()), float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))


# ------------------------------------------------------------------------- examples and plots
def choose_examples(prompts: list[dict], cells: list[str], m: int, n_docs: int, rng: np.random.Generator) -> list[dict]:
    """The example prompts: ``[{"doc", "prompts": [one per cell, in ``cells`` order]}]``, drawn at random (never on a metric).

    A document qualifies iff it has an invalid prompt with ``m`` fields in every cell; ``n_docs`` qualifying documents are drawn
    without replacement (from the sorted ids), then per cell one of the document's such prompts (from the prompts sorted by
    ``ordinal``).
    """
    cand: dict[tuple, list[dict]] = defaultdict(list)
    for q in sorted(prompts, key=lambda q: (q["doc"], q["ordinal"])):
        if q["m"] == m and q["cell"] in cells:
            cand[(q["doc"], q["cell"])].append(q)
    docs = sorted({d for d, _ in cand if all((d, c) in cand for c in cells)})
    assert len(docs) >= n_docs, f"only {len(docs)} documents have a {m}-field invalid prompt in every one of {cells}; need {n_docs}"
    chosen = [docs[i] for i in rng.choice(len(docs), size=n_docs, replace=False)]
    return [{"doc": d, "prompts": [cand[(d, c)][rng.integers(len(cand[(d, c)]))] for c in cells]} for d in chosen]


def pca_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2-d PCA: ``(mean, components [2, d], explained variance ratio [2])``."""
    assert points.ndim == 2 and points.shape[0] >= 3
    mu = points.mean(0)
    _, s, vt = np.linalg.svd(points - mu, full_matrices=False)
    return mu, vt[:2], (s[:2] ** 2) / (s ** 2).sum()


def plot_examples(examples: list[dict], title: str, path: Path, pcfg: dict) -> None:
    """One grid: a row per example document, a column per cell; see the module docstring for the encoding."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n_r, n_c = len(examples), len(examples[0]["prompts"])
    fig, axes = plt.subplots(n_r, n_c, figsize=(pcfg["panel_size"] * n_c, pcfg["panel_size"] * n_r), squeeze=False)
    for r, ex in enumerate(examples):
        mu, comp, var = pca_basis(np.concatenate([p["vecs"] for p in ex["prompts"]]))
        for c, p in enumerate(ex["prompts"]):
            ax = axes[r, c]
            xy = (p["vecs"] - mu) @ comp.T
            for i, j, both_native in p["links"]:
                ax.plot(xy[[i, j], 0], xy[[i, j], 1], ls="--", lw=pcfg["link_width"], zorder=1,
                        color=pcfg["link_native_colour"] if both_native else pcfg["link_incidental_colour"])
            for j in range(len(xy)):
                ax.scatter(xy[j, 0], xy[j, 1], s=pcfg["marker_size"], zorder=2, marker="^" if p["visible"][j] else "o",
                           color=pcfg["corrupted_colour"] if p["corrupted"][j] else pcfg["native_colour"])
                ax.annotate(pcfg["field_names"][p["fields"][j]], xy[j], fontsize=pcfg["label_fontsize"], xytext=(3, 3), textcoords="offset points")
            ax.set_title(f"{p['cell']}  {ex['doc'][:8]}\n{p['note']}", fontsize=pcfg["title_fontsize"])
            ax.set_xlabel(f"PC1 ({var[0]:.0%})", fontsize=pcfg["label_fontsize"])
            ax.set_ylabel(f"PC2 ({var[1]:.0%})", fontsize=pcfg["label_fontsize"])
            ax.tick_params(labelsize=pcfg["label_fontsize"])
    nat, cor = pcfg["native_colour"], pcfg["corrupted_colour"]
    handles = [Line2D([], [], marker="o", ls="", color=nat, label="not corrupted"),
               Line2D([], [], marker="o", ls="", color=cor, label="corrupted"),
               Line2D([], [], marker="^", ls="", color="k", label="triangle: a corruption is shown before it"),
               Line2D([], [], marker="o", ls="", color="k", label="circle: none shown before it"),
               Line2D([], [], ls="--", color=pcfg["link_native_colour"], label="share a line item, both not corrupted"),
               Line2D([], [], ls="--", color=pcfg["link_incidental_colour"], label="share a line item, one corrupted (incidental)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=pcfg["label_fontsize"])
    fig.suptitle(title + "  (one PCA basis per row)", fontsize=pcfg["title_fontsize"])
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(path, dpi=pcfg["dpi"])
    plt.close(fig)


def scan_exclusions(spec: dict, p: dict, runs_root: Path) -> dict:
    """One model's exclusion sets (see ``union_exclusions``): the invalid prompts, those with a field in a (document, field)
    group below ``min_rows_per_doc_field``, and those with an entity centred to zero."""
    split = er.load_split(spec["run_id"], runs_root, spec["layer"], p["max_docs"], "relation_detection", EXTRA_META)
    centred, counts = center_field_doc(split["X"], split["doc_ids"], split["field"])
    zero = zero_rows(centred, p["zero_norm_tol"])
    inv = [g for g in rr.group_rows(split) if not g["valid"]]
    return {"all": {g["pair_id"] for g in inv},
            "small": {g["pair_id"] for g in inv if any(counts[(g["doc"], f)] < p["min_rows_per_doc_field"] for f in g["fields"])},
            "zero": {g["pair_id"] for g in inv if zero[g["rows"]].any()}}


# ------------------------------------------------------------------------- run
def line_items(split_name: str, docs: set[str]) -> dict[str, dict[int, dict]]:
    """``{doc: {line_index: base row}}`` of ``data/vrdu/line/<split>_base.json`` for the given documents."""
    items: dict[str, dict[int, dict]] = defaultdict(dict)
    for it in rr.read_base(REPO_ROOT / "data" / "vrdu" / "line" / f"{split_name}_base.json", rr.LINE_FIELDS, ["document_id", "line_index"]):
        assert it["line_index"] not in items[it["document_id"]]
        items[it["document_id"]][it["line_index"]] = it
    missing = docs - set(items)
    assert not missing, f"documents without line items: {sorted(missing)[:3]}"
    return items


def run(cfg: dict, runs_root: Path) -> dict:
    p, seed = cfg["params"], cfg["seed"]
    out = runs_root / cfg["id"]
    fig_dir = FIGURES_DIR / cfg["id"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    assert set(p["preprocess"]) <= set(PREPROCESSES) and set(p["distances"]) <= set(DISTANCES)
    assert p["plot_preprocess"] in p["preprocess"]
    metrics: dict[str, float] = {}
    examples_out: dict[str, list] = {}
    counts_done = False
    ref_eligible = None
    for spec in p["models"]:
        assert isinstance(spec["layer"], int) and spec["layer"] > 0, "layer must be a positive int (layer 0 is the constant null control)"
    scans = {spec["label"]: scan_exclusions(spec, p, runs_root) for spec in p["models"]}
    excl = union_exclusions(scans)
    for label, sc in scans.items():
        metrics[f"{label}_n_prompts_zero_entity_own"] = len(sc["zero"] - sc["small"])

    for spec in p["models"]:
        label = spec["label"]
        layer = spec["layer"]
        split = er.load_split(spec["run_id"], runs_root, layer, p["max_docs"], "relation_detection", EXTRA_META)
        data = rr.load_line_data(spec["run_id"], runs_root, REPO_ROOT)
        run_params = yaml.safe_load((runs_root / spec["run_id"] / "config.snapshot.yaml").read_text())["params"]
        assert run_params["split"] == "test", f"{spec['run_id']}: split {run_params['split']!r}, this analysis reads the test split"
        groups = rr.group_rows(split)
        by_pair = rr.match_samples(split, groups, data)
        items = line_items(run_params["split"], set(split["doc_ids"].tolist()))
        inv_groups = [g for g in groups if not g["valid"]]
        assert len(inv_groups) == len(by_pair)
        ent = entity_table(split, inv_groups, items, data["eq"])

        # preprocessing: fitted on every row of the run
        centred, _ = center_field_doc(split["X"], split["doc_ids"], split["field"])
        vecs = {k: v for k, v in {"raw": split["X"].astype(np.float64), "field_doc_mean": centred}.items() if k in p["preprocess"]}

        # per-prompt records (labels only: identical for every model / preprocessing / distance)
        prompts = []
        for g in inv_groups:
            rows = g["rows"]   # shown order
            lab = np.array([ent[i]["prov"] for i in rows])
            corr = np.array([ent[i]["corrupted"] for i in rows])
            twin = by_pair[g["pair_id"]][True]
            assert twin["fields"] == g["fields"], g["pair_id"]
            for i, t, f in zip(rows, twin["rows"], g["fields"]):   # native values unchanged in the twin, corrupted ones restored
                assert (str(split["value"][i]) == str(split["value"][t])) != bool(ent[i]["corrupted"]), (g["pair_id"], f, "twin value check")
            pos = np.array([int(split["field_position"][i]) for i in rows])
            prompts.append({"doc": g["doc"], "ordinal": g["ordinal"], "pair_id": g["pair_id"], "cell": g["cell"], "m": len(rows),
                            "fields": g["fields"], "rows": rows, "twin_rows": twin["rows"], "lab": lab, "corrupted": corr,
                            "visible": fill_flags(pos, corr), "eligible": pair_sets(lab) is not None,
                            "links": links([ent[i]["members"] for i in rows], corr)})
        n_all_prompts = len(prompts)
        assert {q["pair_id"] for q in prompts} == excl["all"]
        # excluded for EVERY model and preprocessing, so that all results describe the same prompts: prompts with a field whose
        # (document, field) mean rests on too few rows, and prompts with an entity centred to zero (identical in every row of its
        # group) in ANY model
        prompts = [q for q in prompts if q["pair_id"] not in excl["drop"]]
        assert prompts, "every prompt was excluded (min_rows_per_doc_field / zero entity)"
        metrics["n_prompts_excluded_small_group"] = len(excl["small"])
        metrics["n_prompts_excluded_zero_entity_union"] = len(excl["zero_union"] - excl["small"])
        assert n_all_prompts - len(excl["drop"]) == len(prompts)
        cells = sorted({q["cell"] for q in prompts})
        elig = {c: sum(q["eligible"] for q in prompts if q["cell"] == c) for c in cells}
        for c in cells:
            if c.startswith("cross_item"):
                assert elig[c] == 0, f"{c}: eligible prompts in a cell whose swapped values all belong to the row's own item"
        sig = [q["eligible"] for q in prompts]
        if ref_eligible is None:
            ref_eligible = sig
        else:
            assert ref_eligible == sig, f"{label}: eligibility differs from the first model's (different rows?)"
        if not counts_done:   # model independent
            for c in cells:
                metrics[f"n_prompts_{c.replace(':', '_')}"] = sum(q["cell"] == c for q in prompts)
                metrics[f"n_eligible_{c.replace(':', '_')}"] = elig[c]
            metrics["n_invalid_prompts"] = len(prompts)
            counts_done = True
        print(label, "prompts", len(prompts), "eligible per cell", elig)
        assert sum(elig.values()), "no eligible prompt"
        use = [q for q in prompts if q["eligible"]]

        z0 = er.load_split(spec["run_id"], runs_root, 0, p["max_docs"], "relation_detection", EXTRA_META)["X"].astype(np.float64)

        # metrics
        for pre, X in vecs.items():
            for kind in p["distances"]:
                def stats(rng):
                    rec = []
                    for q in use:
                        s_ = prompt_stats(X[q["rows"]], X[q["twin_rows"]], q["lab"], kind, p["n_permutations"], rng)
                        assert s_ is not None
                        rec.append({**s_, "doc": q["doc"], "cell": q["cell"], "pair_id": q["pair_id"]})
                    return rec

                rec = stats(np.random.default_rng(seed))
                assert rec == stats(np.random.default_rng(seed)), f"{label}/{pre}/{kind}: same-seed rerun differs"
                metrics["control_seed_determinism"] = 1.0
                docs = np.array([r["doc"] for r in rec])
                cellv = np.array([r["cell"] for r in rec])
                for name in ("all", *[c for c in cells if elig[c]]):
                    m = np.ones(len(rec), bool) if name == "all" else cellv == name
                    key = f"{label}_{pre}_{kind}_{name.replace(':', '_')}"
                    brng = np.random.default_rng(seed)
                    for stat in ("win_frac", "mean_within", "mean_between", "twin_win_frac", "contrast", "perm_win_frac", "win_excess", "win_vs_twin"):
                        v = np.array([r[stat] for r in rec])[m]
                        mean, lo, hi = doc_bootstrap_mean(v, docs[m], p["n_boot"], p["ci_level"], brng)
                        metrics[f"{key}_{stat}"] = mean
                        if stat in ("win_frac", "twin_win_frac", "contrast", "win_excess", "win_vs_twin"):
                            metrics[f"{key}_{stat}_ci_lo"], metrics[f"{key}_{stat}_ci_hi"] = lo, hi
                    metrics[f"{key}_n_prompts"] = int(m.sum())
                if pre == p["plot_preprocess"] and kind == p["distances"][0]:
                    np.savez_compressed(out / "artifacts" / f"{label}_prompt_metrics.npz", **{
                        k: np.array([r[k] for r in rec]) for k in rec[0]})

        # random-vector known answer: iid Gaussian vectors, same prompts and labels -> pooled win_frac ~ 0.5
        rrng = np.random.default_rng(seed)
        for kind in p["distances"]:
            w, w0 = [], []
            for q in use:
                d_ = split["X"].shape[1]
                w.append(prompt_stats(rrng.normal(size=(q["m"], d_)), rrng.normal(size=(q["m"], d_)), q["lab"], kind, 1, rrng)["win_frac"])
                # layer-0 known answer: uncentred, one constant vector -> every distance ties -> exactly 0.5
                w0.append(prompt_stats(z0[q["rows"]], z0[q["twin_rows"]], q["lab"], kind, 1, rrng)["win_frac"])
            metrics[f"control_random_{label}_{kind}_win_frac"] = float(np.mean(w))
            metrics[f"control_random_{label}_{kind}_n_prompts"] = len(w)
            if len(w) >= p["random_min_prompts"]:   # below this the mean of random draws is noise, not a check (n is reported)
                assert abs(np.mean(w) - 0.5) <= p["random_tolerance"], f"{label}/{kind}: random-vector win_frac {np.mean(w):.3f} is not ~0.5"
            assert set(w0) == {0.5}, f"{label}/{kind}: layer-0 (constant) win_frac {set(w0)} != 0.5"
            metrics[f"control_layer0_{label}_{kind}_win_frac"] = 0.5
        del z0

        # figures (the same random draw for every model)
        sel = choose_examples(prompts, p["example_cells"], p["example_m"], p["n_example_docs"], np.random.default_rng(seed))
        X = vecs[p["plot_preprocess"]]
        for ex in sel:
            for q in ex["prompts"]:
                q["vecs"] = X[q["rows"]]
                ps = pair_sets(q["lab"])
                q["note"] = (f"win={win_frac(dist_matrix(q['vecs'], p['distances'][0]), ps):.2f}" if ps else "not eligible") + f", {len(q['links'])} links"
        plot_examples(sel, f"{label} layer {layer} ({p['plot_preprocess']})", fig_dir / f"{label}_pca_examples.png", p["plot"])
        examples_out[label] = [{"doc": ex["doc"], "pair_ids": [q["pair_id"] for q in ex["prompts"]]} for ex in sel]

    assert all(v == next(iter(examples_out.values())) for v in examples_out.values()), "models drew different examples"
    (out / "artifacts" / "examples.json").write_text(json.dumps(examples_out, indent=2) + "\n")
    metrics_text = json.dumps(metrics, indent=2) + "\n"
    (out / "metrics.json").write_text(metrics_text)
    (fig_dir / "metrics.json").write_text(metrics_text)
    print(metrics_text)
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
