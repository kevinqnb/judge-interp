"""LEACE erasure of field type / position from entity representations, then the line-item clustering analysis (relation detection, ``line`` test).

Question: does erasing what an entity's representation says about its field and position leave a better "binding vector" -- do entities of one
line item end up closer together than before? Pipeline, per model:

1. Per-(document, field) mean centring of every row of a run (``relation_detection_clustering.center_field_doc``), as in the clustering analysis.
2. LEACE (Belrose et al. 2023) fitted on the TRAIN run's entities (never on the test run), concept Z = one-hot(field) + one-hot(field position),
   then applied to the centred test vectors. ``fit_on: centred`` fits on the centred train vectors; ``raw`` fits on the uncentred ones (centring is
   linear, so applying the raw-fit eraser after centring is the same as erasing and then centring). Note: after per-(document, field) centring the
   field one-hot carries no mean difference, so on centred data the erasure is in effect of position only; ``erasure_rank`` reports the rank erased.
3. The same per-prompt metrics as the clustering analysis (``win_frac``, clean-twin baseline, paired ``win_vs_twin``, Euclidean/cosine, document
   bootstrap) on the same prompts, plus the pooled connected-minus-unconnected distance gap (``relation_grouping_distances``; intra_document cells).
4. PCA grids of the fixed example prompts (those of an earlier clustering run), one file per variant.

Controls: LEACE's defining property on the train data (the cross-covariance with Z vanishes, asserted); a linear probe of position/field from the
transformed vectors falls to the majority-class rate (asserted for the erased variants); erasing a SHUFFLED concept (same rank, no information) leaves
``win_frac`` within a tolerance (asserted); the unerased variant reproduces an earlier clustering run's per-prompt ``win_frac`` (asserted); same-seed
determinism. Config: ``analysis/configs/relation_detection_leace/<id>.yaml``. Outputs: ``metrics.json``, ``artifacts/<label>_prompt_metrics_<variant>.npz``,
figures in ``analysis/figures/<id>/``. Nothing of the earlier runs' outputs is written.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

_here = Path(__file__).parent


def _load(name: str, file: str):
    s = importlib.util.spec_from_file_location(name, _here / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


rc = _load("analysis_relation_detection_clustering", "relation_detection_clustering.py")
gc = _load("analysis_relation_grouping_compare", "relation_grouping_compare.py")
gd = _load("analysis_relation_grouping_distances", "relation_grouping_distances.py")
gc.rc = gd.gc.rc = gd.rc = rc   # one clustering module everywhere
rr, er = rc.rr, rc.er
STATS = ("win_frac", "mean_within", "mean_between", "twin_win_frac", "contrast", "perm_win_frac", "win_excess", "win_vs_twin")
CI_STATS = ("win_frac", "twin_win_frac", "contrast", "win_excess", "win_vs_twin")


# ------------------------------------------------------------------------- LEACE
def one_hot(values: np.ndarray, classes: list) -> np.ndarray:
    """``[n, len(classes)]`` indicator matrix; every value must be one of ``classes``."""
    values = np.asarray(values)
    assert set(np.unique(values).tolist()) <= set(classes), (sorted(set(values.tolist())), classes)
    return np.stack([values == c for c in classes], axis=1).astype(np.float64)


def fit_leace(x: np.ndarray, z: np.ndarray, eig_rel_tol: float, rank_tol: float) -> dict:
    """Fit the LEACE eraser of the linear information about ``z`` in ``x``.

    ``x'  =  x - (x - mu) W P W^+``-style: with ``S`` = cov(x) (pseudo-inverse square roots ``W = S^-1/2``, ``W^+ = S^1/2`` over the eigenvalues above
    ``eig_rel_tol`` times the largest), ``C`` = cov(x, z) and ``U`` the left singular vectors of ``W C`` above ``rank_tol`` times the largest, the
    eraser is ``x' = x - ((x - mu) @ (W U)) @ (W^+ U).T``.

    Returns:
        ``{"mean", "A", "B", "rank"}`` (apply with ``apply_leace``).
    """
    assert x.ndim == 2 and z.ndim == 2 and len(x) == len(z) >= 2
    mu = x.mean(axis=0)
    xc = x - mu
    zc = z - z.mean(axis=0)
    n = len(x)
    cov = xc.T @ xc / n
    c = xc.T @ zc / n
    lam, v = np.linalg.eigh(cov)
    keep = lam > eig_rel_tol * lam.max()
    assert keep.any() and (lam[keep] > 0).all()
    v, lam = v[:, keep], lam[keep]
    w = (v / np.sqrt(lam)) @ v.T
    wp = (v * np.sqrt(lam)) @ v.T
    u, s, _ = np.linalg.svd(w @ c, full_matrices=False)
    r = int((s > rank_tol * s.max()).sum()) if s.max() > 0 else 0
    assert r >= 1, "the concept has no linear information in x"
    u = u[:, :r]
    return {"mean": mu, "A": w @ u, "B": wp @ u, "rank": r}


def apply_leace(e: dict, x: np.ndarray) -> np.ndarray:
    """Erase: ``x - ((x - mean) @ A) @ B.T`` (float64)."""
    x = np.asarray(x, dtype=np.float64)
    return x - ((x - e["mean"]) @ e["A"]) @ e["B"].T


def cross_cov_ratio(e: dict, x: np.ndarray, z: np.ndarray) -> float:
    """Frobenius norm of cov(erased x, z) relative to that of cov(x, z): 0 for a perfect eraser."""
    zc = z - z.mean(axis=0)
    before = np.linalg.norm((x - x.mean(axis=0)).T @ zc)
    xe = apply_leace(e, x)
    return float(np.linalg.norm((xe - xe.mean(axis=0)).T @ zc) / before)


# ------------------------------------------------------------------------- probes
def probe_accuracy(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray, c: float, max_iter: int, seed: int) -> tuple[float, float]:
    """``(test accuracy, test majority-class rate)`` of a multinomial logistic regression fitted on ``x_tr``."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=c, max_iter=max_iter, random_state=seed).fit(x_tr, y_tr)
    acc = float((clf.predict(x_te) == y_te).mean())
    _, counts = np.unique(y_te, return_counts=True)
    return acc, float(counts.max() / counts.sum())


def linear_r2(x_tr: np.ndarray, z_tr: np.ndarray, x_te: np.ndarray, z_te: np.ndarray, alpha: float) -> float:
    """Held-out R^2 of the best (ridge) linear predictor of the one-hot ``z`` from ``x``, summed over the columns.

    This is the property LEACE guarantees (zero cross-covariance: no linear predictor beats the constant under squared loss); it is 0 for an
    erased concept up to noise and can be slightly negative. 0-1 accuracy of a logistic probe is NOT covered by that guarantee.
    """
    from sklearn.linear_model import Ridge
    pred = Ridge(alpha=alpha).fit(x_tr, z_tr).predict(x_te)
    return float(1.0 - ((z_te - pred) ** 2).sum() / ((z_te - z_te.mean(axis=0)) ** 2).sum())


# ------------------------------------------------------------------------- prompts
def build_prompts(split: dict, data: dict, run_params: dict) -> tuple[list[dict], dict]:
    """The invalid prompts of a run, as in ``relation_detection_clustering.run`` (labels, twin rows, edges), and the matched pairs."""
    groups = rr.group_rows(split)
    by_pair = rr.match_samples(split, groups, data)
    items = rc.line_items(run_params["split"], set(split["doc_ids"].tolist()))
    inv = [g for g in groups if not g["valid"]]
    assert len(inv) == len(by_pair)
    ent = rc.entity_table(split, inv, items, data["eq"])
    prompts = []
    for g in inv:
        rows = g["rows"]
        lab = np.array([ent[i]["prov"] for i in rows])
        corr = np.array([ent[i]["corrupted"] for i in rows])
        twin = by_pair[g["pair_id"]][True]
        assert twin["fields"] == g["fields"], g["pair_id"]
        for i, t in zip(rows, twin["rows"]):
            assert (str(split["value"][i]) == str(split["value"][t])) != bool(ent[i]["corrupted"]), (g["pair_id"], "twin value check")
        pos = np.array([int(split["field_position"][i]) for i in rows])
        prompts.append({"doc": g["doc"], "ordinal": g["ordinal"], "pair_id": g["pair_id"], "cell": g["cell"], "m": len(rows), "fields": g["fields"],
                        "rows": rows, "twin_rows": twin["rows"], "lab": lab, "corrupted": corr, "visible": rc.fill_flags(pos, corr),
                        "eligible": rc.pair_sets(lab) is not None, "links": rc.links([ent[i]["members"] for i in rows], corr)})
    return prompts, by_pair


# ------------------------------------------------------------------------- run
def run(cfg: dict, runs_root: Path) -> dict:
    p, seed = cfg["params"], cfg["seed"]
    out = runs_root / cfg["id"]
    fig_dir = rc.FIGURES_DIR / cfg["id"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    names = [v["name"] for v in p["variants"]]
    assert len(set(names)) == len(names) and names[0] == "centred", "the first variant must be the unerased 'centred' one"
    assert all(v["fit_on"] in ("centred", "raw") for v in p["variants"][1:])
    fields_all = list(rr.LINE_FIELDS)
    positions_all = list(range(1, len(fields_all) + 1))
    examples_all = json.loads((runs_root / p["examples_from"] / "artifacts" / "examples.json").read_text())
    metrics: dict[str, float] = {}
    scans = {m["label"]: rc.scan_exclusions({"run_id": m["test_run_id"], "layer": m["layer"]}, p, runs_root) for m in p["models"]}
    excl = rc.union_exclusions(scans)
    metrics["n_prompts_excluded_small_group"] = len(excl["small"])
    metrics["n_prompts_excluded_zero_entity_union"] = len(excl["zero_union"] - excl["small"])
    ref_eligible = None
    for m in p["models"]:
        label, layer = m["label"], m["layer"]
        # ---- test run
        split = er.load_split(m["test_run_id"], runs_root, layer, None, "relation_detection", rc.EXTRA_META)
        data = rr.load_line_data(m["test_run_id"], runs_root, rc.REPO_ROOT)
        run_params = yaml.safe_load((runs_root / m["test_run_id"] / "config.snapshot.yaml").read_text())["params"]
        assert run_params["split"] == "test"
        prompts, _ = build_prompts(split, data, run_params)
        assert {q["pair_id"] for q in prompts} == excl["all"]
        prompts = [q for q in prompts if q["pair_id"] not in excl["drop"]]
        sig = [q["pair_id"] for q in prompts if q["eligible"]]
        if ref_eligible is None:
            ref_eligible = sig
        else:
            assert ref_eligible == sig, f"{label}: eligible prompts differ from the first model's"
        use = [q for q in prompts if q["eligible"]]
        metrics["n_invalid_prompts"], metrics["n_eligible_prompts"] = len(prompts), len(use)
        x_te, _ = rc.center_field_doc(split["X"], split["doc_ids"], split["field"])
        # ---- train run (fit only)
        tr = er.load_split(m["train_run_id"], runs_root, layer, p["train_max_docs"], "relation_detection", rc.EXTRA_META)
        x_tr_c, counts = rc.center_field_doc(tr["X"], tr["doc_ids"], tr["field"])
        ok = np.array([counts[(d, f)] >= p["min_rows_per_doc_field"] for d, f in zip(tr["doc_ids"].tolist(), tr["field"].tolist())]) & ~rc.zero_rows(x_tr_c, p["zero_norm_tol"])
        metrics[f"{label}_n_train_fit_rows"], metrics[f"{label}_n_train_rows"] = int(ok.sum()), len(ok)
        z_tr = np.concatenate([one_hot(tr["field"], fields_all), one_hot(tr["field_position"].astype(int), positions_all)], axis=1)
        rng_fit = np.random.default_rng(seed)
        vecs: dict[str, np.ndarray] = {"centred": x_te}
        erasers: dict[str, dict] = {}
        for v in p["variants"][1:]:
            src = x_tr_c[ok] if v["fit_on"] == "centred" else tr["X"][ok].astype(np.float64)
            z = z_tr[ok]
            if v["shuffle_concept"]:
                z = z[rng_fit.permutation(len(z))]
            e = fit_leace(src, z, p["eig_rel_tol"], p["rank_tol"])
            metrics[f"{label}_{v['name']}_erasure_rank"] = e["rank"]
            ratio = cross_cov_ratio(e, src, z)
            del src
            metrics[f"{label}_{v['name']}_train_cross_cov_ratio"] = ratio
            assert ratio <= p["cross_cov_tolerance"], f"{label}/{v['name']}: cross-covariance with the concept not erased (ratio {ratio:.2e})"
            vecs[v["name"]] = apply_leace(e, x_te)
            erasers[v["name"]] = e
            print(label, v["name"], "erased rank", e["rank"], "cross-cov ratio", f"{ratio:.1e}")
        # ---- probes of field position and field from the transformed vectors (fit on train rows, scored on every test entity):
        # ridge R^2 of the one-hot concept (what LEACE guarantees; asserted) and logistic accuracy vs the majority rate (reported only)
        prng = np.random.default_rng(seed)
        sub = prng.choice(np.flatnonzero(ok), size=min(p["probe_n_train"], int(ok.sum())), replace=False)
        targets = {"position": (tr["field_position"].astype(int), split["field_position"].astype(int), positions_all),
                   "field": (tr["field"].astype(str), split["field"].astype(str), fields_all)}
        for name in names:
            if name not in p["probe_variants"]:
                continue
            v = next(v for v in p["variants"] if v["name"] == name)
            erase = lambda x: x if name == "centred" else apply_leace(erasers[name], x)   # noqa: E731
            xs = erase(x_tr_c[sub])
            xr, xt = erase(x_tr_c[ok]), vecs[name]
            for target, (ytr, yte, classes) in targets.items():
                acc, maj = probe_accuracy(xs, ytr[sub], xt, yte, p["probe_c"], p["probe_max_iter"], seed)
                r2 = linear_r2(xr, one_hot(ytr[ok], classes), xt, one_hot(yte, classes), p["ridge_alpha"])
                metrics[f"{label}_{name}_probe_{target}_acc"], metrics[f"{label}_{name}_probe_{target}_majority"] = acc, maj
                metrics[f"{label}_{name}_probe_{target}_linear_r2"] = r2
                print(label, name, f"{target}: ridge R2 {r2:+.4f}; logistic acc {acc:.3f} (majority {maj:.3f})")
                if target == "position" and name == "centred":   # known answer needs something to erase
                    assert r2 >= p["r2_min_unerased"], f"{label}: unerased position R2 {r2:.4f} < {p['r2_min_unerased']}: nothing for LEACE to remove"
                if target == "position" and v["fit_on"] == "centred" and not v["shuffle_concept"]:
                    assert r2 <= p["r2_tolerance"], f"{label}/{name}: position still linearly predictable after erasure (ridge R2 {r2:.4f} > {p['r2_tolerance']})"
            del xs, xr
        del x_tr_c
        # ---- metrics
        for name, X in vecs.items():
            for kind in p["distances"]:
                def stats(rng):
                    rec = []
                    for q in use:
                        s_ = rc.prompt_stats(X[q["rows"]], X[q["twin_rows"]], q["lab"], kind, p["n_permutations"], rng)
                        assert s_ is not None
                        rec.append({**s_, "doc": q["doc"], "cell": q["cell"], "pair_id": q["pair_id"]})
                    return rec

                rec = stats(np.random.default_rng(seed))
                assert rec == stats(np.random.default_rng(seed)), f"{label}/{name}/{kind}: same-seed rerun differs"
                metrics["control_seed_determinism"] = 1.0
                docs = np.array([r["doc"] for r in rec])
                cellv = np.array([r["cell"] for r in rec])
                for cname in ("all", *sorted(set(cellv.tolist()))):
                    msk = np.ones(len(rec), bool) if cname == "all" else cellv == cname
                    key = f"{label}_{name}_{kind}_{cname.replace(':', '_')}"
                    brng = np.random.default_rng(seed)
                    for stat in STATS:
                        vals = np.array([r[stat] for r in rec])[msk]
                        mean, lo, hi = rc.doc_bootstrap_mean(vals, docs[msk], p["n_boot"], p["ci_level"], brng)
                        metrics[f"{key}_{stat}"] = mean
                        if stat in CI_STATS:
                            metrics[f"{key}_{stat}_ci_lo"], metrics[f"{key}_{stat}_ci_hi"] = lo, hi
                    metrics[f"{key}_n_prompts"] = int(msk.sum())
                if kind == p["distances"][0]:
                    np.savez_compressed(out / "artifacts" / f"{label}_prompt_metrics_{name}.npz", **{k: np.array([r[k] for r in rec]) for k in rec[0]})
                    if name == "centred":   # known answer: the unerased variant IS the earlier clustering run's field_doc_mean result
                        base = np.load(runs_root / p["baseline_prompt_metrics"][label], allow_pickle=True)
                        order = {pid: i for i, pid in enumerate(base["pair_id"].tolist())}
                        assert set(order) == {r["pair_id"] for r in rec}, f"{label}: the prompt set differs from the baseline run's"
                        for stat in ("win_frac", "twin_win_frac", "contrast", "mean_within", "mean_between"):
                            assert np.allclose([r[stat] for r in rec], [base[stat][order[r["pair_id"]]] for r in rec], rtol=1e-9), f"{label}: {stat} differs from the baseline run"
                        metrics[f"control_reproduces_baseline_{label}"] = 1.0
            # connected-minus-unconnected scaled distance gap (intra_document prompts), prompt-level bootstrap
            recs = []
            for q in prompts:
                if not q["cell"].startswith("intra_document"):
                    continue
                lk = {(i, j) for i, j, _ in q["links"]}
                raw = gd.pair_distances(X[q["rows"]], lk)
                if len(raw[0]) and len(raw[1]):
                    recs.append({"raw": raw, "scaled": gd.scaled(*raw)})
            est = gd.boot(recs, "scaled", p["n_boot"], p["ci_level"], np.random.default_rng(seed))
            mc, mu, _ = gd.pooled_diff(recs, "scaled", np.arange(len(recs)))
            metrics.update({f"{label}_{name}_scaled_gap": est[0], f"{label}_{name}_scaled_gap_ci_lo": est[1], f"{label}_{name}_scaled_gap_ci_hi": est[2],
                            f"{label}_{name}_scaled_mean_connected": mc, f"{label}_{name}_scaled_mean_unconnected": mu, f"{label}_{name}_scaled_gap_n_prompts": len(recs)})
        # known answer: the shuffled-concept erasure leaves win_frac where it was
        for v in p["variants"][1:]:
            if v["shuffle_concept"]:
                d = metrics[f"{label}_{v['name']}_{p['distances'][0]}_all_win_frac"] - metrics[f"{label}_centred_{p['distances'][0]}_all_win_frac"]
                metrics[f"control_shuffled_concept_{label}_{v['name']}_win_frac_change"] = d
                assert abs(d) <= p["shuffle_tolerance"], f"{label}/{v['name']}: erasing a shuffled concept changed win_frac by {d:+.3f}"
        # ---- figures: the fixed examples, one grid per variant
        sel_spec = examples_all[label]
        for name, X in vecs.items():
            sel = gc.fixed_examples(sel_spec)(prompts, None, None, None, None)
            for ex in sel:
                ex["prompts"] = [dict(q) for q in ex["prompts"]]
                for q in ex["prompts"]:
                    q["vecs"] = X[q["rows"]]
                    ps = rc.pair_sets(q["lab"])
                    q["note"] = (f"win={rc.win_frac(rc.dist_matrix(q['vecs'], p['distances'][0]), ps):.2f}" if ps else "not eligible") + f", {len(q['links'])} links"
            rc.plot_examples(sel, f"{label} layer {layer} ({name})", fig_dir / f"{label}_pca_{name}.png", p["plot"])
        del vecs
    text = json.dumps(metrics, indent=2) + "\n"
    (out / "metrics.json").write_text(text)
    (fig_dir / "metrics.json").write_text(text)
    k0 = p["distances"][0]
    for m in p["models"]:
        for name in names:
            b = f"{m['label']}_{name}_{k0}_all"
            print(m["label"], f"{name:<22} win {metrics[b + '_win_frac']:.3f} [{metrics[b + '_win_frac_ci_lo']:.3f},{metrics[b + '_win_frac_ci_hi']:.3f}]  "
                  f"win_vs_twin {metrics[b + '_win_vs_twin']:+.3f} [{metrics[b + '_win_vs_twin_ci_lo']:+.3f},{metrics[b + '_win_vs_twin_ci_hi']:+.3f}]  "
                  f"scaled gap {metrics[f'{m['label']}_{name}_scaled_gap']:+.3f}")
    return metrics


def main() -> None:
    cfg_path = Path(sys.argv[1])
    cfg = yaml.safe_load(cfg_path.read_text())
    runs_root = Path(os.environ["RUNS_ROOT"])
    (runs_root / cfg["id"]).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cfg_path, runs_root / cfg["id"] / "config.snapshot.yaml")
    run(cfg, runs_root)


if __name__ == "__main__":
    main()
