"""Distances of connected vs unconnected entity pairs, relation detection vs relation grouping (the same documents and prompts).

For every corrupted full-row prompt of the documents in ``params.document_ids`` (kept after the clustering analysis's exclusions, taken as the
union over all four runs), the Euclidean distance between every two entities of the prompt, on the per-(document, field)-centred vectors the PCA
figures use. A pair is *connected* iff the figures draw an edge between it: its two entities' value-based line-item memberships intersect
(``relation_detection_clustering.links``, edges of either colour); every other pair of the prompt is *unconnected*. Two distributions per run: the
distances of the connected pairs and of the unconnected pairs, and the difference of their means (connected minus unconnected; negative = connected
pairs are closer). Reported on the raw distances and on distances divided by the prompt's own mean pairwise distance (so prompts of different scale
weigh alike). Pairs are pooled over prompts; intervals are percentile bootstraps over PROMPTS (prompts of one document are not independent and there
are few documents, so they are indicative, not document-level). The grouping run's change is the paired difference of the two runs' differences on
the same prompts. Known answer: iid random vectors on the same prompts give a scaled difference near 0.

Config: ``analysis/configs/relation_grouping_distances/<id>.yaml``. Outputs: ``metrics.json``, ``artifacts/pair_distances_<model>_<run>.npz``,
figures in ``analysis/figures/<id>/``.
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
gc.rc = rc   # the compare module patches the loader of ITS clustering module; make it the one used here
rr, er = rc.rr, rc.er
VARIANTS = ("detection", "grouping")


# ------------------------------------------------------------------------- statistics
def pair_distances(v: np.ndarray, link_pairs: set[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Euclidean distances of all entity pairs of a prompt, split into connected (``link_pairs``, ``i < j``) and unconnected."""
    m = len(v)
    assert all(0 <= i < j < m for i, j in link_pairs)
    d = rc.dist_matrix(v, "euclidean")
    iu = [(i, j) for i in range(m) for j in range(i + 1, m)]
    conn = np.array([d[i, j] for i, j in iu if (i, j) in link_pairs])
    unconn = np.array([d[i, j] for i, j in iu if (i, j) not in link_pairs])
    assert len(conn) == len(link_pairs) and len(conn) + len(unconn) == len(iu)
    return conn, unconn


def scaled(conn: np.ndarray, unconn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Divide a prompt's distances by the mean of all its pairwise distances."""
    s = np.concatenate([conn, unconn]).mean()
    assert s > 0
    return conn / s, unconn / s


def pooled_diff(recs: list[dict], key: str, idx: np.ndarray) -> tuple[float, float, float]:
    """``(mean connected, mean unconnected, connected - unconnected)`` over the pairs of the prompts ``idx`` (repeats allowed).

    ``key`` is ``"raw"`` or ``"scaled"``. Pairs are pooled, so a prompt with no connected (or no unconnected) pair adds only to the other side.
    """
    c = np.concatenate([recs[i][key][0] for i in idx])
    u = np.concatenate([recs[i][key][1] for i in idx])
    assert len(c) and len(u), "no connected or no unconnected pair"
    return float(c.mean()), float(u.mean()), float(c.mean() - u.mean())


def boot(recs_a: list[dict], key: str, n_boot: int, ci: float, rng: np.random.Generator, recs_b: list[dict] | None = None) -> tuple[float, float, float]:
    """Bootstrap over prompts of the pooled difference of ``recs_a`` (or, with ``recs_b``, of ``a - b`` on the same prompts): ``(estimate, lo, hi)``."""
    n = len(recs_a)
    assert recs_b is None or len(recs_b) == n

    def stat(idx):
        a = pooled_diff(recs_a, key, idx)[2]
        return a if recs_b is None else a - pooled_diff(recs_b, key, idx)[2]

    est = stat(np.arange(n))
    draws = np.array([stat(rng.integers(n, size=n)) for _ in range(n_boot)])
    a = (1 - ci) / 2
    return est, float(np.quantile(draws, a)), float(np.quantile(draws, 1 - a))


# ------------------------------------------------------------------------- plotting
def plot_distributions(panels: list[list[dict]], row_titles: list[str], col_titles: list[str], title: str, path: Path, pcfg: dict) -> None:
    """Grid of overlaid histograms; ``panels[r][c] = {"conn", "unconn", "note"}``; x range shared along a row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_r, n_c = len(panels), len(panels[0])
    fig, axes = plt.subplots(n_r, n_c, figsize=(pcfg["panel_width"] * n_c, pcfg["panel_height"] * n_r), squeeze=False)
    for r in range(n_r):
        allv = np.concatenate([np.concatenate([p["conn"], p["unconn"]]) for p in panels[r]])
        bins = np.linspace(allv.min(), allv.max(), pcfg["bins"] + 1)
        for c, p in enumerate(panels[r]):
            ax = axes[r, c]
            for vals, col, lab in ((p["unconn"], pcfg["unconnected_colour"], "unconnected"), (p["conn"], pcfg["connected_colour"], "connected (edge)")):
                ax.hist(vals, bins=bins, density=True, alpha=pcfg["alpha"], color=col, label=f"{lab} (n={len(vals)})")
                ax.axvline(vals.mean(), color=col, lw=1.5, ls="--")
            ax.set_title(f"{col_titles[c]}, {row_titles[r]}\n{p['note']}", fontsize=pcfg["title_fontsize"])
            ax.tick_params(labelsize=pcfg["label_fontsize"])
            ax.set_xlabel("distance" if r == 0 else "distance / prompt mean distance", fontsize=pcfg["label_fontsize"])
            if r == 0 and c == 0:
                ax.legend(fontsize=pcfg["label_fontsize"])
    fig.suptitle(title, fontsize=pcfg["title_fontsize"])
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=pcfg["dpi"])
    plt.close(fig)


# ------------------------------------------------------------------------- run
def prompt_records(spec: dict, p: dict, runs_root: Path, keep: set[str]) -> list[dict]:
    """The kept invalid prompts of one run (ordered by pair id): centred vectors' pair distances, connected vs unconnected."""
    split = er.load_split(spec["run_id"], runs_root, spec["layer"], p["max_docs"], "relation_detection", rc.EXTRA_META)
    data = rr.load_line_data(spec["run_id"], runs_root, rc.REPO_ROOT)
    run_params = yaml.safe_load((runs_root / spec["run_id"] / "config.snapshot.yaml").read_text())["params"]
    assert run_params["split"] == "test"
    groups = rr.group_rows(split)
    rr.match_samples(split, groups, data)
    inv = [g for g in groups if not g["valid"] and g["pair_id"] in keep]
    items = rc.line_items(run_params["split"], set(split["doc_ids"].tolist()))
    ent = rc.entity_table(split, inv, items, data["eq"])
    centred, _ = rc.center_field_doc(split["X"], split["doc_ids"], split["field"])
    rng = np.random.default_rng(0)
    out = []
    for g in sorted(inv, key=lambda g: g["pair_id"]):
        rows = g["rows"]
        corr = np.array([ent[i]["corrupted"] for i in rows])
        lk = rc.links([ent[i]["members"] for i in rows], corr)
        link_pairs = {(i, j) for i, j, _ in lk}
        rec = {"pair_id": g["pair_id"], "doc": g["doc"], "cell": g["cell"], "m": len(rows), "n_links": len(link_pairs)}
        rec["raw"] = pair_distances(centred[rows], link_pairs)
        rec["scaled"] = scaled(*rec["raw"])
        # known answer: iid Gaussian vectors with the same prompt structure
        rec["random"] = scaled(*pair_distances(rng.normal(size=(len(rows), split["X"].shape[1])), link_pairs))
        out.append(rec)
    assert len(out) == len(keep), (len(out), len(keep))
    return out


def run(cfg: dict, runs_root: Path) -> dict:
    p, seed = cfg["params"], cfg["seed"]
    out = runs_root / cfg["id"]
    fig_dir = rc.FIGURES_DIR / cfg["id"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    gc.accept_grouping_task()
    docs = set(p["document_ids"])
    scans = {}
    for m in p["models"]:
        for var in VARIANTS:
            sp = {"run_id": m[f"{var}_run_id"], "layer": m["layer"]}
            sc = rc.scan_exclusions(sp, p, runs_root)
            scans[f"{m['label']}/{var}"] = {k: {pid for pid in v if pid.split("|")[0] in docs} for k, v in sc.items()}
    excl = rc.union_exclusions(scans)
    keep = excl["all"] - excl["drop"]
    assert keep, "every prompt was excluded"
    metrics: dict[str, float] = {"n_prompts_all": len(excl["all"]), "n_prompts_excluded": len(excl["drop"]), "n_prompts_kept": len(keep)}
    ci, nb = p["ci_level"], p["n_boot"]
    for m in p["models"]:
        label = m["label"]
        recs = {var: prompt_records({"run_id": m[f"{var}_run_id"], "layer": m["layer"]}, p, runs_root, keep) for var in VARIANTS}
        assert [r["pair_id"] for r in recs["detection"]] == [r["pair_id"] for r in recs["grouping"]]
        assert [r["n_links"] for r in recs["detection"]] == [r["n_links"] for r in recs["grouping"]], "link structure differs between runs"
        cells = ["all", *sorted({r["cell"] for r in recs["detection"]})]
        panels = {key: [] for key in ("raw", "scaled")}
        for key in ("raw", "scaled"):
            for var in VARIANTS:
                allr = recs[var]
                c = np.concatenate([r[key][0] for r in allr])
                u = np.concatenate([r[key][1] for r in allr])
                est = boot(allr, key, nb, ci, np.random.default_rng(seed))
                assert est == boot(allr, key, nb, ci, np.random.default_rng(seed)), "same-seed rerun differs"
                metrics["control_seed_determinism"] = 1.0
                panels[key].append({"conn": c, "unconn": u,
                                    "note": f"mean {c.mean():.3g} vs {u.mean():.3g}; diff {est[0]:+.3g} [{est[1]:+.3g}, {est[2]:+.3g}]"})
                for cell in cells:
                    sub = [r for r in allr if cell == "all" or r["cell"] == cell]
                    cn = cell.replace(":", "_")
                    if not any(len(r[key][0]) for r in sub) or not any(len(r[key][1]) for r in sub):
                        continue
                    mc, mu, d = pooled_diff(sub, key, np.arange(len(sub)))
                    base = f"{label}_{var}_{key}_{cn}"
                    metrics.update({f"{base}_mean_connected": mc, f"{base}_mean_unconnected": mu, f"{base}_diff": d,
                                    f"{base}_n_prompts": len(sub), f"{base}_n_connected": int(sum(len(r[key][0]) for r in sub)),
                                    f"{base}_n_unconnected": int(sum(len(r[key][1]) for r in sub))})
                    if cell == "all":
                        metrics[f"{base}_diff_ci_lo"], metrics[f"{base}_diff_ci_hi"] = est[1], est[2]
            est = boot(recs["grouping"], key, nb, ci, np.random.default_rng(seed), recs["detection"])
            metrics[f"{label}_{key}_all_diff_grouping_minus_detection"] = est[0]
            metrics[f"{label}_{key}_all_diff_grouping_minus_detection_ci_lo"], metrics[f"{label}_{key}_all_diff_grouping_minus_detection_ci_hi"] = est[1], est[2]
        for var in VARIANTS:   # known answer: random vectors, same prompts and links -> scaled difference ~ 0
            d = pooled_diff([{"scaled": r["random"]} for r in recs[var]], "scaled", np.arange(len(recs[var])))[2]
            metrics[f"control_random_{label}_{var}_scaled_diff"] = d
            assert abs(d) <= p["random_tolerance"], f"{label}/{var}: random-vector scaled difference {d:+.3f} is not ~0"
        for var in VARIANTS:
            np.savez_compressed(out / "artifacts" / f"pair_distances_{label}_{var}.npz", pair_id=np.array([r["pair_id"] for r in recs[var]]),
                                cell=np.array([r["cell"] for r in recs[var]]),
                                **{f"{key}_{kind}": np.concatenate([r[key][k] for r in recs[var]]) for key in ("raw", "scaled")
                                   for k, kind in ((0, "connected"), (1, "unconnected"))})
        plot_distributions([panels["raw"], panels["scaled"]], ["raw distances", "scaled by prompt mean"], ["relation detection", "relation grouping"],
                           f"{label} layer {m['layer']}: distance of connected vs unconnected entity pairs (field_doc_mean)", fig_dir / f"{label}_distance_distributions.png", p["plot"])
    text = json.dumps(metrics, indent=2) + "\n"
    (out / "metrics.json").write_text(text)
    (fig_dir / "metrics.json").write_text(text)
    for m in p["models"]:
        for key in ("raw", "scaled"):
            print(m["label"], key, *(f"{var}: diff {metrics[f'{m['label']}_{var}_{key}_all_diff']:+.3f}" for var in VARIANTS),
                  f"grouping-detection {metrics[f'{m['label']}_{key}_all_diff_grouping_minus_detection']:+.3f}")
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
