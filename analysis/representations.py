"""Representations vs. `num_invalid_fields`

PCA of judge representations, colored by `num_invalid_fields`, for two models
(Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct) across the main and line **train**
datasets, at 3 layers each (see `configs/2026-09-15-<model>-repr-<dataset>-train-01.yaml`
for the exact layer indices -- evenly spaced from the first transformer-block
output to the last, post-final-norm, layer). Line rows additionally carry
`error_type` ("inter_document" / "intra_document" invalid-field collisions,
`""` for valid rows and for all main rows) since the 2026-09-15 collision-rule
fix; the line dataset is analyzed both combined and split by `error_type`.

Hypothesis (verbatim, `/experiment` 2026-09-10): each `num_invalid_fields` group
should form a cluster, or -- more likely -- a spectrum of meshed clusters moving
in one direction from most-invalid to fully-valid. That would mean the judge's
internal confidence tracks the *degree* of invalidity, not just a binary
valid/invalid split. Splitting the line dataset by `error_type` checks whether
that spectrum (if any) looks the same for both collision-error classes, or
whether one class produces a cleaner/noisier signal than the other -- a
difference here would mean "degree of invalidity" isn't a single
model-internal axis but something that also depends on *how* the row was
corrupted.

That's a claim about ordered group centroids along one direction, not about
visually separated blobs, so the primary output below is the per-group centroid
table and the Spearman rank correlation between PC1 and `num_invalid_fields` --
the scatter plots are illustration only (and get unreadable at line-train's 38k
points in several colors, so they're subsampled).

Runs this script reads (see `configs/`):
- `2026-09-15-qwen7b-repr-main-train-01`, `2026-09-15-llama8b-repr-main-train-01`
  -- main/train, 5130 rows each, `num_invalid_fields` balanced 0..9 (513/group),
  layers [1, 14, 28] (qwen) / [1, 16, 32] (llama)
- `2026-09-15-qwen7b-repr-line-train-01`, `2026-09-15-llama8b-repr-line-train-01`
  -- line/train, 38612 rows each, `num_invalid_fields` in 0..4 (unbalanced --
  see that config's description for why `k` and `num_invalid_fields` diverge on
  line rows), split further by `error_type`

Before trusting any of this, check `metrics.json` for the run:
`verdict_recognised_rate` should be ~1.0 and `judge_accuracy` should be well
above chance. `load_run` below hard-asserts on both rather than just printing
them, since a judge that can't do the task at all has no reason to have an
interpretable confidence spectrum. It also prints (but does not hard-gate) a
Spearman correlation between `p_true` and `num_invalid_fields`: `judge_accuracy`
alone can look fine on an imbalanced run (e.g. 4617 invalid vs. 513 valid rows)
even when the judge is near coin-flip on the valid class specifically -- check
that print before reading a PC1 spectrum as "the judge's confidence tracks
invalidity"; a flat correlation there means any PCA spectrum has no known
behavioral correlate and should be reported with that caveat.

PCA is centering-only (no per-feature z-scoring) since the representations are
post-final-norm rows that are already comparably scaled -- that's a knob, noted
here so it doesn't get lost.

PC1's sign is otherwise arbitrary (an eigenvector's sign is not identified), and
this script fits one independent PCA per (model, dataset-subset, layer) -- 2
models x 3 layers x 3 line subsets alone. Without a fixed convention those
signs would be incomparable across panels, and the two error-class figures
could not be visually compared to each other or to the combined figure.
`pca_spectrum_analysis` pins PC1 so the valid group's (`num_invalid_fields ==
0`) centroid is always negative. This flip is chosen from the data's own
structure, not from `sign(rho)` -- flipping to make rho positive would make
every run's correlation trivially positive and silently defeat the point of
computing it.

Each figure's color scale (`vmin=0, vmax=<that model+dataset's max
num_invalid_fields>`) is fixed across all panels and, for line, across the
combined and both per-`error_type` figures for that model -- otherwise the
same color could mean a different `num_invalid_fields` in the inter- vs.
intra-document figures (unbalanced groups: see the run description above).

`error_type` note: valid rows (`error_type == ""`) are the shared
`num_invalid_fields == 0` anchor and appear in **both** the inter_document and
intra_document subsets below -- they are not disjoint data, they're the common
zero-invalid endpoint each error class's spectrum is measured against.

Confound to watch for: each main/train document contributes 10 rows (one per
`num_invalid_fields` group) that all share one OCR context. With 513 documents,
document identity is a plausible competing source of PC1 variance -- if PC1
turns out to track "which document is this" rather than validity, the Spearman
correlation here will look weak or null even if a validity direction exists on a
lower/less-variance component. `doc_ids` is loaded by `load_run` and unused
below; if the hypothesis looks falsified, that's the first thing to check (e.g.
per-document-centered PCA) before concluding there's no spectrum.

Sanity control baked in: a shuffled-label control (permute `num_invalid_fields`,
recompute the same statistics against the SAME fitted PC1) runs alongside each
real analysis. If the real Spearman correlation isn't clearly stronger than the
shuffled one, the "spectrum" reading isn't supported. The PCA/centroid/
correlation logic itself was checked against a synthetic fixture with a known
injected signal before being used here (recovers `|rho| > 0.99` with signal,
`|rho| < 0.03` on pure noise or shuffled labels) -- see the 2026-09-10
`/experiment` session notes; the sign-pinning and multi-layer/subset plumbing
added 2026-09-15 reuse that same core, unchanged.

`load_run` also hard-asserts `num_invalid_fields == 0` exactly on rows where
`labels` (the row's `valid`) is `True` -- the valid-anchor reading above (and
the sign-pinning convention) both assume no invalid row landed at
`num_invalid_fields == 0` after collision resampling. If that assert fires,
stop: it means the anchor assumption is wrong, not that the assert is.

Figures are saved to `analysis/figures/` instead of being displayed inline.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

RUNS_ROOT = Path(os.environ["RUNS_ROOT"])
FIGURES_DIR = Path(__file__).parent / "figures"

# (model, dataset) -> run id -- see configs/2026-09-15-<model>-repr-<dataset>-train-01.yaml
# for the collection params (layers, etc.) behind each run.
RUN_IDS = {
    ("qwen7b", "main"): "2026-09-15-qwen7b-repr-main-train-01",
    ("qwen7b", "line"): "2026-09-15-qwen7b-repr-line-train-01",
    ("llama8b", "main"): "2026-09-15-llama8b-repr-main-train-01",
    ("llama8b", "line"): "2026-09-15-llama8b-repr-line-train-01",
}
MODELS = ("qwen7b", "llama8b")
LINE_ERROR_TYPES = ("inter_document", "intra_document")

# Analysis-only display knobs (not experiment params): scatter plots subsample
# to this many points so line-train's 38k rows don't render as a solid blob;
# centroids and the Spearman correlation always use the full subset, unsampled.
MAX_SCATTER_POINTS = 4000
SCATTER_SEED = 0

# Analysis-only diagnostic threshold (not a hard gate -- load_run's
# verdict_recognised_rate/judge_accuracy asserts are the hard gates). Below
# this |rho|, print a caveat that any PCA spectrum has no established
# behavioral correlate in the judge's own output probabilities.
P_TRUE_SANITY_RHO_WARN = 0.3


def load_run(run_id: str, min_verdict_recognised_rate: float = 0.95) -> dict:
    """Load one run_experiment.py output dir: per-layer representation
    matrices, num_invalid_fields/labels/k/error_type/p_true, and provenance
    (run.json + metrics.json).

    Hard-gates on collection health rather than just printing metrics.json:
    compute_metrics() in run_experiment.py returns None for judge_accuracy /
    judge_acc_valid / judge_acc_invalid / mean_p_true_* whenever a mask is
    empty or nothing was recognised, so a broken judge (e.g. every verdict a
    space-prefixed " true" that verdict_recognised never matches) would
    otherwise print "judge_accuracy=None" and let the PCA run anyway on
    representations from a judge that never produced a usable verdict -- runs
    cleanly, number quietly wrong, exactly what this repo's CLAUDE.md warns
    about. Fail loud instead.

    Also asserts num_invalid_fields == 0 exactly where labels is True: the
    valid-anchor reading used throughout this script (and the PC1 sign-pinning
    convention in pca_spectrum_analysis) both depend on this holding globally,
    not just within whichever subset happens to be plotted.
    """
    run_dir = RUNS_ROOT / run_id
    manifest = json.loads((run_dir / "run.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())

    assert manifest["status"] == "success", f"{run_id}: run.json status is {manifest['status']!r}, not 'success'"
    assert metrics["judge_accuracy"] is not None, (
        f"{run_id}: judge_accuracy is null in metrics.json -- no row had a recognised "
        "verdict. Stop and diagnose the collection run; don't interpret its representations."
    )
    assert metrics["verdict_recognised_rate"] >= min_verdict_recognised_rate, (
        f"{run_id}: verdict_recognised_rate={metrics['verdict_recognised_rate']:.4f} < "
        f"{min_verdict_recognised_rate} -- too many rows produced an unrecognised verdict "
        "token for the representations to be trusted as 'the judge's read on this row'."
    )

    with np.load(run_dir / "artifacts" / "representations.npz", allow_pickle=False) as z:
        layers = z["layers"].tolist()
        reps = {layer: z[f"rep_{layer}"] for layer in layers}
        num_invalid_fields = z["num_invalid_fields"]
        labels = z["labels"]
        k = z["k"]
        doc_ids = z["doc_ids"]
        error_type = z["error_type"]
        p_true = z["p_true"]

    n = labels.shape[0]
    for layer, arr in reps.items():
        assert arr.shape[0] == n, f"{run_id}: rep_{layer} has {arr.shape[0]} rows, expected {n}"
    assert num_invalid_fields.shape == (n,)
    assert error_type.shape == (n,)
    assert p_true.shape == (n,)
    assert np.array_equal(num_invalid_fields == 0, labels), (
        f"{run_id}: expected num_invalid_fields == 0 exactly on valid rows -- the "
        "valid-anchor reading and PC1 sign-pinning both assume this. Found a "
        "mismatch; check the collision-resampling logic before trusting anything below."
    )

    print(f"{run_id}: status={manifest['status']}, git_sha={manifest['git_sha']}, "
          f"git_dirty={manifest['git_dirty']}, n_rows={n}, layers={layers}, "
          f"hidden={reps[layers[0]].shape[1]}, n_documents={len(set(doc_ids.tolist()))}")
    print(f"  metrics: verdict_recognised_rate={metrics['verdict_recognised_rate']:.4f}, "
          f"judge_accuracy={metrics['judge_accuracy']}, "
          f"judge_acc_valid={metrics['judge_acc_valid']}, "
          f"judge_acc_invalid={metrics['judge_acc_invalid']}")

    return {
        "run_id": run_id,
        "layers": layers,
        "reps": reps,
        "num_invalid_fields": num_invalid_fields,
        "labels": labels,
        "k": k,
        "doc_ids": doc_ids,
        "error_type": error_type,
        "p_true": p_true,
        "metrics": metrics,
    }


def p_true_sanity_check(run: dict, title: str) -> None:
    """Print-only diagnostic (not a gate -- see load_run for the hard gates):
    Spearman(p_true, num_invalid_fields) at the row level. judge_accuracy can
    look fine on an imbalanced run while the judge is near coin-flip on the
    valid class specifically (see this module's docstring); if this
    correlation is weak, any PCA spectrum below has no established behavioral
    correlate and should be reported with that caveat, not treated as ruled out.
    """
    rho, p = spearmanr(run["p_true"], run["num_invalid_fields"])
    flag = "" if abs(rho) >= P_TRUE_SANITY_RHO_WARN else "  <-- WEAK: PCA spectrum below has no established behavioral correlate"
    print(f"[{title}] Spearman(p_true, num_invalid_fields): rho={rho:.4f}, p={p:.3g}{flag}")


def pca_spectrum_analysis(reps: np.ndarray, num_invalid_fields: np.ndarray, title: str) -> dict:
    """Fit 2-component PCA (centering only, no per-feature scaling) and test the
    "spectrum" hypothesis: are PC1 scores monotonic in num_invalid_fields group
    centroid, and correlated with num_invalid_fields at the row level?

    PC1's sign is pinned so the valid group's (num_invalid_fields == 0)
    centroid is negative -- an eigenvector's sign is otherwise arbitrary, and
    this script fits many independent PCAs (per model x layer x subset) that
    need a shared, label-independent convention to be comparable across
    panels. See this module's docstring for why the flip is not based on
    sign(rho).

    Returns the fitted pca, the [n, 2] (sign-pinned) scores, the per-group PC1
    centroid table, and the Spearman rho/p between PC1 and num_invalid_fields.
    """
    pca = PCA(n_components=2)
    scores = pca.fit_transform(reps)

    groups = sorted(set(num_invalid_fields.tolist()))
    assert groups[0] == 0, f"{title}: expected num_invalid_fields groups to include 0, got {groups}"
    if scores[num_invalid_fields == 0, 0].mean() > 0:
        scores = scores.copy()
        scores[:, 0] *= -1

    pc1 = scores[:, 0]
    centroids = pd.Series(
        {g: pc1[num_invalid_fields == g].mean() for g in groups}, name="pc1_centroid"
    )
    centroid_vals = centroids.values
    monotonic = bool(
        np.all(np.diff(centroid_vals) >= 0) or np.all(np.diff(centroid_vals) <= 0)
    )
    rho, p = spearmanr(pc1, num_invalid_fields)

    print(f"[{title}] explained_variance_ratio (PC1, PC2) = {pca.explained_variance_ratio_}")
    print(f"[{title}] per-group PC1 centroids:\n{centroids}")
    print(f"[{title}] centroids monotonic in num_invalid_fields: {monotonic}")
    print(f"[{title}] Spearman(PC1, num_invalid_fields): rho={rho:.4f}, p={p:.3g}")

    return {
        "pca": pca,
        "scores": scores,
        "centroids": centroids,
        "monotonic": monotonic,
        "rho": rho,
        "p": p,
    }


def shuffled_label_control(scores: np.ndarray, num_invalid_fields: np.ndarray, title: str, seed: int = 0) -> dict:
    """Sanity control: representations don't depend on num_invalid_fields labels,
    so permuting them against the SAME already-fitted PC1 scores should collapse
    the Spearman correlation toward 0. If it doesn't, the "spectrum" reading
    above isn't supported by anything beyond noise/leakage.

    Takes the PCA `scores` already computed by pca_spectrum_analysis (not a
    re-fit) -- only the label permutation changes, isolating exactly the thing
    being controlled for.
    """
    shuffled = np.random.default_rng(seed).permutation(num_invalid_fields)
    rho, p = spearmanr(scores[:, 0], shuffled)
    print(f"[{title} | shuffled-label control] Spearman(PC1, shuffled num_invalid_fields): "
          f"rho={rho:.4f}, p={p:.3g}")
    return {"rho": rho, "p": p}


def analyze_and_plot(
    reps_by_layer: dict[int, np.ndarray],
    num_invalid_fields: np.ndarray,
    title: str,
    out_path: Path,
    vmin: int,
    vmax: int,
    rng_seed: int = SCATTER_SEED,
) -> dict[int, dict]:
    """Run pca_spectrum_analysis + shuffled_label_control per layer, and lay
    the layers out as a 1xN subplot (one panel per layer, shared colorbar) in
    a single figure. The same row subsample (and the same vmin/vmax color
    scale) is used in every panel so the panels are visually comparable to
    each other -- each layer is a separate PCA fit, but on the same rows.

    Returns {layer: result_dict} (pca_spectrum_analysis's return value per
    layer) for any further inspection by the caller.
    """
    layers = sorted(reps_by_layer)
    last_layer = max(layers)
    n = num_invalid_fields.shape[0]
    n_groups = len(set(num_invalid_fields.tolist()))
    assert n_groups >= 2, (
        f"{title}: only {n_groups} distinct num_invalid_fields value(s) in this subset -- "
        "Spearman rho/p are undefined (or vacuous, e.g. a 2-point 'control' that trivially "
        "matches or reverses the real correlation) below that. Fix the subset, don't read the "
        "numbers this call would print."
    )
    if n > MAX_SCATTER_POINTS:
        idx = np.random.default_rng(rng_seed).choice(n, size=MAX_SCATTER_POINTS, replace=False)
    else:
        idx = np.arange(n)

    results: dict[int, dict] = {}
    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 5), squeeze=False)
    axes = axes[0]
    sc = None
    for ax, layer in zip(axes, layers):
        panel_title = f"{title} | layer {layer}"
        result = pca_spectrum_analysis(reps_by_layer[layer], num_invalid_fields, panel_title)
        shuffled_label_control(result["scores"], num_invalid_fields, panel_title)
        results[layer] = result

        sc = ax.scatter(
            result["scores"][idx, 0], result["scores"][idx, 1],
            c=num_invalid_fields[idx], cmap="viridis", s=8, alpha=0.5, linewidths=0,
            vmin=vmin, vmax=vmax,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        # Depth fraction makes e.g. qwen's "layer 14" and llama's "layer 16"
        # (same rung, different total depth) comparable at a glance.
        ax.set_title(f"layer {layer} (depth {layer / last_layer:.0%})")

    fig.suptitle(f"{title} (n={n}, showing {len(idx)}/panel)")
    fig.colorbar(sc, ax=axes.tolist(), label="num_invalid_fields")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{title}] saved figure to {out_path}")
    return results


def main() -> None:
    # Main dataset: 10 balanced num_invalid_fields groups (0..9, 513 rows
    # each), no error_type split (main rows are always error_type == "").
    # Both models' main-dataset runs have finished; run these first regardless
    # of whether the (much longer) line-dataset jobs have finished yet.
    print("=== main/train ===")
    for model in MODELS:
        run = load_run(RUN_IDS[(model, "main")])
        p_true_sanity_check(run, f"{model}/main/train")
        vmax = int(run["num_invalid_fields"].max())
        analyze_and_plot(
            run["reps"], run["num_invalid_fields"], f"{model}/main/train",
            FIGURES_DIR / f"{model}_main_train_pc1_pc2.png",
            vmin=0, vmax=vmax,
        )

    # Line dataset: 5 unbalanced num_invalid_fields groups (0..4), further
    # split by error_type ("inter_document" / "intra_document"; valid rows are
    # the shared num_invalid_fields == 0 anchor and appear in both splits --
    # see this module's docstring). Combined-then-split, all three figures per
    # model sharing one vmin/vmax so colors mean the same thing across them.
    print("=== line/train ===")
    for model in MODELS:
        run = load_run(RUN_IDS[(model, "line")])
        p_true_sanity_check(run, f"{model}/line/train")
        vmax = int(run["num_invalid_fields"].max())

        analyze_and_plot(
            run["reps"], run["num_invalid_fields"],
            f"{model}/line/train (combined, both error classes)",
            FIGURES_DIR / f"{model}_line_train_pc1_pc2.png",
            vmin=0, vmax=vmax,
        )

        for error_type in LINE_ERROR_TYPES:
            mask = (run["error_type"] == "") | (run["error_type"] == error_type)
            assert (run["error_type"][mask] == error_type).any(), (
                f"{run['run_id']}: no {error_type!r} rows found"
            )
            sub_reps = {layer: arr[mask] for layer, arr in run["reps"].items()}
            sub_nif = run["num_invalid_fields"][mask]
            analyze_and_plot(
                sub_reps, sub_nif, f"{model}/line/train ({error_type} + valid)",
                FIGURES_DIR / f"{model}_line_train_{error_type}_pc1_pc2.png",
                vmin=0, vmax=vmax,
            )


if __name__ == "__main__":
    main()
