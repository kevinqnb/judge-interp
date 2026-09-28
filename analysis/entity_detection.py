"""Entity-detection representations: valid vs. invalid, per-document-centered PCA.

Visualization only (2026-09-28) -- no correlational statistics yet. Reads the
2026-09-27 ``entity_detection`` **train**-split runs (Qwen2.5-7B-Instruct and
Llama-3.1-8B-Instruct, ``main`` and ``line`` datasets; see
``configs/2026-09-27-<model>-entity-<dataset>-train-01.yaml``) and, for each
(model, dataset), plots 2-component PCA of the entity representations colored
by ``label_valid``:

    (a) pooled -- every entity type together
    (b) per entity type -- one figure per field in
        ``judge_interp.entity_prompts.FIELDS_BY_DATASET[dataset]``

Hypothesis (2026-09-27-entity-relation-detection-01, verbatim): valid/invalid
entities separate geometrically *within a document*. That is a claim about
document-local structure, not about one global cluster, so before pooling
across documents each entity's representation is centered by subtracting its
*own document's* mean representation (computed over whatever entities are in
the current case -- pooled or one type) -- this removes "which document is
this" as a competing source of variance (the same confound
``analysis/representations.py``'s docstring already flags for record_judge,
here addressed directly rather than left as a caveat). Each case (pooled, or
one entity type) does its own per-document centering and its own independent
PCA fit, exactly mirroring ``analysis/representations.py``'s one-PCA-per-subset
convention -- sklearn's own (now near-zero) mean subtraction on top of that is
harmless double-centering, not a second knob.

Representations are read at each entity's own **content** span (its value's
last token) -- entity_detection also carries a **cue** span per entity (the
"valid?" position used for the verdict), which is a different token position
serving a different role and is deliberately excluded here: pooling content
and cue together would double-count every entity and let span *kind* (not
entity identity) dominate the geometry.

Known, expected-not-a-bug result (predicted in
``configs/2026-09-27-qwen7b-entity-smoke-01.yaml`` before any GPU run, and
confirmed by every real run below): ``verdict_recognised_rate`` at cue spans is
0.0 on every model. The cue sits mid-list, immediately followed by a newline
and the next candidate -- the instructions ask for one "<n>: true/false" line
per candidate only after the full list ends, so the literal next-token argmax
at a cue position is never "true"/"false" regardless of model scale. This
script never reads verdict tokens, only representations, so that 0.0 does not
gate anything here -- it is printed for visibility, not asserted on (unlike
record_judge's ``load_run`` in ``analysis/representations.py``, which does gate
on a nonzero verdict-recognition rate because *that* script's centroid/Spearman
reading depends on the judge having produced a usable verdict at all).

Confound to watch for in the **pooled** figure specifically (per-type figures
are unaffected): per-document centering removes each document's mean, but it
does *not* remove entity-*type* means from the residuals, and the two classes
can have very different type mixes. Measured 2026-09-28: on ``main``, every
field is within ~1 percentage point of its share of the valid vs. invalid
class (e.g. ``advertiser`` is 12.7% of valid entities and 13.1% of invalid --
balanced by construction, since main renders every field's own decoys
1-for-1 against that field's single valid value per document). On ``line``,
the mix differs sharply -- valid entities are 41% ``program_desc`` and only
4% ``channel``, while invalid entities are a near-uniform ~17-22% across all
five fields (line's decoy pool is shared across the whole document rather
than per-field, so it doesn't track the valid mix). Any type-driven cluster
in ``line``'s pooled figure (e.g. a lobe that is mostly ``channel`` values)
will therefore look like a valid/invalid split even if entity *type* is the
only thing separating it. ``load_run`` prints each run's per-type composition
by class and flags this automatically; treat the **per-type** figures as the
clean read for ``line``, and the pooled figure there as suggestive at best.

PCA is centering-only (no per-feature z-scoring), same convention as
``analysis/representations.py``. Colors are the fixed Okabe-Ito
colorblind-safe categorical pair (blue/orange), applied in the same order in
every panel: identity (valid vs. invalid) is never encoded by anything other
than this pair's fixed order. Valid entities are the minority class (~17-19%
pooled, per the 2026-09-27 config descriptions) and are drawn on top and at
higher opacity so they are not visually swamped by the invalid majority; large
subsets are subsampled for the *scatter only* (never for the PCA fit) with the
subsample budget split evenly between the two classes (see
``_stratified_scatter_indices``), so the minority class stays visible even
when both classes individually exceed the budget.

Figures are saved to ``analysis/figures/`` instead of being displayed inline.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from judge_interp.entity_prompts import FIELDS_BY_DATASET

RUNS_ROOT = Path(os.environ["RUNS_ROOT"])
FIGURES_DIR = Path(__file__).parent / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "Times New Roman", "DejaVu Serif"],
})

# Okabe-Ito colorblind-safe categorical pair, fixed order (invalid, valid) in
# every panel -- see this module's docstring.
INVALID_COLOR = "#E69F00"
VALID_COLOR = "#0072B2"

# (model, dataset) -> run id -- see configs/2026-09-27-<model>-entity-<dataset>-train-01.yaml.
# Test splits exist (2026-09-27-*-entity-{main,line}-test-01) but are the
# "not used by the initial train-only analysis" held-out counterparts named in
# those configs' own descriptions -- deliberately excluded here.
RUN_IDS = {
    ("qwen7b", "main"): "2026-09-27-qwen7b-entity-main-train-01",
    ("qwen7b", "line"): "2026-09-27-qwen7b-entity-line-train-01",
    ("llama8b", "main"): "2026-09-27-llama8b-entity-main-train-01",
    ("llama8b", "line"): "2026-09-27-llama8b-entity-line-train-01",
}
MODELS = ("qwen7b", "llama8b")
DATASETS = ("main", "line")

# Analysis-only display knob (not an experiment param): scatter plots
# subsample to this many points; the PCA fit always uses the full subset.
MAX_SCATTER_POINTS = 4000
SCATTER_SEED = 0

# Analysis-only diagnostic threshold (not a hard gate): flag a run's pooled
# figure as type-composition-confounded when some field's share of the valid
# class differs from its share of the invalid class by more than this many
# percentage points. See this module's docstring for the measured line vs.
# main numbers this is calibrated against.
TYPE_COMPOSITION_WARN_PP = 5.0


def load_run(run_id: str) -> dict:
    """Load one entity_detection run: per-layer content-span representations
    plus ``doc_ids`` / ``entity_type`` / ``label_valid``, filtered to
    ``kind == "content"`` (see this module's docstring for why cue spans are
    excluded).

    Hard-gates on ``run.json``'s status and on the content/cue span count
    implied by ``metrics.json`` (entity_detection renders exactly one content
    span and one cue span per entity, so ``n_spans`` must be even and
    ``n_cue_spans`` must equal half of it) -- computed from the run's own
    metrics rather than hardcoded, per this repo's no-magic-numbers rule.
    """
    run_dir = RUNS_ROOT / run_id
    manifest = json.loads((run_dir / "run.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())

    assert manifest["status"] == "success", f"{run_id}: run.json status is {manifest['status']!r}, not 'success'"
    assert manifest["task"] == "entity_detection", f"{run_id}: task is {manifest['task']!r}, not 'entity_detection'"
    assert metrics["n_spans"] % 2 == 0, f"{run_id}: n_spans={metrics['n_spans']} is odd -- expected content+cue pairs"
    expected_content = metrics["n_spans"] // 2
    assert metrics["n_cue_spans"] == expected_content, (
        f"{run_id}: n_cue_spans={metrics['n_cue_spans']} != n_spans/2={expected_content} -- "
        "entity_detection should render exactly one cue span per entity"
    )

    with np.load(run_dir / "artifacts" / "span_representations.npz", allow_pickle=False) as z:
        layers = z["layers"].tolist()
        kind = z["kind"]
        assert set(np.unique(kind).tolist()) == {"content", "cue"}, (
            f"{run_id}: unexpected span kind(s) {sorted(set(kind.tolist()))}"
        )
        mask = kind == "content"
        assert int(mask.sum()) == expected_content, (
            f"{run_id}: {int(mask.sum())} content spans, expected {expected_content}"
        )
        assert int((~mask).sum()) == expected_content, (
            f"{run_id}: {int((~mask).sum())} cue spans, expected {expected_content}"
        )

        doc_ids = z["doc_ids"][mask]
        entity_type = z["entity_type"][mask]
        label_valid = z["label_valid"][mask].astype(bool)
        reps = {L: z[f"rep_{L}"][mask] for L in layers}  # one decompression per layer, masked once here

    n = expected_content
    assert doc_ids.shape == (n,)
    assert entity_type.shape == (n,)
    assert label_valid.shape == (n,)
    for L, arr in reps.items():
        assert arr.shape[0] == n, (L, arr.shape)

    print(
        f"{run_id}: status={manifest['status']}, git_sha={manifest['git_sha']}, "
        f"git_dirty={manifest['git_dirty']}, n_entities={n} "
        f"(valid={int(label_valid.sum())}, invalid={int((~label_valid).sum())}), "
        f"n_documents={len(set(doc_ids.tolist()))}, layers={layers}, "
        f"hidden={reps[layers[0]].shape[1]}"
    )
    print(
        f"  metrics: verdict_recognised_rate={metrics['verdict_recognised_rate']} "
        "(0.0 expected at cue spans on every model -- see this module's docstring; "
        "not gated on since only representations, never verdict tokens, are used here)"
    )
    print_type_composition(entity_type, label_valid, run_id)

    return {
        "run_id": run_id,
        "layers": layers,
        "reps": reps,
        "doc_ids": doc_ids,
        "entity_type": entity_type,
        "label_valid": label_valid,
    }


def print_type_composition(entity_type: np.ndarray, label_valid: np.ndarray, title: str) -> None:
    """Print each entity type's share of the valid class vs. the invalid class,
    and flag (not gate on) a run whose pooled figure is therefore
    type-composition-confounded -- see this module's docstring.
    """
    fields = sorted(set(entity_type.tolist()))
    n_valid, n_invalid = int(label_valid.sum()), int((~label_valid).sum())
    valid_pct = {f: 100.0 * int(((entity_type == f) & label_valid).sum()) / n_valid for f in fields}
    invalid_pct = {f: 100.0 * int(((entity_type == f) & ~label_valid).sum()) / n_invalid for f in fields}
    max_gap = max(abs(valid_pct[f] - invalid_pct[f]) for f in fields)

    print(f"  type composition [{title}] (% of valid class / % of invalid class):")
    for f in fields:
        print(f"    {f}: {valid_pct[f]:.1f}% / {invalid_pct[f]:.1f}%")
    if max_gap > TYPE_COMPOSITION_WARN_PP:
        print(
            f"    WARNING: max valid/invalid composition gap is {max_gap:.1f} percentage points "
            f"(> {TYPE_COMPOSITION_WARN_PP}) -- the POOLED figure for this run is "
            "type-composition-confounded (per-document centering does not remove entity-type "
            "means); treat the per-type figures as the clean read here."
        )


def center_per_document(x: np.ndarray, doc_ids: np.ndarray) -> np.ndarray:
    """Subtract each document's own mean row from its rows, in-place-equivalent.

    Asserts the result: every document's centered mean is ~0. A document
    contributing exactly one row to this subset would zero out entirely (a
    real but harmless degenerate case) -- not possible here, since a
    surviving (document, field) entity list always has >=1 valid and >=1
    invalid entity (``min_valid_per_list`` / ``min_invalid_per_list`` in every
    2026-09-27 entity_detection config), so every document contributes >=2
    rows to any per-type or pooled subset it appears in.
    """
    out = np.empty_like(x)
    for doc in np.unique(doc_ids):
        m = doc_ids == doc
        out[m] = x[m] - x[m].mean(axis=0, keepdims=True)
    for doc in np.unique(doc_ids):
        m = doc_ids == doc
        assert np.allclose(out[m].mean(axis=0), 0.0, atol=1e-3), f"per-document centering failed for {doc!r}"
    return out


def _stratified_scatter_indices(label_valid: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    """Indices to draw in the scatter (not the PCA fit, which always uses every
    row). Splits the budget evenly between the two classes, reallocating any
    unused half to the other class when one class is smaller than half the
    budget -- both the pooled set and several per-type subsets here have
    *both* classes larger than ``max_points`` on their own (e.g. line/train
    pooled: 11762 valid, 49779 invalid, budget 4000), so simply "keep every
    valid point" would silently drop the invalid class to zero rather than
    subsampling it -- caught by eyeballing the first pooled figure, which
    rendered with no invalid points at all despite them being 81% of the data.
    """
    n = label_valid.shape[0]
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    valid_idx_all = np.flatnonzero(label_valid)
    invalid_idx_all = np.flatnonzero(~label_valid)

    half = max_points // 2
    n_valid = min(valid_idx_all.size, half)
    n_invalid = min(invalid_idx_all.size, max_points - n_valid)
    n_valid = min(valid_idx_all.size, max_points - n_invalid)  # reclaim leftover if invalid class was smaller

    valid_idx = valid_idx_all if n_valid == valid_idx_all.size else rng.choice(valid_idx_all, size=n_valid, replace=False)
    invalid_idx = invalid_idx_all if n_invalid == invalid_idx_all.size else rng.choice(invalid_idx_all, size=n_invalid, replace=False)
    return np.concatenate([invalid_idx, valid_idx])


def analyze_and_plot_entities(
    reps_by_layer: dict[int, np.ndarray],
    doc_ids: np.ndarray,
    label_valid: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Per-document-center, fit a 2-component PCA, and plot valid vs. invalid
    as a 1xN-layer subplot figure (one independent PCA fit per layer, all on
    the same rows).
    """
    layers = sorted(reps_by_layer)
    n = label_valid.shape[0]
    assert n >= 2, f"{title}: only {n} row(s) -- nothing to plot"

    scatter_idx = _stratified_scatter_indices(label_valid, MAX_SCATTER_POINTS, SCATTER_SEED)

    fig, axes = plt.subplots(
        1, len(layers), figsize=(5.5 * len(layers), 5.5), squeeze=False, constrained_layout=True
    )
    axes = axes[0]
    for ax, layer in zip(axes, layers):
        centered = center_per_document(reps_by_layer[layer], doc_ids)
        pca = PCA(n_components=2)
        scores = pca.fit_transform(centered)

        inv_idx = scatter_idx[~label_valid[scatter_idx]]
        val_idx = scatter_idx[label_valid[scatter_idx]]
        ax.scatter(
            scores[inv_idx, 0], scores[inv_idx, 1],
            c=INVALID_COLOR, s=8, alpha=0.4, linewidths=0, label="invalid",
        )
        ax.scatter(
            scores[val_idx, 0], scores[val_idx, 1],
            c=VALID_COLOR, s=10, alpha=0.75, linewidths=0, label="valid",
        )
        var1, var2 = pca.explained_variance_ratio_[:2]
        ax.set_xlabel(f"PC1 ({var1:.1%})")
        ax.set_ylabel(f"PC2 ({var2:.1%})")
        ax.set_title(f"layer {layer}")
        ax.set_box_aspect(1)

    axes[0].legend(loc="best", fontsize=9, framealpha=0.9)
    n_shown_valid = int(label_valid[scatter_idx].sum())
    n_shown_invalid = int((~label_valid[scatter_idx]).sum())
    fig.suptitle(
        f"{title}  (per-document-centered PCA; shown: {n_shown_valid}/{int(label_valid.sum())} valid, "
        f"{n_shown_invalid}/{int((~label_valid).sum())} invalid)"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(
        f"[{title}] n={n} (valid={int(label_valid.sum())}, invalid={int((~label_valid).sum())}), "
        f"{len(np.unique(doc_ids))} documents, saved to {out_path}"
    )


def main() -> None:
    for dataset in DATASETS:
        fields = FIELDS_BY_DATASET[dataset]
        for model in MODELS:
            run = load_run(RUN_IDS[(model, dataset)])

            present_types = set(run["entity_type"].tolist())
            unexpected = present_types - set(fields)
            assert not unexpected, (
                f"{run['run_id']}: entity_type(s) {sorted(unexpected)} outside canonical "
                f"{dataset} fields {fields}"
            )

            analyze_and_plot_entities(
                run["reps"], run["doc_ids"], run["label_valid"],
                f"{model}/{dataset}/train (pooled, all entity types)",
                FIGURES_DIR / f"{model}_{dataset}_train_entity_pca_pooled.png",
            )

            for field in fields:
                mask = run["entity_type"] == field
                if not mask.any():
                    continue
                sub_reps = {L: arr[mask] for L, arr in run["reps"].items()}
                analyze_and_plot_entities(
                    sub_reps, run["doc_ids"][mask], run["label_valid"][mask],
                    f"{model}/{dataset}/train ({field})",
                    FIGURES_DIR / f"{model}_{dataset}_train_entity_pca_{field}.png",
                )


if __name__ == "__main__":
    main()
