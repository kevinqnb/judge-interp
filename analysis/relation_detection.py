"""Relation-detection representations, per document: centered PCA colored by
ground-truth source tuple.

Visualization only (2026-09-28) -- no correlational statistics yet. Reads one
of the 2026-09-27 ``relation_detection`` **train**-split runs
(Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct, ``line`` dataset only -- see
``configs/2026-09-27-<model>-relation-line-train-01.yaml``) and, for a single
user-given ``document_id``, plots 2-component PCA of every entity presented to
the model for that document, colored by which valid ground-truth tuple(s) it
belongs to.

Hypothesis (2026-09-27-entity-relation-detection-01, verbatim): entities
cluster by their true source tuple, with shared entities producing "orbiting"
clusters. "Source" here means a ``valid: true`` row in
``artifacts/tuple_membership.json`` -- the actual base line items the
document's entity lists were built from. ``intra_document``-swapped
(invalid) tuples are corruptions, not ground-truth data points, and are never
used as a source here.

**Source collapsing.** Two different valid line items can carry identical
field values across every field (observed in this corpus, e.g. document
``00a83bbc-...``'s ``line_index`` 0 and 1) -- the model has no way to
distinguish such rows from the rendered entity list alone, since entity lists
only ever show *values*, never row identity. Treating each as its own source
would mark every one of their shared entities as spuriously "multi-source".
Valid tuples with byte-identical ``entity_index`` values are therefore
collapsed into one source, labeled by the sorted list of ``line_index``
values that share it (2026-09-28 design decision, confirmed with the user).

**Entity -> source join.** An entity is joined to its source(s) by
``(entity_type, value)`` -- never by ``value`` alone, since a value can recur
verbatim across different field types (e.g. an identical date string in both
``program_start_date`` and ``program_end_date``). Every rendered entity is
asserted to have >=1 matching source: relation-detection's entity lists are
built only from valid rows' non-null values
(``relation_prompts.build_relation_lists`` forces ``min_invalid_per_list=0,
max_invalid_per_list=0``), so a rendered entity with zero matching sources
would mean the join itself is broken, not an expected corpus edge case.
Conversely, every non-null value in every (collapsed) valid source is asserted
to appear among the rendered entities -- with ``min_valid_per_type=1`` no
field with a valid value is dropped from rendering, so a miss there is also a
join bug, not corpus structure.

**Coloring.** Each source gets one color from a golden-angle HSV cycle --
chosen because the number of sources per document ranges 1 to 110 (median 13,
2026-09-28 measurement over line/train's ``tuple_membership.json``), far
beyond any fixed categorical design-system palette (e.g. Okabe-Ito's 8), and a
golden-angle hue step maximizes pairwise hue separation for an arbitrary,
not-known-in-advance count. An entity matching exactly one source is drawn as
a plain filled marker in that source's color; an entity matching multiple
sources is drawn as a pie marker split evenly among its matching sources'
colors -- a literal multi-color mark for "belongs to several sources
simultaneously", per the task spec. Because a full color legend is unreadable
past a handful of sources, this script always prints a per-document summary
(entity count, source count, multi-source count) so a document can be judged
before or after plotting for whether its figure will actually be legible; an
in-plot legend is added only when the source count is small enough to fit one.

PCA here needs no separate per-document centering step (unlike
``analysis/entity_detection.py``'s pooled-across-documents case): the whole
analysis is already scoped to one document, so sklearn's own mean subtraction
inside ``PCA.fit_transform`` *is* the "centered" of "centered PCA
representations" the task asks for.

Figures are saved to ``analysis/figures/`` instead of being displayed inline.
"""

import argparse
import colorsys
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from sklearn.decomposition import PCA

from judge_interp.prompts import LINE_FIELDS

RUNS_ROOT = Path(os.environ["RUNS_ROOT"])
FIGURES_DIR = Path(__file__).parent / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "Times New Roman", "DejaVu Serif"],
})

RUN_IDS = {
    "qwen7b": "2026-09-27-qwen7b-relation-line-train-01",
    "llama8b": "2026-09-27-llama8b-relation-line-train-01",
}

# Above this many sources, an in-figure legend is unreadable -- a full
# color-keyed source table is always printed to stdout regardless (see
# analyze_and_plot_document), but the in-figure legend is only added below
# this count.
MAX_LEGEND_SOURCES = 12

# Marker size (matplotlib scatter `s`, points^2) for a single-source entity's
# plain dot -- pie markers are scaled to match this exactly (see
# _pie_marker_and_size).
MARKER_SIZE = 40

# Sources per document at or above this count still produce a legible figure
# (enough hue separation, printed table still short); used only to suggest
# candidate document_ids in --sweep mode, not to gate anything.
GOOD_SOURCE_COUNT_RANGE = (5, 12)


def load_run(run_id: str) -> dict:
    """Load one relation_detection run: per-layer content-span representations
    plus ``doc_ids`` / ``entity_type`` / ``value``, and its
    ``tuple_membership.json``.

    Hard-gates on ``run.json``'s status and on every span being ``kind ==
    "content"`` -- relation_detection renders no cue spans by design (see
    ``run_experiment.py``'s ``compute_span_metrics`` docstring), so a "cue"
    span here would mean the wrong artifact was read.
    """
    run_dir = RUNS_ROOT / run_id
    manifest = json.loads((run_dir / "run.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())

    assert manifest["status"] == "success", f"{run_id}: run.json status is {manifest['status']!r}, not 'success'"
    assert manifest["task"] == "relation_detection", (
        f"{run_id}: task is {manifest['task']!r}, not 'relation_detection'"
    )
    assert metrics["n_cue_spans"] == 0, f"{run_id}: n_cue_spans={metrics['n_cue_spans']}, expected 0"

    with np.load(run_dir / "artifacts" / "span_representations.npz", allow_pickle=False) as z:
        layers = z["layers"].tolist()
        kind = z["kind"]
        assert set(np.unique(kind).tolist()) == {"content"}, (
            f"{run_id}: unexpected span kind(s) {sorted(set(kind.tolist()))}"
        )
        doc_ids = z["doc_ids"]
        entity_type = z["entity_type"]
        value = z["value"]
        reps = {L: z[f"rep_{L}"] for L in layers}

    n = int(metrics["n_spans"])
    assert doc_ids.shape == (n,)
    assert entity_type.shape == (n,)
    assert value.shape == (n,)
    for L, arr in reps.items():
        assert arr.shape[0] == n, (L, arr.shape)

    tuple_membership = json.loads((run_dir / "artifacts" / "tuple_membership.json").read_text())

    print(
        f"{run_id}: status={manifest['status']}, git_sha={manifest['git_sha']}, "
        f"git_dirty={manifest['git_dirty']}, n_entities={n}, "
        f"n_documents={len(set(doc_ids.tolist()))}, layers={layers}, "
        f"hidden={reps[layers[0]].shape[1]}"
    )

    return {
        "run_id": run_id,
        "layers": layers,
        "reps": reps,
        "doc_ids": doc_ids,
        "entity_type": entity_type,
        "value": value,
        "tuple_membership": tuple_membership,
    }


def _collapse_valid_sources(tuples: list[dict]) -> list[dict]:
    """Collapse ``valid: true`` tuples with byte-identical ``entity_index``
    values into one source each. See this module's docstring.

    Returns one dict per source, sorted by its smallest ``line_index``
    (deterministic order -> deterministic color assignment):
    ``{"line_indices": [int, ...], "entity_index": {field: value}}``.
    """
    valid_tuples = [t for t in tuples if t["valid"]]
    assert valid_tuples, "document has no valid tuples -- cannot build any source"

    by_key: dict[tuple, dict] = {}
    for t in valid_tuples:
        key = tuple(sorted(t["entity_index"].items()))
        line_index = t["row_key"][1]
        if key not in by_key:
            by_key[key] = {"line_indices": [line_index], "entity_index": t["entity_index"]}
        else:
            assert by_key[key]["entity_index"] == t["entity_index"]
            by_key[key]["line_indices"].append(line_index)

    sources = list(by_key.values())
    for src in sources:
        src["line_indices"].sort()
    sources.sort(key=lambda s: s["line_indices"][0])
    return sources


def _golden_angle_colors(n: int) -> list[tuple[float, float, float]]:
    """``n`` RGB colors spaced by the golden angle (~137.5 deg) in hue, fixed
    saturation/value -- maximizes pairwise hue separation for an a-priori
    unknown, potentially large ``n`` (up to 110 sources/document here). See
    this module's docstring for why no fixed categorical palette is used.
    """
    golden_angle = 0.6180339887498949
    return [colorsys.hsv_to_rgb((i * golden_angle) % 1.0, 0.75, 0.85) for i in range(n)]


def _match_sources(entity_type: str, value: str, sources: list[dict]) -> list[int]:
    return [i for i, src in enumerate(sources) if src["entity_index"].get(entity_type) == value]


def _wedge_marker(frac_start: float, frac_end: float, n_arc: int = 24) -> np.ndarray:
    """Vertices of one pie wedge (as a fraction of a full circle,
    ``frac_start``/``frac_end`` in ``[0, 1]``), centered at the origin with
    unit radius, closed back to the origin -- usable directly as a
    ``matplotlib`` scatter ``marker``.
    """
    angles = np.linspace(2 * np.pi * frac_start, 2 * np.pi * frac_end, n_arc)
    x = np.concatenate([[0.0], np.cos(angles), [0.0]])
    y = np.concatenate([[0.0], np.sin(angles), [0.0]])
    return np.column_stack([x, y])


def _wedge_marker_size(verts: np.ndarray, target_size: float) -> float:
    """``scatter``'s ``s`` for ``verts`` so it renders at the same on-screen
    size as a plain circular marker plotted with ``s=target_size``.

    ``scatter`` rescales a custom-path marker to its own vertex extent, so a
    narrow wedge (small ``|verts|.max()``) would otherwise render larger than
    a wide one at the same ``s`` -- multiplying by ``|verts|.max() ** 2``
    (vertices lie on the unit circle here, so this is always 1, but the
    factor is kept so the helper is correct for a non-unit-radius marker too)
    undoes that rescaling.
    """
    return target_size * float(np.abs(verts).max() ** 2)


def _scatter_pie(ax, x: float, y: float, colors: list, target_size: float) -> None:
    """Draw one point as a pie marker split evenly among ``colors``, via
    ``n`` overlaid custom-marker scatter calls -- markers are sized/shaped in
    display space (points), unlike a data-space patch (e.g. ``Wedge``), so
    the result is a round pie regardless of the axes' data aspect ratio or
    which point happens to set the axis limits.
    """
    n = len(colors)
    for i, color in enumerate(colors):
        verts = _wedge_marker(i / n, (i + 1) / n)
        ax.scatter(
            [x], [y], marker=verts, s=_wedge_marker_size(verts, target_size),
            facecolor=color, edgecolors="black", linewidths=0.3,
        )


def _join_document(doc_id: str, keys: list[tuple[str, str]], tuples: list[dict]) -> tuple[list[dict], list[list[int]]]:
    """Collapse ``doc_id``'s valid tuples into sources and join ``keys``
    (``(entity_type, value)`` per rendered entity) to them, asserting the join
    is complete in both directions. Shared by ``analyze_and_plot_document``
    and ``sweep_all_documents`` so their validation can't drift apart.
    """
    assert len(set(keys)) == len(keys), f"{doc_id}: duplicate (entity_type, value) rendered entities"

    sources = _collapse_valid_sources(tuples)
    matches = [_match_sources(et, v, sources) for et, v in keys]
    for (et, v), m in zip(keys, matches):
        assert m, f"{doc_id}: rendered entity ({et!r}, {v!r}) matches no valid source -- join is broken"

    rendered_set = set(keys)
    for src in sources:
        for field in LINE_FIELDS:
            v = src["entity_index"].get(field)
            if v is None:
                continue
            assert (field, v) in rendered_set, (
                f"{doc_id}: source {src['line_indices']}'s {field}={v!r} was never rendered -- join is broken"
            )
    return sources, matches


def sweep_all_documents(run: dict) -> None:
    """Validate the source-collapse + entity<->source join for every document
    in ``run["tuple_membership"]`` (no plotting). The per-document asserts in
    ``analyze_and_plot_document`` were only ever exercised, during
    development, against a handful of hand-picked documents -- this checks
    all of them before a user passes an arbitrary ``document_id``.

    Also prints, per ``entity_type``, the share of that type's rendered
    entities that are multi-source -- e.g. a value like a channel call sign
    or a recurring date is far more likely to be shared across several
    source line items than a free-text ``program_desc`` is, so pie-vs-dot
    placement in a document's figure partly reflects entity *type*, not only
    "how many sources this value happens to have" -- and suggests a few
    ``document_id``s whose source count falls in ``GOOD_SOURCE_COUNT_RANGE``
    (enough sources for the hypothesis to be interesting, few enough for the
    figure and the printed table to stay legible).
    """
    doc_ids = run["doc_ids"]
    entity_type_all = run["entity_type"]
    value_all = run["value"]

    docs = sorted(run["tuple_membership"])
    assert set(docs) == set(np.unique(doc_ids).tolist()), (
        "tuple_membership.json document set != span_representations.npz document set"
    )

    n_multi_by_type: dict[str, int] = {f: 0 for f in LINE_FIELDS}
    n_total_by_type: dict[str, int] = {f: 0 for f in LINE_FIELDS}
    candidates: list[tuple[str, int]] = []

    for doc_id in docs:
        mask = doc_ids == doc_id
        keys = list(zip(entity_type_all[mask].tolist(), value_all[mask].tolist()))
        sources, matches = _join_document(doc_id, keys, run["tuple_membership"][doc_id])

        for (et, _), m in zip(keys, matches):
            n_total_by_type[et] += 1
            if len(m) > 1:
                n_multi_by_type[et] += 1

        if GOOD_SOURCE_COUNT_RANGE[0] <= len(sources) <= GOOD_SOURCE_COUNT_RANGE[1]:
            candidates.append((doc_id, len(sources)))

    print(f"swept {len(docs)} documents: join + reverse-check OK on every one")
    print("multi-source rate by entity_type:")
    for f in LINE_FIELDS:
        if n_total_by_type[f] == 0:
            continue
        pct = 100.0 * n_multi_by_type[f] / n_total_by_type[f]
        print(f"    {f}: {n_multi_by_type[f]}/{n_total_by_type[f]} ({pct:.1f}%) multi-source")
    print(
        f"{len(candidates)} document(s) with {GOOD_SOURCE_COUNT_RANGE[0]}-{GOOD_SOURCE_COUNT_RANGE[1]} "
        "sources (legible figure candidates), e.g.:"
    )
    for doc_id, n_sources in candidates[:10]:
        print(f"    {doc_id} ({n_sources} sources)")


def analyze_and_plot_document(run: dict, document_id: str, out_path: Path) -> None:
    doc_ids = run["doc_ids"]
    mask = doc_ids == document_id
    assert mask.any(), (
        f"document_id {document_id!r} not found among {len(set(doc_ids.tolist()))} documents "
        f"in {run['run_id']}"
    )
    assert document_id in run["tuple_membership"], (
        f"document_id {document_id!r} has entities in {run['run_id']} but no entry in "
        "tuple_membership.json -- inconsistent artifacts"
    )

    entity_type = run["entity_type"][mask]
    value = run["value"][mask]
    n = int(mask.sum())

    keys = list(zip(entity_type.tolist(), value.tolist()))
    n_sources_raw = sum(1 for t in run["tuple_membership"][document_id] if t["valid"])
    sources, matches = _join_document(document_id, keys, run["tuple_membership"][document_id])
    if n_sources_raw != len(sources):
        print(f"[{document_id}] collapsed {n_sources_raw} valid tuples into {len(sources)} distinct source(s)")

    n_multi = sum(1 for m in matches if len(m) > 1)
    print(
        f"[{document_id}] {n} entities, {len(sources)} source(s) "
        f"({n_multi} multi-source entit{'y' if n_multi == 1 else 'ies'})"
    )

    colors = _golden_angle_colors(len(sources))
    layers = run["layers"]

    fig, axes = plt.subplots(
        1, len(layers), figsize=(5.5 * len(layers), 5.5), squeeze=False, constrained_layout=True
    )
    axes = axes[0]
    for ax, layer in zip(axes, layers):
        x = run["reps"][layer][mask]
        pca = PCA(n_components=2)
        scores = pca.fit_transform(x)

        for i in range(n):
            m = matches[i]
            if len(m) == 1:
                ax.scatter(
                    scores[i, 0], scores[i, 1], color=colors[m[0]], s=MARKER_SIZE,
                    edgecolors="black", linewidths=0.3,
                )
            else:
                _scatter_pie(ax, scores[i, 0], scores[i, 1], [colors[j] for j in m], MARKER_SIZE)

        var1, var2 = pca.explained_variance_ratio_[:2]
        ax.set_xlabel(f"PC1 ({var1:.1%})")
        ax.set_ylabel(f"PC2 ({var2:.1%})")
        ax.set_title(f"layer {layer}")
        ax.set_box_aspect(1)

    # Always print a color-keyed source table -- an in-figure legend past
    # MAX_LEGEND_SOURCES is unreadable, but a printed table with no color
    # column is useless for mapping a pie slice back to its source. Printing
    # it unconditionally also covers the in-between case (e.g. the median
    # document's 13 sources, just over MAX_LEGEND_SOURCES=12) with a real
    # color reference rather than nothing.
    print(f"[{document_id}] source table (color, line_indices, entity_index):")
    for i, (color, src) in enumerate(zip(colors, sources)):
        print(f"    source {i} {to_hex(color)}: line_indices={src['line_indices']} entity_index={src['entity_index']}")

    if len(sources) <= MAX_LEGEND_SOURCES:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="black", markersize=8,
                       label=f"line {src['line_indices']}")
            for c, src in zip(colors, sources)
        ]
        axes[0].legend(handles=handles, loc="best", fontsize=7, framealpha=0.9, title="source (line_index)")
    else:
        print(f"[{document_id}] {len(sources)} sources > {MAX_LEGEND_SOURCES} -- skipping in-figure legend, see source table above")

    fig.suptitle(f"relation_detection | doc {document_id} | {run['run_id']}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{document_id}] saved figure to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "document_id", nargs="?", default=None,
        help="document_id to plot (must exist in the chosen run's tuple_membership.json); "
        "omit when using --sweep",
    )
    ap.add_argument("--model", choices=sorted(RUN_IDS), required=True)
    ap.add_argument(
        "--sweep", action="store_true",
        help="validate the source join over every document (no plotting) and suggest legible "
        "document_ids instead of plotting one",
    )
    args = ap.parse_args()
    if args.sweep == (args.document_id is not None):
        ap.error("pass exactly one of a document_id or --sweep")

    run = load_run(RUN_IDS[args.model])
    if args.sweep:
        sweep_all_documents(run)
        return
    out_path = FIGURES_DIR / f"{args.model}_relation_{args.document_id}_pca.png"
    analyze_and_plot_document(run, args.document_id, out_path)


if __name__ == "__main__":
    main()
