<!-- Public devlog entry: judge-interp/devlog/2026-09-15-repr-analysis-multimodel-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-09-15-repr-analysis-multimodel-01
kind: build
---

## Session 2026-09-15

### Prompts

> Let's set up analysis/representations.py for the updated
> 2026-09-15-qwen7b-repr-line-train-01.yaml and 2026-09-15-llama8b-repr-line-train-01.yaml
> experiments (and also the main dataset experiments). What's new is that (a) we have
> a new model -- which should occupy its own figures, (b) we have representations
> from 3 layers for each experiment (for which we should put together in a 3 plot
> subfigure, and (c) that there are now two classes of errors in the line datasets,
> for which we should also split into separate PCA plots.

### Implemented

Rewrote `analysis/representations.py` to cover both new named-model configs
(`configs/2026-09-15-{qwen7b,llama8b}-repr-{main,line}-train-01.yaml`) instead of the
prior single-model, single-layer version. Each (model, dataset) run now gets its own
figure with a 1x3 subplot, one panel per collected layer, sharing a row subsample and
color scale across panels; the line dataset additionally gets two more figures per
model split by the new `error_type` field (`inter_document` / `intra_document`), each
alongside the combined figure, with valid rows shared as the zero-invalid anchor
between the two splits. Because the script now fits many independent PCAs per run
(model x layer x subset), added a fixed PC1 sign convention (valid group's centroid
pinned negative) so panels are comparable, plus a print-only `p_true`-vs-
`num_invalid_fields` sanity check and hard asserts on the valid-anchor invariant and
on degenerate PCA subsets. Verified against the existing tiny fixture
(`repr_tiny_e2e_line`) and ran for real against both finished main-train runs; the two
line-train runs are still in progress on the cluster, and the script correctly
hard-fails on them until their `metrics.json` lands.

### Commits

81caf4c analysis: per-model, multi-layer, error-type-split PCA figures
