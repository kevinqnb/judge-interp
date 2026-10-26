<!-- Public devlog entry: judge-interp/devlog/2026-10-26-analysis-efficiency-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-10-26-analysis-efficiency-01
kind: build
config: analysis/configs/ (2026-10-26 entity_detection, relation_detection, clef, efficiency)
---

## Session 2026-10-26

### Prompts

> Please help me assess the current working state of the `analysis/` directory scripts and their applicability to
> the latest round of experiments (filed under `experiments/experiment-configs/` and dated 2026-10-25).

> write new analysis configs for entity-main, relation-line and Clef. Qwen and llama should be kept separate -- for
> each I want to analyze with the last layer and with a layer slightly before (e.g. 24 or 28). We should fix the
> pairing logic of the line entity analysis. We should also include support for entity detection with the line
> dataset, and a new `efficiency.py` script to analyze computational performance.

> Regarding entity detection for line items, this should be done at the entity level (like we do for main), not over
> pairs. Can you confirm this behavior? Did the entity line experiment collect the necessary data for this?

### Implemented

The assessment found no existing analysis config could run the current scripts (missing `twin_seed`, `ci_level`,
`auroc_n_boot`; retired Clef `kind`; pre-06 run ids) and that entity detection on the line dataset had no analysis path.
`analysis/entity_detection.py` gains `line_span_pairs` (a corrupted span against the same field of its clean twin
sample) behind a required `params.dataset: main|line`; main output is unchanged. New `analysis/efficiency.py` recomputes
wall time by phase (including the per-row cache cleanup, reported separately), seconds per judged row and item,
a compute-vs-tokens fit, GPU memory and storage from each run's per-unit efficiency arrays, cross-checks them against
`efficiency.json`, refuses mixed GPUs or resumed shards, and reports Llama/Qwen and Clef/representation ratios. New
configs per model and layer (Qwen 28 and 24, Llama 32 and 28) plus smoke, Clef and efficiency configs are under
`analysis/configs/`; tests cover each script on hand-built fixtures and the 240-test suite passes. Smoke runs on 20
documents per split completed with the controls holding; no full probe run has been made and walltimes are placeholders.

### Commits

78cc283 entity_detection analysis: pair line-dataset spans with their clean twin's span
0485a52 analysis: add efficiency.py for judge runtime, token, memory and storage cost
3e427df configs: 2026-10-26 analysis configs per model and layer, Clef and efficiency
