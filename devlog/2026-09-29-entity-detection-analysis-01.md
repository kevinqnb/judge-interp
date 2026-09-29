---
id: 2026-09-29-entity-detection-analysis-01
kind: build
config: configs/2026-09-29-qwen7b-entity-mixed-analysis-03.yaml
---

## Session 2026-09-29

### Prompts

> Update the entity detection analysis in `analysis/entity_detection.py` to reflect
> the move of the task to `entity_detection.py`. Load the qwen7b mixed train and
> test runs (layers 0 and last). Preprocess: (a) subtract layer 0 from last layer to
> make a token-normalized set X, (b) mean-center per document, (c) mean-center the
> whole set. PCA on the train split colored by validity, and an identical plot
> colored by error rate k/m. Logistic regression train -> test with precision,
> accuracy, recall, F1, AUROC, then a smooth ECE and calibration diagram following
> scholarlm's `calibration_updated.py`.

Follow-ups (summary): submit as a CPU-only job via `submit.sh`; subtract
per-entity-type means after per-document means and standardize; add a probe-axis
plot; change the error-rate color scale; write figures to `analysis/figures/`.

### Implemented

`analysis/entity_detection.py` is rewritten for the mixed task: token-normalised X
with per-document, optional per-entity-type, and global centering plus optional
standardization; train PCA colored by validity and k/m; a probe-axis plot; and a
logistic-regression probe with metrics, smooth ECE, and shuffled-label and
seed-determinism controls. `scripts/submit.sh` gains an explicit `--analysis` CPU-only
mode, and `relplot` joins the `analysis` extra. Runs are recorded in
`configs/2026-09-29-qwen7b-entity-mixed-analysis-{01,02,03}.yaml`. Unit tests are in
`tests/test_analysis_entity_detection.py`.

### Commits
3dfc3da Rewrite entity_detection analysis for the mixed task
