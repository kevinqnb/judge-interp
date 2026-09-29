---
id: 2026-09-29-relation-detection-analysis-01
kind: build
config: configs/2026-09-29-qwen7b-relation-listed-analysis-01.yaml
---

## Session 2026-09-29

### Prompts

> Update the relation detection analysis in `analysis/relation_detection.py` to reflect the move of the task to `src/judge_interp/relation_detection.py`; scrap the original file if needed. Load the qwen7b relation-listed train and test runs (layers 0 and last). Preprocess: subtract layer 0 from last layer to make token-normalized X, mean-center per document. PCA on the train split colored by source data point (grey for entities in multiple sources), plots for a few documents. Build Z^Tr and Z^Tst by sampling a document, an entity, a valid tuple containing it, keeping k other entities (valid) and replacing q <= k of them with decoys (invalid). Feature: euclidean distance between the entity and the mean of the other entities. Logistic regression train -> test with standard metrics, then smooth ECE and a calibration diagram following scholarlm's `calibration_updated.py`.

Follow-ups (summary): also mean-center per entity type; save figures to `analysis/figures`; diagnose why the distance feature is at chance; try a probe on |x_i - x_j| and controls for it; replace the feature with an m-choose-2 vector of cosine similarities per relation type (0 where absent); try 1 for absent slots; drop samples with no possible decoy swap so both classes share type patterns.

### Implemented

`analysis/relation_detection.py` is rewritten for the listed task: token-normalised X with per-document and per-entity-type centering, train PCA colored by source tuple (grey = multi-source) for example documents, a paired valid/invalid partial-tuple sampler, a 10-dim relation-type cosine feature vector with an explicit `absent_fill`, a logistic-regression probe with metrics, smooth ECE and a reliability diagram, and shuffled-label, seed-determinism, noise-X and presence-only controls with a document-bootstrap AUROC CI. Configs are `configs/2026-09-29-qwen7b-relation-listed-analysis-01.yaml` (fill 0), `-02` (fill 1) and `-smoke-01`. Unit tests are in `tests/test_analysis_relation_detection.py`. Results are in the private note.

### Commits
125725d Rewrite relation_detection analysis for the listed task
