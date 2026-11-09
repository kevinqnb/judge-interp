<!-- Public devlog entry: judge-interp/devlog/2026-11-09-relation-clustering-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-11-09-relation-clustering-01
kind: build
config: analysis/configs/relation_detection_clustering/2026-11-09-relclust-02.yaml
---

## Session 2026-11-09

### Prompts

- Set up an analysis for relation detection: do entities of one line item cluster, compared with the other entities of the same corrupted full-row prompt? Code in `analysis/relation_detection_clustering.py`. Remove the entity's raw-token component first, to isolate a "binding vector". Make 2-d PCA plots for a few chosen examples (colour, marker and fill encode membership, corruption and corruption visibility), chosen across documents and across (error type, k) within a document.
- Design answers: subtract the per-field, per-document mean over all rows of the test run (the layer-0 subtraction turned out to be a constant shift); value-based membership.
- Submit the full run through the wrapper (several times, after fixes); check on the job.
- Exclude prompts with a zero-centred entity, as a union across both models.
- Restyle the plots: colour only by corrupted / not, triangle = a corruption is visible before, circle = hidden, dashed edges between entities that share a ground-truth line item (black if neither is corrupted, light grey otherwise).
- Drop what value-based colouring left behind, re-rank the example selection, and add the paired twin difference.
- Commit the work and write a /devlog note.

### Implemented

`analysis/relation_detection_clustering.py` scores, for each corrupted full-row prompt of a relation-detection test run, whether entity pairs of one line item are closer than pairs of different line items (`win_frac`), on raw and per-(document, field)-centred vectors, with Euclidean and cosine distances and document-bootstrap intervals. Controls: a random-vector and a layer-0 known answer, a per-prompt permutation baseline, seed determinism, and a clean-twin baseline (`twin_win_frac`, `contrast`, and the paired `win_vs_twin`). Prompts with a small (document, field) group or a zero-centred entity in either model are excluded for both models and counted. It writes PCA example grids (random, seeded examples) with the requested encoding. `tests/test_analysis_relation_detection_clustering.py` has 20 hand-built-fixture tests; smoke runs pass and the `-02` smoke reproduces the `-01` smoke's shared metrics exactly. Configs: `analysis/configs/relation_detection_clustering/2026-11-09-relclust-{smoke-01,01,smoke-02,02}.yaml`; the `-01` full run has finished, the `-02` full run is not yet submitted.

### Commits

622e33f analysis: relation detection clustering (within vs between line-item distances in corrupted prompts)
909ee2e configs: relation detection clustering smoke and full runs (-01, -02)
