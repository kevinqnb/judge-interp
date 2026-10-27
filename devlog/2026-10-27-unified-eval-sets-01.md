<!-- Public devlog entry: judge-interp/devlog/2026-10-27-unified-eval-sets-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-10-27-unified-eval-sets-01
kind: build
---

## Session 2026-10-27

### Prompts

- Update `analysis/entity_detection.py` and `analysis/relation_set_detection.py` to evaluate all language models, clef and jev simultaneously in the same file, over unified test sets so methods compare fairly.
- Entity detection: train probes only on `data/vrdu/main/train.json`, evaluate everything on `data/vrdu/main/test.json`.
- Relation detection: train probes on `data/vrdu/line/train_subsets.json`, evaluate everything on the test subsets (`test_subsets.json`; the prompt said train, a typo).
- clef and jev need no training and should already have precomputed results (they do not for the subsets; build with fixtures only).
- Approved replacing the relset pair assertions with row-level metrics as a standalone commit; common set = intersection of what every method scored, drops reported.
- Commit the changes and write a /devlog note.

### Implemented

`analysis/entity_detection.py` now takes `params.models` (probes) and `params.judges` (clef/jev), cuts every method to the test rows all of them scored, builds the minimal pairs per method and asserts they are identical. `analysis/relation_set_detection.py` is rewritten to score the same way on the rows of `data/vrdu/line/test_subsets.json` (unbalanced, label = the row's `valid`), with one metric path for every method; this changes relset eval logic and invalidates earlier relset numbers. Configs: `analysis/configs/entity_detection/2026-10-27-entity-main-all-{smoke-,}01.yaml` and `analysis/configs/relation_set_detection/2026-10-27-relset-subsets-all-01.yaml`. Entity was smoke-run on real runs; relation is tested on fixtures and the real data only, since its runs do not exist yet.

### Commits

08db241 entity detection: probes, Clef and Jev scored together on one common set of test pairs
8fa8d01 relation set detection: EVAL LOGIC CHANGE, score all methods on test_subsets rows
