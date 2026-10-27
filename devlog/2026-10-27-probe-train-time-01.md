---
id: 2026-10-27-probe-train-time-01
kind: build
---

## Session 2026-10-27

### Prompts

- Update `analysis/entity_detection.py` and `analysis/relation_set_detection.py` to record compute time for training the probes, saved in `analysis/results/` for `analysis/efficiency.py` to read later.
- Commit the changes and write a /devlog note.

### Implemented

Both analyses now time the main logistic-regression probe fit and the train-feature build, and write `analysis/results/<run id>/probe_train_time.json` (per probe label: `feature_seconds`, `fit_seconds`, `n_train`, `n_features`). Shared helpers live in `analysis/entity_detection.py` and are reused by `analysis/relation_set_detection.py`. Metrics are unchanged and `efficiency.py` does not read the files yet. Only syntax-checked so far; no config touched.

### Commits

baa602c analysis: record probe training time to analysis/results/<id>/probe_train_time.json
