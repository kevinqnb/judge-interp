<!-- Public devlog entry: judge-interp/devlog/2026-11-05-relset-selection-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-11-05-relset-selection-01
kind: build
config: analysis/configs/relation_set_detection/2026-11-05-relset-corrupted-parents-01.yaml
---

## Session 2026-11-05

### Prompts

- Evaluate all subsets from corrupted parents only (option B). Control for subset size, field presence and corruption count with a one-hot probe baseline; confirm the baseline design before implementing.
- Answers: train the probes and the baseline on corrupted-parent train rows only; drop the twin diagnostic in this mode. Check whether the corruption count belongs in the baseline: it should be a baseline of artificial features, not a ceiling. Then: use just the field-set one-hot.
- Remove the paired selection setting. Clean up the file: a concise, high-level docstring, and function signatures with every input and output described.
- Submit the full corrupted-parents run.
- Commit as a standalone eval-logic commit and write a /devlog note.

### Implemented

`analysis/relation_set_detection.py`: `params.selection` is now `null` (every row) or `corrupted_parents` (every row of a corrupted parent, none of a clean parent, for train and test alike); the paired selection is removed. The structure baseline stays the field-set one-hot, trained on the same restricted train rows and reported, not asserted. The corruption count is deliberately not a feature: it is derived from the label and no method sees it, so it is a report stratum instead. With no clean-parent rows the twin diagnostics and the control report are skipped (`cut(..., link_twins=False)`). The module docstring is shortened and every function has Args / Returns / Raises. This changes eval logic: the paired mode and its metric keys are gone, so the `2026-10-28-relset-paired-*` configs no longer load; with `selection: null` a smoke run's `metrics.json` is identical before and after. Tests: 6 paired-mode tests removed, 6 added. Configs: `analysis/configs/relation_set_detection/2026-11-05-relset-corrupted-parents-{smoke-,}01.yaml`; the full run is submitted, not yet read.

### Commits

b8c51c2 relation set detection: EVAL LOGIC CHANGE, selection = null | corrupted_parents (paired removed)
64d15f7 configs: relation set detection on corrupted-parent subsets (smoke and full)

## Session 2026-11-05 (2)

### Prompts

- Randomly subsample the corrupted-parents set so that subset sizes and field combinations are balanced; make it the behavior of the corrupted-parents setting, not a new selection setting.
- Commit it, add it to the previous devlog, and submit the full balanced run.

### Implemented

`params.selection: corrupted_parents` in `analysis/relation_set_detection.py` now takes every corrupted-parent row and then randomly subsamples, for train and test, to as many valid as invalid rows in every (error type, field set) cell (`balance_cells`, seeded by the config's `seed`; one-class cells are dropped). On the test side the draw is made after the common set, so every method is scored on the same exactly balanced rows and over-length drops are not confused with balancing. The field-set structure baseline is therefore asserted at AUROC exactly 0.5 (`structure_baseline(..., assert_chance)`). This changes eval logic: the unbalanced corrupted-parents behavior is gone and its earlier run is not reproducible; with `selection: null` a smoke run's `metrics.json` is identical before and after. The balanced test size of a smoke run matched a recomputation from the raw subsets files. Configs: `analysis/configs/relation_set_detection/2026-11-05-relset-corrupted-parents-balanced-{smoke-,}01.yaml`; the full run is submitted, not yet read.

### Commits

2ef5629 relation set detection: EVAL LOGIC CHANGE, corrupted_parents is balanced per (error type, field set)
a5f802c configs: relation set detection on balanced corrupted-parent subsets (smoke and full)
