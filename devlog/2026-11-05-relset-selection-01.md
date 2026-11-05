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
