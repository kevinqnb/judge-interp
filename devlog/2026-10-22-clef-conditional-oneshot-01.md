<!-- Public devlog entry: judge-interp/devlog/2026-10-22-clef-conditional-oneshot-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-10-22-clef-conditional-oneshot-01
kind: build
---

## Session 2026-10-22

### Prompts

> Restructure `src/judge_interp/clef.py` (entity and relation currently separate pathways). Update relation detection to an explicitly conditional question: given a primary entity A and conditional entities B and C, evaluate A with respect to B and C; True if A is correctly associated in the document with B and C, False if it is not related to them. This intentionally departs from the prompt in `relation_detection.py`, though the downstream goal is still conditionals.
>
> Add a third pathway, one-shot judgement: given a subset of fields, is the entire record valid? All entities must be present and of the correct type, and all correctly related. Purpose: explore a path to probabilities that do not rely on the conditional relation assessment.

(Summary of answers to design questions.) Subsets will eventually come from `subsets.json`, but build general-purpose base code that takes any input dataset, not tailored to `subsets.json` or VRDU. Conditional: a single question per record, A is the first field, conditions the rest; make primary and conditions explicit input arguments rather than a first-item heuristic. Label: false if any condition is corrupted, even when the primary is valid. Wording approved; commit and /devlog.

### Implemented

Replaced `clef_relation_row` in `src/judge_interp/clef.py` with `clef_relation_conditional` (is an explicit primary entity related to all its conditions) and added `clef_record_oneshot` (is the whole record valid: explicit support, type, and relation). Both read a generic samples file (`params.samples_path`: ordered entities with per-entity validity, explicit primary/conditions), independent of `subsets.json` and VRDU; the label is true iff every entity is valid. `scripts/run_experiment.py` registers the tasks; smoke configs `configs/2026-10-22-clef-conditional-smoke-01.yaml` and `configs/2026-10-22-clef-oneshot-smoke-01.yaml` run over hand-built fixtures in `tests/fixtures/clef_samples_mini/`. Unit tests pass (rung 1); no model run yet. `analysis/clef.py` does not read the new tasks, and the old `clef_relation_row` configs no longer load.

### Commits

be35987 clef: conditional relation and one-shot record tasks over a generic samples file
