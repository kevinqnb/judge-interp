<!-- Public devlog entry: judge-interp/devlog/2026-11-10-relation-grouping-leace-01.md.
     Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-11-10-relation-grouping-leace-01
kind: build
config: analysis/configs/relation_detection_leace/2026-11-10-relleace-01.yaml
---

## Session 2026-11-10

### Prompts

- Try, one at a time, two changes to see whether entity representations separate line items better in the PCA / graph plots, without overwriting the existing clustering results: (1) rerun the 12 plotted settings with instructions that ask the model to group the entities by relational structure (greedy: first group where the entity is related to all of the entities in it), worded for review before implementing, with a small new experiment script; (2) use LEACE to isolate the binding vectors.
- Approved the wording with the "all of the entities" change; submit the checks and runs through the wrapper (the Llama tiny gate skipped).
- Make a distribution plot of the distances between entities joined by a ground-truth edge and between unconnected entities, and the difference of the means.
- Try LEACE; submit the full run as configured.
- Stop there, commit and write the devlog.

### Implemented

A new task `relation_grouping` (`src/judge_interp/relation_grouping.py`, registered in `experiments/run_experiment.py`) reuses the relation-detection items and read-out with a grouping instruction block, on every row of chosen documents; collection configs are in `experiments/experiment-configs/relation-grouping/`. `analysis/relation_grouping_compare.py` runs the clustering analysis on those runs with fixed example prompts and compares with the earlier run; `analysis/relation_grouping_distances.py` compares the distances of connected and unconnected entity pairs; `analysis/relation_detection_leace.py` erases field and position with LEACE (fitted on the train run) and re-runs the clustering metrics, with controls. Configs are under `analysis/configs/relation_grouping_compare/`, `analysis/configs/relation_grouping_distances/` and `analysis/configs/relation_detection_leace/` (full run `2026-11-10-relleace-01`); each module has unit tests, and the known-answer, smoke and full-run steps passed.

### Commits

cc7005a relation grouping task: relation-detection read-out under grouping instructions, per-document runs
3924ac4 analysis: grouping-vs-detection comparison, connected/unconnected distance distributions, LEACE erasure of field/position
