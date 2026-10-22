<!-- Public devlog entry: judge-interp/devlog/2026-10-22-experiments-migration-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-10-22-experiments-migration-01
kind: build
---

## Session 2026-10-22

### Prompts

> Migrate the code for running experiments to `experiments/`: the runners from `scripts/` become high-level runners there, and the configs from `configs/` go into `experiments/experiment-configs/`, organized in subdirectories (entity detection, relation detection, clef in either case).
>
> Clef only needs the vrdu test sets. Route main test to its entity path; for line, create separate experiments for entity detection, relation detection and one-shot validation. Relation and one-shot data come from the randomized test sets in `data/vrdu/subsets.json`, via a helper that treats the first field of each subset as primary and the rest as conditions.

> Analysis configs should be moved to `analysis/configs/{analysis_type}/...`.

### Implemented

Runners (`experiments/run_experiment.py`, `submit.sh`, `_run_experiment_job.sh`) and all run configs now live under `experiments/`, grouped in `experiments/experiment-configs/`; analysis configs moved to `analysis/configs/<type>/`. `submit.sh` resolves an id by searching those trees and requires exactly one match; a layout test checks id uniqueness. `experiments/clef_samples.py` turns `subsets.json` test replicates into Clef samples files (first field primary, rest conditions; one-shot uses all fields), and 12 new configs (`2026-10-22-clef-entity-{main,line}-test-02`, `2026-10-22-clef-{conditional,oneshot}-line-test-rep{0-4}-01`) route the data. Verified with unit tests and a model-free unit-building check only; no GPU run yet.

### Commits

4944510 experiments: move runners and configs under experiments/ and analysis/configs/
07d6f36 clef: route test subsets to conditional and one-shot tasks
