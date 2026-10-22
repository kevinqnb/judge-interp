<!-- Public devlog entry: judge-interp/devlog/2026-10-21-clef-judge-alignment-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-10-21-clef-judge-alignment-01
kind: build
---

## Session 2026-10-21

### Prompts

> Audit `src/judge_interp/clef.py`: are the prompts and tasks comparable with the entity/relation detection ones? Why load the model from a snapshot every time (propose a model config)? Why do we need `run()`, and could it be something more general that an experiment runner calls with an experiment config?

> Start with the housekeeping and the model config / runner refactor, then propose prompt changes for entity and relation detection with clef.

> Align clef to qwen: no full field list in the clef entity question, rename QUERY to EXTRACTED RECORD, clef evaluates the full row like qwen, and neither model shows null lines (fix qwen too).

> Reword criterion 2 only on the clef side. Investigate the qwen closing-quote issue. exact end alignment should be a hard assertion; what would let the format support both qwen and llama? Is the pad space needed? The format must be as clean as possible.

### Implemented

Added `experiments/model-configs/clef.yaml` (pinned revision, release-code hash, dtype, `max_length`) and a task interface (`TASK_PARAMS` + `run(...)`) so `scripts/run_experiment.py` only owns config loading, snapshots, the manifest and metrics. Aligned the Clef prompts with Qwen/Llama (no null lines, `## EXTRACTED RECORD`, whole-row relation question) and removed the random-subset machinery; the 2026-10-14 Clef relation configs that used it no longer load, and `configs/2026-10-21-clef-entity-smoke-02.yaml` and `configs/2026-10-21-clef-relation-smoke-02.yaml` replace the old smokes. Investigated the closing-quote read-out: every entity is read at one merged pad+quote+newline token, which is now a hard assertion in place of `exact_end_alignment`, and a chat-template splice keeps Llama's final newline so Llama reads the same token. The Clef model and Llama have not been run on a GPU since these changes, and all earlier prompt-dependent numbers are invalidated.

### Commits

64ce62a relation_detection: drop duplicate SPAN_META_KEYS and unused _MODEL_KEYS (row_detection's are the source)
9efd974 Model config for clef and a task interface for the runner
0189398 Align Clef prompts with Qwen/Llama; drop null lines; judge relation rows whole
f96229f Clef entity question: criterion 2 says 'as this field is described' (no field list there)
782daa2 Splice the prompt content into the chat template so Llama keeps the query's final newline
a627d79 Read-out guard: every entity must be read at the one pad+quote+newline token (replaces exact_end_alignment)
