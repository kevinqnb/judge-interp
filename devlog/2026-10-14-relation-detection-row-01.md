<!-- Public devlog entry: judge-interp/devlog/2026-10-14-relation-detection-row-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-10-14-relation-detection-row-01
kind: build
config: configs/2026-10-14-qwen7b-relrow-smoke-01.yaml
---

## Session 2026-10-14

### Prompts

> Set up a new `src/judge_interp/relation_detection_row.py` which follows the new changes we've made
> with `entity_detection_row.py`: row-level tasks (only the entities of a single loaded row, still
> separated by type, same prompt shape but different fields and the relation-detection
> instructions), and span collection at the `"\n` token (entities in quotes with a space before the
> closing quote, ended with a newline, mentioned briefly in the instructions).

> Do not borrow old things from `relation_detection.py`; it should be a fresh script so removing it
> later breaks nothing.

Follow-ups (summary): line dataset only, all three line error types selectable; print an example
prompt; explain cross_type vs intra_document errors; wire it into `run_experiment` and run smoke
and tiny end-to-end tests; draft run configs like the entity-row ones, with a smoke config, and
submit the smoke.

### Implemented

New self-contained `src/judge_interp/relation_detection_row.py`: one prompt per `(document, line
row)` with relation instructions (do the row's values belong to the same line item), entities
rendered `"value "` with a trailing newline, and the hidden state read at each closing quote, with
per-entity labels from `field_errors`. Wired into `scripts/run_experiment.py` as task
`relation_detection_row`; tests and a hand-built fixture added, plus
`tests/fixtures/relation_row_tiny_e2e.yaml`. Run configs are
`configs/2026-10-14-qwen7b-relrow-{smoke,line-train,line-test}-01.yaml`; unit tests, a CPU smoke and
a tiny end-to-end (matching its predictions, deterministic across two runs) passed, and the GPU
smoke has been submitted but not yet read.

### Commits
6c5894d relation_detection_row: one prompt per line row, read at the closing quote
d133759 run_experiment: add the relation_detection_row task
