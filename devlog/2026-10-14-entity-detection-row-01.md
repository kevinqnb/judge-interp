<!-- Public devlog entry: judge-interp/devlog/2026-10-14-entity-detection-row-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-10-14-entity-detection-row-01
kind: build
---

## Session 2026-10-14

### Prompts

> Read the open next step of the cross-type-invalids build and plan how to support the new error
> types in `entity_detection.py`.

> The per-document prompting is an issue. Every prompt should be a document + a single data point
> (a row from either the invalid or valid set).

Follow-ups (summary): rewrite entity detection as row-based; line prompts do not include main
fields; each row is a separate prompt; entities are written in quotation marks and the
representation is collected from the quotation, not the entity's last token; make it a fresh module
that does not import the original `entity_detection.py`; end the query with a newline and put a
space before each closing quote so the quote tokenizes uniformly; wire it into `run_experiment` and
run a tiny end-to-end.

### Implemented

New `src/judge_interp/entity_detection_row.py`: one prompt per `(document, row)` of `data/vrdu`, with
labels per entity taken from each row's `field_errors`, so a row can mix correct and wrong entities
(including `cross_type`). Entities are rendered as `"value "` and the hidden state is read at the
closing quote; the padding and trailing newline make that quote a single uniform token. Rows are
re-validated on load, subsetting is a configured stratified draw (`rows_per_stratum`), and
per-prompt label invariants are checked on the output. `scripts/run_experiment.py` gains the
`entity_detection_row` task, with a tiny end-to-end fixture config in
`tests/fixtures/entity_row_tiny_e2e.yaml`. Not yet done: real run configs, a GPU timing run, and
the `analysis/` side.

### Commits
f0b71ae entity_detection_row: one prompt per dataset row, read at the closing quote
c945c79 run_experiment: add the entity_detection_row task
a46cc85 entity_detection_row: pad entities and end the query with a newline
