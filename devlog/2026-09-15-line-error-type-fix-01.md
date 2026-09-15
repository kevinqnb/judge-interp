<!-- Public devlog entry: judge-interp/devlog/2026-09-15-line-error-type-fix-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-09-15-line-error-type-fix-01
kind: build
---

## Session 2026-09-15

### Prompts

> The 2026-09-15-qwen7b-repr-line-train-01 job just failed. Can you help me diagnose
> the issue?

> Yes the error_type field is important, let's make sure it get's carried through
> this experiment.

### Implemented

`2026-09-15-qwen7b-repr-line-train-01` crashed with `KeyError: 'error_type'` in
`prompts.row_key`: a prior commit made `error_type` part of row identity for the
`line` dataset (since the same `document_id`/`line_index`/`k` can now carry both an
`inter_document` and an `intra_document` invalid variant), but `build_items()` in
`scripts/run_experiment.py` and the two internal row-key reconstructions in
`RepresentationLM.collect()` (`src/judge_interp/representationlm.py`) were never
updated to carry it. Threaded `error_type` all the way through: `build_items()`,
`collect()`, `_judge_document()`'s per-document `.npz` shards, and `_aggregate()`'s
final arrays and output columns (`write_rows_jsonl`). Also fixed a related bug this
surfaced — `_aggregate()`'s duplicate-row assertion was keyed on
`(doc_id, line_index, k)` only, which is exactly the collision `error_type` exists to
break, and would have rejected any real `line`-train run once the crash itself was
patched. Added a collision row to the `line` fixture and two unit tests
(`tests/test_run_experiment.py`), then verified with a real tiny end-to-end run
(`tests/fixtures/repr_tiny_e2e_line.yaml`, Qwen2.5-0.5B-Instruct CPU) whose predicted
output — all 3 rows surviving, the two colliding rows disambiguated by `error_type` —
was confirmed directly against the written `.npz` shard and `rows.jsonl`. Full test
suite: 73 passed. Not yet resubmitted to the cluster.

### Commits

539b030 fix: carry error_type through the line-dataset representation-collection path
