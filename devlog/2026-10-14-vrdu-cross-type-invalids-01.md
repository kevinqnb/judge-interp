<!-- Public devlog entry: judge-interp/devlog/2026-10-14-vrdu-cross-type-invalids-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-10-14-vrdu-cross-type-invalids-01
kind: build
config: configs/2026-10-14-vrdu-invalids-03.yaml
---

## Session 2026-10-14

### Prompts

> Fix the entity-detection task so it tests for correct type, not literal string presence:
> (1) the instructions must say a listed entity has to actually correspond to its type in the
> document; (2) add invalid cases where an entity appears in the document but is assigned to the
> wrong type, as their own error type.

Follow-ups (summary): build this into `scripts/build_vrdu_invalids.py`, not the entity task, with
error types explicit for every entity of an invalid row (main splits into error types like line
does); same-kind swaps only (name / date / amount), including across main and line; independent
per-field corruption; keep the date-order check on; drop `channel` from the name kind;
regenerate `data/vrdu`; make the loaders select error types explicitly.

### Implemented

The invalids builder gains a `cross_type` error type: a field takes a same-kind value (name,
date or amount) from another field of the same document, in either dataset, never one that is
already valid for that field. Every row now carries `error_type` and a per-field `field_errors`
map naming the error type and source of each wrong entity. `data/vrdu` was regenerated with
`configs/2026-10-14-vrdu-invalids-03.yaml`; existing error types are unchanged. `load_split` now
requires an explicit `error_types`, `record_judge` gained an `error_types` param, and the entity
instructions state the type-correspondence criterion explicitly. Wiring cross-type decoys into the
mixed entity task and the analysis is still to do.

### Commits
d982e58 analysis: relabel probe-axis plot axes as e @ w + b and PC1 orthogonal to w
06109e1 docs: describe absent relation slots as absent_fill, not 0
2a1ac10 entity_detection: state the type-correspondence criterion explicitly
f749be8 invalids: add cross_type error type and explicit per-entity labels
daec77d invalids: drop channel from the name kind; regenerate data/vrdu (2026-10-14-vrdu-invalids-03)
8fa3855 loaders: select error types explicitly; key main rows on error_type
