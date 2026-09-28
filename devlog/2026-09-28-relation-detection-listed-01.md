<!-- Public devlog entry: judge-interp/devlog/2026-09-28-relation-detection-listed-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-09-28-relation-detection-listed-01
kind: build
config: configs/2026-09-28-qwen7b-relation-listed-smoke-01.yaml
---

## Session 2026-09-28

### Prompts

> Update the relation detection task (built in `representationlm.py`, run on the
> VRDU line dataset via `run_experiment.py`): remove the "valid?" suffixes from
> the entity lists; end each recorded span on the actual entity token, not the
> closing quotation mark (as in `entity_detection.py`); record entities as they
> appear in the prompted list; add brief line-field type descriptions to the
> instructions; record only layer 0 and the last layer, post-norm. Put it in its
> own `relation_detection.py`, wire it to the runner, and add train and test
> configs for qwen7b, reusing `representationlm.py` code where useful.

Follow-ups (summary): compared last-token vs. mean-over-span reads (kept last
token); confirmed the per-document instruction arity; reviewed a full example
prompt and a randomly drawn document for relational structure.

### Implemented

New `src/judge_interp/relation_detection.py` renders one prompt per document with
bare numbered entity lists per line-field type, typed instructions, and spans that
end on the value's last token; a `RelationDetectionLM` subclass reads layer 0 and
post-final-norm only and asserts every read token lies inside the query block.
`scripts/run_experiment.py` gains the `relation_detection_listed` task (the older
`relation_detection` task is unchanged); configs are
`configs/2026-09-28-qwen7b-relation-listed-{smoke,train,test}-01.yaml`. Unit
tests and a tiny CPU end-to-end pass; the GPU smoke has not been run yet.

### Commits
cdef781 Add relation_detection_listed task: cue-free lists, value-end spans, layers 0/last
