<!-- Public devlog entry: judge-interp/devlog/2026-10-21-dataset-spec-refactor-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-10-21-dataset-spec-refactor-01
kind: build
---

## Session 2026-10-21

### Prompts

> Audit and fix `src/judge_interp/`: the prompts, high-level task structures and data processing are
> too specific to VRDU, especially relation detection (defined for line items structured in a very
> specific way). Make the tasks general purpose, and keep VRDU working via dataset configs in
> `experiments/dataset-configs/vrdu_main.py` and `vrdu_line.py` that a runner can load and pass to
> the main modules.

> Justify what you included in `dataset_spec.py`. Do we really need the fine-grained details (e.g.
> the relation spec)? Could the prompts be reworked to be more general?

> Relation detection should judge the relation of all entities as a whole, not each entity against
> the others, while still collecting representations at each entity's closing quote.

> Deduplicate entity_detection and relation_detection; leave analysis code alone for now.

### Implemented

Added `DatasetSpec` (`src/judge_interp/dataset_spec.py`) and moved all VRDU vocabulary and file layout into
`experiments/dataset-configs/vrdu_main.py` / `vrdu_line.py`; the entity, relation and clef tasks and the
runner (`params.dataset_config`) now take a spec, and the prompts are generic. Relation detection asks for a
single whole-record verdict while still reading and labelling per entity. The shared row machinery moved
into `src/judge_interp/row_detection.py`, leaving `entity_detection.py` and `relation_detection.py` as thin
task modules. The analysis import of the line-item field list was redirected to the VRDU line spec in its own
commit, with no logic change. Verified by unit tests, a byte-level comparison of built prompts/rows against
the pre-refactor code (only the reworded instruction text differs), and bitwise-identical tiny CPU end-to-end
outputs before/after the dedup. Old VRDU prompts and cached shards do not apply to new runs.

### Commits

e64cec6 Make judge tasks dataset-independent via DatasetSpec; relation prompt asks for one whole-record verdict
25e7ed4 Deduplicate entity_detection and relation_detection into row_detection
9fd8db5 analysis/relation_detection: take LINE_FIELDS from the vrdu_line dataset config
