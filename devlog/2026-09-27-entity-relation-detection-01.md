<!-- Public devlog entry: judge-interp/devlog/2026-09-27-entity-relation-detection-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-09-27-entity-relation-detection-01
kind: build
---

## Session 2026-09-27

### Prompts

> I would like to set up structure for a new set of experiments in this repository:
> (1) entity detection, presenting one entity type at a time (filtered to the unique
> set present) and asking the model to mark each true/false, collecting each entity's
> last-token representation, to test whether valid/invalid entities separate
> geometrically within a document; (2) relation detection, presenting all entity
> types at once as per-type lists and asking the model to find valid m-tuples (one
> entity per type, mutually supported by the document), collecting the same
> per-entity representations, to test whether entities cluster by their true source
> tuple, with shared entities producing "orbiting" clusters. Please develop a plan;
> don't be afraid to break or remove existing code.

> Yes, auto-accept edits. My only feedback would be for the representation task:
> make sure the ordering of entities is randomized before being presented to the
> model.

> Let's /devlog and get this work committed.

### Implemented

Added two new representation-collection tasks alongside the existing whole-record
judge, reusing the existing VRDU invalids corpus as ground truth instead of new
corruption synthesis: `entity_prompts.py`/`entity_instructions.py` build
per-(document, field) valid/decoy entity lists (entity detection), and
`relation_prompts.py`/`relation_instructions.py` build per-document, per-type
valid-entity lists plus ground-truth tuple membership (relation detection).
`representationlm.py` gained `judge_spans`/`collect_spans`, reading representations
at multiple token positions per prompt via tokenizer offset-mapping, alongside the
existing `judge_one`/`collect` (unchanged). `scripts/run_experiment.py` now
dispatches on a required `task` field with closed-set params validation per task;
all 9 existing `configs/*.yaml` were retrofitted with `task: record_judge`. 44 new
unit tests, plus real CPU end-to-end runs of both new tasks against
Qwen2.5-0.5B-Instruct (`tests/fixtures/entity_tiny_e2e.yaml`,
`relation_tiny_e2e.yaml`).

### Commits
2d57538 Add entity-detection and relation-detection judge tasks
