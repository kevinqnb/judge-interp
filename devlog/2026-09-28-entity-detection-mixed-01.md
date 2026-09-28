<!-- Public devlog entry: judge-interp/devlog/2026-09-28-entity-detection-mixed-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-09-28-entity-detection-mixed-01
kind: build
config: configs/2026-09-28-qwen7b-entity-mixed-smoke-01.yaml
---

## Session 2026-09-28

### Prompts

> Update the entity detection task. Instead of one validity query per entity type
> with invalid decoys added, give the model all entity types (main and line) at
> once. For each document D, F_i is the set of ground-truth fields of type i and
> m = sum |F_i|. For every document and each of s user-defined samples, randomly
> choose an integer k <= m and replace k ground-truth fields with a same-type
> decoy from another document; record m and k. Record the token representations
> of the entity span as it appears in the presented list, with valid/invalid
> labels and the associated m and k for each stored representation. Only record
> layer 0 and the last layer, post-norm. Put it in its own
> `entity_detection.py`, reusing `representationlm.py` where useful.

Follow-ups (summary): keep `decoy_in_ocr` as a per-span flag only (no filtering,
handled in analysis); the judging criteria should be (a) presence in the text and
(b) correspondence to the stated entity type, added as a fixed criterion with
short per-type descriptions in the prompt; show an example rendered prompt.

### Implemented

New `src/judge_interp/entity_detection.py` builds one prompt per (document,
sample) listing every ground-truth value across all 14 entity types, with a
uniformly drawn k of them swapped for same-type decoys from other documents in
the split. It records m, k, the valid/invalid label and a `decoy_in_ocr` flag on
every span, and reads only layer-0 and post-final-norm hidden states (no logits)
through an `EntityDetectionLM` subclass. The instructions add a
type-correspondence criterion and per-type descriptions. `scripts/run_experiment.py`
gains the `entity_detection_mixed` task; configs are
`configs/2026-09-28-qwen7b-entity-mixed-{smoke,train,test}-01.yaml`, with
`tests/test_entity_detection.py` and a tiny CPU end-to-end fixture. Unit tests
and the CPU end-to-end pass; the GPU smoke has not been run yet.

### Commits
d9a7c0a Add mixed-type entity detection task with substituted decoys
