<!-- Public devlog entry: judge-interp/devlog/2026-10-14-row-analysis-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-10-14-row-analysis-01
kind: build
config: configs/2026-10-14-qwen7b-relrow-analysis-smoke-01.yaml
---

## Session 2026-10-14

### Prompts

> Create `analysis/entity_detection_row.py` and `analysis/relation_detection_row.py` for the new
> `src/judge_interp/{entity,relation}_detection_row.py`, mirroring the existing entity/relation
> analyses. Entity: load row activations, subtract only per-document means, PCA coloured by
> validity, a single-evaluation validity probe with a fairly small regularization parameter, smECE
> / Brier skill score / AUROC on a held-out test set, smooth ECE diagrams, accuracy per error type.
> Relation: samples are random subsets of one row's field activations (invalid if any field is
> invalid), features are pairwise Euclidean distances per field pair (0 if absent), train and test
> with the same label rate, same metrics and plots. No results exist yet, so write the scripts and
> smoke-test them.

> Does it make sense to do any sort of standardization before probing? Especially for the relation
> detection task where features are built from distances?

> Leave entity detection as is. Go ahead with standardization of per-pair distances on the relation
> feature dataset.

### Implemented

Two new probe scripts, `analysis/entity_detection_row.py` and `analysis/relation_detection_row.py`,
both config-driven (smoke configs `configs/2026-10-14-qwen7b-entityrow-analysis-smoke-01.yaml` and
`configs/2026-10-14-qwen7b-relrow-analysis-smoke-01.yaml`). The entity script centres per document,
plots PCA by validity, fits a strongly regularised single-entity probe and reports smECE, Brier
skill score, AUROC and a per-error-type breakdown plus a field-identity baseline. The relation
script builds random-subset samples with 10 pairwise-distance features (optionally standardised
per pair on train statistics), class-balanced train and test sets, the same metrics, and reported
(not asserted) presence-only and noise baselines. The shuffled-label control averages over many
permutations. Unit tests cover both scripts; they were smoke-tested on real entity runs and on
synthetic planted-signal data, and the relation script has not yet run on real train/test data.
The new scripts were added to the ask-guarded paths in `.claude/settings.json`.

### Commits
c2478ce analysis: row-task probes for entity and relation detection
