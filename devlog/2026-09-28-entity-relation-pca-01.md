<!-- Public devlog entry: judge-interp/devlog/2026-09-28-entity-relation-pca-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-09-28-entity-relation-pca-01
kind: build
---

## Session 2026-09-28

### Prompts

> Please take a look at the latest round of experiments configs in `configs/` run
> on 2026-09-27. These should have all finished successfully. Our objective now
> is to begin analyzing the results in the manner outlined by
> 2026-09-27-entity-relation-detection-01. Please set up new scripts
> `analysis/entity_detection.py` and `analysis/relation_detection.py` to do this.
> In particular the entity detection script should collect the representations
> of all entities presented to the model and pre-classify them as valid or
> invalid. It should then plot PCA of the data colored by their labels. We
> should do this (a) for the pooled collection of all entities, and (b) for each
> entity type. In each case the data should be mean centered before applying
> PCA. For relation detection we need to start on a more fine grained document
> level. For a single, user input document id, we should plot the centered PCA
> representations for all entities only presented to the model for that
> document. These should be colored by (a) their source, ground truth data
> point if they belong uniquely to a single source, or (b) colored via a
> multi-color label if they belong simultaneously to multiple sources. Don't
> worry just yet about computing any kind of correlational statistics, this is
> purely visualization for now.

Clarified via `AskUserQuestion`: "mean centered" for entity-detection means
per-document centering (removes document identity as a confound before pooling
across documents, per the build note's stated plan); duplicate-valued valid
tuples in relation-detection collapse into one source rather than counting as
separate origins.

### Implemented

`analysis/entity_detection.py` reads the four 2026-09-27 `entity_detection`
train runs (Qwen7B/Llama8B x main/line), per-document-centers each entity's
`content`-span representation, and plots an independent 2-component PCA per
layer for the pooled collection and for each entity type, colored valid/invalid.
It also flags (via a printed diagnostic) a real type-composition confound found
in `line`'s pooled figure: valid and invalid entities have very different
per-type mixes there, unlike `main`. `analysis/relation_detection.py` is a CLI
(`document_id` + `--model`, or `--sweep`) over the two `relation_detection`
line/train runs: for one document, it PCA-plots every presented entity, colored
by its ground-truth valid source tuple(s) — a plain dot for a unique source, a
round multi-color pie marker (built from scaled custom scatter markers, not
data-space patches) for an entity shared across sources. `--sweep` validates
the entity-to-source join over all 509 documents and reports per-type
multi-source rates. Both scripts were run end-to-end against the real
completed 2026-09-27 runs; no config files were added or changed this session.

### Commits
ce3882c Add entity/relation-detection PCA visualization scripts
