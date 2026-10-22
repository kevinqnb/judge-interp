<!-- Public devlog entry: judge-interp/devlog/2026-10-22-vrdu-line-subsets-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-10-22-vrdu-line-subsets-01
kind: build
config: data/vrdu/build-configs/2026-10-21-vrdu-invalids-04.yaml
---

## Session 2026-10-22

### Prompts

> For clef.py, and for the downstream analysis in analysis/relation_detection.py we evaluate the model's ability to predict relation over randomly sampled subsets of relational items. I want to centralize this process by somehow defining those randomized groupings here within the `data/vrdu` directory. The purpose of this is that so we can evaluate both clef and the probe on the same data -- and therefore also compare them more directly on other things like runtime performance. Can you propose a plan for centralizing this process here? I still want to be able to average over randomly chosen subsets of possible relations.

Plan changes (summary): do not change Clef or the probe, only the data structure; keep the loader dataset-specific in `data/vrdu`; line dataset only; each sampled subset gets an explicit random field order, for a chain-rule factorization P(f1,f2,f3) = P(f3|f1,f2) P(f2|f1) P(f1).

### Implemented

`data/vrdu/build_vrdu_subsets.py` draws, per line row, distinct random subsets of the non-null fields with a random field order, labelled valid iff none of its fields is corrupted, in several independent replicates balanced per class, and writes `data/vrdu/subsets.json` (sample ids plus input hashes). `data/vrdu/subsets.py` is the dataset-specific loader (`load_subsets`): it refuses a stale file and re-derives each sample from its row. Values live in `params.subsets` of `data/vrdu/build-configs/2026-10-21-vrdu-invalids-04.yaml`. `data/vrdu/README.md` documents it, including that subset size leaks the label. Clef and the probe are not yet wired to the file.

### Commits

1231eca VRDU line: shared random ordered field subsets (subsets.json) and loader
