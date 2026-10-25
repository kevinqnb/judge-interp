<!-- Public devlog entry: judge-interp/devlog/2026-10-25-vrdu-minimal-pairs-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-10-25-vrdu-minimal-pairs-01
kind: build
config: data/vrdu/build-configs/2026-10-24-vrdu-invalids-06.yaml
---

## Session 2026-10-25

### Prompts

> We recently made a series of changes to the construction of the vrdu datasets in `data/vrdu`. While the
> intention was to create a reliable evaluation set with no test leakage, unfortunately I think this has now
> become overly complicated. I'm looking for ways to simplify, while still preserving a strong, reliable
> evaluation set. ... The `main` train and test datasets ... Error types should be inter-document ... and
> cross-type ..., and should vary over $k$ ... The `line` train and test datasets ... Error types should
> therefore be: intra-document ... and cross-item ... For cross-item, this will have to remain $k = 2$.
> Importantly, this should be created over randomly chosen, ordered subsets of line fields. ... We cannot have
> any label leakage due subset size, or field error type. We need the datasets to be both class-balanced
> across every stratification, as well as large enough to support reliable evaluation.

Decisions (summary): main valid spans from the clean twin row; null-fill off for line only, and recorded for
main ("do we explicitly record if a field was corrupted from null? That way I could stratify and analyze with
and without this set in the end."); "I'd rather keep m=5 in order to maintain full length subsets. Let's use
k<= 3 with that."

> Go ahead and start with build_vrdu_line.py and the fixture tests

> Go ahead with the reader migration

### Implemented

New data version `data/vrdu/build-configs/2026-10-24-vrdu-invalids-06.yaml`, built by `data/vrdu/build_vrdu_main.py`
and `data/vrdu/build_vrdu_line.py`, which replace the -05 invalids / manifest / subsets chain. Every invalid
sample has a clean twin (same item, same ordered fields), so every stratum is 50/50 with identical structure.
Line has 15 equal (error type, m, k) cells; main pairs each corrupted span with its clean-row twin
(`data/vrdu/main_samples.py`), with null-fill recorded. The dataset specs, prompt rendering, row selection, Clef
samples (`experiments/clef_samples.py`) and the three analysis scripts now work on these pairs, and the
analysis asserts that structure-only baselines score exactly 0.5 AUROC. Numbers on -05 data or from the previous
analysis code do not carry over.

### Commits

fda82df vrdu line -06: minimal-pair subset builder
433de9c vrdu main -06: main-only builder and clean-twin span pairs
3723901 vrdu -06 readers: line minimal-pair rows, pairwise draw, clef samples
6216128 analysis: score probes and Clef on vrdu -06 minimal pairs
7d10847 vrdu: remove the -05 builders, manifest, subsets and their tests
4fba69b analysis: structure baselines from each member's own span metadata
6bb942f vrdu -06: refresh build provenance from the committed builders
