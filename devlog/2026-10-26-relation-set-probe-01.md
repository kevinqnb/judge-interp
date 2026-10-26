<!-- Public devlog entry: judge-interp/devlog/2026-10-26-relation-set-probe-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-10-26-relation-set-probe-01
kind: build
config: analysis/configs/relation_set_detection/ (2026-10-26 smoke and four full configs)
---

## Session 2026-10-26

### Prompts

> I want to test some variations of the relation detection analysis script @analysis/relation_detection.py .
> Currently the script analyzes probe performance for relational correctness between pairs of fields within a sample
> record, i.e. prediction of $\mathbb{P}(f_j | f_i)$. [...] My goal is to be able to use a probe to predict
> $\mathbb{P}(f_k | f_{k - 1}, f_{k-2},...,f_1)$ or $\mathbb{P}(f_k, f_{k - 1}, f_{k-2},...,f_1)$. [...] Here's some
> things I'd like to try:
> 1. Additive representations (joint probability): Sum representation vectors for all $f_i$ from a single record;
>    train a probe to predict the validity label for the summed vectors
> 2. Additive conditional representations (conditional probability): Concatenate the representation vector for $f_k$
>    with the sum of representation vectors for $f_{k-1} + f_{k-2} + ... + f_1$. Train a probe to predict the validity
>    label for the concatenated vectors

> Ok let's keep an eye on this going forward. Write configs for the full runs, use only L24 for qwen and L28 for llama.

### Implemented

New `analysis/relation_set_detection.py` probes sample validity on minimal pairs of samples, with a required
`params.feature`: `sum` (sum of all shown fields' activations) or `cond` (last-shown field concatenated with the sum of
the rest), plus `last`/`context` ablations (`params.ablations`), since the last field's activation has already attended
to the earlier fields. It reports the pair probe's metrics plus a last-shown-corrupted stratum and detection on invalid
samples with no invalid field pair, and asserts the label rule on each whole sample, the structure baseline, the
shuffled-label control and seed determinism. `analysis/relation_detection.py` only had its sample matching extracted as
`match_samples` (behaviour unchanged). Configs are under `analysis/configs/relation_set_detection/`: a smoke config and
`sum`/`cond` configs for Qwen L24 and Llama L28. Unit tests, a smoke run and a full-data count check passed before
the full runs.

### Commits

4f1c631 relation_detection analysis: extract match_samples from build_pairs
ab563b5 analysis: add relation_set_detection.py, a sample-level relation probe
1372fce configs: 2026-10-26 relation_set_detection smoke and full configs
