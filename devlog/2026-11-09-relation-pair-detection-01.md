<!-- Public devlog entry: judge-interp/devlog/2026-11-09-relation-pair-detection-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-11-09-relation-pair-detection-01
kind: build
config: experiments/experiment-configs/relation-pair-detection/2026-11-09-qwen7b-relpair-line-tiny-01.yaml
---

## Session 2026-11-09

### Prompts

- Create a new `relation_pair_detection` experiment setup and analysis: prompt the model with all unordered (x, y) pairs of entities from one line item, and have it decide for each pair whether it is relationally valid. Each entity must be present and of the correct type (as in entity detection), and the pair must be explicitly related by the document, i.e. both entities describe a common subject (as in relation set detection).
- Use the VRDU line items, clean and corrupted (intra-document, cross-item). Collect the ` "\n` token representation at the end of each pair, as for entity and relation set detection. Precompute pair labels: valid if the values appear together in any ground-truth line item. Randomize the pair order and the field order within a pair.
- Analysis: a linear probe for pair validity, subsampled to 50/50 valid / invalid and balanced per (field type 1, field type 2) combination, evaluated with the same setup as relation set and entity detection.
- Design answers (summary): analyse corrupted parents only; 50/50 within each field-pair cell; one-line `fa: "va", fb: "vb "` format; include Clef / Jev as new runs oriented like the LM prompt.
- Give a high-level overview of the instructions given to the model; commit the work and write the devlog.

### Implemented

New task `relation_pair_detection` (`src/judge_interp/relation_pair_detection.py`, registered in `experiments/run_experiment.py`): one prompt per line row listing every entity pair in a seeded order shared with the row's clean twin, read at each pair line's closing token, labelled from the size-2 rows of `data/vrdu/line/<split>_subsets.json`; `row_detection.RowDetectionLM` gained a small per-task hook (span columns, invariants, metrics). `experiments/clef_samples.py --source line_pairs` writes matching Clef / Jev conditional samples. `analysis/relation_pair_detection.py` trains the probe and scores it with the judges on balanced corrupted-parent pairs, with a structure baseline, label-shuffle, determinism, layer-0 and twin-prefix known-answer controls (see `analysis/EVALUATION.md`). Configs: `experiments/experiment-configs/relation-pair-detection/2026-11-09-*` (LM tiny / train / test, Clef and Jev tiny / test) and `analysis/configs/relation_pair_detection/2026-11-09-relpair-{smoke-01,01}.yaml`; unit tests and a CPU smoke run pass, no GPU run yet.

### Commits

0aeb7a6 row detection: per-task span columns, invariants and metrics hook (behaviour unchanged)
551c762 relation pair detection: task, line_pairs judge samples, fixtures and tests
28f471f analysis: relation pair detection (probe + Clef/Jev on balanced corrupted-parent pairs)
e908213 configs: relation pair detection collection, judges and analysis (2026-11-09)
