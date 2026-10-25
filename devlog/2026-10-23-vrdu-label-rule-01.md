<!-- Public devlog entry: judge-interp/devlog/2026-10-23-vrdu-label-rule-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-10-23-vrdu-label-rule-01
kind: build
config: data/vrdu/build-configs/2026-10-22-vrdu-invalids-05.yaml
---

## Session 2026-10-23

### Prompts

> Please investigate the issues written up in @data/vrdu/ISSUES.md and design a plan to fix them.

> I think it would be better to create a new error type, 'cross_item', which explicitly tests for type
> correctness for items from the same line. ... inter-document and cross-type errors become the main error
> types to evaluate for entity detection ... intra-document and cross-item errors become the main error types
> to evaluate for relation detection ... adjust the dataset error type shares to 25% each.

> taking a valid line item subset, create a cross-item invalid error by randomly choosing two field types to
> directly swap values for ... always have k=2 ... try not to change the total amounts that each previous error
> type contributed ... intra-document errors should now be limited to only same-type errors ... we still need
> to test if the constructed subset co-occurs with any valid subset in the document.

Follow-ups (summary): pair-probe labels by co-occurrence; keep the README high-level; rewrite the docstrings
in the touched files concisely, with goal / inputs / outputs for the `data/vrdu` functions.

### Implemented

New data version `data/vrdu/build-configs/2026-10-22-vrdu-invalids-05.yaml`. One label rule,
`build_vrdu_invalids.matches_item` (shown values are valid iff they are one real line item's), is now used by
the row builder, the subsets builder and loader, and the pair probe in `analysis/relation_detection.py`; -04
labelled some samples, rows and probe pairs invalid that were in fact valid. `cross_type` no longer draws from
the row's own line item, and a new subset-only `cross_item` type swaps two shown fields of a clean subset.
`subsets.json` now gives each of the four error types the same number of invalid samples (25% each). The
rebuilt data was checked for determinism and against the label rule; numbers on -04 data do not carry over.

### Commits

a50a729 settings: guard the label and scoring code paths
9fee761 vrdu data -05: label rule, cross_item error type, 25% per type
c83f0bd relation-detection analysis: label pairs by the data's label rule
4e836a4 vrdu builders: concise goal/Args/Returns docstrings
