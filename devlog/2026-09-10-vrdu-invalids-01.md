---
id: 2026-09-10-vrdu-invalids-01
kind: build
config: configs/2026-09-10-vrdu-invalids-01.yaml
---

## Session 2026-09-10

### Prompts

> Now that we have training and testing datasets initialized, we need to populate
> them with invalid entries. [...] randomly choosing one or more fields to invalidate
> by giving them incorrect values. We'll do that differently for the main and line
> datasets.
>
> **Main:** compile, per field, a list of every value seen across all documents. For
> `train.json` / `test.json`: include all valid base rows, then for k in 1-9 create
> one invalid row per valid row by choosing k fields and resampling each from the
> all-document list (without replacement until the value differs from the original).
> Constraint: `flight_from` before `flight_to`.
>
> **Line:** same shape but the value list is per single document, k runs 1-4, and
> each invalid row gets `num_invalid_fields: k`. Constraint: `program_start_date`
> before `program_end_date`. Channel is often constant across line items; if the
> sampled channel can't be changed, sample a different field instead.

Follow-up decisions: date pools restricted to parseable values (fragments never
emitted as invalid dates); ordering is `<=`; corrupt-to is always non-null but a null
field may be corrupted to non-null; an infeasible `(row, k)` falls back to the
largest achievable count `<= k` (`k` and the realized `num_invalid_fields` both
recorded), a row supporting nothing is skipped.

Mid-session direction change for the line date fields:

> Maintain the per document pool, and do NOT use sentinels on the line dataset. [...]
> invalids where field values have been replaced by others that explicitly appear in
> the context. If you cannot find any other dates that appear on the page which would
> satisfy the date requirements (and you may expand the pool to include dates from
> the "flight_from" or "flight_to" fields), then we should not attempt to replace the
> value at all. Instead, we should sample another field to edit, or [...] accept less
> than k edits.

Closing decisions: keep main on the cross-document pool (unchanged); commit the
23 MB `line/train.json`; the `choose_fields` date weighting is fine as is.

### Implemented

`scripts/build_vrdu_invalids.py` (config-driven, run directly — local data prep, no
cluster) reads the four `*_base.json` files and writes
`data/vrdu/{main,line}/{train,test}.json` (valid base rows + generated invalid rows)
plus `data/vrdu/invalids.json` (config snapshot, seed, python version, pool sizes,
`num_invalid_fields` distribution, realized per-field corruption frequency, and every
skipped/capped row). Each row carries `valid`, `k` (requested field count),
`num_invalid_fields` (realized), and `invalid_fields`; `(document_id[, line_index],
k)` is unique per split. Main corrupt values come from a per-split, cross-document,
per-field pool (date fields also carry two synthetic ordering sentinels); line
corrupt values come only from the same document — the two line date fields share one
pool of every parseable date on the page (both program-date columns plus the
document's `flight_from`/`flight_to`), with no sentinels. A coupled feasibility check
(`_date_feasibility` / `achievable_counts` / `_choose_fields`) keeps `from <= to` on
every row and, when the page can't supply a valid date, drops the date field and
samples another (or accepts fewer than k edits). `configs/2026-09-10-vrdu-invalids-01.yaml`
holds the seed, both k-ranges, pool scopes, null policy, year pivot, sentinels, and
line context fields. `tests/test_build_vrdu_invalids.py` adds 28 tests over a
hand-built 10-document fixture (`pyyaml` added as a dependency). Ladder: unit tests →
fixture smoke → fixture end-to-end (counts predicted and matched) → full run;
determinism confirmed by a byte-identical second run, and an independent validation
pass (pools rebuilt from scratch) reports zero invariant failures on all four splits.
Realized: main 513/128 valid + 4 617/1 152 invalid; line 7 732/1 431 valid +
30 880/5 716 invalid (14 line rows skipped as fully un-invalidatable).

### Commits

- `a0a38c1` feat: add VRDU invalids builder
- `6bae260` data: build VRDU main/line invalids (2026-09-10-vrdu-invalids-01)
