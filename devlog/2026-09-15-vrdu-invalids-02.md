---
id: 2026-09-15-vrdu-invalids-02
kind: build
config: configs/2026-09-15-vrdu-invalids-02.yaml
---

## Session 2026-09-15

### Prompts

> I want to adjust how the line version of the dataset is built [...]. First, two
> bugs: (1) `LINE_CARRIED_FIELDS` should be `[]` — line rows shouldn't carry
> document-level context. (2) `num_invalid_fields` isn't correctly labeled when a
> row can't support as many invalid fields as requested; fix this by explicitly
> counting and labeling the actual number of invalids given.
>
> Next, expand the line dataset's invalids into two independent error types:
> **inter-document** (fields filled with values drawn exclusively from *other*
> documents, same approach as the main dataset) and **intra-document** (fields
> filled from other valid line rows in the *same* document, but now every borrowed
> injection must come from a separate donor row — once a row has been borrowed
> from for one field, it can't be borrowed from again for another). Keep the
> existing "one invalid row per valid case per num_invalid_fields value" process,
> but do it twice — once per error type — and label every row with its error type.
> Also raise the intra-document cap from 4 to 5 (all line fields), and make sure
> intra-document generation can't accidentally reproduce a full copy of a valid
> entry.

### Implemented

Set `LINE_CARRIED_FIELDS = []` in `scripts/build_vrdu_dataset.py`. Rewrote the line
half of `scripts/build_vrdu_invalids.py`: inter-document generation reuses the
main-dataset machinery against a pool built from every *other* document; intra-
document generation is a chain that draws one donor line row per corrupted field
(never reusing a donor), stopping — and recording the true realized count, never
assumed to be the max — the moment no eligible (field, donor) pair remains; a pair
that would reproduce another valid row in the document verbatim is excluded rather
than aborting the build (two line items differing by exactly one field is plausible
real data). Every invalid line row now carries `error_type`
(`inter_document`/`intra_document`); row identity widened to `(document_id,
line_index, error_type, k)`, and `k` now always equals the realized
`num_invalid_fields` (the original bug: capped rows kept their requested `k`,
letting two different `k` values duplicate the same realized content). Main-dataset
generation is untouched — verified `main/train.json` byte-identical after the
rebuild, and `main` provenance (counts, distributions, skipped/capped sets) matches
old vs. new exactly. `configs/2026-09-15-vrdu-invalids-02.yaml` replaces `-01` for
the line dataset (`k_range: [1, 5]`, no more `line_context_fields`).
`src/judge_interp/prompts.py`'s `row_key` updated for the new line identity — a
known follow-up is that `run_experiment.py`/`representationlm.py` don't yet
propagate `error_type`, so the next line-dataset representation-collection run will
need that wired through. `tests/test_build_vrdu_invalids.py` rewritten with unit
tests for the new pool/chain logic and a fixture end-to-end with counts predicted
from the fixture's construction. Full local rebuild on the real corpus (641 docs, no
cluster job): line/train now 7 732 valid + 38 660 inter-document + 22 313
intra-document rows, line/test 1 431 + 7 155 + 4 157; this invalidates every prior
number computed from the old line dataset (`2026-09-10-vrdu-invalids-01`).

Main dataset generation (unchanged):

```
for split in [train, test]:
    pool[field] = every non-null value of field across the split's base rows
    for each base_row, for k in 1..9:
        j = largest achievable field count <= k   # self-caps
        choose j fields, resample each from pool (must differ from original)
        emit invalid row: k=k, num_invalid_fields=j
```

Line dataset generation (new):

```
for split in [train, test]:
    # inter-document: independent per row
    for each document d:
        pool[field] = non-null values of field from every OTHER document
    for each row r:
        for k in achievable num_invalid_fields values (1..5, no capping):
            choose k fields, resample each from d's pool
            emit invalid row: error_type=inter_document, k=num_invalid_fields=k

    # intra-document: a chain per row, one donor consumed per field
    for each row r in document d:
        donors = every OTHER row in d; used = {}; corrupted = []
        while len(corrupted) < 5:
            feasible = fields not yet corrupted with >=1 unused, eligible donor
                       (differs from r's original, keeps date order, and would
                        not reproduce another valid row verbatim)
            if none feasible: stop
            pick a field uniformly, then a donor uniformly among its eligible ones
            apply it; mark field corrupted, donor used
            emit invalid row: error_type=intra_document, k=num_invalid_fields=len(corrupted)
```

### Commits

- `1b66374` fix: drop line carried fields, correct k/num_invalid_fields, split line invalids into inter/intra-document errors
- `fa45711` data: rebuild VRDU line dataset with dropped carried fields; regenerate invalids (2026-09-15-vrdu-invalids-02)
