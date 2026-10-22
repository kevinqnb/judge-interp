# data/vrdu

VRDU ad-buy-form corpus (641 documents) turned into extraction-validation rows for the judge:
every row is one extraction, either **valid** (copied from the annotations) or **invalid**
(valid values deliberately corrupted). Four scripts build everything, in this order:

| script | writes | config |
|---|---|---|
| `build_vrdu_dataset.py` | `main/`, `line/` `*_base.json` (valid rows only), `ocr/`, `directory.json` | CLI args (split ratio, seed 0) |
| `build_vrdu_invalids.py` | `main/`, `line/` `{train,test}.json`, `invalids.json` | `data/vrdu/build-configs/2026-10-21-vrdu-invalids-04.yaml` |
| `build_vrdu_manifest.py` | `manifest.json` (matched evaluation subset) | `params.manifest` in the same config |
| `build_vrdu_subsets.py` | `subsets.json` (shared random field subsets, line only) | `params.subsets` in the same config |

Build configs live in `build-configs/` (one per invalids version, `-01` to `-04`; only the newest
describes the data on disk). Run each script directly (local data prep, no cluster job), e.g.
`uv run --python 3.12 python data/vrdu/build_vrdu_invalids.py --config data/vrdu/build-configs/2026-10-21-vrdu-invalids-04.yaml`.
Output is deterministic for a fixed seed. The large row files (`{main,line}/{train,test}.json`,
~110 MB) and `ocr/` are **gitignored**: regenerate them with the commands above, then check them
against the committed `manifest.json`, which stores the sha256 of each split file
(`build_vrdu_manifest.load_manifest_split` raises if a regenerated file differs). Numbers from earlier invalids configs (`-01`..`-03`) do
not carry over to `-04`.

## Two datasets

- **main**: one row per document, 9 fields (`advertiser`, `agency`, `property`, `tv_address`,
  `product`, `contract_num`, `flight_from`, `flight_to`, `gross_amount`). Many are null.
- **line**: one row per line item (broadcast spot), 5 fields (`channel`, `program_desc`,
  `program_start_date`, `program_end_date`, `sub_amount`); 9,163 line items in 636 documents.

The split is by document (513 train / 128 test, seed 0), so no document is in both splits.

## Row format

Every row has `document_id` (+ `line_index` for line), `valid`, `error_type` (null if valid),
`k` (number of corrupted fields; 0 if valid), `invalid_fields`, one value per field, and
`field_errors`: one entry per field, `null` if that entity is correct, otherwise
`{error_type, source}` saying where the wrong value came from. A row's identity is
`(document_id, [line_index], error_type, k)`. Rows are single-error-type: all corrupted fields
of a row were built the same way.

## How the error types are made

An invalid row starts from a valid row of the same split and replaces exactly `k` of its fields.
One invalid row is emitted per achievable `k` (never a requested `k` the row can't reach).

| type | datasets | where the wrong value comes from |
|---|---|---|
| `inter_document` | main, line | another document of the same split (main: any row's value for that field; line: any other document's line rows). Not in this document's text by construction. This is intended, so it tests whether the judge checks the value against the document. |
| `intra_document` | line | the **same field** taken from other line items of the **same** document (a same-type error: line A's `program_start_date` gets line B's `program_start_date`), one distinct donor item per corrupted field. A chain: fields and donors are drawn one at a time until `k` is reached or no eligible pair is left. |
| `cross_type` | main, line | a real value of this document that belongs to a **different field of the same kind** (name: advertiser, agency, property, tv_address, product; date: the four date fields; amount: gross_amount, sub_amount), in either dataset. The value is genuine but filed under the wrong type. `contract_num`, `channel`, `program_desc` have no kind and are never touched. Two fields that pull each other's values are marked `exchanged_with`. |

`intra_document` and `cross_type` never overlap: `intra_document` keeps the field's type (same field, another
line item), `cross_type` changes it (a different field). A `cross_type` value may not already be valid for
its target field anywhere in the document, which excludes exactly the same-field values `intra_document` uses.
In a row's `field_errors`, an intra source has `field` equal to the corrupted field, a cross source a
different one.

Rules that apply to every type:

- **A corruption must change the value.** "Same value" is judged by an equivalence key (parsed
  date, numeric amount, case- and whitespace-folded text), so `2/18/20` -> `02/18/20` or
  `ISSUE` -> `Issue` is never an error. An intra/cross row also may not reproduce another valid
  row of the document, and a cross value may not already be valid for its target field anywhere
  in the document.
- **Null fields can be filled; values are never nulled.** Filling a null adds an entity that
  was absent; every corrupted source carries `null_fill` (true/false) so it can be stratified
  (13-22% of corrupted entities in `inter`/`cross`, 5% in `intra`).
- **Date order.** The two dates of a record keep `from <= to`, enforced whenever both dates parse
  (no valid row is reversed, so a reversed invalid row would be a free tell).
- **Field choice is uniform** over the fields that can be corrupted for that row.

## Composition (train / test)

| dataset | valid | inter_document | intra_document | cross_type |
|---|---|---|---|---|
| main | 513 / 128 | 4,617 / 1,152 | - | 3,836 / 944 |
| line | 7,732 / 1,431 | 38,660 / 7,155 | 24,893 / 4,633 | 18,009 / 3,118 |

Rows per `k` (train; test has the same shape, e.g. line intra 1,278 / 1,257 / 1,040 / 853 / 205):

| | k=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| main inter_document | 513 | 513 | 513 | 513 | 513 | 513 | 513 | 513 | 513 |
| main cross_type | 513 | 513 | 513 | 513 | 512 | 507 | 484 | 281 | - |
| line inter_document | 7,732 | 7,732 | 7,732 | 7,732 | 7,732 | - | - | - | - |
| line intra_document | 7,334 | 7,146 | 5,560 | 3,992 | 861 | - | - | - | - |
| line cross_type | 7,728 | 6,566 | 3,715 | - | - | - | - | - | - |

The types differ in reach: inter always reaches every `k`, intra and cross only where the
document has enough donors (line cross is capped at k=3, its three kinded fields). Cells at high
`k` are therefore made of different, richer documents. Skipped rows (no achievable corruption) are
listed in `invalids.json` (`skipped_rows`), along with pool sizes, `k` distributions and
`date_fragment_rates`. Valid rows are ~6% (main) and ~9% (line) of each split; balance them when
scoring. Rows from one base row at different `k` are not independent (and intra rows are nested).

## OCR-fragment dates

About 20% of line date values (12-18% in main) are OCR fragments that don't parse as a date, e.g.
`/08/20` or `02/21/`. They are valid ground truth. To keep "is a fragment" from predicting "is correct",
corruptions can also produce fragments, at the same rate they occur in the valid rows, and date order
(`from <= to`) is only enforced when both dates parse. The corrupted and kept fragment rates are recorded
per error type in `invalids.json` (`date_fragment_rates`) and match everywhere except one place:
**line `cross_type` over-produces fragments** (~28-33% of corrupted dates vs ~18-20% kept), so there a
fragment is weakly predictive of "corrupted".

## Matched subset (`manifest.json`)

The full set is not comparable across types (sizes and documents differ, above). The manifest
fixes that: a base row qualifies only if **every** error type of its dataset has a row at
**every** `k` in `k_values`; each selected base row contributes its valid row plus one row per
(error type, `k`). Every (error type, `k`) cell then has the same size, the same documents and the
same exact `k`. `k_values`: main 1-6 (507 / 126 base rows), line 1-3 (2,509 / 428 base rows).
The manifest records the invalids config id and sha256 of each split file, and
`load_manifest_split` refuses one that is stale. `data/vrdu/build_vrdu_manifest.py` documents the
selection; it is not yet wired into `row_detection`.

## Random field subsets (`subsets.json`, line only)

The shared samples for comparing Clef and the probe, and for averaging over random subsets.
A **sample** is an ordered subset of one line row's non-null fields: `fields` is f1..fn in a random
order, for the chain-rule factorisation P(f1..fn) = P(fn|f1..fn-1)...P(f2|f1)P(f1). It is **valid iff none
of its fields is in the row's `invalid_fields`**, so a subset of an invalid row that avoids every
corrupted field is a valid sample. Per source row, `samples_per_row` distinct *unordered* subsets of size
>= `min_subset_size` are drawn uniformly, then each gets a uniformly random permutation (ordered tuples
are not enumerated; that would weight a subset by n!). Each replicate is cut to exactly `n_per_class`
valid and invalid samples. Replicate r of a split uses `default_rng([seed, split index, r])`, so replicates
are independent and raising `n_replicates` leaves the earlier ones unchanged. The source rows are
`manifest.json`'s rows (`source: manifest`) or all rows (`all_rows`); `params.subsets` in the config holds
every value (currently 5 replicates, 4 subsets per row, size >= 2, 3,800 / 950 per class train / test).

The file stores `sample_id` strings, `doc|line_index|error_type or valid|k|f1+f2+...` (the order is part
of the id), plus the sha256 of every input. Read it only through `subsets.load_subsets(data_root, split,
replicate)`: it refuses a stale file, re-derives each sample's row, fields and label, and checks the class
balance. Run it from the repo root, after the manifest:
`uv run --python 3.12 python data/vrdu/build_vrdu_subsets.py --config data/vrdu/build-configs/2026-10-21-vrdu-invalids-04.yaml`.

Caveats:
- **Subset size leaks the label.** A larger subset is more likely to contain a corrupted field. In
  replicate 0 (train) 58% of valid samples have 2 fields against 39% of invalid ones. Report per-size
  results or a size-only baseline.
- **Most valid samples come from corrupted rows** (e.g. 1,420 of 3,800 train valid samples in replicate 0
  are from valid rows), so "valid" mixes clean rows with the clean parts of corrupted ones.
- Source rows from the manifest carry its selection bias (see Known limitations); `n_rows_too_small`
  and `n_rows_fewer_subsets_than_requested` per replicate are in the file's `stats`.
- Replicates overlap very little (pairwise Jaccard of ids ~0.005), partly because the order is part of the id.
- A row's samples, and the rows of one base row, are not independent; resample by document.

## Known limitations

- `inter_document` values almost never occur in the document text (by design); `intra` and
  `cross` values do. Report types separately.
- The matched subset is not a random sample. A line row qualifies only if intra and cross reach k=3, which
  needs several other line items with varied values, so only ~32% of line base rows (train) and ~30%
  (test) qualify, skewed toward longer, more varied line-item tables. Main keeps ~99%. Results on the
  manifest describe those documents, not the whole corpus.
- ~5% of test line rows are identical (all fields) to a train line row, and many test advertisers
  also occur in train; documents are split, but contracts recur.
