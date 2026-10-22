# data/vrdu

VRDU ad-buy-form corpus (641 documents) turned into extraction-validation rows for the judge:
every row is one extraction, either **valid** (copied from the annotations) or **invalid**
(valid values deliberately corrupted). Three scripts build everything, in this order:

| script | writes | config |
|---|---|---|
| `build_vrdu_dataset.py` | `main/`, `line/` `*_base.json` (valid rows only), `ocr/`, `directory.json` | CLI args (split ratio, seed 0) |
| `build_vrdu_invalids.py` | `main/`, `line/` `{train,test}.json`, `invalids.json` | `data/vrdu/build-configs/2026-10-21-vrdu-invalids-04.yaml` |
| `build_vrdu_manifest.py` | `manifest.json` (matched evaluation subset) | `params.manifest` in the same config |

Build configs live in `build-configs/` (one per invalids version, `-01` to `-04`; only the newest
describes the data on disk). Run each script directly (local data prep, no cluster job), e.g.
`uv run --python 3.12 python data/vrdu/build_vrdu_invalids.py --config data/vrdu/build-configs/2026-10-21-vrdu-invalids-04.yaml`.
Output is deterministic for a fixed seed. Numbers from earlier invalids configs (`-01`..`-03`) do
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
| `intra_document` | line | other line items of the **same** document, one distinct donor item per corrupted field. A chain: fields and donors are drawn one at a time until `k` is reached or no eligible pair is left. |
| `cross_type` | main, line | a real value of this document that belongs to a **different field of the same kind** (name: advertiser, agency, property, tv_address, product; date: the four date fields; amount: gross_amount, sub_amount), in either dataset. The value is genuine but filed under the wrong type. `contract_num`, `channel`, `program_desc` have no kind and are never touched. Two fields that pull each other's values are marked `exchanged_with`. |

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

About 20% of line date values (17.6% main train, 12.3% main test) are OCR fragments that do not
parse as a date, e.g. `/08/20` or `02/21/`. They are valid ground truth and sit in the valid rows.
If corruptions only ever produced clean dates, "is a fragment" would predict "is correct", so
fragments are ordinary values in every pool and every error type:

- they are candidates like any other value, and order is only enforced between two **parseable**
  dates (a fragment has no position in time);
- order filtering would still over-select them (a fragment always passes), so the fragment/real
  class is drawn first at the split's own fragment rate among valid dates
  (`fragment_target_rate` in `invalids.json`), then a value within the class;
- `date_fragment_rates` records corrupted vs kept fragment rates per type. They match for main
  and for line `inter_document` / `intra_document`. **Known residual:** line `cross_type`
  corrupts to a fragment 27.7% of the time (train) / 33.2% (test) vs ~18-20% kept, because the
  document's own dates minus values already valid for the target often leave only fragments.
  Treat a fragment as weakly predictive of "corrupted" there.

## Matched subset (`manifest.json`)

The full set is not comparable across types (sizes and documents differ, above). The manifest
fixes that: a base row qualifies only if **every** error type of its dataset has a row at
**every** `k` in `k_values`; each selected base row contributes its valid row plus one row per
(error type, `k`). Every (error type, `k`) cell then has the same size, the same documents and the
same exact `k`. `k_values`: main 1-6 (507 / 126 base rows), line 1-3 (2,509 / 428 base rows).
The manifest records the invalids config id and sha256 of each split file, and
`load_manifest_split` refuses one that is stale. `data/vrdu/build_vrdu_manifest.py` documents the
selection; it is not yet wired into `row_detection`.

## Known limitations

- `inter_document` values almost never occur in the document text (by design); `intra` and
  `cross` values do. Report types separately.
- The common-support selection favours documents with several varied line items.
- ~5% of test line rows are identical (all fields) to a train line row, and many test advertisers
  also occur in train; documents are split, but contracts recur.
