# data/vrdu

VRDU ad-buy-form corpus (641 documents) turned into extraction-validation rows for the judge:
each row is **valid** (copied from the annotations) or **invalid** (valid values deliberately
corrupted). Two datasets, each with its own task:

- **main** (entity detection): one row per document, 9 fields (`advertiser`, `agency`, `property`,
  `tv_address`, `product`, `contract_num`, `flight_from`, `flight_to`, `gross_amount`).
- **line** (relation detection): one line item's non-null fields among `channel`, `program_desc`,
  `program_start_date`, `program_end_date`, `sub_amount`, plus all their subsets.

The split is by document (513 train / 128 test, seed 0).

| script | writes | config |
|---|---|---|
| `build_vrdu_dataset.py` | `{main,line}/*_base.json` (valid rows), `ocr/`, `directory.json` | CLI args (split ratio, seed 0) |
| `build_vrdu_main.py` | `main/{train,test}.json`, `main_build.json` | `build-configs/2026-10-24-vrdu-invalids-06.yaml` |
| `build_vrdu_line.py` | `line/{train,test}.json`, `line/{train,test}_subsets.json`, `line_build.json` | `build-configs/2026-10-27-vrdu-invalids-07.yaml` |

Run each script with `uv run --python 3.12 python data/vrdu/<script> --config <config>`. Output is
deterministic for a fixed seed. The row files are **gitignored**. The committed `{main,line}_build.json`
record each file's sha256, and readers refuse a stale file (`vrdu_common.check_build`). Numbers
from invalids `-05` and earlier do not carry over to `-06`. The `-05` scripts (`build_vrdu_invalids.py`,
`build_vrdu_manifest.py`, `build_vrdu_subsets.py`) and their outputs were removed after `-06`; see git history.

## Rules shared by both datasets

- **Label rule.** Shown values are valid iff they are one real item's values (`matches_item`; nulls
  are not shown). "Same value" is judged by an equivalence key (parsed date, numeric amount,
  folded text), so a reformat is never an error.
- **Date order.** `from <= to` holds whenever both dates parse, so a reversed date is never a tell.
- **OCR-fragment dates** (~20% of dates, e.g. `/08/20`) are valid ground truth. Corruptions draw
  fragments at the same rate, so a fragment does not predict the label.

## main

One row per (document, error type, achievable k), plus the clean row. Values are corrupted as follows:

- `inter_document` takes values from another document (k = 1-9);
- `cross_type` takes this document's value of a different field of the same kind (name, date or amount; k = 1-8).

Null fields can be filled (`corrupt_from_null`). Every corrupted source records `null_fill`.

| | valid | inter_document | cross_type |
|---|---|---|---|
| rows (train / test) | 513 / 128 | 4,617 / 1,152 | 3,836 / 944 |

**Evaluation:** `main_samples.main_entity_pairs(rows, seed)` gives one pair per corrupted span:
- the invalid span;
- its clean twin, the same (document, field) span in the clean row.

A null-fill span has no clean value, so its twin is the same field from another document
(`null_fill=True`; drop these pairs to evaluate without null-fill). This gives 39,453 / 9,751 pairs,
of which 5,598 / 1,472 are null-fill. `cross_type` thins out at k ≥ 7 (test: 109 docs at k=7, 69 at k=8).

## line

Two files per split, built in two steps. Rows carry the 5 field values (null if not shown), plus `fields`
(always in the standard field order), `m`, `cell`, `pair_id` and the usual label keys.

**1. Full-length pairs** (`line/<split>.json`). An item is shown with all its non-null fields (m = 2-5). Errors
are generated only here, on the full item, with k <= m. Each invalid sample has one clean twin (same item,
same fields, original values), and an item is in one pair per split.

| cell | how | k |
|---|---|---|
| `intra_document` | the same field from another item of the same document, a distinct donor per field | 1-3 |
| `cross_item` | two fields swap values | 2 |

4 cells (`<type>:k<k>`), 1,250 / 250 pairs each (train / test): 10,000 / 2,000 samples, half invalid. m varies
within a cell (train, invalid samples: m=2: 511, m=3: 425, m=4: 1,448, m=5: 2,616). Weakest cell:
`intra_document:k3`, 291 / 61 documents. No null-fill.

**2. Subsets** (`line/<split>_subsets.json`). Every subset of 2 to m fields of every full-length sample, valid
and invalid parents alike (train / test: 172,310 / 35,366 rows; train by size 2-5: 73,268 / 64,754 / 29,056 /
5,232). No new errors are made: a subset keeps its parent's values. Each row records its parent
(`parent_id`, `parent_index`), `size`, `invalid_fields` (the parent's corrupted fields it still shows) and
`error_type` (the parent's, if any is shown). `valid` is recomputed with `matches_item` against all line items
of the document, not inherited:
- a subset with no corrupted field is valid;
- a subset with some can still be valid if it equals a real item (~40% at size 2), so `error_type` and `valid`
  can disagree. Use `valid` as the label;
- the full-size subset is the parent, with its label.

Clean and invalid parents share subsets, so rows repeat by content. Per-cell document counts and fragment
rates are in `line_build.json`.

## Known limitations

- `inter_document` values almost never occur in the document text. `intra`, `cross_type` and
  `cross_item` values do. Report the types separately.
- Rows and samples from one document are not independent, so resample by document.
- ~5% of test line items are identical to a train line item. Documents are split, but contracts recur.
