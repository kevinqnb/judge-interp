# data/vrdu

VRDU ad-buy-form corpus (641 documents) turned into extraction-validation rows for the judge:
each row is **valid** (copied from the annotations) or **invalid** (valid values deliberately
corrupted). Two datasets, each with its own task:

- **main** (entity detection): one row per document, 9 fields (`advertiser`, `agency`, `property`,
  `tv_address`, `product`, `contract_num`, `flight_from`, `flight_to`, `gross_amount`).
- **line** (relation detection): random ordered subsets of one line item's 5 fields (`channel`,
  `program_desc`, `program_start_date`, `program_end_date`, `sub_amount`).

The split is by document (513 train / 128 test, seed 0).

| script | writes | config |
|---|---|---|
| `build_vrdu_dataset.py` | `{main,line}/*_base.json` (valid rows), `ocr/`, `directory.json` | CLI args (split ratio, seed 0) |
| `build_vrdu_main.py` | `main/{train,test}.json`, `main_build.json` | `build-configs/2026-10-24-vrdu-invalids-06.yaml` |
| `build_vrdu_line.py` | `line/{train,test}.json`, `line_build.json` | the same config |

Run each script with `uv run --python 3.12 python data/vrdu/<script> --config <config>`. Output is
deterministic for a fixed seed. The row files are **gitignored**. The committed `{main,line}_build.json`
record each file's sha256, and readers refuse a stale file (`vrdu_common.check_build`). Numbers
from invalids `-05` and earlier do not carry over to `-06`. The `-05` scripts (`build_vrdu_invalids.py`,
`build_vrdu_manifest.py`, `build_vrdu_subsets.py`) and their outputs are retired.

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

**Minimal pairs.** Each invalid sample has one clean twin: the same line item, the same ordered
fields, with the original values. Every cell is therefore 50/50, and the field set, order, size, item
and document are identical across the two classes.

| cell | how | m (fields shown) | k (corrupted, within the subset) |
|---|---|---|---|
| `intra_document` | the same field from another item of the same document, a distinct donor per field | 2-5 | 1 to min(m, 3) |
| `cross_item` | two shown fields swap values | 2-5 | 2 |

There are 15 cells (`<type>:m<m>:k<k>`), each with 1,250 / 250 pairs (train / test). That makes
37,500 / 7,500 samples. Items are drawn round-robin over documents. The weakest cell is
`intra_document:m5:k3`, with 188 / 42 documents. No null-fill. Rows carry the 5 field values (null
if not shown), plus:
- `fields` (shown order), `m`, `cell`, `pair_id`, `sample_id`;
- the usual label keys.

Per-cell document counts and fragment rates are in `line_build.json`.

## Known limitations

- `inter_document` values almost never occur in the document text. `intra`, `cross_type` and
  `cross_item` values do. Report the types separately.
- Rows and samples from one document are not independent, so resample by document.
- ~5% of test line items are identical to a train line item. Documents are split, but contracts recur.
