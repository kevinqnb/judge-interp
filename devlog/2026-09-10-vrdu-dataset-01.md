---
id: 2026-09-10-vrdu-dataset-01
kind: build
---

## Session 2026-09-10

### Prompts

> Today we are setting up one dataset to be used in this repo. We are going to start
> by using the `../vrdu/ad-buy-form/main` dataset (641 PDFs + a `dataset.jsonl` of
> OCR text with entity annotations). Organize it according to the needs of this
> repository.
>
> Step 1: split documents ~80/20 train/test; keep a `data/vrdu/directory.json`
> listing document IDs, their split assignments, and relevant metadata.
>
> Step 2: write each document's `ocr.text` to its own file (named by document id) in
> `data/vrdu/ocr/`.
>
> Step 3: compile document-level datapoints (advertiser, agency, property,
> tv_address, product, contract_num, flight_from, flight_to, gross_amount) into
> `data/vrdu/main/{train,test}_base.json` by split, each with a `document_id` and a
> `valid` field set to `True`. Compile line items (channel, program_desc,
> program_start_date, program_end_date, sub_amount — one datapoint per line item)
> into `data/vrdu/line/{train,test}_base.json`, also carrying the document's
> advertiser/agency/property/tv_address/product/contract_num. Every field is always
> present; unreported fields are null.

Decisions in follow-ups: conflicting annotation occurrences for a document-level
field are resolved by majority vote over the stripped strings with the earliest
occurrence breaking ties (all distinct occurrences kept in `directory.json`); build
shape is a standalone script with split ratio and seed as required CLI args, no
config file; keep a per-line-item `line_index`; commit the structured files and
gitignore the regenerable `data/vrdu/ocr/`.

### Implemented

`scripts/build_vrdu_dataset.py` (stdlib only) reads the upstream
`ad-buy-form/main/dataset.jsonl` and writes `data/vrdu/`: `directory.json` (split +
per-document metadata + a `build` provenance block with git sha/dirty, seed,
`test_frac`, counts, field lists), `ocr/<id>.txt`, and `main/` + `line/`
`{train,test}_base.json` datapoint files. Split ratio and seed are required CLI args
with no defaults; there is no `configs/<id>.yaml`. Value selection for document-level
fields is majority vote with an earliest-occurrence tie-break, with all distinct
occurrences recorded per conflicted field. The script asserts at every boundary
(1:1 document↔PDF match, one annotation entry per document-level field, row counts,
key sets, id resolution) and carries baked-in known-answer checks (641 documents,
9163 line items, 5 documents with no line items, exact per-field non-null counts).
`tests/test_build_vrdu_dataset.py` adds 10 tests over a hand-built 3-document fixture
(`pytest` added as a dev dependency). Realized: 513/128 train/test documents,
7732/1431 line items.

### Commits

- `89b8da0` feat: add VRDU ad-buy-form dataset builder
- `c5656c4` data: build VRDU ad-buy-form dataset (2026-09-10-vrdu-dataset-01)
