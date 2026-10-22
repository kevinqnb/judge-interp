<!-- Public devlog entry: judge-interp/devlog/2026-10-22-vrdu-invalids-shortcuts-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-10-22-vrdu-invalids-shortcuts-01
kind: build
config: data/vrdu/build-configs/2026-10-21-vrdu-invalids-04.yaml
---

## Session 2026-10-22

### Prompts

> Please help me audit and clean up `scripts/build_vrdu_dataset.py` and `scripts/build_vrdu_invalids.py`:
> move them into `data/vrdu/`, look for small errors or inconsistencies, check that each invalid type is
> built cleanly and faithfully to its intent (especially the cross-type errors, and whether dates being
> irreplaceable because of ordering requirements is true or necessary), and check that everything is set
> up to create a fair experimentation setup for entity and relation detection.

> I am most concerned about the judge learning shortcuts, however inter document errors are intended to be
> this way. We are explicitly testing the model's ability on this. The important thing is that we have a
> comparable number of other error types to look at as well.

Follow-ups (summary): accepted treating OCR-fragment dates as ordinary values and a matched evaluation
manifest ("go ahead with the plan as proposed"); chose the manifest `k_values`; asked for a concise
`data/vrdu/README.md` (how error types are made, composition, a note on fragments); moved the dataset
build configs into `data/vrdu/build-configs/`; gitignored the large row files.

### Implemented

The two builders now live in `data/vrdu/` with their configs in `data/vrdu/build-configs/`. A new invalids
config, `data/vrdu/build-configs/2026-10-21-vrdu-invalids-04.yaml`, removes judge shortcuts found in the
audit: OCR-fragment dates are ordinary values in every pool and error type (date order is enforced only when
both dates parse, and fragments are drawn at a controlled rate), a "corruption" must differ from the original
by an equivalence key (so reformats are never errors), the synthetic date sentinels are gone, and every
corrupted entity carries an explicit `null_fill` flag. `data/vrdu/build_vrdu_manifest.py` writes
`manifest.json`, a matched subset where every error type is present at every configured `k` for the same
base rows; it records the sha256 of each split file and refuses a stale manifest. `row_detection.check_row`
accepts the new flag. The large `{main,line}/{train,test}.json` files are now gitignored (regenerate them
with the build scripts; `data/vrdu/README.md` describes the data). The manifest is not yet wired into
`row_detection.select_rows`.

### Commits
0229be2 Move VRDU dataset/invalids builders into data/vrdu/
4922163 invalids -04: fragments as ordinary values, reformat-proof corruptions, no sentinels, null_fill tag
994827a Add matched-subset manifest builder for the VRDU invalid rows
e60c0fa invalids -04: draw date fragments at the split's valid fragment rate
6e910f5 Move VRDU invalids build configs into data/vrdu/build-configs/
f5a9975 Add data/vrdu/README.md: how the VRDU rows and error types are built
1c2082d Gitignore the large VRDU invalid-row files; keep provenance committed
80f13a4 VRDU -04 data: provenance, matched-subset manifest, manifest k_values
