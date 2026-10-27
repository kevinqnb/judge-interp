<!-- Public devlog entry: judge-interp/devlog/2026-10-27-vrdu-line-fulllength-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-10-27-vrdu-line-fulllength-01
kind: build
config: data/vrdu/build-configs/2026-10-27-vrdu-invalids-07.yaml
---

## Session 2026-10-27

### Prompts

> Please help me make a targeted change to how the line dataset is build with `data/vrdu/build_vrdu_line.py`. Here's how the process should look:
> 1. First, generate full length ($m = 5$) invalid items to be paired with their full length valid (clean) counterparts. Keep these as their own separate train/test datasets.
> 2. Starting from the full length sets, enumerate all possible subsets of each entry. Record its parent's id and/or index, its validity label (as determined by running `match_item()` across all other line items from the doc), the error type present, and the subset size. Save these as their own separate train/test subsets.

Decisions: replace `line/{train,test}.json` with the full-length set; subsets are order-preserving, sizes 2-5, in a standard field order (no random ordering).

> I should clarify: a full length item does not need to have $m=5$ necessarily. The intention is just to separate the original records from artificially created subsets.

> Please confirm: when we generate errors at different $k$ values, is that always within the full length items? Please outline the composition of the dataset now, and make sure this is described cleanly at a high level in the readme.

> Please say more about why 38% of size-2 train subsets that still show a corrupted field are labelled valid

> Please update stale fixtures, and any downstream consumers in the main `src/judge_interp` source code. I will update analysis consumers later.

### Implemented

`data/vrdu/build_vrdu_line.py` now builds full-length minimal pairs (each item shown with all its non-null fields in standard order, one pair per item, cells `<type>:k<k>`) and, separately, every subset of 2 to m fields of each pair, written to `line/<split>_subsets.json` with parent id/index, size, error type and a label recomputed by `matches_item`. New config `data/vrdu/build-configs/2026-10-27-vrdu-invalids-07.yaml` (line only; main stays on -06). Tests, the line fixtures (`tests/fixtures/make_line_fixtures.py`) and the hand-worked expectations in four `src`-facing test files were migrated; `src/judge_interp` needed no change. The README's line section is rewritten. Analysis scripts and their tests are not yet migrated.

### Commits
dc9531f vrdu line: full-length pairs plus a separate subsets file (-07)

## Session 2026-10-27 (continued)

### Prompts

> Please draft configs for running both entity detection and relation detection with llama and qwen on the new line dataset -- only on the full `train.json` and `test.json` sets. Also draft configs for running these tasks on the line dataset with clef and jev. IN this case, we ONLY want to run these with the `test_subsets.json` set.

> Yes, use the subset's own valid label instead of per-entity flags. Build out support for running on the subsets sample files.

> Commit the changes and add to the devlog note.

### Implemented

Drafted 8 configs for Llama and Qwen entity/relation detection on the full -07 train and test sets (`experiments/experiment-configs/{entity,relation}-detection/2026-10-27-*-line-{train,test}-01.yaml`, 4 cells each) and two Clef/Jev conditional configs on the test subsets (`relation-detection/{clef,jev}/2026-10-27-*-conditional-subsets-test-01.yaml`). To run on subsets, Clef/Jev sample files now carry an explicit `label_valid` instead of deriving it from the entity flags, and `experiments/clef_samples.py` takes `--source pairs|subsets`; a subset is labelled by its own `matches_item` result. Existing samples files lack the new key and must be regenerated. Nothing has been run.

### Commits
1e59a46 clef/jev: run on the line subsets; configs for the -07 line sets
