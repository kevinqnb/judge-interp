<!-- Public devlog entry: judge-interp/devlog/2026-10-27-row-prefix-cache-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-10-27-row-prefix-cache-01
kind: build
---

## Session 2026-10-27

### Prompts

> Can you investigate four main files in this repo for adequate usage of prompt caching? These will use different models / serve setups, so things will be different in each case. But the overall setup and motivation remains the same: in each case we may submit multiple queries for the same document. In that case, we want to make sure that we're not overusing tokens by reinstantiating the prompt every time. Instead, the document should be cached and reused. [entity_detection, relation_detection, clef, jev]

- Do not change the Clef model's forward pass; if it cannot handle caching without that, it is just at a disadvantage. Still order requests by document like Jev, in case the serving process caches automatically.
- Asked whether vLLM's automatic prefix caching could be used for entity/relation detection.
- When recording token usage, account for caching, so genuinely fewer processed tokens are evident in the result.
- "Yes, start with the HF path; use atol 1e-2"
- "I think you missed some of the newest configs. I want prefix caching to be turned on for those."
- Commit the prefix cache changes and write a /devlog note.

### Implemented

Entity and relation detection (`src/judge_interp/row_detection.py`) now compute the KV of instructions + context once per document and run only the record for its other rows (`prefix_cache`). Three required config params control it: `prefix_cache`, `prefix_cache_check_n` (first n rows are also run uncached) and `prefix_cache_atol` (1e-2 here); a mismatch fails the run. The check caught a silent masking bug while this was written, fixed with an explicit all-ones attention mask on the cached pass. Token accounting now separates `reused_n_tokens` from `computed_n_tokens` (new `eff_reused_*` / `eff_computed_*` metrics). Existing entity/relation configs got `prefix_cache: false`; the `2026-10-27` line configs (`experiments/experiment-configs/{entity,relation}-detection/2026-10-27-*-line-*-01.yaml`) have it on. Clef (no cache without changing its forward pass) and Jev (no visible cache control) are not changed; tests are in `tests/test_prefix_cache.py`, GPU runs are still to do.

### Commits

67c688f row detection: per-document KV prefix cache with token accounting and an equivalence check
