# judge-interp

Interpretability for LLM judges in information-extraction validation tasks. When an
LLM acts as a judge — scoring or validating the output of an extraction pipeline —
this repo studies *why* it decides the way it does: what features of the candidate
extraction, the source document, and the rubric drive the verdict, and how stable
those decisions are. The aim is a mechanistic and behavioural account of judge
reliability that can feed back into how extraction validation is designed.

<!-- The description above is drawn from a one-line README and is deliberately thin;
tighten it once the first build lands. -->

## Harness

This repo is developed through the harness. Read `notes/hub/conventions.md` first —
it is the source of truth for the config standard, the experiment ID scheme, what a
run writes, and the `/develop` `/devlog` `/experiment` `/explog` `/debrief` loop.

**No magic numbers.** Every value that would change between runs lives in a
`configs/<id>.yaml` `params` block (or the repo's own committed source-of-truth
config for fixed, repo-wide values) — never hardcoded in runner code.

**Public devlog.** `<repo>/devlog/<id>.md` is the committed, curated record of
AI-assisted work on a build or experiment — my prompts, a short summary, commit
hashes. Written by `/devlog` as a trimmed copy of the private note; same review bar
as a commit message. Plain refactors and bug fixes don't get one.

This is research code. Its output goes into papers. The failure mode that matters is
not a crash — it is code that runs cleanly and produces a number that is quietly
wrong. Optimize for catching that.

## Fail loud

Defensive coding is an anti-pattern here. It converts crashes, which I would notice,
into wrong results, which I would not.

- No bare `except:` and no `except Exception` that swallows and continues.
- No default values for missing config keys. A missing key is a hard error.
- No silent fallbacks — no "if the GPU isn't available, use CPU," no "if the file is
  missing, skip it," no substituting an empty result for a failure.
- Assert at every pipeline boundary: shapes, dtypes, row counts, ID sets, and that
  joins didn't drop or duplicate rows.
- If something is genuinely optional, it gets an explicit flag, not an inferred
  default.

## Staged gates before any full run

Never go from "code written" to "full experiment submitted." Walk the ladder, and say
which rung we are on:

1. **Unit tests** on a tiny hand-built fixture where I can verify the expected output
   by inspection.
2. **Smoke run** — one step, one batch, smallest possible input, run directly in the
   current shell session — never by opening a new interactive or batch session
   yourself. Confirms plumbing, not results.
3. **Tiny end-to-end** whose result I can predict in advance. If it doesn't match the
   prediction, stop; do not scale up.
4. **Full run**, submitted through the wrapper.

Propose this sequence yourself rather than waiting for me to ask.

## Sanity controls, not just tests

Unit tests catch broken code. These catch broken _experiments_. Include them in the
experiment plan, not as an afterthought:

- **Shuffled-label / permutation control** — performance should collapse to chance.
  If it doesn't, something is leaking.
- **Known-answer case** — an input whose correct output I can state up front.
- **Seed determinism** — two runs with the same seed produce identical output. If they
  don't, find out why before interpreting anything.
- **Ablation direction** — removing a component I believe matters should hurt. A
  component that can be deleted with no effect is either useless or unused.

## Guarding the evaluation code

Changes to metric, scoring, or evaluation logic are the highest-risk edits in the
repo, because the natural debugging loop — adjust until the numbers look reasonable —
is indistinguishable from fitting the metric to the hypothesis.

- Never modify eval or metric code as part of "getting the experiment to run." If the
  eval breaks, report it and stop.
- Any change to eval logic is a separate, standalone commit with its own
  justification, and it invalidates every prior number computed with the old version.
  Say that explicitly when proposing such a change.
- When results look surprising, the first hypothesis is a bug in _our_ code, not a
  real effect. Investigate in that order.

## Guarding the eval/metrics paths specifically

This repo has no scoring or evaluation code yet, so the global
`Edit(**/eval/**)` / `Edit(**/metrics/**)` ask-rule currently protects nothing here.
When the first judge-scoring or metric code lands, check where it lives: that rule
only matches a literal directory *segment* named `eval` or `metrics`, so a flat
`analysis/metrics.py` or a `scoring.py` at the repo root is unguarded. If the real
scoring code doesn't sit under an `eval/` or `metrics/` directory, add a project-level
`.claude/settings.json` enumerating the actual file paths under `ask` — see
`hub/safety.md` for the rationale and scholarlm's `.claude/settings.json` for an
example.
