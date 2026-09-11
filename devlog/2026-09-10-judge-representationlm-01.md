<!-- Public devlog entry: judge-interp/devlog/2026-09-10-judge-representationlm-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-09-10-judge-representationlm-01
kind: build
---

## Session 2026-09-10

### Prompts

Verbatim first request:

> The core of this library should contain code for running an LLM judge, which is
> used to judge or 'validate' individual entries from our datasets. A single judge
> instance should therefore be given (a) instructions for performing validation, (b)
> the full OCR context, and (c) a single data point to judge for validity.
>
> An `instruction_prompts.py` file should be used to define a string with
> instructions for the model. In particular, for a data point to be valid:
> * The value for each field must appear explicitly in the context, or be left as
>   null
> * If there is one or more choices for a field, then the field must hold a value
>   which is *cohesive* with other field values. For example, if there are two
>   addresses in the context, "63 Chestnut St. Boston, MA" and "44 Milton St. Albany,
>   NY", then a data point such as `{street: 44 Milton St., city: Boston, state: NY}`
>   would be invalid. To be cohesive, the value for the city field must be Albany.
> * If any one field does not satisfy the criteria, then the whole point is invalid
>
> A `representationlm.py` script should then form the main framework for running the
> model with `nnsight`, and for generating and passing queries to the model. While
> running the model we should collect last token output representations from a user
> given list of intermediate layers. This should default to just the last layer in
> the model. These outputs should then be cached and saved.

Follow-up decisions: package as `src/judge_interp/` (added a hatchling build);
instruct model + chat template (not base + answer cue); a single prefill trace
reading the residual and verdict probability at the last prompt token (not
generate-then-read); judge model `Qwen/Qwen2.5-7B-Instruct`, `uv sync --extra gpu`.

### Implemented

Added `src/judge_interp/`: `instruction_prompts.py` (the validation rubric, with a
separable cohesion criterion toggled by `include_cohesion` — an ablation control, not
just a flag), `prompts.py` (pure query rendering + layer resolution, no GPU deps),
and `representationlm.py` (`RepresentationLM`: one nnsight prefill trace per data
point, last-prompt-token residual + casing-marginalised `P(true)`, a
`verify_read_point` determinism gate, and a per-document-sharded, resumable
`collect()`). Added the harness adapter `scripts/run_experiment.py` and
`scripts/submit.sh`/`_run_experiment_job.sh` (GPU resources read from the config).
`pyproject.toml` gained a `gpu` extra and a real package build. 61 tests (39 new),
including a hand-built fixture corpus and a runnable tiny end-to-end config
(`tests/fixtures/repr_tiny_e2e.yaml`). Ladder: unit tests, a CPU smoke run, and a
tiny end-to-end run all passed (after fixing two real bugs the smoke/e2e rungs
caught); the first real cluster run (rung 4) was not attempted this session.

### Commits

cc8cef4 feat: add judge-interp core (instruction prompts, RepresentationLM, run adapter)
