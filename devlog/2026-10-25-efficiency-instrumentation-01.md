<!-- Public devlog entry: judge-interp/devlog/2026-10-25-efficiency-instrumentation-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-10-25-efficiency-instrumentation-01
kind: build
config: experiments/experiment-configs/ (17 configs dated 2026-10-25)
---

## Session 2026-10-25

### Prompts

> Please help me write updated experiment configs within `experiments/experiment-configs/`: entity detection
> (vrdu main train/test and line train/test; `qwen-2.5-7b` and `llama-3.1-8b` representations, and clef on main),
> relation detection (line train/test; both models and clef), one-shot validation (line test; clef).
> It's very important to me that while running these experiments we are collecting the necessary information to
> measure runtime and computational efficiency. For the language models collecting activations, this would mean
> runtime, token usage, and memory usage to collect and store a layer's representations. For clef, this will mean
> runtime, token usage (input/output tokens), and memory usage. Before writing any configs, please assess the
> experimental setup for collecting this information. If necessary, we should implement any changes needed.

Decisions: pin the GPU type to L40S; per-prompt and per-layer storage cost only (no early-stopped forward or
layer ablation); full data everywhere.

### Implemented

New `src/judge_interp/efficiency.py` plus measurement hooks in `row_detection.py` and `clef.py`: synchronized
per-unit timing by phase, peak GPU memory, host RSS, input tokens and per-layer storage bytes, written to
`artifacts/efficiency.*` and `eff_*` metrics. Output is unchanged (bitwise-identical `span_representations.npz`
on the entity fixture and a -06 line smoke). Seventeen configs dated 2026-10-25 cover the requested runs, with
placeholder walltimes pending a GPU pilot; the Clef path is not yet exercised on a GPU.

### Commits

6b66789 efficiency: measure runtime, tokens and memory in the representation and Clef tasks
ab9f17a configs: 2026-10-25 entity, relation and one-shot runs on vrdu -06
