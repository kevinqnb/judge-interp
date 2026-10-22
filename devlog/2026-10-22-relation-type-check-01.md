<!-- Public devlog entry: judge-interp/devlog/2026-10-22-relation-type-check-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-10-22-relation-type-check-01
kind: build
---

## Session 2026-10-22

### Prompts

> Please help me update the relation detection tasks in `src/judge_interp/relation_detection.py` and `src/judge_interp/clef.py`. Currently these only ask about relational correctness, but we need to ask them something (relatively brief) about correctness with respect to the field type. For example in `src/judge_interp/entity_detection.py`, in addition to document presence, we ask about type correctness. A similar check needs to be implemented for the relation detection task: we may have a value which is correctly related to others it's being compared against, but if that value was listed under the wrong type then the relation should fail. Be careful and crystal clear about how you word this. But do not be overly dense or wordy.
>
> Note that this does not apply to the one shot task for clef, since that already subsumes the presence and type criteria from entity detection.

> In your assessment, is this now too complex a task for models to handle?

> commit this and write a /devlog note

### Implemented

The relation-detection instructions (`src/judge_interp/relation_detection.py`) and the Clef conditional question (`src/judge_interp/clef.py`) now state that field names are part of the relation: a value of the right item listed under the wrong field fails it. The conditional question no longer takes the conditions "as given", since they are checked too, consistent with its label. The one-shot question is unchanged. Tests assert the new wording; no config touched. This is a prompt change, so relation and conditional results from earlier prompts are not comparable with new runs.

### Commits

dacd6a9 relation prompts: field names are part of the relation (type check)
