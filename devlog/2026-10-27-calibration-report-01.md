<!-- Public devlog entry: judge-interp/devlog/2026-10-27-calibration-report-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-10-27-calibration-report-01
kind: build
config: analysis/configs/entity_detection/2026-10-27-entity-main-all-01.yaml
---

## Session 2026-10-27

### Prompts

> Update the figures and results of entity detection and relation set detection, always stratified by error type (inter-document / cross-type; intra-document / cross-item); alert me if the data does not split cleanly.
> 1. Figures: calibration only, one per error type comparing qwen, llama, clef, jev; pastel tab10; separate legend PDF; ACL-style font; 95% bands from 2000 document resamples.
> 2. Tables: LaTeX smECE / BSS / AUROC with 95% document-bootstrap intervals, rows = four models + shuffled-label baseline, columns grouped by error type.
> 3. Relation control: keep full-length corrupted parents and their clean twins plus strict subsets of corrupted parents; drop strict subsets of clean parents.

> Commit the changes and write a /devlog note.

### Implemented

New `analysis/calibration_report.py` produces, per error type, a calibration diagram (PDF, separate legend), a LaTeX table and a CSV, with intervals over 2000 document resamples and a shuffled-label baseline; pooled values are asserted equal to the existing metrics. It is wired into `analysis/entity_detection.py` and `analysis/relation_set_detection.py` (which also writes the control report), the other figures are removed, and `params.report` was added to `analysis/configs/entity_detection/2026-10-27-entity-main-all-01.yaml`, its smoke config and `analysis/configs/relation_set_detection/2026-10-27-relset-subsets-all-01.yaml`. Unit tests and an entity smoke run pass; the relation-set analysis has not been run end to end.

### Commits
16fb071 clef: analyse_run can skip its own figures (fig_dir=None)
9f2326d calibration report: per-error-type document-bootstrap figures and tables, relation control
