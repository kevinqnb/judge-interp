# Evaluation

How entity detection and relation detection are scored. Code: `entity_detection.py`, `relation_set_detection.py`,
`relation_pair_detection.py` (full details in their docstrings); datasets: `data/vrdu/README.md`.

**Methods.** All are scored on the same test items, with the same code path.
- *Probes* (per model): logistic regression on hidden-state activations, trained on train-run activations, scored on test.
- *Judges* (Clef, Jev): direct judges, nothing trained.

**Metrics.** Label = valid (1) / invalid (0). smECE, Brier skill score, AUROC, accuracy. Reported overall and per stratum
(a stratum with one class has no AUROC). Every interval is a bootstrap over **documents** (items from one document are
resampled together), and the pooled values in the calibration report are asserted equal to the metrics.

**Common set.** The test items every method scored (judges drop over-length documents, LMs have a context limit).
Every method is cut to it, and each method's drops are reported.

---

## Entity detection (`main`)

**Item: a patched pair.** For each corrupted span with a non-null original value: the corrupted span (invalid) and the
same field in the *patched* row (valid), meaning the corrupted row with only that span restored (k = 1: the clean row).
Only the target span differs; the rest of the context is identical. Null-fill spans have no counterpart and are dropped.
Pairs must be identical across methods (asserted). Each (error type, k, field) stratum is exactly 50/50.

**k stratum.** `train_k` / `test_k` choose the corrupted-field counts kept (null = all). A patched row counts as its
parent's k.

**Probe.** Features are the activation of the target entity at one layer, with no centering (the pair members share the
document). Train rows are `train_eval.json`, test rows `test_eval.json`.

**Strata.** Error type (`inter_document`, `cross_type`) and (error type, k).

**Diagnostic.** Row context: the uncorrupted spans of corrupted rows against the same spans in the clean row (both
valid). AUROC ~ 0.5 means the method does not just sense "this row has a corruption somewhere".

**Controls.**
- Field-identity baseline: AUROC exactly 0.5 on pairs (asserted).
- Train-label shuffle (probes refit on permuted train labels, scored on real test labels): AUROC ~ 0.5 (asserted).
- Seed determinism.
- For judges, the report's shuffled-label baseline is the only label-permutation control.

---

## Relation detection (`line` subsets)

**Item: a subset row.** 2 to m fields of a full-length sample, labelled by its own `valid` (label rule against the
document's line items). Rows are not minimal pairs and repeat by content, so the Brier reference is the evaluation
set's own valid rate.

**Selection (`params.selection`, applied to train and test).**
- `null`: every row.
- `corrupted_parents`: only rows of corrupted parents (so every row has a corrupted prompt as context), subsampled with
  the config seed to equal valid and invalid counts in every (error type, field set) cell. Cells with one class are
  dropped. On the test side the draw is made from the common set, so every method is scored on exactly the same
  balanced rows. Twin diagnostics and the control report are skipped (no clean parents).

**Probe.** Logistic regression on the sum of the subset members' activations (last token of the value, in the parent's
prompt, per-document mean-centred). Ablations: `last` (the member shown last) and `context` (the sum of the others). A
judge sees only the subset; a probe's member has attended to every field shown before it in the parent.

**Strata.** Error type, cell, size, corrupted members, parent validity, and row kind:
`invalid`; `clean_parent`; `corrupted_member` (valid despite a corrupted member); `hidden` / `visible` (valid, no
corrupted member; every corrupted parent field is shown after all members / before one).
With `selection: null`, a context diagnostic compares each valid row of a corrupted parent with the same subset of the
clean parent.

**Controls.**
- Structure baseline (all methods): a probe on the field set. AUROC exactly 0.5 under `corrupted_parents` (asserted);
  only reported under `null`.
- Train-label shuffle: AUROC ~ 0.5 (asserted, probes only).
- Seed determinism (asserted, probes only).
- `hidden` prefix known answer: a hidden row's features equal its twin's up to `prefix_match_rtol` (asserted, probes only).
- Sum vs ablation: document-bootstrap AUROC difference.
- With `selection: null`, a second control report drops the strict subsets of clean parents (all valid), so "clean
  parent" and "short" are not confounded with the label.

---

## Relation pair detection (`line` pairs)

**Item: an entity pair.** Two fields of a full-length sample (a size-2 row of `line/<split>_subsets.json`), labelled by
its own `valid` (label rule). The LM prompt lists all C(m, 2) pairs of the row, one per line, in a seeded random order and
orientation shared by the row and its clean twin, and reads each pair at its line's `' "\n'`. Judges (Clef, Jev
`conditional`) see one pair per prompt, in the LM's orientation (asserted).

**Selection (train and test).** Pairs of corrupted parents only, randomly balanced to 50/50 in every (error type, field
pair) cell; the test draw is made from the common set.

**Probe.** Logistic regression on the pair's activation at one layer (per-document mean-centred).

**Strata.** Error type, cell, field pair, corrupted members (0/1/2), pair position, parent size, and whether a corrupted
pair was shown earlier in the prompt.

**Controls.**
- Structure baseline: field-pair one-hot, AUROC exactly 0.5 (asserted).
- Train-label shuffle: AUROC ~ 0.5 (asserted, probes only). Seed determinism (asserted, probes only).
- Layer 0 constant over all pairs (asserted).
- Prefix known answer: pairs shown before a corrupted prompt's first corrupted pair equal the clean twin's up to
  `prefix_match_rtol` (asserted, probes only).
