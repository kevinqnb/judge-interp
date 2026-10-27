"""TypeSafe Jev as a direct judge for entity, conditional relation and one-shot record validity; no layer representations.

Mirror of ``clef``: the same tasks, rows, samples, labels and question wording, with Jev
(a System One model, called over the TypeSafe API) in place of ``Cloudflare/clef-flash``. A
``noul`` question returns the probability that its statement is true, so ``P(true)`` is read
straight from the response. The model is pinned by a versioned id (``model_config.load_jev_model_config``).

Tasks (as in ``clef``):
- ``jev_entity_row``: rows selected by ``entity_detection.build_items``, the same rows the
  Qwen/Llama/Clef tasks judge (a null field is not shown, as there). One request per row, one
  ``noul`` question per non-null field.
- ``jev_relation_conditional`` and ``jev_record_oneshot`` judge *samples* read from a samples
  file (``params.samples_path``, see ``clef.load_samples``, which is reused): one request and one
  ``noul`` question per sample. Conditional: is the primary entity correctly associated with all
  of the conditions, every value under its correct field? Label: true iff the primary and every
  condition is valid. One-shot: is the whole record valid? Label: true iff every entity is valid.

Differences from ``clef``:
- The state is a JSON object with named fields (``context``, ``extracted_record`` or
  ``primary_entity`` / ``conditions``), and a question refers to a part by its path in backticks,
  not a markdown string with ``##`` sections. The questions are otherwise worded as in ``clef``.
- Only ``P(true)`` comes back (no logits). ``logit_true`` / ``logit_false`` do not exist.
- There is no local tokenizer, so records are not pre-filtered for length: a request over the API's
  context limit raises ``TypeSafeAPIError``. There is no ``overlong_docs`` param.
- Calls are remote, so the determinism check compares ``P(true)`` of a rescoring within
  ``params.determinism_atol`` and reports the largest difference, instead of requiring bitwise equality.
- Efficiency is per-request wall time and API-reported token counts; no GPU measurements.

Pipeline: ``entity_units`` / ``conditional_units`` / ``oneshot_units`` build records,
``JevScorer`` scores them one request at a time, and ``run`` writes
``artifacts/jev_predictions.npz`` and ``artifacts/efficiency_records.npz`` and returns flat metrics.
The API key is ``TYPESAFE_API_KEY`` in the environment (e.g. ``uv run --env-file .env ...``).
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from typesafe_sdk import Noul, RetryPolicy, TypeSafeClient

from judge_interp import clef
from judge_interp import entity_detection as edr
from judge_interp.dataset_spec import DatasetSpec
from judge_interp.prompts import load_context

SAMPLE_TASKS = ("jev_relation_conditional", "jev_record_oneshot")
TASKS = ("jev_entity_row", *SAMPLE_TASKS)
_CLEF_TASK = {
    "jev_entity_row": "clef_entity_row", "jev_relation_conditional": "clef_relation_conditional",
    "jev_record_oneshot": "clef_record_oneshot",
}
_COMMON_PARAMS = (
    "data_root", "model_config", "dataset_config", "context_control", "determinism_check_n", "determinism_atol",
    "known_answer_check",
)
TASK_PARAMS: dict[str, tuple[str, ...]] = {
    "jev_entity_row": (*_COMMON_PARAMS, "split", "error_types", "rows_per_stratum"),
    "jev_relation_conditional": (*_COMMON_PARAMS, "samples_path"),
    "jev_record_oneshot": (*_COMMON_PARAMS, "samples_path"),
}
# Per-request measurements of ``JevScorer.score``.
EFFICIENCY_KEYS = ("seconds_request", "input_tokens", "output_tokens", "n_questions")

_ENTITY_QUESTION = """In the `extracted_record` of the state, which is {record_description}, consider the value of the field `extracted_record.{field}`. This field denotes {desc}.

Is this value present in `context` (the {context_description}) and of the correct type for this field? That is: is it a VALID value for `{field}` in this document? It is valid only if BOTH criteria hold:

{criteria}

Judge only against `context`. Do not use outside knowledge about the entities involved, and do not guess when the evidence is unclear: unclear evidence fails a criterion. Judge this field on its own, independently of the record's other fields."""

_CONDITIONAL_QUESTION = """The state has a primary entity, `primary_entity`, and conditions, `conditions`, which together come from {record_description}. Each is a `field: value` pair.

The primary entity is the field `{primary}`: {primary_desc}.
The conditions are:
{conditions}

The document may describe many items of the kind this record describes. Is the primary entity correctly associated with the conditions in `context` (the {context_description})? That is: does the primary entity belong to the same single item as every one of the conditions? Answer true only if it is related to all of them. Answer false if it is not related to one or more of them, for example because it belongs to a different item than they do. A value that does not appear in `context` cannot be associated with anything in it.

The field names are part of the association. The primary entity and each condition must be what its field denotes for that item, as described above. A value that belongs to the same item but is listed under the wrong field (it is actually the value of a different field) makes the association fail: answer false.

Judge only against `context`. Do not use outside knowledge about the entities involved, and do not guess when the evidence is unclear: unclear evidence means false."""

_ONESHOT_QUESTION = """The `extracted_record` of the state is {record_description}, a set of `field: value` pairs. The fields present are:
{types}

Is the ENTIRE record valid? It is valid only if ALL of the following hold:

{criteria}
(3) Relation. All the values must be related: taken together they describe the same single item in `context` (the {context_description}). The document may describe many items of the kind this record describes. If one or more values belong to a different item than the others, this criterion fails, even though each of those values may be a genuine value of the document.

Criteria (1) and (2) apply to every value of the record; criterion (3) applies to the record as a whole. If any value fails (1) or (2), or the values are not all related, answer false. Answer true only if the whole record is valid.

Judge only against `context`. Do not use outside knowledge about the entities involved, and do not guess when the evidence is unclear: unclear evidence fails a criterion."""


# ----------------------------------------------------------------------
# State / question construction (model-free)


def record_state(context: str, values: dict[str, str | None], fields: list[str]) -> dict:
    """Assemble the Jev ``state`` of an entity or one-shot record.

    Args:
        context: context text of the document.
        values: field -> value; every field in ``fields`` must have a ``str`` value.
        fields: the fields to show, in output order.

    Returns:
        ``{"context": str, "extracted_record": {field: value}}``, fields in ``fields`` order.
    """
    assert fields and all(isinstance(values[f], str) for f in fields), fields
    return {"context": context, "extracted_record": {f: values[f] for f in fields}}


def conditional_state(context: str, primary: tuple[str, str], conditions: list[tuple[str, str]]) -> dict:
    """Assemble the Jev ``state`` of a conditional record.

    Args:
        context: context text of the document.
        primary: ``(field, value)`` of the primary entity.
        conditions: ``(field, value)`` of each condition, in order.

    Returns:
        ``{"context", "primary_entity": {field: value}, "conditions": {field: value, ...}}``.
    """
    assert conditions and all(isinstance(v, str) for _, v in [primary, *conditions])
    return {"context": context, "primary_entity": dict([primary]), "conditions": dict(conditions)}


def entity_question(field: str, spec: DatasetSpec) -> Noul:
    """Build the ``noul`` question asking whether one field's value is valid.

    Args:
        field: the field whose value is judged.
        spec: the dataset; supplies the record, context and field descriptions.
    """
    return Noul(instructions=_ENTITY_QUESTION.format(
        record_description=spec.record_description, context_description=spec.context_description,
        field=field, desc=spec.field_descriptions[field], criteria=edr.build_criteria(spec, "as this field is described"),
    ))


def conditional_question(primary: str, conditions: list[str], spec: DatasetSpec) -> Noul:
    """Build the ``noul`` question asking whether the primary entity is related to all conditions.

    Args:
        primary: the primary field.
        conditions: the condition fields: at least one, distinct, none equal to ``primary``.
        spec: the dataset; must support the relation tasks.
    """
    spec.require_relation()
    assert conditions and len({primary, *conditions}) == len(conditions) + 1 and {primary, *conditions} <= set(spec.fields), (
        primary, conditions)
    return Noul(instructions=_CONDITIONAL_QUESTION.format(
        record_description=spec.record_description, context_description=spec.context_description,
        primary=primary, primary_desc=spec.field_descriptions[primary],
        conditions="\n".join(f"- `conditions.{f}`: {spec.field_descriptions[f]}" for f in conditions),
    ))


def oneshot_question(fields: list[str], spec: DatasetSpec) -> Noul:
    """Build the ``noul`` question asking whether the whole record is valid.

    Args:
        fields: at least two distinct fields of ``spec`` present in the record.
        spec: the dataset; must support the relation tasks.
    """
    spec.require_relation()
    assert len(set(fields)) == len(fields) >= 2 and set(fields) <= set(spec.fields), fields
    return Noul(instructions=_ONESHOT_QUESTION.format(
        record_description=spec.record_description, context_description=spec.context_description,
        types="\n".join(f"- {f}: {spec.field_descriptions[f]}" for f in fields),
        criteria=edr.build_criteria(spec, edr.QWEN_DESCRIBED),
    ))


def entity_units(items: list[dict], spec: DatasetSpec, contexts: dict[str, str]) -> list[dict]:
    """Build one Jev unit per row.

    Args:
        items: items from ``entity_detection.build_items``.
        spec: the dataset.
        contexts: document_id -> the context text shown for that document's rows (its own
            unless the context control swapped it).

    Returns:
        One dict per row with ``doc_id``, ``row_ordinal``, ``spans`` and ``record`` (``id``,
        ``state``, and one question per non-null field, keyed by field).
    """
    units = []
    for it in items:
        values = clef.item_values(it, spec)
        fields = [sp["field"] for sp in it["spans"]]
        assert all(values[f] is not None for f in fields) and set(fields) == {f for f in spec.fields if values[f] is not None}
        units.append({
            "doc_id": it["document_id"],
            "row_ordinal": it["row_ordinal"],
            "spans": it["spans"],
            "record": {
                "id": f"{it['document_id']}|{it['row_ordinal']}",
                "state": record_state(contexts[it["document_id"]], values, fields),
                "questions": {f: entity_question(f, spec) for f in fields},
            },
        })
    return units


def _sample_unit(s: dict, ents: list[dict], state: dict, question: Noul) -> dict:
    """The unit fields the two sample tasks share (``fields`` are in the order shown)."""
    bad = [e["field"] for e in ents if not e["valid"]]
    return {
        "sample_id": s["sample_id"],
        "doc_id": s["document_id"],
        **{k: s[k] for k in clef._CARRIED_KEYS},
        "fields": [e["field"] for e in ents],
        "invalid_fields": bad,
        "label_valid": not bad,
        "record": {"id": s["sample_id"], "state": state, "questions": {"q": question}},
    }


def conditional_units(samples: list[dict], spec: DatasetSpec, contexts: dict[str, str]) -> list[dict]:
    """Build one Jev unit per conditional sample (output of ``clef.load_samples``).

    Returns:
        One dict per sample with the keys of ``clef.conditional_units`` (``fields`` primary first,
        ``label_valid`` = primary and every condition valid) and ``record`` (one ``noul`` question).
    """
    units = []
    for s in samples:
        prim, conds = s["primary"], s["conditions"]
        assert conds, s["sample_id"]
        state = conditional_state(
            contexts[s["document_id"]], (prim["field"], prim["value"]), [(e["field"], e["value"]) for e in conds])
        u = _sample_unit(s, [prim, *conds], state, conditional_question(prim["field"], [e["field"] for e in conds], spec))
        u.update(primary=prim["field"], conditions=[e["field"] for e in conds], primary_valid=prim["valid"],
                 invalid_conditions=[e["field"] for e in conds if not e["valid"]])
        assert u["label_valid"] == (u["primary_valid"] and not u["invalid_conditions"])
        units.append(u)
    return units


def oneshot_units(samples: list[dict], spec: DatasetSpec, contexts: dict[str, str]) -> list[dict]:
    """Build one Jev unit per one-shot sample (output of ``clef.load_samples``).

    Returns:
        One dict per sample with the keys of ``clef.oneshot_units`` (``label_valid`` = every entity
        valid) and ``record`` (one ``noul`` question over the whole record).
    """
    units = []
    for s in samples:
        ents = s["entities"]
        fields = [e["field"] for e in ents]
        state = record_state(contexts[s["document_id"]], {e["field"]: e["value"] for e in ents}, fields)
        units.append(_sample_unit(s, ents, state, oneshot_question(fields, spec)))
    return units


# ----------------------------------------------------------------------
# Model side


class JevScorer:
    """Scores one record per API request with a pinned Jev release.

    Args:
        model: versioned model id (``jev-X.Y.Z``); sent on every request and asserted on every response.
        retry: ``max_retries``, ``backoff_max`` and ``timeout`` (seconds) of ``typesafe_sdk.RetryPolicy``.

    Raises:
        RuntimeError: ``TYPESAFE_API_KEY`` is not set.
    """

    def __init__(self, model: str, retry: dict):
        if not os.environ.get("TYPESAFE_API_KEY"):
            raise RuntimeError("TYPESAFE_API_KEY is not set (run with `uv run --env-file .env ...`)")
        self.model = model
        self.client = TypeSafeClient(model=model, retry=RetryPolicy(**retry))

    def score(self, record: dict) -> dict:
        """Score one record.

        Args:
            record: ``id``, ``state`` (JSON object) and ``questions`` (name -> ``Noul``).

        Returns:
            ``p_true`` (``float64 [n_questions]``, in the order of ``record["questions"]``),
            ``n_tokens`` (input tokens as reported by the API), ``request_id`` and ``efficiency``
            (see ``EFFICIENCY_KEYS``).

        Raises:
            typesafe_sdk.TypeSafeAPIError: the request failed after retries (including a request over the context limit).
            AssertionError: the response's model differs from the pinned one, an answer is missing or
                not a probability, or token counts are not reported.
        """
        names = list(record["questions"])
        t0 = time.perf_counter()
        resp = self.client.system_one(record["state"], record["questions"], model=self.model)
        seconds = time.perf_counter() - t0
        assert resp.model == self.model, f"{record['id']!r}: answered by {resp.model!r}, pinned {self.model!r}"
        assert set(resp.answers) == set(names) and set(resp.nouls) == set(names), (sorted(resp.answers), names)
        p_true = np.array([resp.nouls[k].noul for k in names], dtype=np.float64)
        assert np.isfinite(p_true).all() and ((0 <= p_true) & (p_true <= 1)).all(), p_true
        assert resp.usage.input_tokens is not None and resp.usage.output_tokens is not None, f"{record['id']!r}: usage not reported"
        efficiency = {
            "seconds_request": seconds, "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens, "n_questions": len(names),
        }
        assert set(efficiency) == set(EFFICIENCY_KEYS)
        return {"p_true": p_true, "n_tokens": resp.usage.input_tokens, "request_id": resp.request_id, "efficiency": efficiency}

    def known_answer_check(self) -> dict:
        """Score a hand-built record whose answers are known: a total of 1250 is above 1000, not above 2000.

        Returns:
            ``known_answer_p_yes`` and ``known_answer_p_no`` (``P(true)`` of each question).

        Raises:
            AssertionError: the model does not answer the two questions yes and no.
        """
        rec = {"id": "known-answer", "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD"}},
               "questions": {"above_1000": Noul(instructions="Is `invoice.total` above 1000 USD?"),
                             "above_2000": Noul(instructions="Is `invoice.total` above 2000 USD?")}}
        p = self.score(rec)["p_true"]
        assert p[0] > 0.5 > p[1], f"known-answer case failed: P(true) = {p.tolist()} (expected [>0.5, <0.5])"
        return {"known_answer_p_yes": float(p[0]), "known_answer_p_no": float(p[1])}


# ----------------------------------------------------------------------
# Run


def _score_units(scorer: JevScorer, units: list[dict], determinism_check_n: int, determinism_atol: float,
                 verbose: bool = True) -> tuple[list[dict], float]:
    """Score every unit, then rescore the first ``determinism_check_n`` and require ``P(true)`` within ``determinism_atol``.

    Returns:
        ``(results, max_abs_diff)``: ``JevScorer.score`` output per unit, in order, and the largest
        ``|P(true) - P(true) of the rescoring|`` (0.0 if nothing was rescored).
    """
    assert 0 <= determinism_check_n <= len(units) and determinism_atol >= 0
    results = []
    for i, u in enumerate(units):
        r = scorer.score(u["record"])
        assert len(r["p_true"]) == len(u["record"]["questions"])
        results.append(r)
        if verbose and (i + 1) % 50 == 0:
            print(f"scored {i + 1}/{len(units)}")
    max_diff = 0.0
    for u, r in zip(units[:determinism_check_n], results):
        again = scorer.score(u["record"])
        assert again["n_tokens"] == r["n_tokens"], f"{u['record']['id']}: rescoring counted different input tokens"
        diff = float(np.abs(again["p_true"] - r["p_true"]).max())
        assert diff <= determinism_atol, f"{u['record']['id']}: rescoring moved P(true) by {diff} > determinism_atol {determinism_atol}"
        max_diff = max(max_diff, diff)
    return results, max_diff


def _entity_columns(units: list[dict], results: list[dict]) -> dict[str, np.ndarray]:
    """Flatten entity units and scores to one array entry per span.

    Returns:
        Arrays ``doc_ids``, every ``entity_detection.SPAN_META_KEYS`` column, ``context_doc``,
        ``p_true`` and ``n_tokens`` (input tokens of the row's request).
    """
    cols: dict[str, list] = defaultdict(list)
    for u, r in zip(units, results):
        for qi, sp in enumerate(u["spans"]):
            cols["doc_ids"].append(u["doc_id"])
            for k in edr.SPAN_META_KEYS:
                cols[k].append(sp[k])
            cols["context_doc"].append(u["context_doc"])
            cols["p_true"].append(r["p_true"][qi])
            cols["n_tokens"].append(r["n_tokens"])
    out = {k: np.array(v) for k, v in cols.items()}
    n = len(out["doc_ids"])
    assert all(len(v) == n for v in out.values())
    return out


def _sample_columns(units: list[dict], results: list[dict]) -> dict[str, np.ndarray]:
    """Flatten sample units and scores to one array entry per sample.

    Returns:
        The columns of ``clef._sample_columns`` except ``logit_true`` / ``logit_false``.
    """
    cols: dict[str, list] = defaultdict(list)
    for u, r in zip(units, results):
        assert len(r["p_true"]) == 1
        cols["sample_id"].append(u["sample_id"])
        cols["doc_ids"].append(u["doc_id"])
        for k in clef._CARRIED_KEYS:
            cols[k].append(u[k])
        cols["size"].append(len(u["fields"]))
        cols["fields"].append("|".join(u["fields"]))
        cols["invalid_fields"].append("|".join(u["invalid_fields"]))
        cols["n_invalid_fields"].append(len(u["invalid_fields"]))
        cols["label_valid"].append(u["label_valid"])
        if "primary" in u:
            cols["primary"].append(u["primary"])
            cols["conditions"].append("|".join(u["conditions"]))
            cols["primary_valid"].append(u["primary_valid"])
            cols["invalid_conditions"].append("|".join(u["invalid_conditions"]))
        cols["context_doc"].append(u["context_doc"])
        cols["p_true"].append(r["p_true"][0])
        cols["n_tokens"].append(r["n_tokens"])
    out = {k: np.array(v) for k, v in cols.items()}
    assert all(len(v) == len(units) for v in out.values())
    return out


def efficiency_metrics(results: list[dict]) -> tuple[dict, dict[str, np.ndarray]]:
    """Per-request efficiency arrays and their flat ``eff_*`` summary.

    Args:
        results: ``_score_units`` output.

    Returns:
        ``(metrics, records)``: scalar metrics (request count, total / mean / max request seconds,
        total input and output tokens, total questions) and one array per ``EFFICIENCY_KEYS`` entry.
    """
    recs = {k: np.array([r["efficiency"][k] for r in results]) for k in EFFICIENCY_KEYS}
    sec = recs["seconds_request"]
    m = {
        "eff_n_requests": len(results), "eff_seconds_request_total": float(sec.sum()),
        "eff_seconds_request_mean": float(sec.mean()), "eff_seconds_request_max": float(sec.max()),
        "eff_input_tokens_total": int(recs["input_tokens"].sum()), "eff_output_tokens_total": int(recs["output_tokens"].sum()),
        "eff_n_questions_total": int(recs["n_questions"].sum()),
    }
    return m, recs


def validate_params(p: dict) -> None:
    """Check the config values that ``run`` branches on.

    Raises:
        ValueError: ``context_control`` is not allowed, or ``determinism_check_n`` /
            ``determinism_atol`` is negative.
    """
    if p["context_control"] not in clef.CONTEXT_CONTROLS:
        raise ValueError(f"context_control must be one of {clef.CONTEXT_CONTROLS}, got {p['context_control']!r}")
    if not (isinstance(p["determinism_check_n"], int) and p["determinism_check_n"] >= 0):
        raise ValueError(f"determinism_check_n must be a non-negative int, got {p['determinism_check_n']!r}")
    if not (isinstance(p["determinism_atol"], (int, float)) and p["determinism_atol"] >= 0):
        raise ValueError(f"determinism_atol must be a non-negative number, got {p['determinism_atol']!r}")


def build_units(task: str, p: dict, seed: int, spec: DatasetSpec, repo_root: Path) -> tuple[list[dict], dict]:
    """Load the task's input and build one Jev unit per row or sample (no API call).

    Args:
        task: one of ``TASKS``.
        p: the config ``params``.
        seed: run seed.
        spec: the dataset.
        repo_root: directory that ``p["data_root"]`` (and ``p["samples_path"]``) is relative to.

    Returns:
        ``(units, metrics)``: the units, each with ``context_doc`` set, and ``n_items`` /
        ``n_units``.
    """
    assert task in TASKS, task
    if task == "jev_entity_row":
        item_params = {k: p[k] for k in ("data_root", "split", "error_types", "rows_per_stratum")}
        item_params["context_token_limit"] = None  # no local tokenizer: the API rejects an over-length request
        items = edr.build_items(spec, item_params, seed, repo_root, None)
        contexts = {it["document_id"]: it["context"] for it in items}
    else:
        items = clef.load_samples(repo_root / p["samples_path"], spec, _CLEF_TASK[task])
        contexts = {d: load_context(repo_root / p["data_root"], spec, d) for d in sorted({s["document_id"] for s in items})}

    shown = clef.swap_contexts(contexts, seed) if p["context_control"] == "other_document" else contexts
    context_doc = {d: d for d in contexts}
    if p["context_control"] == "other_document":
        inv = {c: d for d, c in contexts.items()}
        assert len(inv) == len(contexts), "two documents with identical context text"
        context_doc = {d: inv[shown[d]] for d in contexts}

    if task == "jev_entity_row":
        units = entity_units(items, spec, shown)
    elif task == "jev_relation_conditional":
        units = conditional_units(items, spec, shown)
    else:
        units = oneshot_units(items, spec, shown)
    assert len(units) == len(items), "unit count != input count"
    for u in units:
        u["context_doc"] = context_doc[u["doc_id"]]
    return units, {"n_items": len(items), "n_units": len(units)}


def run(cfg: dict, spec: DatasetSpec, run_dir: Path, repo_root: Path, model_cfg: dict) -> dict:
    """Build records, score them with Jev and write ``artifacts/jev_predictions.npz``.

    Args:
        cfg: the config (``task``, ``seed`` and ``params`` for the Jev tasks).
        spec: the dataset named by ``params["dataset_config"]``.
        run_dir: run output directory; ``artifacts/`` is created inside it.
        repo_root: directory that ``params["data_root"]`` is relative to.
        model_cfg: the validated model config (``model_config.load_jev_model_config``).

    Returns:
        Flat scalar metrics: item and unit counts, class counts, max input tokens, mean ``P(true)``
        per class, ``determinism_max_abs_diff``, ``eff_*`` totals and the known-answer
        probabilities if enabled.
    """
    task, p, seed = cfg["task"], cfg["params"], int(cfg["seed"])
    validate_params(p)
    t0 = time.perf_counter()
    units, metrics = build_units(task, p, seed, spec, repo_root)
    assert units, "no records to score"
    metrics["build_units_seconds"] = time.perf_counter() - t0

    scorer = JevScorer(model_cfg["model"], model_cfg["retry"])
    if p["known_answer_check"]:
        metrics.update(scorer.known_answer_check())
    t0 = time.perf_counter()
    results, max_diff = _score_units(scorer, units, p["determinism_check_n"], p["determinism_atol"])
    metrics["score_loop_seconds"] = time.perf_counter() - t0
    metrics["determinism_check_n"] = p["determinism_check_n"]
    metrics["determinism_max_abs_diff"] = max_diff

    cols = _entity_columns(units, results) if task == "jev_entity_row" else _sample_columns(units, results)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / "artifacts" / "jev_predictions.npz", **cols)
    metrics.update(clef.summarize(cols))
    eff_metrics, eff_records = efficiency_metrics(results)
    assert (eff_records["input_tokens"] == np.array([u_r["n_tokens"] for u_r in results])).all()
    np.savez_compressed(run_dir / "artifacts" / "efficiency_records.npz", **eff_records)
    metrics.update(eff_metrics)
    (run_dir / "artifacts" / "request_ids.json").write_text(json.dumps(
        {u["record"]["id"]: r["request_id"] for u, r in zip(units, results)}, indent=2))
    return metrics
