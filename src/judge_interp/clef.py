"""Clef-flash as a direct judge for entity, conditional relation and one-shot record validity; no layer representations.

``Cloudflare/clef-flash`` takes a ``state`` and typed questions and returns one logit per
allowed option per question, so ``P(true)`` of a ``noul`` question is a probability read
straight from the model. The model runs through its own release code
(``joint_schema_model.py``, pinned by ``revision``).

Tasks:
- ``clef_entity_row``: rows selected by ``entity_detection.build_items``, the same rows the
  Qwen/Llama tasks judge (a null field is not shown, as there). One record per row (state =
  context text + the row's non-null fields), one ``noul`` question per non-null field.
- ``clef_relation_conditional`` and ``clef_record_oneshot`` judge *samples* read from a samples
  file (``params.samples_path``, see ``load_samples``), not rows: a sample is a document id, an
  ordered list of entities (field, value, whether the value is valid) and, for the conditional
  task, an explicit split into one primary entity and its conditions. Nothing about where the
  samples came from (which dataset, how they were drawn) is known here.
  - conditional: one record per sample, one ``noul`` question: is the primary entity correctly
    associated, in the document, with all of the conditions, with every value under its correct
    field? Label: true iff the primary and every
    condition is valid (a valid primary with a corrupted condition is false).
  - one-shot: one record per sample, one ``noul`` question: is the whole record valid, i.e. is
    every entity present and of the right type, and are they all related? Label: true iff every
    entity is valid.

Pipeline: ``entity_units`` / ``conditional_units`` / ``oneshot_units`` build records, ``filter_overlong`` handles
records that do not fit ``max_length`` (``overlong_docs``: ``"exclude"`` drops and records
them, ``"error"`` raises), ``ClefScorer`` scores records one at a time, and ``run`` writes
``artifacts/clef_predictions.npz`` and returns flat metrics. ``context_control`` can swap
each document's context text for another document's as a control.

Guards: the question option set must be exactly ``("true", "false")``; ``encode_checked``
raises instead of letting ``encode_record`` silently truncate the state. Questions of one
record share attention in the model, so they are not independent.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from judge_interp import entity_detection as edr
from judge_interp.dataset_spec import DatasetSpec, load_dataset_spec
from judge_interp.prompts import load_context

OPTION_IDS = ("true", "false")  # joint_schema_model.question_options order for a noul question
SAMPLE_TASKS = ("clef_relation_conditional", "clef_record_oneshot")
TASKS = ("clef_entity_row", *SAMPLE_TASKS)
_COMMON_PARAMS = (
    "data_root", "model_config", "dataset_config", "device", "overlong_docs", "context_control",
    "determinism_check_n", "known_answer_check",
)
_ROW_PARAMS = (*_COMMON_PARAMS, "split", "error_types", "rows_per_stratum")
_SAMPLE_PARAMS = (*_COMMON_PARAMS, "samples_path")
TASK_PARAMS: dict[str, tuple[str, ...]] = {
    "clef_entity_row": _ROW_PARAMS, "clef_relation_conditional": _SAMPLE_PARAMS, "clef_record_oneshot": _SAMPLE_PARAMS,
}
OVERLONG_MODES = ("exclude", "error")
CONTEXT_CONTROLS = ("none", "other_document")
_DTYPES = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}

_ENTITY_QUESTION = """In the EXTRACTED RECORD of the state, which is {record_description}, consider the value of the field `{field}`. This field denotes {desc}.

Is this value present in the CONTEXT (the {context_description}) and of the correct type for this field? That is: is it a VALID value for `{field}` in this document? It is valid only if BOTH criteria hold:

{criteria}

Judge only against the CONTEXT. Do not use outside knowledge about the entities involved, and do not guess when the evidence is unclear: unclear evidence fails a criterion. Judge this field on its own, independently of the record's other fields."""

_CONDITIONAL_QUESTION = """The state has a PRIMARY ENTITY and CONDITIONS, which together come from {record_description}. Each is listed as one `field: value` per line.

The primary entity is the field `{primary}`: {primary_desc}.
The conditions are:
{conditions}

The document may describe many items of the kind this record describes. Is the PRIMARY ENTITY correctly associated with the CONDITIONS in the CONTEXT (the {context_description})? That is: does the primary entity belong to the same single item as every one of the conditions? Answer true only if it is related to all of them. Answer false if it is not related to one or more of them, for example because it belongs to a different item than they do. A value that does not appear in the CONTEXT cannot be associated with anything in it.

The field names are part of the association. The primary entity and each condition must be what its field denotes for that item, as described above. A value that belongs to the same item but is listed under the wrong field (it is actually the value of a different field) makes the association fail: answer false.

Judge only against the CONTEXT. Do not use outside knowledge about the entities involved, and do not guess when the evidence is unclear: unclear evidence means false."""

_ONESHOT_QUESTION = """The EXTRACTED RECORD of the state is {record_description}, listed as one `field: value` per line. The fields present are:
{types}

Is the ENTIRE record valid? It is valid only if ALL of the following hold:

{criteria}
(3) Relation. All the values must be related: taken together they describe the same single item in the CONTEXT (the {context_description}). The document may describe many items of the kind this record describes. If one or more values belong to a different item than the others, this criterion fails, even though each of those values may be a genuine value of the document.

Criteria (1) and (2) apply to every value of the record; criterion (3) applies to the record as a whole. If any value fails (1) or (2), or the values are not all related, answer false. Answer true only if the whole record is valid.

Judge only against the CONTEXT. Do not use outside knowledge about the entities involved, and do not guess when the evidence is unclear: unclear evidence fails a criterion."""


# ----------------------------------------------------------------------
# State / question construction (model-free)


def render_record_text(values: dict[str, str | None], fields: list[str]) -> str:
    """Render a record as one ``field: <json>`` line per field.

    Args:
        values: field -> value; every field in ``fields`` must have a ``str`` value.
        fields: the fields to render, in output order.

    Returns:
        The newline-joined record text.
    """
    assert fields and all(isinstance(values[f], str) for f in fields), fields
    return "\n".join(f"{f}: {json.dumps(values[f])}" for f in fields)


def build_state(context: str, record_text: str) -> str:
    """Assemble the Clef ``state`` string.

    Args:
        context: context text of the document.
        record_text: rendered record (see ``render_record_text``).

    Returns:
        The state with a ``## CONTEXT`` and a ``## EXTRACTED RECORD`` section.
    """
    return f"## CONTEXT\n{context}\n\n## EXTRACTED RECORD\n{record_text}"


def build_conditional_state(context: str, primary_text: str, conditions_text: str) -> str:
    """Assemble the Clef ``state`` of a conditional record.

    Args:
        context: context text of the document.
        primary_text: rendered primary entity (see ``render_record_text``).
        conditions_text: rendered conditions, in order.

    Returns:
        The state with ``## CONTEXT``, ``## PRIMARY ENTITY`` and ``## CONDITIONS`` sections.
    """
    return f"## CONTEXT\n{context}\n\n## PRIMARY ENTITY\n{primary_text}\n\n## CONDITIONS\n{conditions_text}"


def entity_question(field: str, spec: DatasetSpec) -> dict:
    """Build the ``noul`` question asking whether one field's value is valid.

    Args:
        field: the field whose value is judged.
        spec: the dataset; supplies the record, context and field descriptions.

    Returns:
        A ``{"type": "noul", "instructions": str}`` question.
    """
    return {
        "type": "noul",
        "instructions": _ENTITY_QUESTION.format(
            record_description=spec.record_description, context_description=spec.context_description,
            field=field, desc=spec.field_descriptions[field], criteria=edr.build_criteria(spec, "as this field is described"),
        ),
    }


def conditional_question(primary: str, conditions: list[str], spec: DatasetSpec) -> dict:
    """Build the ``noul`` question asking whether the primary entity is related to all conditions.

    Args:
        primary: the primary field.
        conditions: the condition fields: at least one, distinct, none equal to ``primary``.
        spec: the dataset; must support the relation tasks.

    Returns:
        A ``{"type": "noul", "instructions": str}`` question.
    """
    spec.require_relation()
    assert conditions and len({primary, *conditions}) == len(conditions) + 1 and {primary, *conditions} <= set(spec.fields), (
        primary, conditions)
    return {"type": "noul", "instructions": _CONDITIONAL_QUESTION.format(
        record_description=spec.record_description, context_description=spec.context_description,
        primary=primary, primary_desc=spec.field_descriptions[primary],
        conditions="\n".join(f"- `{f}`: {spec.field_descriptions[f]}" for f in conditions),
    )}


def oneshot_question(fields: list[str], spec: DatasetSpec) -> dict:
    """Build the ``noul`` question asking whether the whole record is valid.

    Args:
        fields: at least two distinct fields of ``spec`` present in the record.
        spec: the dataset; must support the relation tasks.

    Returns:
        A ``{"type": "noul", "instructions": str}`` question.
    """
    spec.require_relation()
    assert len(set(fields)) == len(fields) >= 2 and set(fields) <= set(spec.fields), fields
    return {"type": "noul", "instructions": _ONESHOT_QUESTION.format(
        record_description=spec.record_description, context_description=spec.context_description,
        types="\n".join(f"- {f}: {spec.field_descriptions[f]}" for f in fields),
        criteria=edr.build_criteria(spec, edr.QWEN_DESCRIBED),
    )}


def item_values(item: dict, spec: DatasetSpec) -> dict[str, str | None]:
    """Recover a row's field values from an item's spans.

    Args:
        item: an item from ``entity_detection.build_items``.
        spec: the dataset.

    Returns:
        Field -> value for every field of ``spec`` (``None`` for a null field).
    """
    vals = {sp["field"]: sp["value"] for sp in item["spans"]}
    assert len(vals) == len(item["spans"]), "two spans for one field"
    assert set(vals) <= set(spec.fields), sorted(vals)
    return {f: vals.get(f) for f in spec.fields}


def entity_units(items: list[dict], spec: DatasetSpec, contexts: dict[str, str]) -> list[dict]:
    """Build one Clef unit per row.

    Args:
        items: items from ``entity_detection.build_items``.
        spec: the dataset.
        contexts: document_id -> the context text shown for that document's rows (its own
            unless the context control swapped it).

    Returns:
        One dict per row with ``doc_id``, ``row_ordinal``, ``spans`` and ``record`` (the
        Clef record: ``id``, ``state``, and one question per non-null field).
    """
    units = []
    for it in items:
        values = item_values(it, spec)
        fields = [sp["field"] for sp in it["spans"]]
        assert all(values[f] is not None for f in fields) and set(fields) == {f for f in spec.fields if values[f] is not None}
        rec_text = render_record_text(values, fields)
        units.append({
            "doc_id": it["document_id"],
            "row_ordinal": it["row_ordinal"],
            "spans": it["spans"],
            "record": {
                "id": f"{it['document_id']}|{it['row_ordinal']}",
                "state": build_state(contexts[it["document_id"]], rec_text),
                "questions": {f: entity_question(f, spec) for f in fields},
            },
        })
    return units


# ----------------------------------------------------------------------
# Samples (conditional and one-shot tasks)

ENTITY_KEYS = {"field", "value", "valid"}
SAMPLE_KEYS = {
    "clef_relation_conditional": {"sample_id", "document_id", "error_type", "primary", "conditions"},
    "clef_record_oneshot": {"sample_id", "document_id", "error_type", "entities"},
}


def _check_entity(e: dict, spec: DatasetSpec, where: str) -> None:
    assert isinstance(e, dict) and set(e) == ENTITY_KEYS, f"{where}: entity keys {sorted(e) if isinstance(e, dict) else e!r}"
    assert e["field"] in spec.fields, f"{where}: field {e['field']!r} is not a field of dataset {spec.name!r}"
    assert isinstance(e["value"], str) and isinstance(e["valid"], bool), f"{where}/{e['field']}: value/valid types"


def load_samples(path: Path, spec: DatasetSpec, task: str) -> list[dict]:
    """Read and validate a samples file.

    The file is a JSON list. Every sample has ``sample_id`` (unique), ``document_id``,
    ``error_type`` (``""`` if none; carried through to the predictions) and entities, each
    ``{"field", "value", "valid"}`` (``valid``: whether the value is correct for its field in the
    document, the ground truth). A conditional sample has ``primary`` (one entity) and
    ``conditions`` (a non-empty list of entities); a one-shot sample has ``entities`` (a list of at
    least two). The order given is the order shown to the judge. A field occurs at most once per
    sample. Nothing here depends on how the samples were drawn.

    Args:
        path: the samples file.
        spec: the dataset; supplies the field set.
        task: ``clef_relation_conditional`` or ``clef_record_oneshot``.

    Returns:
        The samples, unchanged, in file order.

    Raises:
        AssertionError: the file is empty or malformed.
    """
    assert task in SAMPLE_TASKS, task
    samples = json.loads(Path(path).read_text(encoding="utf-8"))
    assert isinstance(samples, list) and samples, f"{path}: expected a non-empty list"
    ids = [s.get("sample_id") if isinstance(s, dict) else None for s in samples]
    assert all(isinstance(i, str) and i for i in ids) and len(set(ids)) == len(ids), f"{path}: sample_id missing or not unique"
    for s in samples:
        where = s["sample_id"]
        assert set(s) == SAMPLE_KEYS[task], f"{where}: keys {sorted(s)} != {sorted(SAMPLE_KEYS[task])}"
        assert isinstance(s["document_id"], str) and s["document_id"] and isinstance(s["error_type"], str), where
        ents = [s["primary"], *s["conditions"]] if task == "clef_relation_conditional" else s["entities"]
        assert isinstance(ents, list) and len(ents) >= 2, f"{where}: needs at least two entities"
        for e in ents:
            _check_entity(e, spec, where)
        fields = [e["field"] for e in ents]
        assert len(set(fields)) == len(fields), f"{where}: a field occurs twice: {fields}"
    return samples


def _sample_unit(s: dict, ents: list[dict], state: str, question: dict) -> dict:
    """The unit fields the two sample tasks share (``fields`` are in the order shown)."""
    bad = [e["field"] for e in ents if not e["valid"]]
    return {
        "sample_id": s["sample_id"],
        "doc_id": s["document_id"],
        "error_type": s["error_type"],
        "fields": [e["field"] for e in ents],
        "invalid_fields": bad,
        "label_valid": not bad,
        "record": {"id": s["sample_id"], "state": state, "questions": {"q": question}},
    }


def conditional_units(samples: list[dict], spec: DatasetSpec, contexts: dict[str, str]) -> list[dict]:
    """Build one Clef unit per conditional sample.

    Args:
        samples: output of ``load_samples`` for ``clef_relation_conditional``.
        spec: the dataset; must support the relation tasks.
        contexts: document_id -> the context text shown for that document (its own unless the
            context control swapped it).

    Returns:
        One dict per sample with ``sample_id``, ``doc_id``, ``error_type``, ``primary``,
        ``conditions``, ``fields`` (primary first, then the conditions in order),
        ``primary_valid``, ``invalid_conditions``, ``invalid_fields``, ``label_valid`` (primary
        and every condition valid) and ``record`` (one ``noul`` question).
    """
    units = []
    for s in samples:
        prim, conds = s["primary"], s["conditions"]
        assert conds, s["sample_id"]
        values = {e["field"]: e["value"] for e in [prim, *conds]}
        state = build_conditional_state(
            contexts[s["document_id"]], render_record_text(values, [prim["field"]]),
            render_record_text(values, [e["field"] for e in conds]))
        u = _sample_unit(s, [prim, *conds], state, conditional_question(prim["field"], [e["field"] for e in conds], spec))
        u.update(primary=prim["field"], conditions=[e["field"] for e in conds], primary_valid=prim["valid"],
                 invalid_conditions=[e["field"] for e in conds if not e["valid"]])
        assert u["label_valid"] == (u["primary_valid"] and not u["invalid_conditions"])
        units.append(u)
    return units


def oneshot_units(samples: list[dict], spec: DatasetSpec, contexts: dict[str, str]) -> list[dict]:
    """Build one Clef unit per one-shot sample.

    Args:
        samples: output of ``load_samples`` for ``clef_record_oneshot``.
        spec: the dataset; must support the relation tasks.
        contexts: document_id -> the context text shown for that document.

    Returns:
        One dict per sample with ``sample_id``, ``doc_id``, ``error_type``, ``fields`` (in the
        order shown), ``invalid_fields``, ``label_valid`` (every entity valid) and ``record``
        (one ``noul`` question over the whole record).
    """
    units = []
    for s in samples:
        ents = s["entities"]
        fields = [e["field"] for e in ents]
        state = build_state(contexts[s["document_id"]], render_record_text({e["field"]: e["value"] for e in ents}, fields))
        units.append(_sample_unit(s, ents, state, oneshot_question(fields, spec)))
    return units


# ----------------------------------------------------------------------
# Context control


def swap_contexts(contexts: dict[str, str], seed: int) -> dict[str, str]:
    """Pair every document with a different document's context text.

    Args:
        contexts: document_id -> context text; needs at least two documents.
        seed: seeds the permutation; the pairing is a cyclic shift, so no document maps to itself.

    Returns:
        document_id -> the context text of its partner document.
    """
    docs = sorted(contexts)
    assert len(docs) >= 2, "the other_document context control needs at least two documents"
    order = np.random.default_rng([seed, 99]).permutation(len(docs))
    nxt = {docs[order[i]]: docs[order[(i + 1) % len(docs)]] for i in range(len(docs))}
    assert all(k != v for k, v in nxt.items())
    return {d: contexts[nxt[d]] for d in docs}


# ----------------------------------------------------------------------
# Model side


def load_release_code(model: str, revision: str, filename: str, sha256: str):
    """Import the release code at a pinned revision (no weights), after checking its hash.

    Args:
        model: HF repo id.
        revision: HF commit hash to download the file from.
        filename: the release-code file in the repo (``joint_schema_model.py``).
        sha256: expected sha256 hex digest of the file.

    Returns:
        The imported module.

    Raises:
        ValueError: the downloaded file's digest differs from ``sha256``.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(model, filename, revision=revision)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError(f"{model}@{revision}/{filename}: sha256 {digest} != expected {sha256}")
    spec = importlib.util.spec_from_file_location("clef_joint_schema_model", path)
    jsm = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = jsm  # its dataclasses (postponed annotations) look themselves up here
    spec.loader.exec_module(jsm)
    return jsm


def count_tokens(jsm, tokenizer, record: dict) -> tuple[int, int]:
    """Count a record's tokens without truncation.

    Args:
        jsm: the release module from ``load_release_code``.
        tokenizer: the model's tokenizer.
        record: a Clef record.

    Returns:
        ``(fixed, total)``: tokens of the record with an empty state, and with its full state.
    """
    empty = jsm.encode_record(tokenizer, {**record, "state": ""}, max_length=10**9)
    state_n = len(tokenizer(jsm.render(record["state"]), add_special_tokens=False).input_ids)
    return len(empty.input_ids), len(empty.input_ids) + state_n


def encode_checked(jsm, tokenizer, record: dict, max_length: int):
    """Encode a record, refusing to truncate it.

    Args:
        jsm: the release module from ``load_release_code``.
        tokenizer: the model's tokenizer.
        record: a Clef record.
        max_length: token budget for the encoded record.

    Returns:
        The encoded record.

    Raises:
        ValueError: the record needs more than ``max_length`` tokens.
        AssertionError: a question's options are not ``OPTION_IDS``.
    """
    _, total = count_tokens(jsm, tokenizer, record)
    if total > max_length:
        raise ValueError(f"record {record['id']!r}: {total} tokens > max_length {max_length} (would be truncated)")
    enc = jsm.encode_record(tokenizer, record, max_length=max_length)
    assert len(enc.input_ids) == total, f"{record['id']!r}: encoded {len(enc.input_ids)} != counted {total}"
    assert [q.question_id for q in enc.questions] == [str(k) for k in record["questions"]]
    for q in enc.questions:
        assert q.option_ids == OPTION_IDS, f"{q.question_id}: options {q.option_ids}"
    return enc


def filter_overlong(units: list[dict], jsm, tokenizer, max_length: int, mode: str) -> tuple[list[dict], list[dict]]:
    """Set ``n_tokens`` on each unit and split off units longer than ``max_length``.

    Args:
        units: units with a ``record``; ``n_tokens`` is added in place.
        jsm: the release module from ``load_release_code``.
        tokenizer: the model's tokenizer.
        max_length: token budget per record.
        mode: ``"exclude"`` or ``"error"``.

    Returns:
        ``(kept, excluded)``.

    Raises:
        ValueError: ``mode == "error"`` and at least one unit is too long.
    """
    assert mode in OVERLONG_MODES, mode
    kept, excluded = [], []
    for u in units:
        _, u["n_tokens"] = count_tokens(jsm, tokenizer, u["record"])
        (excluded if u["n_tokens"] > max_length else kept).append(u)
    if excluded and mode == "error":
        raise ValueError(f"{len(excluded)} record(s) exceed max_length {max_length}, e.g. {excluded[0]['record']['id']!r}")
    assert len(kept) + len(excluded) == len(units)
    return kept, excluded


class ClefScorer:
    """Loads the pinned Clef release and scores one record at a time.

    Args:
        jsm: the release module from ``load_release_code``.
        model: HF repo id.
        revision: HF revision to load.
        device: torch device string; a CUDA device must be available if requested.
        dtype: ``"bfloat16"``, ``"float16"`` or ``"float32"``.
        max_length: token budget per record; longer records raise.
    """

    def __init__(self, jsm, model: str, revision: str, device: str, dtype: str, max_length: int):
        import torch
        from huggingface_hub import snapshot_download

        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {dtype!r}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"device {device!r} requested but CUDA is not available")
        self.jsm, self.torch, self.max_length, self.device = jsm, torch, max_length, device
        path = snapshot_download(model, revision=revision)
        assert Path(path).name == revision, f"snapshot {path} is not commit {revision}"
        self.model, self.processor = jsm.load_release_model(path, device=device, dtype=getattr(torch, _DTYPES[dtype]))
        self.tokenizer = self.processor.tokenizer

    def score(self, record: dict) -> dict:
        """Score one record.

        Args:
            record: a Clef record.

        Returns:
            ``logits`` (``float32 [n_questions, 2]`` in ``OPTION_IDS`` order), ``p_true``
            (``float64 [n_questions]``) and ``n_tokens``.
        """
        torch = self.torch
        enc = encode_checked(self.jsm, self.tokenizer, record, self.max_length)
        batch = self.jsm.collate_records([enc], self.tokenizer.pad_token_id, torch.device(self.device))
        with torch.inference_mode():
            out = self.model(batch)[0]
        assert len(out) == len(enc.questions)
        logits = np.zeros((len(out), len(OPTION_IDS)), dtype=np.float32)
        for qi, (q, ql) in enumerate(zip(enc.questions, out)):
            assert ql.shape == (len(OPTION_IDS),), ql.shape
            logits[qi] = ql.float().cpu().numpy()[[q.option_ids.index(o) for o in OPTION_IDS]]
        assert np.isfinite(logits).all()
        z = logits.astype(np.float64)
        p_true = np.exp(z[:, 0] - np.logaddexp(z[:, 0], z[:, 1]))
        assert ((0 <= p_true) & (p_true <= 1)).all()
        return {"logits": logits, "p_true": p_true, "n_tokens": len(enc.input_ids)}

    def known_answer_check(self) -> dict:
        """Score a hand-built record whose answers are known: a total of 1250 is above 1000, not above 2000.

        Returns:
            ``known_answer_p_yes`` and ``known_answer_p_no`` (``P(true)`` of each question).

        Raises:
            AssertionError: the model does not answer the two questions yes and no.
        """
        rec = {"id": "known-answer", "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD"}},
               "questions": {"above_1000": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
                             "above_2000": {"type": "noul", "instructions": "Is the total above 2000 USD?"}}}
        p = self.score(rec)["p_true"]
        assert p[0] > 0.5 > p[1], f"known-answer case failed: P(true) = {p.tolist()} (expected [>0.5, <0.5])"
        return {"known_answer_p_yes": float(p[0]), "known_answer_p_no": float(p[1])}


# ----------------------------------------------------------------------
# Run


def _score_units(scorer: ClefScorer, units: list[dict], determinism_check_n: int, verbose: bool = True) -> list[dict]:
    """Score every unit, then rescore the first ``determinism_check_n`` and require identical logits.

    Args:
        scorer: a loaded ``ClefScorer``.
        units: units with ``record`` and ``n_tokens``.
        determinism_check_n: number of leading units to rescore.
        verbose: print progress every 50 units.

    Returns:
        ``ClefScorer.score`` output per unit, in order.
    """
    results = []
    for i, u in enumerate(units):
        r = scorer.score(u["record"])
        assert r["n_tokens"] == u["n_tokens"]
        assert len(r["p_true"]) == len(u["record"]["questions"])
        results.append(r)
        if verbose and (i + 1) % 50 == 0:
            print(f"scored {i + 1}/{len(units)}")
    for u, r in zip(units[:determinism_check_n], results):
        again = scorer.score(u["record"])
        assert np.array_equal(again["logits"], r["logits"]), f"{u['record']['id']}: rescoring gave different logits"
    return results


def _entity_columns(units: list[dict], results: list[dict]) -> dict[str, np.ndarray]:
    """Flatten entity units and scores to one array entry per span.

    Args:
        units: output of ``entity_units`` after ``filter_overlong``, with ``context_doc``.
        results: ``_score_units`` output, parallel to ``units``.

    Returns:
        Arrays ``doc_ids``, every ``entity_detection.SPAN_META_KEYS`` column,
        ``context_doc``, ``p_true``, ``logit_true``, ``logit_false`` and ``n_tokens``.
    """
    cols: dict[str, list] = defaultdict(list)
    for u, r in zip(units, results):
        for qi, sp in enumerate(u["spans"]):
            cols["doc_ids"].append(u["doc_id"])
            for k in edr.SPAN_META_KEYS:
                cols[k].append(sp[k])
            cols["context_doc"].append(u["context_doc"])
            cols["p_true"].append(r["p_true"][qi])
            cols["logit_true"].append(r["logits"][qi, 0])
            cols["logit_false"].append(r["logits"][qi, 1])
            cols["n_tokens"].append(r["n_tokens"])
    out = {k: np.array(v) for k, v in cols.items()}
    n = len(out["doc_ids"])
    assert all(len(v) == n for v in out.values())
    return out


def _sample_columns(units: list[dict], results: list[dict]) -> dict[str, np.ndarray]:
    """Flatten sample units and scores to one array entry per sample.

    Args:
        units: output of ``conditional_units`` / ``oneshot_units`` after ``filter_overlong``, with ``context_doc``.
        results: ``_score_units`` output, parallel to ``units``.

    Returns:
        Arrays ``sample_id``, ``doc_ids``, ``error_type``, ``size``, ``fields`` (``|``-joined, in
        the order shown), ``invalid_fields``, ``n_invalid_fields``, ``label_valid``,
        ``context_doc``, ``p_true``, ``logit_true``, ``logit_false`` and ``n_tokens``; for
        conditional units also ``primary``, ``conditions``, ``primary_valid`` and ``invalid_conditions``.
    """
    cols: dict[str, list] = defaultdict(list)
    for u, r in zip(units, results):
        assert len(r["p_true"]) == 1
        cols["sample_id"].append(u["sample_id"])
        cols["doc_ids"].append(u["doc_id"])
        cols["error_type"].append(u["error_type"])
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
        cols["logit_true"].append(r["logits"][0, 0])
        cols["logit_false"].append(r["logits"][0, 1])
        cols["n_tokens"].append(r["n_tokens"])
    out = {k: np.array(v) for k, v in cols.items()}
    assert all(len(v) == len(units) for v in out.values())
    return out


def validate_params(p: dict) -> None:
    """Check the config values that ``run`` branches on.

    Args:
        p: the config ``params``.

    Raises:
        ValueError: ``overlong_docs`` or ``context_control`` is not an allowed value.
    """
    if p["overlong_docs"] not in OVERLONG_MODES:
        raise ValueError(f"overlong_docs must be one of {OVERLONG_MODES}, got {p['overlong_docs']!r}")
    if p["context_control"] not in CONTEXT_CONTROLS:
        raise ValueError(f"context_control must be one of {CONTEXT_CONTROLS}, got {p['context_control']!r}")


def build_units(task: str, p: dict, seed: int, spec: DatasetSpec, repo_root: Path) -> tuple[list[dict], dict]:
    """Load the task's input and build one Clef unit per row or sample (no model needed).

    Args:
        task: one of ``TASKS``.
        p: the config ``params``.
        seed: run seed.
        spec: the dataset.
        repo_root: directory that ``p["data_root"]`` (and ``p["samples_path"]``) is relative to.

    Returns:
        ``(units, metrics)``: the units, each with ``context_doc`` set, and the counts
        (``n_items``, ``n_units_before_filter``) known before scoring.
    """
    assert task in TASKS, task
    if task == "clef_entity_row":
        item_params = {k: p[k] for k in ("data_root", "split", "error_types", "rows_per_stratum")}
        item_params["context_token_limit"] = None  # over-length handling is Clef's own (overlong_docs)
        items = edr.build_items(spec, item_params, seed, repo_root, None)
        contexts = {it["document_id"]: it["context"] for it in items}
    else:
        items = load_samples(repo_root / p["samples_path"], spec, task)
        contexts = {d: load_context(repo_root / p["data_root"], spec, d) for d in sorted({s["document_id"] for s in items})}

    shown = swap_contexts(contexts, seed) if p["context_control"] == "other_document" else contexts
    context_doc = {d: d for d in contexts}
    if p["context_control"] == "other_document":
        inv = {c: d for d, c in contexts.items()}
        assert len(inv) == len(contexts), "two documents with identical context text"
        context_doc = {d: inv[shown[d]] for d in contexts}

    metrics: dict = {"n_items": len(items)}
    if task == "clef_entity_row":
        units = entity_units(items, spec, shown)
    elif task == "clef_relation_conditional":
        units = conditional_units(items, spec, shown)
    else:
        units = oneshot_units(items, spec, shown)
    assert len(units) == len(items), "unit count != input count"
    for u in units:
        u["context_doc"] = context_doc[u["doc_id"]]
    metrics["n_units_before_filter"] = len(units)
    return units, metrics


def summarize(cols: dict[str, np.ndarray]) -> dict:
    """Flat scalar metrics of a prediction table.

    Args:
        cols: output of ``_entity_columns`` or ``_sample_columns``.

    Returns:
        Prediction, class and token counts and mean ``P(true)`` per class (``None`` for an empty class).
    """
    lab = cols["label_valid"].astype(bool)
    return {
        "n_predictions": int(len(lab)), "n_valid": int(lab.sum()), "n_invalid": int((~lab).sum()),
        "max_n_tokens": int(cols["n_tokens"].max()),
        "mean_p_true_valid": float(cols["p_true"][lab].mean()) if lab.any() else None,
        "mean_p_true_invalid": float(cols["p_true"][~lab].mean()) if (~lab).any() else None,
    }


def run(cfg: dict, spec: DatasetSpec, run_dir: Path, repo_root: Path, model_cfg: dict) -> dict:
    """Build records, score them with Clef and write ``artifacts/clef_predictions.npz``.

    Args:
        cfg: the config (``task``, ``seed`` and ``params`` for the Clef tasks).
        spec: the dataset named by ``params["dataset_config"]``.
        run_dir: run output directory; ``artifacts/`` is created inside it.
        repo_root: directory that ``params["data_root"]`` is relative to.
        model_cfg: the validated model config (``model_config.load_clef_model_config``).

    Returns:
        Flat scalar metrics: item, unit and exclusion counts, class counts, max token
        count and mean ``P(true)`` per class, plus the known-answer probabilities if enabled.
    """
    task, p, seed = cfg["task"], cfg["params"], int(cfg["seed"])
    validate_params(p)
    max_length = model_cfg["max_length"]

    units, metrics = build_units(task, p, seed, spec, repo_root)

    rc = model_cfg["release_code"]
    jsm = load_release_code(model_cfg["model"], model_cfg["revision"], rc["file"], rc["sha256"])
    scorer = ClefScorer(jsm, model_cfg["model"], model_cfg["revision"], p["device"], model_cfg["dtype"], max_length)
    units, excluded = filter_overlong(units, jsm, scorer.tokenizer, max_length, p["overlong_docs"])
    metrics["n_excluded_overlong"] = len(excluded)
    metrics["n_excluded_overlong_docs"] = len({u["doc_id"] for u in excluded})
    assert units, "every record was excluded"
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts" / "excluded_overlong.json").write_text(json.dumps(
        [{"record_id": u["record"]["id"], "doc_id": u["doc_id"], "n_tokens": u["n_tokens"]} for u in excluded], indent=2))

    if p["known_answer_check"]:
        metrics.update(scorer.known_answer_check())
    results = _score_units(scorer, units, p["determinism_check_n"])
    metrics["determinism_check_n"] = p["determinism_check_n"]

    cols = _entity_columns(units, results) if task == "clef_entity_row" else _sample_columns(units, results)
    np.savez_compressed(run_dir / "artifacts" / "clef_predictions.npz", **cols)
    metrics.update(summarize(cols))
    return metrics
