"""Clef-flash as a direct judge for entity and relation detection; no layer representations.

``Cloudflare/clef-flash`` takes a ``state`` and typed questions and returns one logit per
allowed option per question, so ``P(true)`` of a ``noul`` question is a probability read
straight from the model. The model runs through its own release code
(``joint_schema_model.py``, pinned by ``revision``).

Tasks:
- ``clef_entity_row``: one record per row selected by ``entity_detection.build_items``
  (state = context text + the row), one ``noul`` question per non-null field.
- ``clef_relation_row``: rows selected by ``relation_detection.build_items``; from each row,
  random subsets of its non-null fields are drawn (``draw_samples``) and class-balanced
  (``balance``) with one ``default_rng([seed, subset_rng_stream])``. One record per subset
  (state = context text + only the subset's fields), one ``noul`` question; a subset is valid
  iff every field in it is valid.

Pipeline: ``entity_units`` / ``relation_units`` build records, ``filter_overlong`` handles
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
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from judge_interp import entity_detection as edr
from judge_interp import relation_detection as rdr
from judge_interp.dataset_spec import DatasetSpec, load_dataset_spec

OPTION_IDS = ("true", "false")  # joint_schema_model.question_options order for a noul question
TASKS = ("clef_entity_row", "clef_relation_row")
_COMMON_PARAMS = (
    "data_root", "model_config", "dataset_config", "split", "device", "overlong_docs", "context_control",
    "determinism_check_n", "known_answer_check", "error_types", "rows_per_stratum",
)
TASK_PARAMS: dict[str, tuple[str, ...]] = {
    "clef_entity_row": _COMMON_PARAMS,
    "clef_relation_row": _COMMON_PARAMS + ("samples_per_row", "min_subset_size", "n_per_class", "subset_rng_stream"),
}
OVERLONG_MODES = ("exclude", "error")
CONTEXT_CONTROLS = ("none", "other_document")
_DTYPES = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}

_ENTITY_QUESTION = """In the EXTRACTED RECORD of the state, which is {record_description}, consider the value of the field `{field}`. This field denotes {desc}.

Is this value present in the CONTEXT (the {context_description}) and of the correct type for this field? That is: is it a VALID value for `{field}` in this document? It is valid only if BOTH criteria hold:

{criteria}

Judge only against the CONTEXT; unclear evidence fails a criterion. Judge this field on its own, independently of the record's other fields."""

# TODO(wording): relation question wording is a first draft, to be reviewed before a full run.
_RELATION_QUESTION = """The EXTRACTED RECORD of the state is {record_description}, listed as one `field: value` per line. The fields present are:
{types}

Are all of these values correctly related? That is: does every value genuinely serve as the field it is listed under, and do all the values together describe the same single item in the CONTEXT (the {context_description})? Answer false if any value belongs to a different item or to another part of the document, or is not what its field denotes. A value being present in the CONTEXT is not enough; judge whether the values belong together.

Judge only against the CONTEXT; unclear evidence means false."""


# ----------------------------------------------------------------------
# State / question construction (model-free)


def render_record_text(values: dict[str, str | None], fields: list[str]) -> str:
    """Render a record as one ``field: <json>`` line per field.

    Args:
        values: field -> value (``None`` is rendered as ``null``).
        fields: the fields to render, in output order.

    Returns:
        The newline-joined record text.
    """
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
            field=field, desc=spec.field_descriptions[field], criteria=edr.build_criteria(spec),
        ),
    }


def relation_question(fields: list[str], spec: DatasetSpec) -> dict:
    """Build the ``noul`` question asking whether a subset of line fields belongs together.

    Args:
        fields: at least two distinct fields of ``spec`` present in the record.
        spec: the dataset; must support the relation tasks.

    Returns:
        A ``{"type": "noul", "instructions": str}`` question.
    """
    spec.require_relation()
    assert set(fields) <= set(spec.fields) and len(set(fields)) == len(fields) >= 2, fields
    types = "\n".join(f"- {f}: {spec.field_descriptions[f]}" for f in fields)
    return {"type": "noul", "instructions": _RELATION_QUESTION.format(
        record_description=spec.record_description, context_description=spec.context_description, types=types,
    )}


def item_values(item: dict, spec: DatasetSpec) -> dict[str, str | None]:
    """Recover a row's field values from an item's spans.

    Args:
        item: an item from ``entity_detection.build_items`` or ``relation_detection.build_items``.
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
        rec_text = render_record_text(values, spec.fields)
        fields = [sp["field"] for sp in it["spans"]]
        assert all(values[f] is not None for f in fields)
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
# Relation subsets


def flatten_items(items: list[dict]) -> dict[str, np.ndarray]:
    """Flatten items' spans into parallel arrays.

    Args:
        items: items from ``relation_detection.build_items``, sorted by
            ``(document_id, row_ordinal)``.

    Returns:
        Arrays, one entry per span, in ``(document, row_ordinal, field_position)`` order:
        ``doc_ids``, ``row_ordinal``, ``field``, ``field_position``, ``label_valid``,
        ``row_error_type``, ``field_error_type``, ``value``, ``line_index``, ``row_k``.
    """
    cols: dict[str, list] = defaultdict(list)
    keys = []
    for it in items:
        for sp in it["spans"]:
            keys.append((it["document_id"], it["row_ordinal"], sp["field_position"]))
            cols["doc_ids"].append(it["document_id"])
            cols["row_ordinal"].append(it["row_ordinal"])
            for k in ("field", "field_position", "label_valid", "row_error_type", "field_error_type",
                      "value", "line_index", "row_k"):
                cols[k].append(sp[k])
    assert keys == sorted(keys), "items are not in (document, row_ordinal, field_position) order"
    return {k: np.array(v) for k, v in cols.items()}


def group_rows(split: dict, spec: DatasetSpec) -> list[dict]:
    """Group span indices by row.

    Args:
        split: flat arrays from ``flatten_items``.
        spec: the dataset.

    Returns:
        One dict per ``(document, row_ordinal)`` with ``doc``, ``ordinal``, ``rows`` (span
        indices in ``field_position`` order), ``fields`` and ``error_type``.
    """
    by_key: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, (d, o) in enumerate(zip(split["doc_ids"].tolist(), split["row_ordinal"].tolist())):
        by_key[(d, o)].append(i)
    groups = []
    for (d, o), rows in by_key.items():
        rows = sorted(rows, key=lambda i: int(split["field_position"][i]))
        fields = [str(split["field"][i]) for i in rows]
        assert len(set(fields)) == len(fields) and set(fields) <= set(spec.fields), (d, o, fields)
        ets = {str(split["row_error_type"][i]) for i in rows}
        assert len(ets) == 1, f"{d}/{o}: row_error_type varies within a row: {ets}"
        groups.append({"doc": d, "ordinal": o, "rows": rows, "fields": fields, "error_type": ets.pop()})
    return groups


def draw_samples(groups: list[dict], label_valid: np.ndarray, samples_per_row: int, min_subset_size: int,
                 rng: np.random.Generator, max_fields: int) -> dict:
    """Draw random subsets of each row's non-null fields.

    Args:
        groups: output of ``group_rows``.
        label_valid: per-span validity, indexed by the span indices in ``groups``.
        samples_per_row: subsets drawn per row (fewer if the row has fewer subsets).
        min_subset_size: smallest subset size, at least 2.
        rng: generator used for all draws.
        max_fields: number of fields of the dataset; ``rows`` is padded to this width.

    Returns:
        Arrays, one entry per sample: ``rows`` (span indices padded with -1 to ``max_fields``), ``label`` (all fields valid), ``size``, ``doc``, ``ordinal``,
        ``error_type``; plus ``stats`` (counts of rows skipped or short of subsets).
    """
    assert min_subset_size >= 2, "a subset needs at least two fields"
    out: dict[str, list] = defaultdict(list)
    stats = {"n_rows": len(groups), "n_rows_too_small": 0, "n_rows_fewer_subsets_than_requested": 0}
    for g in groups:
        n = len(g["rows"])
        if n < min_subset_size:
            stats["n_rows_too_small"] += 1
            continue
        subsets = [c for r in range(min_subset_size, n + 1) for c in itertools.combinations(range(n), r)]
        take = min(samples_per_row, len(subsets))
        if take < samples_per_row:
            stats["n_rows_fewer_subsets_than_requested"] += 1
        for j in rng.choice(len(subsets), size=take, replace=False):
            rows = [g["rows"][k] for k in subsets[int(j)]]
            out["rows"].append(rows + [-1] * (max_fields - len(rows)))
            out["label"].append(bool(label_valid[rows].all()))
            out["size"].append(len(rows))
            out["doc"].append(g["doc"])
            out["ordinal"].append(g["ordinal"])
            out["error_type"].append(g["error_type"])
    assert out["rows"], "no samples drawn"
    res = {k: np.array(v) for k, v in out.items()}
    res["stats"] = stats
    assert res["rows"].shape[1] == max_fields and (res["rows"] >= 0).sum(axis=1).tolist() == res["size"].tolist()
    assert not ((res["error_type"] == "") & ~res["label"]).any(), "invalid sample from a clean row"
    return res


def balance(samples: dict, n_per_class: int, rng: np.random.Generator) -> dict:
    """Subsample to ``n_per_class`` valid and ``n_per_class`` invalid samples.

    Args:
        samples: output of ``draw_samples``.
        n_per_class: samples kept per class.
        rng: generator used for the draws.

    Returns:
        ``samples`` restricted to the kept samples (original order), with ``stats`` kept.

    Raises:
        ValueError: a class has fewer than ``n_per_class`` samples.
    """
    assert isinstance(n_per_class, int) and n_per_class >= 1
    label = samples["label"]
    keep = []
    for lab, name in ((True, "valid"), (False, "invalid")):
        pool = np.flatnonzero(label == lab)
        if len(pool) < n_per_class:
            raise ValueError(f"only {len(pool)} {name} samples available, {n_per_class} requested")
        keep.append(rng.choice(pool, size=n_per_class, replace=False))
    keep = np.sort(np.concatenate(keep))
    res = {k: v[keep] for k, v in samples.items() if k != "stats"}
    res["stats"] = samples["stats"]
    assert int(res["label"].sum()) == n_per_class and int((~res["label"]).sum()) == n_per_class
    return res


def draw_relation_samples(flat: dict, p: dict, seed: int, spec: DatasetSpec) -> dict:
    """Draw and class-balance relation samples with one generator.

    Args:
        flat: output of ``flatten_items``.
        p: config params with ``subset_rng_stream``, ``samples_per_row``, ``min_subset_size``
            and ``n_per_class``.
        seed: run seed; the generator is ``default_rng([seed, subset_rng_stream])``.
        spec: the dataset.

    Returns:
        The balanced samples (see ``draw_samples``).
    """
    rng = np.random.default_rng([seed, p["subset_rng_stream"]])
    s = draw_samples(group_rows(flat, spec), flat["label_valid"], p["samples_per_row"], p["min_subset_size"], rng,
                     len(spec.fields))
    return balance(s, p["n_per_class"], rng)


def relation_units(flat: dict, samples: dict, contexts: dict[str, str], spec: DatasetSpec) -> list[dict]:
    """Build one Clef unit per relation sample.

    Args:
        flat: output of ``flatten_items``.
        samples: balanced samples from ``draw_relation_samples``.
        contexts: document_id -> the context text shown for that document.
        spec: the dataset.

    Returns:
        One dict per sample with ``doc_id``, ``row_ordinal``, ``sample_index``, ``rows``,
        ``fields``, ``invalid_fields``, ``label_valid``, ``row_error_type``, ``line_index``
        and ``record`` (one ``related`` question over the subset's fields).
    """
    units = []
    for i in range(len(samples["label"])):
        rows = [int(r) for r in samples["rows"][i] if r >= 0]
        fields = [str(flat["field"][r]) for r in rows]
        assert fields == sorted(fields, key=spec.fields.index), "subset not in canonical field order"
        doc = str(samples["doc"][i])
        assert all(flat["doc_ids"][r] == doc for r in rows)
        values = {str(flat["field"][r]): str(flat["value"][r]) for r in rows}
        bad = [str(flat["field"][r]) for r in rows if not flat["label_valid"][r]]
        assert (not bad) == bool(samples["label"][i]), "subset label != all-valid"
        units.append({
            "doc_id": doc,
            "row_ordinal": int(samples["ordinal"][i]),
            "sample_index": i,
            "rows": samples["rows"][i],
            "fields": fields,
            "invalid_fields": bad,
            "label_valid": bool(samples["label"][i]),
            "row_error_type": str(samples["error_type"][i]),
            "line_index": int(flat["line_index"][rows[0]]),
            "record": {
                "id": f"{doc}|{int(samples['ordinal'][i])}|{i}",
                "state": build_state(contexts[doc], render_record_text(values, fields)),
                "questions": {"related": relation_question(fields, spec)},
            },
        })
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


def _relation_columns(units: list[dict], results: list[dict]) -> dict[str, np.ndarray]:
    """Flatten relation units and scores to one array entry per sample.

    Args:
        units: output of ``relation_units`` after ``filter_overlong``, with ``context_doc``.
        results: ``_score_units`` output, parallel to ``units``.

    Returns:
        Arrays ``doc_ids``, ``row_ordinal``, ``sample_index``, ``line_index``, ``rows``,
        ``size``, ``fields``, ``invalid_fields``, ``n_invalid_fields``, ``label_valid``,
        ``row_error_type``, ``context_doc``, ``p_true``, ``logit_true``, ``logit_false``
        and ``n_tokens``.
    """
    cols: dict[str, list] = defaultdict(list)
    for u, r in zip(units, results):
        cols["doc_ids"].append(u["doc_id"])
        cols["row_ordinal"].append(u["row_ordinal"])
        cols["sample_index"].append(u["sample_index"])
        cols["line_index"].append(u["line_index"])
        cols["rows"].append(u["rows"])
        cols["size"].append(len(u["fields"]))
        cols["fields"].append("|".join(u["fields"]))
        cols["invalid_fields"].append("|".join(u["invalid_fields"]))
        cols["n_invalid_fields"].append(len(u["invalid_fields"]))
        cols["label_valid"].append(u["label_valid"])
        cols["row_error_type"].append(u["row_error_type"])
        cols["context_doc"].append(u["context_doc"])
        cols["p_true"].append(r["p_true"][0])
        cols["logit_true"].append(r["logits"][0, 0])
        cols["logit_false"].append(r["logits"][0, 1])
        cols["n_tokens"].append(r["n_tokens"])
    return {k: np.array(v) for k, v in cols.items()}


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
    """Select rows and build one Clef unit per entity row or relation sample (no model needed).

    Args:
        task: ``clef_entity_row`` or ``clef_relation_row``.
        p: the config ``params``.
        seed: run seed.
        spec: the dataset.
        repo_root: directory that ``p["data_root"]`` is relative to.

    Returns:
        ``(units, metrics)``: the units, each with ``context_doc`` set, and the counts
        (``n_items``, ``sample_*``, ``n_units_before_filter``) known before scoring.
    """
    assert task in TASKS, task
    item_params = {k: p[k] for k in ("data_root", "split", "error_types", "rows_per_stratum")}
    item_params["context_token_limit"] = None  # over-length handling is Clef's own (overlong_docs)
    if task == "clef_entity_row":
        items = edr.build_items(spec, item_params, seed, repo_root, None)
    else:
        items = rdr.build_items(spec, item_params, seed, repo_root, None)

    contexts = {it["document_id"]: it["context"] for it in items}
    shown = swap_contexts(contexts, seed) if p["context_control"] == "other_document" else contexts
    context_doc = {d: d for d in contexts}
    if p["context_control"] == "other_document":
        inv = {c: d for d, c in contexts.items()}
        assert len(inv) == len(contexts), "two documents with identical context text"
        context_doc = {d: inv[shown[d]] for d in contexts}

    metrics: dict = {"n_items": len(items)}
    if task == "clef_entity_row":
        units = entity_units(items, spec, shown)
    else:
        flat = flatten_items(items)
        samples = draw_relation_samples(flat, p, seed, spec)
        for k, v in samples["stats"].items():
            metrics[f"sample_{k}"] = int(v)
        units = relation_units(flat, samples, shown, spec)
    for u in units:
        u["context_doc"] = context_doc[u["doc_id"]]
    metrics["n_units_before_filter"] = len(units)
    return units, metrics


def summarize(cols: dict[str, np.ndarray]) -> dict:
    """Flat scalar metrics of a prediction table.

    Args:
        cols: output of ``_entity_columns`` or ``_relation_columns``.

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

    cols = _entity_columns(units, results) if task == "clef_entity_row" else _relation_columns(units, results)
    np.savez_compressed(run_dir / "artifacts" / "clef_predictions.npz", **cols)
    metrics.update(summarize(cols))
    return metrics
