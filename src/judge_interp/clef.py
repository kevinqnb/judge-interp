"""Clef-flash as a direct judge: entity detection and relation detection, no probe.

``Cloudflare/clef-flash`` is a decision model: given a ``state`` and typed questions it
returns one logit per allowed option per question, so ``P(true)`` of a ``noul`` question is
a calibrated probability read straight from the model. Nothing is trained here and no layer
representation is collected.

The model is served through its own release code (``joint_schema_model.py``, pinned by
``revision``), with transformers; the model card documents probability output only for that
path. Facts about that code that the design rests on (read from the pinned revision):

- a ``noul`` question has exactly the options ``("true", "false")``; P(true) is selected by
  option id and the option set is asserted for every question;
- ``encode_record`` **silently truncates the state** to fit ``max_length``; ``encode`` here
  counts the tokens first and raises instead, so a truncated record can never be scored;
- the head's decoder layers self-attend across all questions of one record, so questions of
  one record are *not* independent (matters for the entity task, one record per row);
- records are scored one at a time (batch size 1): no padding is ever involved.

Entity task (``clef_entity_row``): the rows are exactly those of
``entity_detection_row.build_items`` (same stratified draw). One record per row; the state is
the document's OCR text plus the row; one ``noul`` question per non-null field.

Relation task (``clef_relation_row``): rows as ``relation_detection_row.build_items``; from each
row random subsets of its non-null fields are drawn and balanced exactly as
``analysis/relation_detection_row.py`` does (``draw_samples`` then ``balance`` on one
``default_rng([seed, subset_rng_stream])``; stream 1 = the probe's test split, so the subsets
equal the probe's -- asserted by tests). One record per subset (state = OCR text plus only the
subset's fields), one ``noul`` question. A subset is valid iff every field in it is valid.

Documents whose record does not fit ``max_length`` are handled per ``overlong_docs``
(``"exclude"`` drops them and records them; ``"error"`` raises). For the relation task the
exclusion is applied *after* drawing and balancing, so every surviving subset is the same
subset the probe sees.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from judge_interp import entity_detection_row as edr
from judge_interp import relation_detection_row as rdr
from judge_interp.instruction_prompts import _EXPLICIT_CRITERION
from judge_interp.prompts import LINE_FIELDS, QUERY_FIELDS

OPTION_IDS = ("true", "false")  # joint_schema_model.question_options order for a noul question
OVERLONG_MODES = ("exclude", "error")
CONTEXT_CONTROLS = ("none", "other_document")
_DTYPES = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}

_ENTITY_QUESTION = """In the EXTRACTED RECORD of the state, which is {noun}, consider the value of the field `{field}`. This field denotes {desc}.

Is this value present in the CONTEXT (the OCR text of the source document) and of the correct type for this field? That is: is it a VALID value for `{field}` in this document? It is valid only if BOTH criteria hold:

{criteria}

Judge only against the CONTEXT; unclear evidence fails a criterion. Judge this field on its own, independently of the record's other fields."""

# TODO(wording): relation question wording is a first draft, to be reviewed before a full run.
_RELATION_QUESTION = """The EXTRACTED RECORD of the state lists values of one line item (one broadcast spot) of an advertising order, one `field: value` per line. The fields present are:
{types}

Are all of these values correctly related? That is: does every value genuinely serve as the field it is listed under, and do all the values together describe the same single line item (the same broadcast spot) of the order in the CONTEXT (the OCR text of the source document)? Answer false if any value belongs to a different line item or to another part of the order, or is not what its field denotes. A value being present in the CONTEXT is not enough; judge whether the values belong together.

Judge only against the CONTEXT; unclear evidence means false."""


# ----------------------------------------------------------------------
# State / question construction (model-free)


def render_record_text(values: dict[str, str | None], fields: list[str]) -> str:
    """``field: <json>`` per field in ``fields`` order (``null`` for a missing value)."""
    return "\n".join(f"{f}: {json.dumps(values[f])}" for f in fields)


def build_state(context: str, record_text: str) -> str:
    return f"## CONTEXT\n{context}\n\n## EXTRACTED RECORD\n{record_text}"


def entity_question(field: str, dataset: str) -> dict:
    criteria = "\n".join([
        _EXPLICIT_CRITERION.format(n=1),
        edr._TYPE_CRITERION.format(n=2, other_place=edr._OTHER_PLACE[dataset]),
    ])
    return {
        "type": "noul",
        "instructions": _ENTITY_QUESTION.format(
            noun=edr._RECORD_NOUN[dataset], field=field, desc=edr.TYPE_DESCRIPTIONS[field], criteria=criteria
        ),
    }


def relation_question(fields: list[str]) -> dict:
    assert set(fields) <= set(LINE_FIELDS) and len(set(fields)) == len(fields) >= 2, fields
    types = "\n".join(f"- {f}: {rdr.TYPE_DESCRIPTIONS[f]}" for f in fields)
    return {"type": "noul", "instructions": _RELATION_QUESTION.format(types=types)}


def item_values(item: dict, dataset: str) -> dict[str, str | None]:
    """Field -> value of an item's row (``None`` for a null field), canonical fields only."""
    vals = {sp["field"]: sp["value"] for sp in item["spans"]}
    assert len(vals) == len(item["spans"]), "two spans for one field"
    assert set(vals) <= set(QUERY_FIELDS[dataset]), sorted(vals)
    return {f: vals.get(f) for f in QUERY_FIELDS[dataset]}


def entity_units(items: list[dict], dataset: str, contexts: dict[str, str]) -> list[dict]:
    """One unit per row: ``record`` (for Clef), the span metadata and ``context_doc``.

    ``contexts[doc_id]`` is the OCR text shown for that document's rows (the document's own
    unless the context control swapped it).
    """
    units = []
    for it in items:
        values = item_values(it, dataset)
        rec_text = render_record_text(values, QUERY_FIELDS[dataset])
        fields = [sp["field"] for sp in it["spans"]]
        assert all(values[f] is not None for f in fields)
        units.append({
            "doc_id": it["document_id"],
            "row_ordinal": it["row_ordinal"],
            "spans": it["spans"],
            "record": {
                "id": f"{it['document_id']}|{it['row_ordinal']}",
                "state": build_state(contexts[it["document_id"]], rec_text),
                "questions": {f: entity_question(f, dataset) for f in fields},
            },
        })
    return units


# ----------------------------------------------------------------------
# Relation subsets (must equal analysis/relation_detection_row.py's draw_samples / balance)


def flatten_items(items: list[dict]) -> dict[str, np.ndarray]:
    """Flat span arrays in ``RelationRowLM.collect_rows`` order (sorted doc, row_ordinal, position)."""
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


def group_rows(split: dict) -> list[dict]:
    by_key: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, (d, o) in enumerate(zip(split["doc_ids"].tolist(), split["row_ordinal"].tolist())):
        by_key[(d, o)].append(i)
    groups = []
    for (d, o), rows in by_key.items():
        rows = sorted(rows, key=lambda i: int(split["field_position"][i]))
        fields = [str(split["field"][i]) for i in rows]
        assert len(set(fields)) == len(fields) and set(fields) <= set(LINE_FIELDS), (d, o, fields)
        ets = {str(split["row_error_type"][i]) for i in rows}
        assert len(ets) == 1, f"{d}/{o}: row_error_type varies within a row: {ets}"
        groups.append({"doc": d, "ordinal": o, "rows": rows, "fields": fields, "error_type": ets.pop()})
    return groups


def draw_samples(groups: list[dict], label_valid: np.ndarray, samples_per_row: int, min_subset_size: int,
                 rng: np.random.Generator) -> dict:
    """Same algorithm as ``analysis/relation_detection_row.draw_samples`` (tests assert equality)."""
    max_fields = len(LINE_FIELDS)
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
    """Same algorithm as ``analysis/relation_detection_row.balance``."""
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


def draw_relation_samples(flat: dict, p: dict, seed: int) -> dict:
    """Subsets of every row, balanced to ``n_per_class`` valid + invalid (one rng, as the probe does)."""
    rng = np.random.default_rng([seed, p["subset_rng_stream"]])
    s = draw_samples(group_rows(flat), flat["label_valid"], p["samples_per_row"], p["min_subset_size"], rng)
    return balance(s, p["n_per_class"], rng)


def relation_units(flat: dict, samples: dict, contexts: dict[str, str]) -> list[dict]:
    """One unit per sample: ``record`` plus sample metadata (fields in ``field_position`` order)."""
    units = []
    for i in range(len(samples["label"])):
        rows = [int(r) for r in samples["rows"][i] if r >= 0]
        fields = [str(flat["field"][r]) for r in rows]
        assert fields == sorted(fields, key=LINE_FIELDS.index), "subset not in canonical field order"
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
                "questions": {"related": relation_question(fields)},
            },
        })
    return units


# ----------------------------------------------------------------------
# Context control


def swap_contexts(contexts: dict[str, str], seed: int) -> dict[str, str]:
    """Map every document to a *different* document's OCR text (a seeded cyclic shift: no fixed point)."""
    docs = sorted(contexts)
    assert len(docs) >= 2, "the other_document context control needs at least two documents"
    order = np.random.default_rng([seed, 99]).permutation(len(docs))
    nxt = {docs[order[i]]: docs[order[(i + 1) % len(docs)]] for i in range(len(docs))}
    assert all(k != v for k, v in nxt.items())
    return {d: contexts[nxt[d]] for d in docs}


# ----------------------------------------------------------------------
# Model side


def load_release_code(model: str, revision: str):
    """Import the pinned revision's ``joint_schema_model.py`` (code only; no weights)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(model, "joint_schema_model.py", revision=revision)
    spec = importlib.util.spec_from_file_location("clef_joint_schema_model", path)
    jsm = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = jsm  # its dataclasses (postponed annotations) look themselves up here
    spec.loader.exec_module(jsm)
    return jsm


def count_tokens(jsm, tokenizer, record: dict) -> tuple[int, int]:
    """``(fixed, total)``: tokens of the record without its state, and with it (untruncated)."""
    empty = jsm.encode_record(tokenizer, {**record, "state": ""}, max_length=10**9)
    state_n = len(tokenizer(jsm.render(record["state"]), add_special_tokens=False).input_ids)
    return len(empty.input_ids), len(empty.input_ids) + state_n


def encode_checked(jsm, tokenizer, record: dict, max_length: int):
    """``encode_record`` that refuses to truncate; also asserts each question is a 2-option noul."""
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
    """Attach ``n_tokens`` to each unit; return ``(kept, excluded)`` (or raise if ``mode == "error"``)."""
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
    """Loads the pinned release and scores one record at a time."""

    def __init__(self, jsm, model: str, revision: str, device: str, dtype: str, max_length: int):
        import torch
        from huggingface_hub import snapshot_download

        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {dtype!r}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"device {device!r} requested but CUDA is not available")
        self.jsm, self.torch, self.max_length, self.device = jsm, torch, max_length, device
        path = snapshot_download(model, revision=revision)
        self.model, self.processor = jsm.load_release_model(path, device=device, dtype=getattr(torch, _DTYPES[dtype]))
        self.tokenizer = self.processor.tokenizer

    def score(self, record: dict) -> dict:
        """``logits`` ``[n_questions, 2]`` in ``OPTION_IDS`` order, ``p_true`` ``[n_questions]``, ``n_tokens``."""
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
        """Hand-built case with a stated answer: a total of 1250 is above 1000, and not above 2000."""
        rec = {"id": "known-answer", "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD"}},
               "questions": {"above_1000": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
                             "above_2000": {"type": "noul", "instructions": "Is the total above 2000 USD?"}}}
        p = self.score(rec)["p_true"]
        assert p[0] > 0.5 > p[1], f"known-answer case failed: P(true) = {p.tolist()} (expected [>0.5, <0.5])"
        return {"known_answer_p_yes": float(p[0]), "known_answer_p_no": float(p[1])}


# ----------------------------------------------------------------------
# Run


def _score_units(scorer: ClefScorer, units: list[dict], determinism_check_n: int, verbose: bool = True) -> list[dict]:
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


def run(cfg: dict, run_dir: Path, repo_root: Path) -> dict:
    """Build records, score them with Clef, write ``artifacts/clef_predictions.npz``; return flat metrics."""
    task, p, seed = cfg["task"], cfg["params"], int(cfg["seed"])
    assert task in ("clef_entity_row", "clef_relation_row"), task
    if p["overlong_docs"] not in OVERLONG_MODES:
        raise ValueError(f"overlong_docs must be one of {OVERLONG_MODES}, got {p['overlong_docs']!r}")
    if p["context_control"] not in CONTEXT_CONTROLS:
        raise ValueError(f"context_control must be one of {CONTEXT_CONTROLS}, got {p['context_control']!r}")
    max_length = p["max_length"]
    assert isinstance(max_length, int) and max_length > 0

    item_params = {k: p[k] for k in ("data_root", "dataset", "split", "error_types", "rows_per_stratum")}
    item_params["context_token_limit"] = None  # over-length handling is Clef's own (overlong_docs)
    if task == "clef_entity_row":
        items = edr.build_items(item_params, seed, repo_root, None)
    else:
        items = rdr.build_items(item_params, seed, repo_root, None)
    dataset = p["dataset"]

    contexts = {it["document_id"]: it["context"] for it in items}
    shown = swap_contexts(contexts, seed) if p["context_control"] == "other_document" else contexts
    context_doc = {d: d for d in contexts}
    if p["context_control"] == "other_document":
        inv = {c: d for d, c in contexts.items()}
        assert len(inv) == len(contexts), "two documents with identical OCR text"
        context_doc = {d: inv[shown[d]] for d in contexts}

    metrics: dict = {"n_items": len(items)}
    if task == "clef_entity_row":
        units = entity_units(items, dataset, shown)
    else:
        flat = flatten_items(items)
        samples = draw_relation_samples(flat, p, seed)
        for k, v in samples["stats"].items():
            metrics[f"sample_{k}"] = int(v)
        units = relation_units(flat, samples, shown)
    for u in units:
        u["context_doc"] = context_doc[u["doc_id"]]
    metrics["n_units_before_filter"] = len(units)

    jsm = load_release_code(p["model"], p["revision"])
    scorer = ClefScorer(jsm, p["model"], p["revision"], p["device"], p["dtype"], max_length)
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

    lab = cols["label_valid"].astype(bool)
    metrics.update({
        "n_predictions": int(len(lab)), "n_valid": int(lab.sum()), "n_invalid": int((~lab).sum()),
        "max_n_tokens": int(cols["n_tokens"].max()),
        "mean_p_true_valid": float(cols["p_true"][lab].mean()) if lab.any() else None,
        "mean_p_true_invalid": float(cols["p_true"][~lab].mean()) if (~lab).any() else None,
    })
    return metrics
