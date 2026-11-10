"""Relation pair detection: every unordered pair of a line item's entities, one prompt per row, read per pair.

A row (one line item shown with its m non-null fields, clean or corrupted) becomes one prompt whose record lists
all C(m, 2) pairs of its entities, one pair per line::

    sub_amount: "$100.00", channel: "7 "
    program_desc: "Movie", program_start_date: "01/05/20 "

The judge is asked, pair by pair, whether the two entities are each present and of the right field AND describe the
same item of the document. The hidden state is read at the end of every pair line: the second value carries the pad
space, so its closing quote and the newline merge into ``row_detection.READ_TOKEN`` as in the row tasks.

Order: ``pair_order`` shuffles the pairs and each pair's two fields with a seed derived from the row's ``pair_id``
(shared by a corrupted row and its clean twin), so twins show the same pairs in the same order and orientation, and
the pair lines shown before the first line with a corrupted member have identical prompt prefixes in both.

Labels are not derived from ``field_errors``: a pair is valid iff its two shown values are one real line item's (the
dataset's label rule), read from the size-2 rows of the subsets file (``params.subsets_path``), joined by
``subset_id``. A pair with a corrupted member can be valid (the corrupted value may be the partner's real value on
another item).

The model side is ``RowDetectionLM`` with this task's span columns (``PAIR_META_KEYS``), invariants and metrics.

Variant ``relation_pair_values`` (a targeted diff of the prompt, everything else equal: rows, pair order and
orientation, labels, span columns): the instructions ask only whether the two entities describe the same item
(``_VALUES_HEADER``: no presence / type criteria, no field list), and each line shows the two values without field
names, ``("$100.00", "7 ")``; the hidden state is read at the merged ``' ")\n'`` token (``VALUES_READ_TOKEN``).
``field_a`` / ``field_b`` are still recorded (not shown). The labels are the field-aware ones; the analysis drops the
pairs whose label would differ if field assignment were ignored.
"""
from __future__ import annotations

import itertools
import json
import math
import random
from pathlib import Path

import numpy as np

from judge_interp.dataset_spec import DatasetSpec
from judge_interp.entity_detection import QWEN_DESCRIBED, build_criteria
from judge_interp.prompts import load_context, load_split, shown_fields, sub_seed
from judge_interp.row_detection import (
    _PAD, READ_TOKEN, ROW_PARAMS, RowDetectionLM, _prompt_groups, check_row, row_index, run_rows, select_rows,
)

# Span metadata columns, in shard order. All values are str / int / bool.
PAIR_META_KEYS = (
    "dataset", "row_ordinal", "line_index", "sample_id", "pair_id", "cell", "row_error_type", "row_k",
    "parent_valid", "parent_m", "subset_id", "pair_position", "field_a", "field_b", "value_a", "value_b",
    "corrupted_a", "corrupted_b", "n_corrupted", "label_valid",
)
PAIR_ORDER_SALT = "relation_pair_order"


_HEADER = """You are validating the relationships between entities extracted from a document by an information-extraction pipeline. The entities are the values of one record: {record_description}.

You are given:
1) In ## CONTEXT: the full {context_description}.
2) In ## EXTRACTED RECORD: every pair of the record's extracted entities, one pair per line, in the form `field: value, field: value`, each line ending with a newline. A value is a JSON string in double quotes. The second value of each line is followed by one space before its closing quote; that padding space is not part of the value. The closing quote and the newline after it end the pair.

The fields that may appear are:
{types}

The document may describe many items of the kind this record describes. For each pair, decide whether it is a VALID pair. A pair is valid only if each of its two entities meets criteria (1) and (2), and the pair meets criterion (3):

{criteria}
(3) Relation. The two entities must be explicitly related by the document: taken together, they describe the same single item in the document. A pair whose entities belong to different items is not valid, even though each of them may be a genuine value of the document.

Judge only against the context. Do not use outside knowledge about the entities involved, and do not guess when the evidence is unclear — unclear evidence is a failed criterion. Judge every pair on its own: one pair's validity has no bearing on another's.

For each pair, answer with exactly one line in the form `<field>, <field>: true` or `<field>, <field>: false`, in the order shown."""


_VALUES_HEADER = """You are validating the relationships between entities extracted from a document by an information-extraction pipeline. The entities are the values of one record: {record_description}.

You are given:
1) In ## CONTEXT: the full {context_description}.
2) In ## EXTRACTED RECORD: every pair of the record's extracted entities, one pair per line, in the form `(value, value)`, each line ending with a newline. A value is a JSON string in double quotes. The second value of each line is followed by one space before its closing quote; that padding space is not part of the value. The closing parenthesis and the newline after it end the pair.

The document may describe many items of the kind this record describes. For each pair, decide whether its two entities are RELATED. Two entities are related if they are both used in the document to describe the same single item. A pair whose entities describe different items is not related, even though each of them may appear in the document.

Judge only against the context. Do not use outside knowledge about the entities involved, and do not guess when the evidence is unclear: unclear evidence means not related. Judge every pair on its own: one pair's answer has no bearing on another's.

For each pair, answer with exactly one line, `true` (related) or `false` (not related), in the order shown."""

VALUES_READ_TOKEN = _PAD + '")\n'   # pad, closing quote, closing parenthesis and newline merged


def build_instructions(spec: DatasetSpec) -> str:
    """The instruction block, fixed for a run.

    Raises:
        ValueError: the dataset does not support the relation tasks.
    """
    spec.require_relation()
    types = "\n".join(f"- {f}: {spec.field_descriptions[f]}" for f in spec.fields)
    return _HEADER.format(record_description=spec.record_description, context_description=spec.context_description,
                          types=types, criteria=build_criteria(spec, QWEN_DESCRIBED))


def build_values_instructions(spec: DatasetSpec) -> str:
    """The ``relation_pair_values`` instruction block: relatedness only, no field names.

    Raises:
        ValueError: the dataset does not support the relation tasks.
    """
    spec.require_relation()
    return _VALUES_HEADER.format(record_description=spec.record_description, context_description=spec.context_description)


def pair_order(fields: list[str], pair_id: str, seed: int) -> list[tuple[str, str]]:
    """Every unordered pair of ``fields``, in a random order and each in a random orientation.

    Seeded by ``pair_id`` (not the row), so a corrupted row and its clean twin get the same order.

    Args:
        fields: the row's shown fields (at least two, distinct).
        pair_id: the row's minimal-pair id.
        seed: run seed.

    Returns:
        ``C(len(fields), 2)`` ``(field_a, field_b)`` tuples, ``field_a`` shown first.
    """
    assert len(fields) >= 2 and len(set(fields)) == len(fields), fields
    rng = random.Random(sub_seed(seed, PAIR_ORDER_SALT, pair_id))
    pairs = list(itertools.combinations(fields, 2))
    rng.shuffle(pairs)
    return [(b, a) if rng.random() < 0.5 else (a, b) for a, b in pairs]


def render_pairs(row: dict, order: list[tuple[str, str]]) -> dict:
    """Render a row's pairs as the record text and locate each pair's last closing quote.

    Args:
        row: a dataset row; every field in ``order`` must hold a ``str``.
        order: ``pair_order`` output.

    Returns:
        ``{"text": str, "spans": [(char_start, char_end), ...]}``: one span per pair, in ``order``, covering exactly
        the closing ``"`` of the second value (offsets into ``text``, exclusive end).
    """
    lines, spans, offset = [], [], 0
    for fa, fb in order:
        va, vb = row[fa], row[fb]
        assert isinstance(va, str) and isinstance(vb, str), (fa, va, fb, vb)
        line = f"{fa}: {json.dumps(va)}, {fb}: {json.dumps(vb)[:-1]}{_PAD}\""
        close = offset + len(line) - 1
        spans.append((close, close + 1))
        lines.append(line)
        offset += len(line) + 1
    assert lines, "row with no pair"
    text = "\n".join(lines) + "\n"
    for a, b in spans:
        assert text[a:b] == '"' and text[a - 1] == _PAD and text[b] == "\n", (a, text)
    return {"text": text, "spans": spans}


def render_value_pairs(row: dict, order: list[tuple[str, str]]) -> dict:
    """``render_pairs`` without field names: one ``("va", "vb ")`` line per pair.

    Returns:
        As ``render_pairs``; each span covers exactly the closing ``"`` of the second value, which is followed by
        ``)`` and the newline (together with the pad: ``VALUES_READ_TOKEN``).
    """
    lines, spans, offset = [], [], 0
    for fa, fb in order:
        va, vb = row[fa], row[fb]
        assert isinstance(va, str) and isinstance(vb, str), (fa, va, fb, vb)
        line = f"({json.dumps(va)}, {json.dumps(vb)[:-1]}{_PAD}\")"
        close = offset + len(line) - 2
        spans.append((close, close + 1))
        lines.append(line)
        offset += len(line) + 1
    assert lines, "row with no pair"
    text = "\n".join(lines) + "\n"
    for a, b in spans:
        assert text[a - 1:b + 2] == VALUES_READ_TOKEN, (a, text)
    return {"text": text, "spans": spans}


def subset_id(row: dict, a: str, b: str, spec: DatasetSpec) -> str:
    """The id of the size-2 subset row of ``row`` showing fields ``a`` and ``b`` (fields in ``spec.fields`` order)."""
    return f"{row['sample_id']}|{'+'.join(f for f in spec.fields if f in (a, b))}"


def load_pair_labels(path: Path, spec: DatasetSpec) -> dict[str, dict]:
    """The size-2 rows of a subsets file, by ``subset_id``.

    Raises:
        FileNotFoundError: no such file.
        AssertionError: an empty file, a duplicate id, or a row whose fields are not two of ``spec.fields``.
    """
    if not path.is_file():
        raise FileNotFoundError(f"no such subsets file: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(rows, list) and rows, f"{path}: expected a non-empty list"
    out = {}
    for r in rows:
        if r["size"] != 2:
            continue
        assert len(r["fields"]) == 2 and set(r["fields"]) <= set(spec.fields), (r["subset_id"], r["fields"])
        assert r["subset_id"] not in out, f"{path}: duplicate subset_id {r['subset_id']!r}"
        out[r["subset_id"]] = r
    assert out, f"{path}: no size-2 row"
    return out


def build_items(spec: DatasetSpec, params: dict, seed: int, repo_root: Path, tokenizer=None) -> list[dict]:
    """``relation_pair_detection`` items: ``_build_items`` with ``build_instructions`` and ``render_pairs``."""
    return _build_items(spec, params, seed, repo_root, tokenizer, build_instructions, render_pairs)


def build_values_items(spec: DatasetSpec, params: dict, seed: int, repo_root: Path, tokenizer=None) -> list[dict]:
    """``relation_pair_values`` items: ``_build_items`` with ``build_values_instructions`` and ``render_value_pairs``."""
    return _build_items(spec, params, seed, repo_root, tokenizer, build_values_instructions, render_value_pairs)


def _build_items(spec: DatasetSpec, params: dict, seed: int, repo_root: Path, tokenizer, instructions_fn, render_fn) -> list[dict]:
    """Build one ``RelationPairDetectionLM.collect_rows`` item per selected row.

    Rows are drawn as in ``row_detection.build_items`` (pair by pair, per cell). Each pair's label is its size-2
    subset row's ``valid``; that row must agree with the parent on document, values and corrupted members.

    Args:
        spec: the dataset; must have ``supports_relation`` and ``pair_key``.
        params: config ``params`` (``data_root``, ``split``, ``error_types``, ``rows_per_stratum``,
            ``context_token_limit``, ``subsets_path`` with a ``{split}`` placeholder, relative to ``data_root``).
        seed: run seed: the stratified draw and the pair order.
        repo_root: directory that ``params["data_root"]`` is relative to.
        tokenizer: must be given iff ``params["context_token_limit"]`` is not ``None``.
        instructions_fn: ``spec -> instructions``.
        render_fn: ``(row, pair_order) -> {text, spans}`` (``render_pairs`` / ``render_value_pairs``).

    Returns:
        Items in ``select_rows`` order: ``document_id``, ``row_ordinal``, ``instructions``, ``context``, ``query`` and
        ``spans`` (one per pair, ``char_start``, ``char_end`` and every ``PAIR_META_KEYS`` column).

    Raises:
        ValueError: tokenizer/limit mismatch, or a context longer than ``context_token_limit`` tokens.
        AssertionError: a parent without exactly C(m, 2) size-2 subsets, or a subset row that disagrees with it.
    """
    spec.require_relation()
    assert spec.pair_key is not None, f"{spec.name}: pair detection needs a paired dataset (pair_key)"
    data_root = repo_root / params["data_root"]
    split, limit = params["split"], params["context_token_limit"]
    if (limit is None) != (tokenizer is None):
        raise ValueError("tokenizer must be given iff context_token_limit is not null")
    instructions = instructions_fn(spec)

    rows = load_split(data_root, spec, split, params["error_types"])
    for r in rows:
        check_row(r, spec)
        assert "patch_target" not in r, f"{r['sample_id']}: patched rows are not supported"
    labels = load_pair_labels(data_root / params["subsets_path"].format(split=split), spec)
    n_per_parent: dict[str, int] = {}
    for s in labels.values():
        n_per_parent[s["parent_id"]] = n_per_parent.get(s["parent_id"], 0) + 1
    selected = select_rows(rows, spec, split, params["rows_per_stratum"], seed)

    contexts: dict[str, str] = {}
    ordinal: dict[str, int] = {}
    items: list[dict] = []
    for r in selected:
        doc_id, sid = r["document_id"], r["sample_id"]
        if doc_id not in contexts:
            text = load_context(data_root, spec, doc_id)
            if limit is not None:
                n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
                if n_tokens > limit:
                    raise ValueError(f"context for {doc_id!r} is {n_tokens} tokens > context_token_limit {limit}")
            contexts[doc_id] = text
        j = ordinal.get(doc_id, 0)
        ordinal[doc_id] = j + 1
        shown = shown_fields(r, spec)
        assert n_per_parent.get(sid) == math.comb(len(shown), 2), (
            f"{sid}: {n_per_parent.get(sid)} size-2 subsets, expected C({len(shown)}, 2)")
        order = pair_order(shown, r[spec.pair_key], seed)
        rendered = render_fn(r, order)
        spans = []
        for pos, ((fa, fb), (c0, c1)) in enumerate(zip(order, rendered["spans"]), start=1):
            s = labels[subset_id(r, fa, fb, spec)]
            assert s["parent_id"] == sid and s["document_id"] == doc_id and s["pair_id"] == r[spec.pair_key], s["subset_id"]
            assert s[fa] == r[fa] and s[fb] == r[fb], f"{s['subset_id']}: values differ from the parent's"
            bad_a, bad_b = fa in r["invalid_fields"], fb in r["invalid_fields"]
            assert set(s["invalid_fields"]) == {f for f, bad in ((fa, bad_a), (fb, bad_b)) if bad}, s["subset_id"]
            assert s["valid"] or bad_a or bad_b, f"{s['subset_id']}: invalid pair without a corrupted member"
            spans.append({
                "char_start": c0, "char_end": c1, "dataset": spec.name, "row_ordinal": j,
                "line_index": row_index(r, spec), "sample_id": sid, "pair_id": r[spec.pair_key], "cell": r["cell"],
                "row_error_type": "" if r["valid"] else r["error_type"], "row_k": r["k"], "parent_valid": r["valid"],
                "parent_m": len(shown), "subset_id": s["subset_id"], "pair_position": pos,
                "field_a": fa, "field_b": fb, "value_a": r[fa], "value_b": r[fb],
                "corrupted_a": bad_a, "corrupted_b": bad_b, "n_corrupted": int(bad_a) + int(bad_b),
                "label_valid": bool(s["valid"]),
            })
        items.append({"document_id": doc_id, "row_ordinal": j, "instructions": instructions,
                      "context": contexts[doc_id], "query": rendered["text"], "spans": spans})
    assert len(items) == len(selected), "row count changed while building items"
    return items


# ----------------------------------------------------------------------
# Checks and metrics


def check_pair_invariants(result: dict) -> None:
    """Assert per-prompt pair bookkeeping on the aggregated arrays.

    Within every ``(document, row_ordinal)`` prompt: one ``sample_id``, ``parent_valid``, ``row_error_type`` and
    ``parent_m``; ``C(parent_m, 2)`` pairs at positions 1..C(parent_m, 2) in order; distinct ``subset_id``. Per pair:
    ``n_corrupted = corrupted_a + corrupted_b``; a clean parent's pairs are uncorrupted and valid; an invalid pair has
    a corrupted member; ``row_error_type`` is empty iff the parent is valid.

    Raises:
        AssertionError: any of the conditions above fails.
    """
    uniq, inv = _prompt_groups(result)
    first = np.unique(inv, return_index=True)[1]
    for k in ("sample_id", "parent_valid", "row_error_type", "parent_m", "row_k"):
        assert (result[k] == result[k][first][inv]).all(), f"{k} is not constant within a prompt"
    n_per = np.bincount(inv, minlength=len(uniq))
    m = result["parent_m"][first]
    assert (n_per == m * (m - 1) // 2).all(), "a prompt does not hold C(m, 2) pairs"
    order = np.lexsort((np.arange(len(inv)), inv))
    pos, grp = result["pair_position"][order], inv[order]
    starts = np.r_[True, grp[1:] != grp[:-1]]
    assert (pos[starts] == 1).all() and (pos[1:][~starts[1:]] == pos[:-1][~starts[1:]] + 1).all(), "pair_position is not 1..n"
    assert len(np.unique(result["subset_id"])) == len(inv), "a subset_id occurs twice"
    ca, cb = result["corrupted_a"].astype(bool), result["corrupted_b"].astype(bool)
    assert (result["n_corrupted"] == ca.astype(int) + cb.astype(int)).all(), "n_corrupted != corrupted_a + corrupted_b"
    valid, parent_valid = result["label_valid"].astype(bool), result["parent_valid"].astype(bool)
    assert (valid[parent_valid]).all() and (result["n_corrupted"][parent_valid] == 0).all(), "a clean parent's pair is corrupted or invalid"
    assert (result["n_corrupted"][~valid] > 0).all(), "an invalid pair without a corrupted member"
    assert ((result["row_error_type"] == "") == parent_valid).all(), "row_error_type empty iff the parent is valid"
    assert (result["field_a"] != result["field_b"]).all(), "a pair of one field"


def compute_metrics(result: dict, spec: DatasetSpec) -> dict:
    """Pair, prompt and label counts of a run; asserts every pair was read at ``READ_TOKEN``'s single id.

    Returns:
        Flat scalars: ``n_spans`` (pairs), ``n_prompts``, ``n_valid_spans``, ``n_invalid_spans``, the same split by
        clean / corrupted parent, ``n_spans_<error type>`` (pairs of corrupted parents of that type),
        ``n_valid_spans_with_corrupted_member``, ``n_spans_ncorr<k>`` and the prompt length range.
    """
    read_ids = np.unique(result["token_id"])
    assert len(read_ids) == 1, f"pairs were read at {len(read_ids)} different tokens: {read_ids.tolist()}"
    valid, parent_valid = result["label_valid"].astype(bool), result["parent_valid"].astype(bool)
    out = {
        "n_spans": int(len(valid)), "n_prompts": int(len(_prompt_groups(result)[0])),
        "n_valid_spans": int(valid.sum()), "n_invalid_spans": int((~valid).sum()),
        "n_spans_clean_parent": int(parent_valid.sum()), "n_spans_corrupted_parent": int((~parent_valid).sum()),
        "n_valid_spans_corrupted_parent": int((valid & ~parent_valid).sum()),
        "n_valid_spans_with_corrupted_member": int((valid & (result["n_corrupted"] > 0)).sum()),
        "read_token_id": int(read_ids[0]),
        "min_prompt_n_tokens": int(result["prompt_n_tokens"].min()),
        "max_prompt_n_tokens": int(result["prompt_n_tokens"].max()),
    }
    for et in spec.error_types:
        out[f"n_spans_{et}"] = int((result["row_error_type"] == et).sum())
    assert sum(out[f"n_spans_{et}"] for et in spec.error_types) == out["n_spans_corrupted_parent"]
    for k in range(3):
        out[f"n_spans_ncorr{k}"] = int((result["n_corrupted"] == k).sum())
    assert sum(out[f"n_spans_ncorr{k}"] for k in range(3)) == out["n_spans"]
    return out


class RelationPairDetectionLM(RowDetectionLM):
    """``RowDetectionLM`` with one span per pair (``PAIR_META_KEYS``)."""

    span_meta_keys = PAIR_META_KEYS

    def check_invariants(self, result: dict) -> None:
        check_pair_invariants(result)

    def compute_metrics(self, result: dict, spec: DatasetSpec) -> dict:
        return compute_metrics(result, spec)


class RelationPairValuesLM(RelationPairDetectionLM):
    """``RelationPairDetectionLM`` read at ``VALUES_READ_TOKEN`` (``relation_pair_values``)."""

    read_token = VALUES_READ_TOKEN


assert READ_TOKEN == _PAD + '"\n'   # render_pairs relies on it
TASK_PARAMS = {"relation_pair_detection": (*ROW_PARAMS, "subsets_path"),
               "relation_pair_values": (*ROW_PARAMS, "subsets_path")}
_TASKS = {"relation_pair_detection": (build_items, RelationPairDetectionLM),
          "relation_pair_values": (build_values_items, RelationPairValuesLM)}


def run(cfg: dict, spec: DatasetSpec, run_dir: Path, repo_root: Path, model_cfg: None) -> dict:
    """Run ``cfg["task"]`` (one of ``TASK_PARAMS``): see ``row_detection.run_rows``. ``model_cfg`` must be ``None``."""
    assert model_cfg is None
    build_fn, lm_class = _TASKS[cfg["task"]]
    return run_rows(cfg, spec, run_dir, repo_root, build_fn, lm_class)
