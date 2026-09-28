"""Adapter: run one representation-collection experiment per the harness contract.

Usage
-----
    uv run --python 3.12 python scripts/run_experiment.py configs/<id>.yaml

Reads the standardized envelope (``id``, ``project``, ``description``, ``seed``,
``params``), builds one judge prompt per dataset row, runs
``judge_interp.RepresentationLM`` over them, and writes the contract's output to
``$RUNS_ROOT/<id>/``:

    run.json                manifest
    config.snapshot.yaml    verbatim copy of the config
    metrics.json            flat scalar map (judge accuracy, mean P(true), ...)
    log.txt                 stdout/stderr (written by the caller / submit.sh)
    artifacts/representations.npz   per-layer [n, hidden] arrays + provenance
    artifacts/rows.jsonl            one JSON object per row (scalars only)
    artifacts/cache/<document_id>.npz   per-document shards (resume support)

Every ``params`` key is required — a missing key is a hard error, not a default.

``params``
----------
    data_root          repo-relative dir holding ``vrdu/{main,line,ocr}/``
    model              HF instruct-model repo id
    dataset            "main" | "line"
    split              "train" | "test"
    layers             list of ints and/or "last" (see prompts.resolve_layers)
    dtype              "float32" | "float16" | "bfloat16"
    device             device_map string ("cuda", "cuda:0", "cpu")
    include_cohesion   bool — include the cohesion criterion in the instructions
    context_token_limit null | int — hard error if any OCR context's raw text
                       tokenizes (via params["model"]'s tokenizer) to more than
                       this many tokens. Checks the raw OCR context alone, not
                       the full instructions+query+chat-template prompt built
                       at judge time — see the configs for the margin this
                       needs to leave under max_position_embeddings.
    row_subset         null (all rows) | {n: int, seed: int} (deterministic sample)
    verify_read_point  bool — run RepresentationLM.verify_read_point on the first
                       item before judging anything (determinism / read-point
                       gate; see its docstring). Leave true unless this exact
                       (model, device, dtype) has already been checked.

Task ``entity_detection_mixed`` (judge_interp/entity_detection.py) takes no
``dataset``/``row_subset``; instead:
    layers             must be [0, "last"]
    doc_subset         null (all documents) | {n: int, seed: int}
    samples_per_doc    s -- prompts drawn per document
    k_min              smallest k (decoys per prompt) that can be drawn; k is
                       uniform on [k_min, m]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from judge_interp import entity_prompts, relation_prompts
from judge_interp.entity_instructions import build_entity_detection_instructions
from judge_interp.instruction_prompts import build_instructions
from judge_interp.prompts import LINE_FIELDS, load_ocr_context, load_split, render_query
from judge_interp.relation_instructions import build_relation_detection_instructions

_TASK_PARAMS: dict[str, tuple[str, ...]] = {
    "record_judge": (
        "data_root", "model", "dataset", "split", "layers", "dtype", "device",
        "include_cohesion", "context_token_limit", "row_subset", "verify_read_point",
    ),
    "entity_detection": (
        "data_root", "model", "dataset", "split", "layers", "dtype", "device",
        "context_token_limit", "row_subset", "verify_read_point",
        "min_valid_per_list", "min_invalid_per_list", "max_invalid_per_list",
    ),
    "relation_detection": (
        "data_root", "model", "split", "layers", "dtype", "device",
        "context_token_limit", "row_subset", "verify_read_point", "min_valid_per_type",
    ),
    # All entity types (main + line) in one substituted-decoy list per
    # (document, sample) -- see judge_interp/entity_detection.py. layers must
    # be [0, "last"].
    "entity_detection_mixed": (
        "data_root", "model", "split", "layers", "dtype", "device",
        "context_token_limit", "doc_subset", "verify_read_point",
        "samples_per_doc", "k_min",
    ),
}
# Allowed for any task -- read only by scripts/submit.sh, ignored here.
_SUBMIT_ONLY_PARAMS = frozenset({"gpu_type", "gpu_memory", "gpu_c", "omp", "walltime"})


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def _require(d: dict, key: str, where: str):
    if key not in d:
        raise KeyError(f"{where}: required key {key!r} is missing")
    return d[key]


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("id") != path.stem:
        raise ValueError(f"config id {cfg.get('id')!r} != filename stem {path.stem!r}")
    if cfg.get("project") != "judge-interp":
        raise ValueError(f"config project must be 'judge-interp', got {cfg.get('project')!r}")
    for key in ("task", "description", "seed", "params"):
        _require(cfg, key, "config")

    task = cfg["task"]
    if task not in _TASK_PARAMS:
        raise ValueError(f"task must be one of {sorted(_TASK_PARAMS)}, got {task!r}")

    params = cfg["params"]
    required = set(_TASK_PARAMS[task])
    missing = required - set(params)
    if missing:
        raise KeyError(f"params: required key(s) missing for task {task!r}: {sorted(missing)}")
    extra = set(params) - required - _SUBMIT_ONLY_PARAMS
    if extra:
        raise KeyError(f"params: key(s) not valid for task {task!r}: {sorted(extra)}")

    if task in ("record_judge", "entity_detection") and params["dataset"] not in ("main", "line"):
        raise ValueError(f"params.dataset must be 'main' or 'line', got {params['dataset']!r}")
    return cfg


def build_items(params: dict) -> list[dict]:
    """One judge item per row: identity + label metadata + context + query."""
    data_root = REPO_ROOT / params["data_root"]
    dataset, split = params["dataset"], params["split"]
    rows = load_split(data_root, dataset, split)

    limit = params["context_token_limit"]
    tokenizer = AutoTokenizer.from_pretrained(params["model"]) if limit is not None else None
    contexts: dict[str, str] = {}
    for doc_id in {r["document_id"] for r in rows}:
        text = load_ocr_context(data_root, doc_id)
        if limit is not None:
            n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            if n_tokens > limit:
                raise ValueError(
                    f"OCR context for {doc_id!r} is {n_tokens} tokens > "
                    f"context_token_limit {limit}"
                )
        contexts[doc_id] = text

    items = []
    for r in rows:
        items.append({
            "document_id": r["document_id"],
            "line_index": r["line_index"] if dataset == "line" else None,
            "error_type": r["error_type"] if dataset == "line" else None,
            "label": bool(r["valid"]),
            "k": r["k"],
            "num_invalid_fields": r["num_invalid_fields"],
            "invalid_fields": list(r["invalid_fields"]),
            "context": contexts[r["document_id"]],
            "query": render_query(r, dataset),
        })

    subset = params["row_subset"]
    if subset is not None:
        n, seed = _require(subset, "n", "row_subset"), _require(subset, "seed", "row_subset")
        if n > len(items):
            raise ValueError(f"row_subset.n {n} > available rows {len(items)}")
        rng = random.Random(seed)
        items = sorted(rng.sample(items, n), key=lambda it: (it["document_id"], it["line_index"] if it["line_index"] is not None else -1, it["k"]))
    return items


def _flatten_entity_spans(rendered: dict, field: str, kinds: tuple[str, ...]) -> list[dict]:
    """Flatten ``entity_prompts.render_entity_list``'s per-entity span dicts
    into ``representationlm``-ready flat span dicts, one per (entity, kind)."""
    spans = []
    for span in rendered["spans"]:
        for kind in kinds:
            spans.append({
                "char_start": span[f"{kind}_start"],
                "char_end": span[f"{kind}_end"],
                "kind": kind,
                "entity_type": field,
                "value": span["value"],
                "label_valid": span["label_valid"],
                "list_position": span["index"],
            })
    return spans


def build_entity_items(params: dict, seed: int) -> list[dict]:
    """One judge item per surviving ``(document, field)`` entity list.

    Each item carries its own ``instructions`` (entity-detection instructions
    name the specific field being judged, so they vary across items in the
    same run -- unlike ``record_judge``'s single fixed instruction block).
    """
    data_root = REPO_ROOT / params["data_root"]
    dataset, split = params["dataset"], params["split"]
    result = entity_prompts.build_entity_lists(
        data_root, dataset, split, seed=seed,
        min_valid_per_list=params["min_valid_per_list"],
        min_invalid_per_list=params["min_invalid_per_list"],
        max_invalid_per_list=params["max_invalid_per_list"],
    )

    limit = params["context_token_limit"]
    tokenizer = AutoTokenizer.from_pretrained(params["model"]) if limit is not None else None
    contexts: dict[str, str] = {}

    items = []
    for (doc_id, field), entities in sorted(result["lists"].items()):
        if doc_id not in contexts:
            text = load_ocr_context(data_root, doc_id)
            if limit is not None:
                n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
                if n_tokens > limit:
                    raise ValueError(
                        f"OCR context for {doc_id!r} is {n_tokens} tokens > "
                        f"context_token_limit {limit}"
                    )
            contexts[doc_id] = text

        rendered = entity_prompts.render_entity_list(entities)
        items.append({
            "document_id": doc_id,
            "field": field,
            "instructions": build_entity_detection_instructions(field),
            "context": contexts[doc_id],
            "query": rendered["text"],
            "spans": _flatten_entity_spans(rendered, field, ("content", "cue")),
        })

    subset = params["row_subset"]
    if subset is not None:
        n, subset_seed = _require(subset, "n", "row_subset"), _require(subset, "seed", "row_subset")
        if n > len(items):
            raise ValueError(f"row_subset.n {n} > available items {len(items)}")
        rng = random.Random(subset_seed)
        items = sorted(rng.sample(items, n), key=lambda it: (it["document_id"], it["field"]))
    return items


def build_relation_items(params: dict, seed: int) -> tuple[list[dict], dict]:
    """One judge item per document with >= 1 renderable entity-type list.

    Returns ``(items, tuple_membership)`` -- ``tuple_membership`` is the
    ground-truth valid/invalid tuple bookkeeping (never rendered into a
    prompt), restricted to documents that made it into ``items`` (after the
    ``row_subset`` sample), written verbatim to
    ``artifacts/tuple_membership.json``.
    """
    data_root = REPO_ROOT / params["data_root"]
    split = params["split"]
    result = relation_prompts.build_relation_lists(
        data_root, split, seed=seed, min_valid_per_type=params["min_valid_per_type"]
    )

    limit = params["context_token_limit"]
    tokenizer = AutoTokenizer.from_pretrained(params["model"]) if limit is not None else None

    by_doc_lists: dict[str, dict[str, list]] = {}
    for (doc_id, field), entities in result["lists"].items():
        by_doc_lists.setdefault(doc_id, {})[field] = entities

    instructions = build_relation_detection_instructions(LINE_FIELDS)

    items = []
    for doc_id in sorted(by_doc_lists):
        text = load_ocr_context(data_root, doc_id)
        if limit is not None:
            n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            if n_tokens > limit:
                raise ValueError(
                    f"OCR context for {doc_id!r} is {n_tokens} tokens > "
                    f"context_token_limit {limit}"
                )

        rendered = relation_prompts.render_relation_prompt(by_doc_lists[doc_id])
        spans = []
        for field, field_spans in rendered["spans"].items():
            rendered_field = {"spans": field_spans}
            spans.extend(_flatten_entity_spans(rendered_field, field, ("content",)))
        items.append({
            "document_id": doc_id,
            "instructions": instructions,
            "context": text,
            "query": rendered["text"],
            "spans": spans,
        })

    subset = params["row_subset"]
    if subset is not None:
        n, subset_seed = _require(subset, "n", "row_subset"), _require(subset, "seed", "row_subset")
        if n > len(items):
            raise ValueError(f"row_subset.n {n} > available items {len(items)}")
        rng = random.Random(subset_seed)
        items = sorted(rng.sample(items, n), key=lambda it: it["document_id"])

    kept_docs = {it["document_id"] for it in items}
    tuple_membership = {d: t for d, t in result["tuples"].items() if d in kept_docs}
    return items, tuple_membership


def compute_metrics(result: dict) -> dict:
    """Flat scalar metrics. Judge accuracy is measured only over rows whose
    argmax verdict was a recognised token ('true'/'false').

    A mask with no rows (e.g. no invalid rows in a subset) yields ``None``
    rather than NaN for that metric — ``json.dumps`` writes bare ``NaN``, which
    is not valid JSON, and ``metrics.json`` must stay parseable by every reader.
    """
    labels = result["labels"].astype(bool)
    verdict_true = result["verdict_true"].astype(bool)
    recognised = result["verdict_recognised"].astype(bool)
    p_true = result["p_true"].astype(np.float64)
    n = len(labels)

    def _acc(mask: np.ndarray) -> float | None:
        m = mask & recognised
        if not m.any():
            return None
        return float((verdict_true[m] == labels[m]).mean())

    def _mean_p(mask: np.ndarray) -> float | None:
        return float(p_true[mask].mean()) if mask.any() else None

    return {
        "n_rows": int(n),
        "n_valid": int(labels.sum()),
        "n_invalid": int((~labels).sum()),
        "verdict_recognised_rate": float(recognised.mean()),
        "judge_accuracy": _acc(np.ones(n, bool)),
        "judge_acc_valid": _acc(labels),
        "judge_acc_invalid": _acc(~labels),
        "mean_p_true_valid": _mean_p(labels),
        "mean_p_true_invalid": _mean_p(~labels),
    }


def compute_span_metrics(result: dict) -> dict:
    """Flat scalar metrics over span rows (entity_detection / relation_detection).

    Verdict accuracy is only meaningful at ``"cue"``-kind spans -- a
    ``"content"`` span's logits are not a verdict distribution (see
    ``RepresentationLM.judge_spans``). ``relation_detection`` items carry no
    cue spans at all, so their accuracy fields are ``None``, not 0 -- there is
    no verdict to be right or wrong about.
    """
    n = len(result["doc_ids"])
    kind = result["kind"]
    cue_mask = kind == "cue"

    out = {
        "n_spans": int(n),
        "n_cue_spans": int(cue_mask.sum()),
        "exact_end_alignment_rate": float(result["exact_end_alignment"].mean()),
        "verdict_recognised_rate": None,
        "judge_accuracy": None,
    }
    if cue_mask.any():
        labels = result["label_valid"].astype(bool)
        verdict_true = result["verdict_true"].astype(bool)
        recognised = result["verdict_recognised"].astype(bool)
        out["verdict_recognised_rate"] = float(recognised[cue_mask].mean())
        m = cue_mask & recognised
        out["judge_accuracy"] = float((verdict_true[m] == labels[m]).mean()) if m.any() else None
    return out


def _write_span_representations(result: dict, path: Path) -> None:
    payload = {f"rep_{L}": result["representations"][L] for L in result["layers"].tolist()}
    for k, v in result.items():
        if k == "representations":
            continue
        payload[k] = v
    np.savez_compressed(path, **payload)


def write_rows_jsonl(result: dict, path: Path) -> None:
    scalar_keys = [
        "doc_ids", "line_indices", "error_type", "k", "num_invalid_fields", "labels",
        "p_true", "p_false", "logit_p_true", "logit_p_false", "verdict_true",
        "verdict_recognised", "prompt_n_tokens",
    ]
    cols = {k: result[k].tolist() for k in scalar_keys}
    with open(path, "w") as f:
        for i in range(len(result["doc_ids"])):
            f.write(json.dumps({k: cols[k][i] for k in scalar_keys}) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    params = cfg["params"]
    task = cfg["task"]

    runs_root = os.environ.get("RUNS_ROOT")
    if not runs_root:
        raise RuntimeError("RUNS_ROOT is not set")
    run_dir = Path(runs_root) / cfg["id"]
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    (run_dir / "config.snapshot.yaml").write_text(args.config.read_text())

    manifest = {
        "id": cfg["id"],
        "project": "judge-interp",
        "task": task,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "started_at": _now(),
        "finished_at": None,
        "status": "running",
        "host": os.uname().nodename,
        "job_id": os.environ.get("JOB_ID"),
        "config_path": str(args.config),
    }
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2))

    # Determinism: one seed drives python / numpy / torch.
    import torch

    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    from judge_interp.representationlm import RepresentationLM

    lm_class = RepresentationLM
    if task == "entity_detection_mixed":
        from judge_interp.entity_detection import EntityDetectionLM

        lm_class = EntityDetectionLM

    rlm = lm_class(
        model_name=params["model"],
        layers=params["layers"],
        device=params["device"],
        dtype=params["dtype"],
        verbose=True,
    )

    if task == "record_judge":
        items = build_items(params)
        instructions = build_instructions(include_cohesion=params["include_cohesion"])
        print(f"{cfg['id']}: {len(items)} rows, dataset={params['dataset']}/{params['split']}, "
              f"include_cohesion={params['include_cohesion']}")

        result = rlm.collect(
            items, instructions, params["dataset"],
            cache_dir=run_dir / "artifacts" / "cache",
            verify_read_point=params["verify_read_point"],
        )

        npz_path = run_dir / "artifacts" / "representations.npz"
        payload = {f"rep_{L}": result["representations"][L] for L in result["layers"].tolist()}
        for k, v in result.items():
            if k == "representations":
                continue
            payload[k] = v
        np.savez_compressed(npz_path, **payload)
        write_rows_jsonl(result, run_dir / "artifacts" / "rows.jsonl")
        metrics = compute_metrics(result)

    elif task == "entity_detection":
        items = build_entity_items(params, seed)
        print(f"{cfg['id']}: {len(items)} entity list(s), dataset={params['dataset']}/{params['split']}")

        result = rlm.collect_spans(
            items,
            cache_dir=run_dir / "artifacts" / "cache",
            verify_read_point=params["verify_read_point"],
        )
        _write_span_representations(result, run_dir / "artifacts" / "span_representations.npz")
        metrics = compute_span_metrics(result)

    elif task == "relation_detection":
        items, tuple_membership = build_relation_items(params, seed)
        print(f"{cfg['id']}: {len(items)} document prompt(s), split={params['split']}")

        result = rlm.collect_spans(
            items,
            cache_dir=run_dir / "artifacts" / "cache",
            verify_read_point=params["verify_read_point"],
        )
        _write_span_representations(result, run_dir / "artifacts" / "span_representations.npz")
        (run_dir / "artifacts" / "tuple_membership.json").write_text(
            json.dumps(tuple_membership, indent=2)
        )
        metrics = compute_span_metrics(result)

    elif task == "entity_detection_mixed":
        from judge_interp import entity_detection

        limit = params["context_token_limit"]
        tokenizer = AutoTokenizer.from_pretrained(params["model"]) if limit is not None else None
        items = entity_detection.build_items(params, seed, REPO_ROOT, tokenizer)
        print(f"{cfg['id']}: {len(items)} prompt(s) "
              f"({params['samples_per_doc']}/doc), split={params['split']}")

        result = rlm.collect_entities(
            items,
            cache_dir=run_dir / "artifacts" / "cache",
            verify_read_point=params["verify_read_point"],
        )
        _write_span_representations(result, run_dir / "artifacts" / "span_representations.npz")
        metrics = entity_detection.compute_metrics(result)

    else:
        raise AssertionError(f"unhandled task {task!r} -- load_config should have rejected this")

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("metrics:", json.dumps(metrics, indent=2))

    manifest["finished_at"] = _now()
    manifest["status"] = "success"
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
