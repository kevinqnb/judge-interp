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
    context_char_limit null | int — hard error if any OCR context exceeds it
    row_subset         null (all rows) | {n: int, seed: int} (deterministic sample)
    verify_read_point  bool — run RepresentationLM.verify_read_point on the first
                       item before judging anything (determinism / read-point
                       gate; see its docstring). Leave true unless this exact
                       (model, device, dtype) has already been checked.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from judge_interp.instruction_prompts import build_instructions
from judge_interp.prompts import load_ocr_context, load_split, render_query

_REQUIRED_PARAMS = (
    "data_root", "model", "dataset", "split", "layers", "dtype", "device",
    "include_cohesion", "context_char_limit", "row_subset", "verify_read_point",
)


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
    for key in ("description", "seed", "params"):
        _require(cfg, key, "config")
    params = cfg["params"]
    for key in _REQUIRED_PARAMS:
        _require(params, key, "params")
    if params["dataset"] not in ("main", "line"):
        raise ValueError(f"params.dataset must be 'main' or 'line', got {params['dataset']!r}")
    return cfg


def build_items(params: dict) -> list[dict]:
    """One judge item per row: identity + label metadata + context + query."""
    data_root = REPO_ROOT / params["data_root"]
    dataset, split = params["dataset"], params["split"]
    rows = load_split(data_root, dataset, split)

    limit = params["context_char_limit"]
    contexts: dict[str, str] = {}
    for doc_id in {r["document_id"] for r in rows}:
        text = load_ocr_context(data_root, doc_id)
        if limit is not None and len(text) > limit:
            raise ValueError(
                f"OCR context for {doc_id!r} is {len(text)} chars > context_char_limit {limit}"
            )
        contexts[doc_id] = text

    items = []
    for r in rows:
        items.append({
            "document_id": r["document_id"],
            "line_index": r["line_index"] if dataset == "line" else None,
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


def write_rows_jsonl(result: dict, path: Path) -> None:
    scalar_keys = [
        "doc_ids", "line_indices", "k", "num_invalid_fields", "labels", "p_true",
        "p_false", "logit_p_true", "logit_p_false", "verdict_true",
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

    runs_root = os.environ.get("RUNS_ROOT")
    if not runs_root:
        raise RuntimeError("RUNS_ROOT is not set")
    run_dir = Path(runs_root) / cfg["id"]
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    (run_dir / "config.snapshot.yaml").write_text(args.config.read_text())

    manifest = {
        "id": cfg["id"],
        "project": "judge-interp",
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

    items = build_items(params)
    instructions = build_instructions(include_cohesion=params["include_cohesion"])
    print(f"{cfg['id']}: {len(items)} rows, dataset={params['dataset']}/{params['split']}, "
          f"include_cohesion={params['include_cohesion']}")

    rlm = RepresentationLM(
        model_name=params["model"],
        layers=params["layers"],
        device=params["device"],
        dtype=params["dtype"],
        verbose=True,
    )
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
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("metrics:", json.dumps(metrics, indent=2))

    manifest["finished_at"] = _now()
    manifest["status"] = "success"
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
