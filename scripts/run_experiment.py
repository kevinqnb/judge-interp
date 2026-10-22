"""Run one representation-collection or direct-judge experiment from a config.

Usage
-----
    uv run --python 3.12 python scripts/run_experiment.py configs/<id>.yaml

Reads the config envelope (``id``, ``project``, ``task``, ``description``, ``seed``,
``params``), runs the task, and writes the harness output to ``$RUNS_ROOT/<id>/``:

    run.json                         manifest (task, git sha, status, timestamps)
    config.snapshot.yaml             verbatim copy of the config
    dataset_config.snapshot.py       verbatim copy of the dataset-config module (prompt vocabulary)
    metrics.json                     flat scalar metrics
    artifacts/span_representations.npz   per-layer ``rep_<L>`` arrays + span metadata
    artifacts/cache/<document_id>.npz    per-document shards (resume support)

Every ``params`` key is required for its task and unknown keys are rejected; the
GPU-resource keys read by ``scripts/submit.sh`` are the only keys allowed on every task.

Tasks
-----
``entity_detection`` (``judge_interp.entity_detection``): one prompt per selected
    dataset row, with the hidden state read at each non-null entity's closing quote.
``relation_detection`` (``judge_interp.relation_detection``): the same read-out with
    instructions asking whether a row's values belong to the same item; the dataset must
    have ``supports_relation``.
``clef_entity_row`` / ``clef_relation_row`` (``judge_interp.clef``): the Cloudflare
    clef-flash model as a direct judge; no representations are collected.

``params`` of the two representation tasks
    data_root          repo-relative dir the dataset config's ``split_path`` / ``context_path``
                       are relative to
    model              HF instruct-model repo id
    dataset_config     repo-relative path of a ``.py`` file defining ``DATASET``, a
                       ``judge_interp.dataset_spec.DatasetSpec``
                       (e.g. ``experiments/dataset-configs/vrdu_line.py``)
    split              "train" | "test"
    layers             ints and/or "last" (see ``prompts.resolve_layers``)
    dtype              "float32" | "float16" | "bfloat16"
    device             device_map string ("cuda", "cuda:0", "cpu")
    context_token_limit null | int; hard error if any OCR context tokenizes to more
                       than this many tokens
    verify_read_point  bool; run ``RepresentationLM.verify_read_point`` on the first
                       item before judging
    error_types        invalid-row error types to include alongside every valid row
                       (members of the dataset config's ``error_types``)
    rows_per_stratum   {"valid": n, "<error_type>": n, ...}; keys must equal "valid" +
                       ``error_types``; n is null (all rows) or an int >= 1, and a
                       stratum smaller than n is a hard error. Drawn with ``seed``.
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

from judge_interp.dataset_spec import load_dataset_spec  # noqa: E402


_ROW_PARAMS = (
    "data_root", "model", "dataset_config", "split", "layers", "dtype", "device",
    "context_token_limit", "verify_read_point", "error_types", "rows_per_stratum",
)
_CLEF_PARAMS = (
    "data_root", "model", "revision", "dataset_config", "split", "device", "dtype", "max_length",
    "overlong_docs", "context_control", "determinism_check_n", "known_answer_check",
    "error_types", "rows_per_stratum",
)
_TASK_PARAMS: dict[str, tuple[str, ...]] = {
    "entity_detection": _ROW_PARAMS,
    "relation_detection": _ROW_PARAMS,
    "clef_entity_row": _CLEF_PARAMS,
    "clef_relation_row": _CLEF_PARAMS + ("samples_per_row", "min_subset_size", "n_per_class", "subset_rng_stream"),
}
# Allowed for any task -- read only by scripts/submit.sh, ignored here.
_SUBMIT_ONLY_PARAMS = frozenset({"gpu_type", "gpu_memory", "gpu_c", "omp", "walltime"})


def _now() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(*args: str) -> str:
    """Run ``git`` in the repo root and return its stripped stdout; raises on a non-zero exit."""
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def _require(d: dict, key: str, where: str):
    """Return ``d[key]``, raising ``KeyError`` that names ``where`` if the key is missing."""
    if key not in d:
        raise KeyError(f"{where}: required key {key!r} is missing")
    return d[key]


def load_config(path: Path) -> dict:
    """Load and validate a run config.

    Args:
        path: ``configs/<id>.yaml``; its stem must equal the config's ``id``.

    Returns:
        The config dict (``id``, ``project``, ``task``, ``description``, ``seed``, ``params``).

    Raises:
        KeyError: a required envelope key or task param is missing, or a param is not
            valid for the task.
        ValueError: the id does not match the filename, the project is not
            ``judge-interp``, the task is unknown, or the dataset config does not support
            a relation task that the task needs.
    """
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

    spec = load_dataset_spec(REPO_ROOT / params["dataset_config"])
    if task in ("relation_detection", "clef_relation_row"):
        spec.require_relation()
    return cfg


def _write_span_representations(result: dict, path: Path) -> None:
    """Write a ``collect_rows`` result to a compressed ``.npz``.

    Args:
        result: ``collect_rows`` output; ``representations`` is stored as ``rep_<layer>``
            and every other entry under its own key.
        path: destination file.
    """
    payload = {f"rep_{L}": result["representations"][L] for L in result["layers"].tolist()}
    for k, v in result.items():
        if k == "representations":
            continue
        payload[k] = v
    np.savez_compressed(path, **payload)


def main(argv=None) -> None:
    """Run the experiment named by a config and write its outputs under ``$RUNS_ROOT/<id>/``.

    Args:
        argv: command-line arguments (``config`` path); ``None`` reads ``sys.argv``.

    Raises:
        RuntimeError: ``RUNS_ROOT`` is not set.
    """
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
    dataset_config_path = REPO_ROOT / params["dataset_config"]
    (run_dir / "dataset_config.snapshot.py").write_text(dataset_config_path.read_text())
    spec = load_dataset_spec(dataset_config_path)

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

    if task in ("clef_entity_row", "clef_relation_row"):
        from judge_interp import clef

        metrics = clef.run(cfg, run_dir, REPO_ROOT)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print("metrics:", json.dumps(metrics, indent=2))
        manifest["finished_at"] = _now()
        manifest["status"] = "success"
        (run_dir / "run.json").write_text(json.dumps(manifest, indent=2))
        print(f"wrote {run_dir}")
        return

    if task == "entity_detection":
        from judge_interp import entity_detection as task_module

        lm_class = task_module.EntityDetectionLM
    elif task == "relation_detection":
        from judge_interp import relation_detection as task_module

        lm_class = task_module.RelationDetectionLM
    else:
        raise AssertionError(f"unhandled task {task!r} -- load_config should have rejected this")

    rlm = lm_class(
        model_name=params["model"],
        layers=params["layers"],
        device=params["device"],
        dtype=params["dtype"],
        verbose=True,
    )

    limit = params["context_token_limit"]
    tokenizer = AutoTokenizer.from_pretrained(params["model"]) if limit is not None else None
    items = task_module.build_items(spec, params, seed, REPO_ROOT, tokenizer)
    print(f"{cfg['id']}: {len(items)} row prompt(s), dataset={spec.name}, split={params['split']}")

    result = rlm.collect_rows(
        items,
        cache_dir=run_dir / "artifacts" / "cache",
        verify_read_point=params["verify_read_point"],
    )
    _write_span_representations(result, run_dir / "artifacts" / "span_representations.npz")
    metrics = task_module.compute_metrics(result, spec)

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("metrics:", json.dumps(metrics, indent=2))

    manifest["finished_at"] = _now()
    manifest["status"] = "success"
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
