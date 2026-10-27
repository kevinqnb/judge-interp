"""Run one representation-collection or direct-judge experiment from a config.

Usage
-----
    uv run --python 3.12 python experiments/run_experiment.py experiments/experiment-configs/<group>/<id>.yaml

Reads the config envelope (``id``, ``project``, ``task``, ``description``, ``seed``,
``params``), runs the task, and writes the harness output to ``$RUNS_ROOT/<id>/``:

    run.json                         manifest (task, git sha, status, timestamps)
    config.snapshot.yaml             verbatim copy of the config
    dataset_config.snapshot.py       verbatim copy of the dataset-config module (prompt vocabulary)
    model_config.snapshot.yaml       verbatim copy of the model config (tasks with ``model_config`` only)
    metrics.json                     flat scalar metrics
    artifacts/span_representations.npz   per-layer ``rep_<L>`` arrays + span metadata
    artifacts/cache/<document_id>.npz    per-document shards (resume support)

Every ``params`` key is required for its task and unknown keys are rejected; the
GPU-resource keys read by ``experiments/submit.sh`` are the only keys allowed on every task.

Tasks
-----
``entity_detection`` (``judge_interp.entity_detection``): one prompt per selected
    dataset row, with the hidden state read at each non-null entity's closing quote.
``relation_detection`` (``judge_interp.relation_detection``): the same read-out with
    instructions asking whether a row's values belong to the same item; the dataset must
    have ``supports_relation``.
``clef_entity_row`` / ``clef_relation_conditional`` / ``clef_record_oneshot`` (``judge_interp.clef``):
    the Cloudflare clef-flash model as a direct judge; no representations are collected. The
    entity task judges dataset rows; the other two judge the samples of a samples file
    (conditional: is a primary entity related to all its conditions; one-shot: is the whole
    record valid). The dataset must have ``supports_relation`` for the latter two.
``jev_entity_row`` / ``jev_relation_conditional`` / ``jev_record_oneshot`` (``judge_interp.jev``):
    the same three tasks with TypeSafe Jev, called over the API (needs ``TYPESAFE_API_KEY``;
    ``params.model_config`` is a ``load_jev_model_config`` file, e.g. ``experiments/model-configs/jev.yaml``).
    Params are listed in ``jev.TASK_PARAMS``.

Each task module exposes ``TASK_PARAMS`` (its required params) and
``run(cfg, spec, run_dir, repo_root, model_cfg)``, which returns the metrics; the runner owns
config loading, snapshots, the manifest (``status`` is ``success`` or ``failed``), seeding and
``metrics.json``. ``model_cfg`` is the loaded model config for tasks whose params include
``model_config``, else ``None``.

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
                       (members of the dataset config's ``error_types``); for a paired
                       dataset (line), the pairs whose invalid row has one of these types
    rows_per_stratum   {"valid": n, "<error_type>": n, ...}; keys must equal "valid" +
                       ``error_types``; n is null (all rows) or an int >= 1, and a
                       stratum smaller than n is a hard error. Drawn with ``seed``.
                       For a paired dataset: {"<cell>": n, ...} over the cells present,
                       n pairs per cell (both rows of a pair are kept).

``params`` of the three Clef tasks: ``data_root`` and ``dataset_config`` as above, plus
    model_config       repo-relative path of a model config
                       (e.g. ``experiments/model-configs/clef.yaml``: model, pinned revision,
                       release-code hash, dtype, ``max_length``)
    device             torch device string; CUDA must be available if requested
    overlong_docs      "exclude" | "error"; records over the model config's ``max_length``
    context_control    "none" | "other_document"
    determinism_check_n, known_answer_check   sanity controls (see ``judge_interp.clef``)
and, for ``clef_entity_row``, ``split``, ``error_types`` and ``rows_per_stratum`` as above; for the
two sample tasks instead ``samples_path`` (repo-relative samples file, see ``clef.load_samples``;
the context of each sample's document is read from ``data_root``).
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

from judge_interp import clef, entity_detection, jev, relation_detection  # noqa: E402
from judge_interp.dataset_spec import load_dataset_spec  # noqa: E402
from judge_interp.model_config import load_clef_model_config, load_jev_model_config  # noqa: E402


# task name -> module. Each module exposes TASK_PARAMS (task -> required params) and
# run(cfg, spec, run_dir, repo_root, model_cfg) -> metrics.
_TASK_MODULES = {
    "entity_detection": entity_detection,
    "relation_detection": relation_detection,
    "clef_entity_row": clef,
    "clef_relation_conditional": clef,
    "clef_record_oneshot": clef,
    "jev_entity_row": jev,
    "jev_relation_conditional": jev,
    "jev_record_oneshot": jev,
}
_TASK_PARAMS: dict[str, tuple[str, ...]] = {
    task: module.TASK_PARAMS[task] for task, module in _TASK_MODULES.items()
}
# Allowed for any task -- read only by experiments/submit.sh, ignored here.
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
        path: ``experiments/experiment-configs/<group>/<id>.yaml``; its stem must equal the config's ``id``.

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
    if task in ("relation_detection", *clef.SAMPLE_TASKS, *jev.SAMPLE_TASKS):
        spec.require_relation()
    return cfg


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
    model_cfg = None
    if "model_config" in _TASK_PARAMS[task]:
        model_config_path = REPO_ROOT / params["model_config"]
        model_cfg = (load_jev_model_config if task in jev.TASKS else load_clef_model_config)(model_config_path)
        (run_dir / "model_config.snapshot.yaml").write_text(model_config_path.read_text())

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
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    if task not in jev.TASKS:   # the Jev tasks call an API and load no local model: torch is not needed
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    try:
        metrics = _TASK_MODULES[task].run(cfg, spec, run_dir, REPO_ROOT, model_cfg)
    except BaseException:
        manifest["finished_at"] = _now()
        manifest["status"] = "failed"
        (run_dir / "run.json").write_text(json.dumps(manifest, indent=2))
        raise

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("metrics:", json.dumps(metrics, indent=2))
    manifest["finished_at"] = _now()
    manifest["status"] = "success"
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
