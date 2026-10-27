"""Model configs: the committed record of which model a run used and how it is loaded.

A model config (``experiments/model-configs/<name>.yaml``) holds what identifies a model and
its input encoding; an experiment config points at it with ``params.model_config``. For a
Clef model the keys are ``CLEF_KEYS``.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

CLEF_KEYS = frozenset({"model", "revision", "release_code", "dtype", "max_length"})
_RELEASE_CODE_KEYS = frozenset({"file", "sha256"})
DTYPES = ("bfloat16", "float16", "float32")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def load_clef_model_config(path: Path) -> dict:
    """Load and validate a Clef model config.

    Args:
        path: a ``experiments/model-configs/*.yaml`` file.

    Returns:
        The config dict with exactly the keys of ``CLEF_KEYS``.

    Raises:
        KeyError: a key is missing or unknown.
        ValueError: ``revision`` is not a full commit hash, ``release_code.sha256`` is not a
            sha256 hex digest, ``dtype`` is not in ``DTYPES``, or ``max_length`` is not a
            positive int.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if set(cfg) != CLEF_KEYS:
        raise KeyError(f"{path}: keys must be exactly {sorted(CLEF_KEYS)}, got {sorted(cfg)}")
    if set(cfg["release_code"]) != _RELEASE_CODE_KEYS:
        raise KeyError(f"{path}: release_code keys must be exactly {sorted(_RELEASE_CODE_KEYS)}")
    if not _COMMIT.fullmatch(cfg["revision"]):
        raise ValueError(f"{path}: revision must be a full 40-hex commit hash, got {cfg['revision']!r}")
    if not _SHA256.fullmatch(cfg["release_code"]["sha256"]):
        raise ValueError(f"{path}: release_code.sha256 must be a 64-hex digest")
    if cfg["dtype"] not in DTYPES:
        raise ValueError(f"{path}: dtype must be one of {DTYPES}, got {cfg['dtype']!r}")
    if not (isinstance(cfg["max_length"], int) and not isinstance(cfg["max_length"], bool) and cfg["max_length"] > 0):
        raise ValueError(f"{path}: max_length must be a positive int, got {cfg['max_length']!r}")
    return cfg


JEV_KEYS = frozenset({"model", "retry"})
_RETRY_KEYS = frozenset({"max_retries", "backoff_max", "timeout"})
_JEV_VERSIONED = re.compile(r"jev-\d+\.\d+\.\d+")


def load_jev_model_config(path: Path) -> dict:
    """Load and validate a Jev model config.

    Args:
        path: a ``experiments/model-configs/*.yaml`` file.

    Returns:
        The config dict with exactly the keys of ``JEV_KEYS``.

    Raises:
        KeyError: a key is missing or unknown.
        ValueError: ``model`` is not a versioned id (an alias such as ``jev-latest`` moves between
            releases), ``retry.max_retries`` is not a non-negative int, or ``retry.backoff_max`` /
            ``retry.timeout`` is not a positive number.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if set(cfg) != JEV_KEYS:
        raise KeyError(f"{path}: keys must be exactly {sorted(JEV_KEYS)}, got {sorted(cfg)}")
    if set(cfg["retry"]) != _RETRY_KEYS:
        raise KeyError(f"{path}: retry keys must be exactly {sorted(_RETRY_KEYS)}")
    if not _JEV_VERSIONED.fullmatch(cfg["model"]):
        raise ValueError(f"{path}: model must be a versioned id like 'jev-1.14.0', not an alias; got {cfg['model']!r}")
    r = cfg["retry"]
    if not (isinstance(r["max_retries"], int) and not isinstance(r["max_retries"], bool) and r["max_retries"] >= 0):
        raise ValueError(f"{path}: retry.max_retries must be a non-negative int, got {r['max_retries']!r}")
    for k in ("backoff_max", "timeout"):
        if not (isinstance(r[k], (int, float)) and not isinstance(r[k], bool) and r[k] > 0):
            raise ValueError(f"{path}: retry.{k} must be a positive number of seconds, got {r[k]!r}")
    return cfg
