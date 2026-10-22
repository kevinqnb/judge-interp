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
