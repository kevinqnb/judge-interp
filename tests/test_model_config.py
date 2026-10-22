"""Unit tests for judge_interp.model_config (validation of the committed Clef model config)."""
from pathlib import Path

import pytest
import yaml

from judge_interp.model_config import load_clef_model_config

CLEF = Path(__file__).resolve().parent.parent / "experiments" / "model-configs" / "clef.yaml"


def _write(tmp_path, cfg):
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_committed_clef_config_loads():
    cfg = load_clef_model_config(CLEF)
    assert cfg["model"] == "Cloudflare/clef-flash" and len(cfg["revision"]) == 40


@pytest.mark.parametrize("key", ["model", "revision", "release_code", "dtype", "max_length"])
def test_missing_key_is_an_error(tmp_path, key):
    cfg = yaml.safe_load(CLEF.read_text())
    del cfg[key]
    with pytest.raises(KeyError):
        load_clef_model_config(_write(tmp_path, cfg))


def test_unknown_key_is_an_error(tmp_path):
    cfg = {**yaml.safe_load(CLEF.read_text()), "device": "cuda"}
    with pytest.raises(KeyError):
        load_clef_model_config(_write(tmp_path, cfg))


@pytest.mark.parametrize("key,bad", [
    ("revision", "main"), ("revision", "8b2e5fd"), ("dtype", "int8"), ("max_length", 0), ("max_length", True),
    ("max_length", "16384"),
])
def test_bad_value_is_an_error(tmp_path, key, bad):
    cfg = {**yaml.safe_load(CLEF.read_text()), key: bad}
    with pytest.raises(ValueError):
        load_clef_model_config(_write(tmp_path, cfg))


def test_bad_release_code_digest_is_an_error(tmp_path):
    cfg = yaml.safe_load(CLEF.read_text())
    cfg["release_code"]["sha256"] = "abc"
    with pytest.raises(ValueError):
        load_clef_model_config(_write(tmp_path, cfg))
