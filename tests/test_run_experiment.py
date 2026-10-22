"""Unit tests for experiments/run_experiment.py's config loading (required keys, task set)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import run_experiment as run  # noqa: E402


def test_entity_detection_fixture_config_loads_and_requires_every_key(tmp_path):
    import yaml

    path = Path(__file__).resolve().parent / "fixtures" / "entity_detection_tiny_e2e.yaml"
    cfg = run.load_config(path)
    assert cfg["task"] == "entity_detection"
    for key in ("rows_per_stratum", "error_types", "layers"):
        broken = {**cfg, "params": {k: v for k, v in cfg["params"].items() if k != key}}
        p = tmp_path / "entity_detection_tiny_e2e.yaml"
        p.write_text(yaml.safe_dump(broken))
        with pytest.raises(KeyError, match=key):
            run.load_config(p)
    bad = {**cfg, "params": {**cfg["params"], "row_subset": None}}
    p = tmp_path / "entity_detection_tiny_e2e.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(KeyError, match="row_subset"):
        run.load_config(p)


def test_relation_detection_fixture_config_loads_and_requires_every_key(tmp_path):
    import yaml

    path = Path(__file__).resolve().parent / "fixtures" / "relation_detection_tiny_e2e.yaml"
    cfg = run.load_config(path)
    assert cfg["task"] == "relation_detection"
    for key in ("rows_per_stratum", "error_types", "layers"):
        broken = {**cfg, "params": {k: v for k, v in cfg["params"].items() if k != key}}
        p = tmp_path / "relation_detection_tiny_e2e.yaml"
        p.write_text(yaml.safe_dump(broken))
        with pytest.raises(KeyError, match=key):
            run.load_config(p)
    bad = {**cfg, "params": {**cfg["params"], "row_subset": None}}
    p = tmp_path / "relation_detection_tiny_e2e.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(KeyError, match="row_subset"):
        run.load_config(p)
    main = {**cfg, "params": {**cfg["params"], "dataset_config": "experiments/dataset-configs/vrdu_main.py"}}
    p.write_text(yaml.safe_dump(main))
    with pytest.raises(ValueError, match="does not support the relation tasks"):
        run.load_config(p)


@pytest.mark.parametrize("task", ["record_judge", "entity_detection_mixed", "relation_detection_listed", "entity_detection_row"])
def test_removed_tasks_are_rejected(tmp_path, task):
    import yaml

    cfg = run.load_config(Path(__file__).resolve().parent / "fixtures" / "entity_detection_tiny_e2e.yaml")
    p = tmp_path / "entity_detection_tiny_e2e.yaml"
    p.write_text(yaml.safe_dump({**cfg, "task": task}))
    with pytest.raises(ValueError, match="task must be one of"):
        run.load_config(p)


CONFIG_ROOT = Path(__file__).resolve().parent.parent / "experiments" / "experiment-configs"
CLEF_CONFIG = CONFIG_ROOT / "relation-detection" / "clef" / "2026-10-22-clef-conditional-smoke-01.yaml"
CLEF_ONESHOT_CONFIG = CONFIG_ROOT / "one-shot" / "clef" / "2026-10-22-clef-oneshot-smoke-01.yaml"


def test_clef_config_requires_model_config_and_rejects_inline_model_keys(tmp_path):
    import yaml

    cfg = run.load_config(CLEF_CONFIG)
    assert cfg["task"] == "clef_relation_conditional"
    broken = {**cfg, "params": {k: v for k, v in cfg["params"].items() if k != "model_config"}}
    p = tmp_path / f"{cfg['id']}.yaml"
    p.write_text(yaml.safe_dump(broken))
    with pytest.raises(KeyError, match="model_config"):
        run.load_config(p)
    for key in ("revision", "dtype", "max_length", "model"):
        p.write_text(yaml.safe_dump({**cfg, "params": {**cfg["params"], key: "x"}}))
        with pytest.raises(KeyError, match=key):
            run.load_config(p)


def test_every_task_module_implements_the_task_interface():
    for task, module in run._TASK_MODULES.items():
        assert task in module.TASK_PARAMS and callable(module.run)
    assert set(run._TASK_PARAMS) == set(run._TASK_MODULES)


def test_clef_sample_task_params_and_removed_relation_row(tmp_path):
    import yaml

    cfg = run.load_config(CLEF_ONESHOT_CONFIG)
    assert cfg["task"] == "clef_record_oneshot"
    p = tmp_path / f"{cfg['id']}.yaml"
    for drop, match in (("samples_path", "samples_path"),):
        p.write_text(yaml.safe_dump({**cfg, "params": {k: v for k, v in cfg["params"].items() if k != drop}}))
        with pytest.raises(KeyError, match=match):
            run.load_config(p)
    for extra in ("rows_per_stratum", "split"):  # row-task keys are hard errors for sample tasks
        p.write_text(yaml.safe_dump({**cfg, "params": {**cfg["params"], extra: None}}))
        with pytest.raises(KeyError, match=extra):
            run.load_config(p)
    p.write_text(yaml.safe_dump({**cfg, "task": "clef_relation_row"}))  # the retired task fails loudly
    with pytest.raises(ValueError, match="task must be one of"):
        run.load_config(p)
