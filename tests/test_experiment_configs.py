"""Layout checks for experiments/experiment-configs/ and analysis/configs/: submit.sh looks a config up by its id, so ids must be unique."""
from collections import Counter
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
EXPERIMENT_ROOT = REPO / "experiments" / "experiment-configs"
ANALYSIS_ROOT = REPO / "analysis" / "configs"
CONFIG_ROOTS = (EXPERIMENT_ROOT, ANALYSIS_ROOT)


def _configs(roots=CONFIG_ROOTS) -> list[Path]:
    paths = sorted(p for root in roots for p in root.rglob("*.yaml"))
    assert paths, f"no configs under {roots}"
    return paths


def test_every_config_id_equals_its_filename_stem():
    for path in _configs():
        assert yaml.safe_load(path.read_text())["id"] == path.stem, path


def test_config_ids_are_unique_across_groups():
    dupes = {s: n for s, n in Counter(p.stem for p in _configs()).items() if n > 1}
    assert not dupes, dupes


def test_configs_sit_in_a_group_subdirectory():
    assert all(p.parent not in CONFIG_ROOTS for p in _configs())


def test_experiment_configs_have_a_task_and_analysis_configs_do_not():
    assert all("task" in yaml.safe_load(p.read_text()) for p in _configs((EXPERIMENT_ROOT,)))
    assert not any("task" in yaml.safe_load(p.read_text()) for p in _configs((ANALYSIS_ROOT,)))
