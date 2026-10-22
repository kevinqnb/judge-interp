"""The VRDU dataset specs, loaded the way the experiment runner loads them."""
from pathlib import Path

from judge_interp.dataset_spec import load_dataset_spec

_DIR = Path(__file__).resolve().parents[1] / "experiments" / "dataset-configs"
MAIN = load_dataset_spec(_DIR / "vrdu_main.py")
LINE = load_dataset_spec(_DIR / "vrdu_line.py")
SPECS = {"main": MAIN, "line": LINE}
