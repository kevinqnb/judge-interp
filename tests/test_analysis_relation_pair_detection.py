"""Tests for analysis/relation_pair_detection.py on tests/fixtures/relation_pair_detection_mini (see
tests/test_relation_pair_detection.py for its pairs) and on synthetic runs written from its items.

A synthetic run's activation for a pair is a function of the record text up to and including that pair's line (plus a
label signal), so two prompts with the same prefix get the same activation: the prefix known answer holds exactly.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from judge_interp import relation_pair_detection as rp
from dataset_specs import LINE

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = "tests/fixtures/relation_pair_detection_mini"
LINE_DIR = REPO_ROOT / FIXTURE / "vrdu" / "line"

_spec = importlib.util.spec_from_file_location("analysis_relation_pair_detection", REPO_ROOT / "analysis" / "relation_pair_detection.py")
ap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ap)

HIDDEN = 8
P = dict(lr_C=1.0, lr_max_iter=2000, lr_tol=1e-6, decision_threshold=0.5)


def _frame():
    return ap.pair_frame(json.loads((LINE_DIR / "train_subsets.json").read_text()), json.loads((LINE_DIR / "train.json").read_text()))


def _items():
    p = dict(data_root=FIXTURE, split="train", error_types=list(LINE.error_types), rows_per_stratum={"cross_item:k2": None, "intra_document:k1": None},
             context_token_limit=None, subsets_path="vrdu/line/{split}_subsets.json")
    return rp.build_items(LINE, p, 0, REPO_ROOT)


def _vec(text: str, label: bool) -> np.ndarray:
    rng = np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest(), 16) % 2**32)
    v = rng.normal(size=HIDDEN).astype(np.float32)
    v[0] += 3.0 if label else -3.0
    return v


def write_run(runs_root: Path, run_id: str, items: list[dict], layer: int = 5, data_root: str = FIXTURE, split: str = "train",
              dataset_config: str = "experiments/dataset-configs/vrdu_line.py", task: str = "relation_pair_detection") -> Path:
    """A synthetic pair run of ``items`` (the shape ``collect_rows`` writes)."""
    d = runs_root / run_id
    (d / "artifacts").mkdir(parents=True)
    sp = [(it, s) for it in items for s in it["spans"]]
    cols = {k: np.asarray([s[k] for _, s in sp]) for k in rp.PAIR_META_KEYS}
    x = np.stack([_vec(it["query"][: s["char_end"]], s["label_valid"]) for it, s in sp])
    np.savez_compressed(d / "artifacts" / "span_representations.npz", layers=np.array([0, layer]), rep_0=np.ones((len(sp), HIDDEN), np.float32),
                        **{f"rep_{layer}": x}, doc_ids=np.asarray([it["document_id"] for it, _ in sp]), **cols)
    (d / "run.json").write_text(json.dumps({"status": "success", "task": task}))
    (d / "metrics.json").write_text(json.dumps({"n_spans": len(sp), "n_valid_spans": int(cols["label_valid"].sum())}))
    (d / "config.snapshot.yaml").write_text(yaml.safe_dump({"seed": 0, "params": {
        "data_root": data_root, "dataset_config": dataset_config, "split": split, "subsets_path": "vrdu/line/{split}_subsets.json"}}))
    return d


# ---------------------------------------------------------------- frame
def test_pair_frame_hand_worked():
    fr = _frame()
    assert len(fr["y"]) == 52 and int((~fr["parent_valid"]).sum()) == 26
    bad = ~fr["parent_valid"]
    assert int((~fr["y"][bad]).sum()) == 13 and fr["y"][fr["parent_valid"]].all()
    (i,) = np.flatnonzero(fr["y"] & (fr["n_corrupted"] > 0))
    assert fr["fields"][i] == "channel+program_start_date" and fr["cell"][i] == "intra_document:k1"
    assert set(fr["error_type"].tolist()) == {"intra_document", "cross_item"}


def test_pair_frame_refuses_a_parent_with_a_missing_pair():
    subsets = [r for r in json.loads((LINE_DIR / "train_subsets.json").read_text())]
    drop = next(i for i, r in enumerate(subsets) if r["size"] == 2)
    with pytest.raises(AssertionError, match="pairs for m"):
        ap.pair_frame(subsets[:drop] + subsets[drop + 1:], json.loads((LINE_DIR / "train.json").read_text()))


def test_balanced_cells_and_the_structure_baseline_is_chance():
    fr = _frame()
    pos, counts = ap.rs.balance_cells(fr, np.flatnonzero(~fr["parent_valid"]), 0)
    ev = ap.cut(fr, pos)
    ap.check_balanced(ev)
    assert counts["n_balanced_rows"] == 2 * int(ev["y"].sum()) > 0
    m = ap.structure_baseline(ev, ev, P, 0)
    assert m["test_baseline_fields_auroc"] == pytest.approx(0.5, abs=1e-12)
    with pytest.raises(AssertionError, match="not balanced"):
        ap.check_balanced(ap.cut(fr, np.flatnonzero(~fr["parent_valid"])))


# ---------------------------------------------------------------- runs
def test_after_corruption_hand_worked():
    run = {"doc_ids": np.array(["d"] * 4 + ["e"] * 2), "row_ordinal": np.array([0, 0, 0, 0, 0, 0]),
           "pair_position": np.array([2, 1, 3, 4, 1, 2]), "n_corrupted": np.array([1, 0, 0, 2, 1, 0])}
    # prompt d in shown order: pos1 clean, pos2 corrupted, pos3 clean, pos4 corrupted -> after = F, F, T, T
    assert ap.after_corruption(run).tolist() == [False, False, True, True, False, True]


def test_load_match_and_prefix_known_answer(tmp_path):
    items = _items()
    write_run(tmp_path, "r", items)
    run = ap.load_pair_run("r", tmp_path, 5, None, "relation_pair_detection")
    assert run["parents_path"] == LINE_DIR / "train.json" and run["subsets_path"] == LINE_DIR / "train_subsets.json"
    fr = _frame()
    pos = ap.match_run(fr, run, np.ones(len(fr["y"]), bool))
    assert (run["subset_id"][pos] == fr["subset_id"]).all()
    dev, n = ap.prefix_known_answer(run)
    expected = sum(int(np.argmax([s["n_corrupted"] > 0 for s in it["spans"]])) for it in items if not it["spans"][0]["parent_valid"])
    assert dev == 0.0 and n == expected > 0
    # a pre-corruption pair that differs from its twin breaks the known answer
    bad = next(i for i in range(len(run["X"])) if not run["parent_valid"][i] and not run["after"][i] and run["n_corrupted"][i] == 0)
    run["X"][bad] += 10.0
    assert ap.prefix_known_answer(run)[0] > 0.5


def test_a_leaking_layer_0_and_a_mislabelled_run_are_refused(tmp_path):
    d = write_run(tmp_path, "r", _items())
    with np.load(d / "artifacts" / "span_representations.npz") as z:
        arrays = dict(z)
    arrays["rep_0"][0, 0] = 2.0
    np.savez_compressed(d / "artifacts" / "span_representations.npz", **arrays)
    with pytest.raises(AssertionError, match="layer-0"):
        ap.load_pair_run("r", tmp_path, 5, None, "relation_pair_detection")
    write_run(tmp_path, "s", _items())
    run = ap.load_pair_run("s", tmp_path, 5, None, "relation_pair_detection")
    run["label_valid"][0] = ~run["label_valid"][0]
    with pytest.raises(AssertionError, match="label differs"):
        ap.match_run(_frame(), run, np.ones(52, bool))


def test_judge_orientation_check():
    layout = {"pair_position": np.array([1, 2, -1]), "field_a": np.array(["channel", "sub_amount", ""]),
              "field_b": np.array(["sub_amount", "program_desc", ""])}
    ok = {"idx": np.array([0, 1, 2]), "primary": np.array(["channel", "sub_amount", "x"]), "condition": np.array(["sub_amount", "program_desc", "y"])}
    ap.check_judge_orientation("j", layout, ok)
    flipped = {**ok, "primary": np.array(["sub_amount", "sub_amount", "x"])}
    with pytest.raises(AssertionError, match="primary"):
        ap.check_judge_orientation("j", layout, flipped)


# ---------------------------------------------------------------- probe
def test_probe_separates_the_signal_and_the_controls_hold(tmp_path):
    items = _items()
    write_run(tmp_path, "r", items)
    run = ap.load_pair_run("r", tmp_path, 5, None, "relation_pair_detection")
    fr = _frame()
    pos = ap.match_run(fr, run, np.ones(len(fr["y"]), bool))
    ev = ap.cut(fr, np.flatnonzero(~fr["parent_valid"]))
    for c in ap.LAYOUT_COLS:
        ev[c] = run[c][pos][~fr["parent_valid"]]
    x = run["X"][pos][~fr["parent_valid"]]
    p = {**P, "prefix_match_rtol": 0.25, "run_controls": True, "n_train_label_shuffles": 5, "train_label_shuffle_auroc_tolerance": 0.5}
    m, probs = ap.run_lm("lm", 5, p, 0, x, ev["y"], x, ev, (0.0, 3), tmp_path)
    assert m["test_auroc"] == 1.0 and m["control_seed_determinism"] == 1.0
    assert "test_auroc_intra_document" in m and "test_n_ncorr2" in m and "test_n_after_corruption" in m
    with pytest.raises(AssertionError, match="twins"):
        ap.run_lm("lm", 5, p, 0, x, ev["y"], x, ev, (0.9, 3), tmp_path)


# ---------------------------------------------------------------- whole analysis on synthetic runs
def write_clef_run(runs_root: Path, run_id: str, samples: list[dict], flip: int | None = None) -> None:
    """A synthetic Clef ``conditional`` run on a ``line_pairs`` samples file: P(true) 0.8 on valid pairs, 0.3 on invalid."""
    d = runs_root / run_id
    (d / "artifacts").mkdir(parents=True)
    prim = [s["primary"]["field"] for s in samples]
    cond = [s["conditions"][0]["field"] for s in samples]
    if flip is not None:
        prim[flip], cond[flip] = cond[flip], prim[flip]
    p = np.array([0.8 if s["label_valid"] else 0.3 for s in samples])
    np.savez_compressed(d / "artifacts" / "clef_predictions.npz", p_true=p, logit_true=np.log(p), logit_false=np.log(1 - p),
                        label_valid=np.array([s["label_valid"] for s in samples]), sample_id=np.array([s["sample_id"] for s in samples]),
                        doc_ids=np.array([s["document_id"] for s in samples]), cell=np.array([s["cell"] for s in samples]),
                        primary=np.array(prim), conditions=np.array(cond))
    (d / "run.json").write_text(json.dumps({"status": "success", "task": "clef_relation_conditional"}))
    (d / "metrics.json").write_text(json.dumps({"n_predictions": len(samples), "n_excluded_overlong": 0, "determinism_max_abs_diff": 0.0}))
    (d / "config.snapshot.yaml").write_text(yaml.safe_dump({"params": {"samples_path": "experiments/clef-samples/conditional-test-pairs.json"}}))


def test_run_end_to_end_on_synthetic_runs(tmp_path, monkeypatch):
    import sys

    import test_build_vrdu_line as tl
    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    import clef_samples as cs

    repo, runs = tmp_path / "repo", tmp_path / "runs"
    vrdu = repo / "data" / "vrdu"
    vrdu.mkdir(parents=True)
    # the builder's own fixture, widened to 5 documents of 6 items, so a document bootstrap of the report keeps both classes
    monkeypatch.setattr(tl, "make_base", lambda split: [tl.item(f"{tl.PREFIX[split]}{d}", li) for d in "ABCDE" for li in range(6)])
    line = {**tl.LINE, "n_per_cell": {"train": 4, "test": 4}}
    tl.b.main(["--config", str(tl.write_root(vrdu, line))])   # train / test line files, subsets and line_build.json (check_build passes)
    dataset_config = str(REPO_ROOT / "experiments/dataset-configs/vrdu_line.py")
    parents = {s: json.loads((vrdu / "line" / f"{s}.json").read_text()) for s in ("train", "test")}
    (vrdu / "ocr").mkdir()
    for s in parents.values():
        for r in s:
            (vrdu / "ocr" / f"{r['document_id']}.txt").write_text("CONTRACT\n")
    for split in ("train", "test"):
        p = dict(data_root="data", split=split, error_types=list(LINE.error_types), context_token_limit=None,
                 subsets_path="vrdu/line/{split}_subsets.json", rows_per_stratum=dict.fromkeys({r["cell"] for r in parents[split]}))
        items = rp.build_items(LINE, p, 0, repo)
        for model in ("m1", "m2"):
            write_run(runs, f"{model}-{split}", items, data_root="data", split=split, dataset_config=dataset_config)
    samples = cs.pair_samples(parents["test"], json.loads((vrdu / "line" / "test_subsets.json").read_text()), 0)
    write_clef_run(runs, "clef-run", samples)

    monkeypatch.setattr(ap, "REPO_ROOT", repo)
    monkeypatch.setattr(ap, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(ap.er, "RESULTS_DIR", tmp_path / "results")
    p = {**P, "train_subsets": "data/vrdu/line/train_subsets.json", "test_subsets": "data/vrdu/line/test_subsets.json",
         "train_parents": "data/vrdu/line/train.json", "test_parents": "data/vrdu/line/test.json",
         "models": [{"label": m, "train_run_id": f"{m}-train", "test_run_id": f"{m}-test", "layer": 5, "task": "relation_pair_detection"}
                    for m in ("m1", "m2")],
         "error_types": ["intra_document", "cross_item"], "drop_field_free_mismatch": False, "comparisons": [],
         "judges": [{"label": "clef", "judge": "clef", "run_id": "clef-run"}],
         "prefix_match_rtol": 1e-6, "ci_level": 0.9, "run_controls": True, "n_train_label_shuffles": 3,
         "train_label_shuffle_auroc_tolerance": 0.5, "max_docs": None,
         "report": {"n_boot": 20, "methods": {"m1": "m1", "m2": "m2", "clef": "clef"},
                    "error_type_names": {"intra_document": "Intra-document", "cross_item": "Cross-item"}, "font": "DejaVu Serif",
                    "white_frac": 0.35, "mesh_pts": 21, "alpha_floor": 0.15, "band_alpha": 0.35, "alpha_levels": 4,
                    "shuffled_auroc_tolerance": 0.5, "figsize": [3.0, 3.0]}}
    m = ap.run({"id": "r1", "seed": 0, "params": p}, runs)
    assert m["n_common_test_pairs"] == m["n_selected_test_pairs"] == len(samples)   # every corrupted-parent test pair
    assert all(m[f"{k}_n_dropped_pairs"] == 0 for k in ("m1", "m2", "clef"))
    assert m["test_baseline_fields_auroc"] == pytest.approx(0.5, abs=1e-12)
    assert m["m1_test_auroc"] == m["m2_test_auroc"] == 1.0 and m["clef_test_auroc"] == 1.0
    assert m["m1_prefix_match_dev"] == 0.0 and m["m1_prefix_match_n"] > 0 and m["m1_control_seed_determinism"] == 1.0
    assert m["m1_n"] == 2 * m["m1_n_valid"] and m["test_n_balanced_rows"] == m["m1_n"]
    assert json.loads((runs / "r1" / "metrics.json").read_text()) == m
    assert (tmp_path / "figures" / "r1" / "calibration_table.tex").is_file()

    # a judge that saw one pair the other way round is refused
    write_clef_run(runs, "clef-flipped", samples, flip=0)
    with pytest.raises(AssertionError, match="primary"):
        ap.run({"id": "r2", "seed": 0, "params": {**p, "judges": [{"label": "clef", "judge": "clef", "run_id": "clef-flipped"}]}}, runs)
    # a model spec naming the other prompt variant is refused
    with pytest.raises(AssertionError, match="task"):
        ap.run({"id": "r3", "seed": 0, "params": {**p, "models": [{**p["models"][0], "task": "relation_pair_values"}]}}, runs)


# ---------------------------------------------------------------- field-free labels, intra-only, prompt variants
def _overlap_repo(tmp_path, monkeypatch):
    """A synthetic line build whose item li's end date is item li+1's start date (so a swapped-in start date can be
    the shown item's own end date: invalid, but field-free valid)."""
    import test_build_vrdu_line as tl

    repo = tmp_path / "repo"
    vrdu = repo / "data" / "vrdu"
    vrdu.mkdir(parents=True)
    monkeypatch.setattr(tl, "make_base", lambda split: [tl.item(f"{tl.PREFIX[split]}{d}", li, program_end_date=f"01/0{li + 2}/20")
                                                        for d in "ABCDEFGH" for li in range(6)])
    tl.b.main(["--config", str(tl.write_root(vrdu, {**tl.LINE, "n_per_cell": {"train": 8, "test": 8}}))])
    (vrdu / "ocr").mkdir()
    for split in ("train", "test"):
        for r in json.loads((vrdu / "line" / f"{split}.json").read_text()):
            (vrdu / "ocr" / f"{r['document_id']}.txt").write_text("CONTRACT\n")
    return repo


def test_field_free_valid_finds_the_values_of_one_item_under_other_fields(tmp_path, monkeypatch):
    repo = _overlap_repo(tmp_path, monkeypatch)
    line = repo / "data" / "vrdu" / "line"
    fr = ap.load_pairs(line / "train_subsets.json", line / "train.json")
    free = ap.field_free_valid(fr, line / "train_subsets.json", line / "train.json")
    assert (free | ~fr["y"]).all() and free[fr["parent_valid"]].all()
    base = json.loads((line / "train_base.json").read_text())
    rows = {r["subset_id"]: r for r in json.loads((line / "train_subsets.json").read_text()) if r["size"] == 2}
    mismatch = np.flatnonzero(free != fr["y"])
    assert len(mismatch) > 0, "the overlapping dates should make some invalid pair field-free valid"
    for i in mismatch:   # brute force on the raw strings (the synthetic values have one spelling each)
        r = rows[fr["subset_id"][i]]
        vals = {r[f] for f in r["fields"]}
        assert not fr["y"][i] and any(vals <= {v for k, v in it.items() if k in ap.rr.LINE_FIELDS}
                                      for it in base if it["document_id"] == r["document_id"])


def test_run_intra_only_with_two_prompt_variants_and_dropped_mismatches(tmp_path, monkeypatch):
    repo = _overlap_repo(tmp_path, monkeypatch)
    runs = tmp_path / "runs"
    dataset_config = str(REPO_ROOT / "experiments/dataset-configs/vrdu_line.py")
    for split in ("train", "test"):
        cells = {r["cell"] for r in json.loads((repo / "data/vrdu/line" / f"{split}.json").read_text()) if r["cell"].startswith("intra")}
        p = dict(data_root="data", split=split, error_types=["intra_document"], context_token_limit=None,
                 subsets_path="vrdu/line/{split}_subsets.json", rows_per_stratum=dict.fromkeys(cells))
        write_run(runs, f"full-{split}", rp.build_items(LINE, p, 0, repo), data_root="data", split=split, dataset_config=dataset_config)
        write_run(runs, f"vals-{split}", rp.build_values_items(LINE, p, 0, repo), data_root="data", split=split,
                  dataset_config=dataset_config, task="relation_pair_values")

    monkeypatch.setattr(ap, "REPO_ROOT", repo)
    monkeypatch.setattr(ap, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(ap.er, "RESULTS_DIR", tmp_path / "results")
    p = {**P, "train_subsets": "data/vrdu/line/train_subsets.json", "test_subsets": "data/vrdu/line/test_subsets.json",
         "train_parents": "data/vrdu/line/train.json", "test_parents": "data/vrdu/line/test.json",
         "models": [{"label": "full", "train_run_id": "full-train", "test_run_id": "full-test", "layer": 5, "task": "relation_pair_detection"},
                    {"label": "vals", "train_run_id": "vals-train", "test_run_id": "vals-test", "layer": 5, "task": "relation_pair_values"}],
         "judges": [], "error_types": ["intra_document"], "drop_field_free_mismatch": True, "comparisons": [["vals", "full"]],
         "prefix_match_rtol": 1e-6, "ci_level": 0.9, "run_controls": True, "n_train_label_shuffles": 3,
         "train_label_shuffle_auroc_tolerance": 0.5, "max_docs": None,
         "report": {"n_boot": 20, "methods": {"full": "full", "vals": "values"}, "error_type_names": {"intra_document": "Intra-document"},
                    "font": "DejaVu Serif", "white_frac": 0.35, "mesh_pts": 21, "alpha_floor": 0.15, "band_alpha": 0.35, "alpha_levels": 4,
                    "shuffled_auroc_tolerance": 0.5, "figsize": [3.0, 3.0]}}
    m = ap.run({"id": "v1", "seed": 0, "params": p}, runs)
    line = repo / "data/vrdu/line"
    fr = ap.load_pairs(line / "test_subsets.json", line / "test.json")
    intra = ~fr["parent_valid"] & (fr["error_type"] == "intra_document")
    free = ap.field_free_valid(fr, line / "test_subsets.json", line / "test.json")
    assert m["n_field_free_mismatch_test"] == int((intra & (free != fr["y"])).sum())
    assert m["n_selected_test_pairs"] == int(intra.sum()) - m["n_field_free_mismatch_test"]
    assert m["full_test_auroc"] == m["vals_test_auroc"] == 1.0
    assert m["vals_minus_full_test_auroc"] == 0.0
    assert m["vals_minus_full_test_auroc_ci_lo"] == pytest.approx(0.0, abs=1e-12) and m["vals_minus_full_test_auroc_ci_hi"] == pytest.approx(0.0, abs=1e-12)
    ev = np.load(runs / "v1" / "artifacts" / "full_test_predictions.npz")
    sel = {fr["subset_id"][i] for i in np.flatnonzero(intra & (free == fr["y"]))}
    assert set(ev["subset_id"].tolist()) <= sel   # intra-document, no mismatch
    # without the drop, the mismatches are kept (as invalid pairs)
    m2 = ap.run({"id": "v2", "seed": 0, "params": {**p, "drop_field_free_mismatch": False}}, runs)
    assert m2["n_selected_test_pairs"] == int(intra.sum())
