"""Unit tests for analysis/calibration_report.py on hand-built evaluation sets."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import relplot

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analysis_calibration_report", ROOT / "analysis" / "calibration_report.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)

REP = {"n_boot": 50, "methods": {"m1": "one", "m2": "two"}, "error_type_names": {"a_type": "Type A", "b_type": "Type B"},
       "font": "DejaVu Serif", "white_frac": 0.3, "mesh_pts": 21, "min_density_frac": 0.02, "shuffled_auroc_tolerance": 0.1,
       "figsize": [3.0, 3.0]}
ETS = ["a_type", "b_type"]


def synthetic(n_docs=40, per_doc=20, seed=0) -> dict:
    """Two error types; m1 is informative (probs follow the label), m2 is noise."""
    rng = np.random.default_rng(seed)
    n = n_docs * per_doc
    y = rng.random(n) < 0.5
    f1 = np.clip(np.where(y, 0.75, 0.25) + rng.normal(0, 0.15, n), 0.01, 0.99)
    return {"y": y, "docs": np.repeat([f"d{i}" for i in range(n_docs)], per_doc),
            "error_type": np.where(np.arange(n) % 2 == 0, "a_type", "b_type"),
            "probs": {"m1": f1, "m2": rng.random(n)}}


def test_score_sample_hand_worked():
    y = np.array([True, False, True, False])
    f = np.array([0.9, 0.2, 0.6, 0.4])
    sc, sigma = cr.score_sample(y, f)
    # Brier = (0.01 + 0.04 + 0.16 + 0.16) / 4 = 0.0925; reference (rate 0.5) = 0.25 -> BSS = 1 - 0.37
    assert sc["bss"] == pytest.approx(0.63)
    assert sc["auroc"] == 1.0                       # both valid items outrank both invalid ones
    assert sc["smece"] == pytest.approx(relplot.smECE(f, y.astype(float)))
    assert sigma > 0


def test_score_sample_needs_both_classes():
    with pytest.raises(AssertionError):
        cr.score_sample(np.ones(3, bool), np.array([0.1, 0.5, 0.9]))


def test_doc_draws_keep_whole_documents():
    docs = np.array(["a", "b", "a", "c", "b", "a"])
    members, draws = cr.doc_draws(docs, 7, 0)
    assert [m.tolist() for m in members] == [[0, 2, 5], [1, 4], [3]]   # documents in sorted order, rows in file order
    assert draws.shape == (7, 3) and draws.min() >= 0 and draws.max() < 3
    idx = np.concatenate([members[j] for j in [0, 0, 2]])              # a, a, c
    assert docs[idx].tolist() == ["a", "a", "a", "a", "a", "a", "c"]
    again = cr.doc_draws(docs, 7, 0)[1]
    assert np.array_equal(draws, again)                                 # seed determinism


def test_split_report_counts_and_alerts():
    ev = synthetic()
    counts = cr.split_report(ev, ETS)
    assert counts["n_a_type"] + counts["n_b_type"] == len(ev["y"])
    assert counts["n_valid_a_type"] + counts["n_invalid_a_type"] == counts["n_a_type"]
    with pytest.raises(AssertionError, match="error types"):
        cr.split_report(ev, ["a_type", "b_type", "c_type"])
    one_class = {**ev, "y": ev["y"] | (ev["error_type"] == "b_type")}   # every b_type row valid
    with pytest.raises(AssertionError, match="b_type"):
        cr.split_report(one_class, ETS)


def test_check_evset_rejects_wrong_methods_and_range():
    ev = synthetic()
    with pytest.raises(AssertionError, match="methods"):
        cr.check_evset(ev, ["m1"])
    bad = {**ev, "probs": {**ev["probs"], "m2": ev["probs"]["m2"] + 1.0}}
    with pytest.raises(AssertionError, match="outside"):
        cr.check_evset(bad, ["m1", "m2"])


def test_assert_matches_existing():
    ev = synthetic()
    existing = {}
    for label, f in ev["probs"].items():
        sc, _ = cr.score_sample(ev["y"], f)
        existing.update({f"{label}_test_{m}": v for m, v in sc.items()})
    cr.assert_matches_existing(ev, existing)
    existing["m1_test_bss"] += 1e-6
    with pytest.raises(AssertionError, match="bss"):
        cr.assert_matches_existing(ev, existing)


def test_analysis_known_answers():
    ev = synthetic()
    res = cr.analyse(ev, REP, ETS, 0.95, seed=0)
    for et in ETS:
        r = res[et]
        m1, m2 = r["methods"]["m1"], r["methods"]["m2"]
        for m in cr.METRICS:                                            # point estimate inside its interval
            assert m1[m][1] <= m1[m][0] <= m1[m][2], (et, m)
        assert m1["auroc"][0] > 0.95 and m1["auroc"][1] > 0.9           # informative probe
        assert abs(m2["auroc"][0] - 0.5) < 0.1                          # noise
        assert m1["bss"][0] > m2["bss"][0]
        assert abs(r[cr.SHUFFLED]["auroc"][0] - 0.5) < 0.1              # shuffled-label baseline is chance
        assert r[cr.SHUFFLED]["bss"][0] < m1["bss"][0]
        c = m1["curve"]
        assert (c["lo"] <= c["hi"]).all() and c["supported"].any()
        sup = c["supported"]
        assert (c["mu"][sup] >= -1e-6).all() and (c["mu"][sup] <= 1 + 1e-6).all()
        assert r["n"] == int((ev["error_type"] == et).sum())


def test_analysis_is_seed_deterministic():
    ev = synthetic()
    a, b = cr.analyse(ev, REP, ETS, 0.95, seed=3), cr.analyse(ev, REP, ETS, 0.95, seed=3)
    for et in ETS:
        assert a[et]["methods"]["m1"]["auroc"] == b[et]["methods"]["m1"]["auroc"]
        assert a[et][cr.SHUFFLED]["smece"] == b[et][cr.SHUFFLED]["smece"]
    c = cr.analyse(ev, REP, ETS, 0.95, seed=4)
    assert a["a_type"]["methods"]["m1"]["auroc"][1:] != c["a_type"]["methods"]["m1"]["auroc"][1:]   # another seed, other resamples


def test_methods_share_resamples():
    """A method whose probabilities are another's exactly has identical intervals (same resamples)."""
    ev = synthetic()
    ev["probs"]["m2"] = ev["probs"]["m1"].copy()
    res = cr.analyse(ev, REP, ETS, 0.95, seed=0)
    assert res["a_type"]["methods"]["m1"]["bss"] == res["a_type"]["methods"]["m2"]["bss"]


def test_pastel_mixes_with_white():
    assert cr.pastel((0.0, 0.0, 0.0), 0.5) == (0.5, 0.5, 0.5)
    assert cr.pastel((1.0, 1.0, 1.0), 0.3) == (1.0, 1.0, 1.0)
    with pytest.raises(AssertionError):
        cr.pastel("red", 1.0)


def test_write_report_files_and_table_shape(tmp_path):
    ev = synthetic()
    fig_dir, out_dir = tmp_path / "fig", tmp_path / "out"
    fig_dir.mkdir()
    out_dir.mkdir()
    counts = cr.write_report(ev, REP, ETS, 0.95, 0, fig_dir, out_dir, "calibration", "Extra.")
    assert counts["calibration_n_a_type"] == int((ev["error_type"] == "a_type").sum())
    for name in ("calibration_a_type.pdf", "calibration_b_type.pdf", "calibration_legend.pdf", "calibration_results.csv", "calibration_table.tex"):
        assert (fig_dir / name).stat().st_size > 0, name
        assert (out_dir / name).exists() or name.endswith(".pdf"), name
    tex = (out_dir / "calibration_table.tex").read_text()
    assert tex.count(r"\multicolumn{3}{c}") == 2 and tex.count(r"\cmidrule") == 2
    assert r"smECE $\downarrow$" in tex and r"BSS $\uparrow$" in tex and r"AUROC $\uparrow$" in tex
    assert "Shuffled labels" in tex and "Type A" in tex and "Type B" in tex and "50 bootstrap" in tex
    body = [ln for ln in tex.splitlines() if ln.startswith(("one", "two", "Shuffled"))]
    assert len(body) == 3 and all(ln.count("[") == 6 for ln in body)    # 2 types x 3 metrics, each with an interval
    rows = list(csv.DictReader((out_dir / "calibration_results.csv").open()))
    assert len(rows) == 2 * (2 + 1 + 2) * 3                             # types x (methods + shuffled mean + shuffled per method) x metrics
    assert {r["method"] for r in rows} == {"one", "two", "shuffled", "shuffled:one", "shuffled:two"}
    assert all(float(r["ci_lo"]) <= float(r["point"]) + 1e-12 or r["method"].startswith("shuffled") or r["metric"] != "auroc" for r in rows)
    assert (fig_dir / "calibration_results.csv").read_text() == (out_dir / "calibration_results.csv").read_text()


def test_write_report_rejects_other_report_keys(tmp_path):
    with pytest.raises(AssertionError, match="report keys"):
        cr.write_report(synthetic(), {**REP, "extra": 1}, ETS, 0.95, 0, tmp_path, tmp_path, "c", "")


def test_missing_font_is_an_error():
    with pytest.raises(Exception):
        cr.set_style("No Such Font 123")


def test_evset_roundtrip(tmp_path):
    ev = synthetic()
    cr.save_evset(tmp_path / "p.npz", ev, control=np.ones(len(ev["y"]), bool))
    got, extra = cr.load_evset(tmp_path / "p.npz")
    assert np.array_equal(got["y"], ev["y"]) and np.array_equal(got["docs"], ev["docs"])
    assert sorted(got["probs"]) == ["m1", "m2"] and np.array_equal(got["probs"]["m1"], ev["probs"]["m1"])
    assert list(extra) == ["control"]
