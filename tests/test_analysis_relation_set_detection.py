"""Unit tests for analysis/relation_set_detection.py on a hand-built line fixture (no model, no run on disk
except the synthetic ones the tests write).

Doc ``d`` (fields shown: CH, DESC, ED; the two other line fields are never shown) has three items:
    li0 = (7, Movie, 01/10/20)   li1 = (8, News, 01/31/20)   li2 = (8, Movie, 02/10/20)
Two pairs, each an invalid parent and its clean twin:
    p1: li0 with ED <- "01/31/20" (li1's)   -> invalid parent (7, Movie, 01/31/20)
    p2: li2 with CH <- "7"        (li0's)   -> invalid parent (7, Movie, 02/10/20)
Subsets (size 2-3) of the invalid parents, by the label rule:
    p1: (CH,DESC)=(7,Movie) is li0's -> valid, no corrupted member, ED (corrupted) is shown after both -> hidden
        (CH,ED), (DESC,ED), (CH,DESC,ED) -> invalid
    p2: (CH,DESC)=(7,Movie) is li0's -> valid despite the corrupted CH -> corrupted_member
        (DESC,ED)=(Movie,02/10/20) is li2's -> valid, no corrupted member, CH shown before -> visible
        (CH,ED), (CH,DESC,ED) -> invalid
Every subset of a clean parent is valid. 16 rows: 5 invalid, 11 valid (8 clean_parent, 1 each corrupted_member,
visible, hidden; 3 context rows in all).
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import zlib
from pathlib import Path

import numpy as np
import pytest
import yaml

import test_analysis_clef as tac
from test_analysis_relation_detection import EQ, KINDS, rr
from vrdu_common import item_keys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analysis_relation_set_detection", ROOT / "analysis" / "relation_set_detection.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)

CH, DESC, SD, ED, AMT = rr.LINE_FIELDS
LF = rr.LINE_FIELDS
SHOWN = [CH, DESC, ED]
_ITEMS = [{CH: "7", DESC: "Movie", SD: None, ED: "01/10/20", AMT: None},
          {CH: "8", DESC: "News", SD: None, ED: "01/31/20", AMT: None},
          {CH: "8", DESC: "Movie", SD: None, ED: "02/10/20", AMT: None}]
ITEMS = {"d": _ITEMS, "dtr": _ITEMS}   # the test document and the train document have the same items
P1_CLEAN, P1_BAD = {CH: "7", DESC: "Movie", ED: "01/10/20"}, {CH: "7", DESC: "Movie", ED: "01/31/20"}
P2_CLEAN, P2_BAD = {CH: "8", DESC: "Movie", ED: "02/10/20"}, {CH: "7", DESC: "Movie", ED: "02/10/20"}


def parent(doc, pid, valid, values, corrupt, cell):
    return {"sample_id": f"{pid}|{'valid' if valid else 'invalid'}", "pair_id": pid, "document_id": doc, "cell": cell, "valid": valid,
            "fields": [f for f in LF if f in values], "invalid_fields": corrupt, **{f: values.get(f) for f in LF},
            "field_errors": {f: ({"error_type": "intra_document"} if f in corrupt else None) for f in LF}}


def parents_of(doc):
    return [parent(doc, f"{doc}-p1", False, P1_BAD, [ED], "intra_document:k1"), parent(doc, f"{doc}-p1", True, P1_CLEAN, [], "intra_document:k1"),
            parent(doc, f"{doc}-p2", False, P2_BAD, [CH], "intra_document:k1"), parent(doc, f"{doc}-p2", True, P2_CLEAN, [], "intra_document:k1")]


def subsets_of(parents, items=ITEMS):
    keys = {d: item_keys(its, LF, EQ) for d, its in items.items()}
    rows = []
    for par in parents:
        for k in range(2, len(par["fields"]) + 1):
            for members in itertools.combinations(par["fields"], k):
                inv = [f for f in members if f in par["invalid_fields"]]
                rows.append({"subset_id": f"{par['sample_id']}|{'+'.join(members)}", "parent_id": par["sample_id"], "pair_id": par["pair_id"],
                             "document_id": par["document_id"], "cell": par["cell"], "parent_valid": par["valid"], "fields": list(members),
                             "size": k, "valid": rr.is_item_pair({f: par[f] for f in members}, keys[par["document_id"]], EQ),
                             "error_type": "intra_document" if inv else "", "invalid_fields": inv, "num_invalid_fields": len(inv),
                             **{f: par[f] if f in members else None for f in LF}})
    return rows


def write_files(root: Path, split: str, doc: str) -> tuple[Path, Path]:
    (root / "line").mkdir(exist_ok=True, parents=True)
    par = parents_of(doc)
    (root / "line" / f"{split}.json").write_text(json.dumps(par))
    (root / "line" / f"{split}_subsets.json").write_text(json.dumps(subsets_of(par)))
    return root / "line" / f"{split}_subsets.json", root / "line" / f"{split}.json"


@pytest.fixture(autouse=True)
def no_build_check(monkeypatch):
    monkeypatch.setattr(rs.rr, "check_build", lambda *a, **k: None)


def frame(tmp_path, split="test", doc="d"):
    return rs.load_subsets(*write_files(tmp_path, split, doc))


def sid(pid, valid, *members, doc="d"):
    return f"{doc}-{pid}|{'valid' if valid else 'invalid'}|{'+'.join(members)}"


# ------------------------------------------------------------------------ subsets frame
def test_load_subsets_hand_worked(tmp_path):
    f = frame(tmp_path)
    assert len(f["subset_id"]) == 16 and int(f["y"].sum()) == 11
    kinds = dict(zip(f["subset_id"].tolist(), f["kind"].tolist()))
    assert kinds[sid("p1", False, CH, DESC)] == "hidden"
    assert kinds[sid("p2", False, CH, DESC)] == "corrupted_member"
    assert kinds[sid("p2", False, DESC, ED)] == "visible"
    assert sum(k == "invalid" for k in kinds.values()) == 5
    assert sum(k == "clean_parent" for k in kinds.values()) == 8
    assert not f["y"][f["index"][sid("p1", False, CH, ED)]] and f["y"][f["index"][sid("p1", True, CH, ED)]]
    assert f["n_corrupted"][f["index"][sid("p1", False, CH, DESC, ED)]] == 1 and f["fields"][0] == f"{CH},{DESC}"
    assert set(f["error_type"].tolist()) == {"intra_document"} and set(f["size"].tolist()) == {2, 3}


def test_load_subsets_refuses_inconsistent_rows(tmp_path):
    sub_path, par_path = write_files(tmp_path, "test", "d")
    rows = json.loads(sub_path.read_text())
    bad = [{**r, "fields": list(reversed(r["fields"]))} if r["subset_id"] == sid("p1", False, CH, DESC) else r for r in rows]
    sub_path.write_text(json.dumps(bad))
    with pytest.raises(AssertionError, match="shown order"):
        rs.load_subsets(sub_path, par_path)
    bad = [{**r, "invalid_fields": []} if r["subset_id"] == sid("p1", False, CH, ED) else r for r in rows]
    sub_path.write_text(json.dumps(bad))
    with pytest.raises(AssertionError, match="corrupted fields are not the parent's"):
        rs.load_subsets(sub_path, par_path)
    sub_path.write_text(json.dumps(rows + rows[:1]))
    with pytest.raises(AssertionError, match="not unique"):
        rs.load_subsets(sub_path, par_path)


def test_cut_adds_the_clean_parent_twin(tmp_path):
    f = frame(tmp_path)
    ev = rs.cut(f, np.arange(16))
    for p in ("p1", "p2"):
        for members in ((CH, DESC), (DESC, ED), (CH, DESC, ED)):
            i = ev["index"][sid(p, False, *members)]
            if ev["kind"][i] in rs.CONTEXT_KINDS:
                assert ev["subset_id"][ev["twin"][i]] == sid(p, True, *members)
    assert (ev["twin"] >= 0).sum() == 3 and (ev["twin"][ev["kind"] == "invalid"] == -1).all()
    drop_twin = np.array([i for i in range(16) if f["subset_id"][i] != sid("p1", True, CH, DESC)])
    with pytest.raises(AssertionError, match="twin is not among the rows"):
        rs.cut(f, drop_twin)


# ------------------------------------------------------------------------ probe features
def span_content(group_values: list[str], pos: int) -> np.ndarray:
    """A causal activation: a function of the field position and every value up to and including this one."""
    seed = zlib.crc32("|".join([str(pos), *group_values[: pos + 1]]).encode())
    return np.random.default_rng(seed).standard_normal(4).astype(np.float32)


def make_split(parents, with_x=True):
    cols = {c: [] for c in ("doc_ids", "row_ordinal", "field", "field_position", "value", "label_valid", "row_error_type", "pair_id", "cell")}
    xs, ordinal = [], {}
    for par in parents:
        o = ordinal[par["document_id"]] = ordinal.get(par["document_id"], -1) + 1
        vals = [par[f] for f in par["fields"]]
        for pos, f in enumerate(par["fields"]):
            for c, v in (("doc_ids", par["document_id"]), ("row_ordinal", o), ("field", f), ("field_position", pos + 1), ("value", par[f]),
                         ("label_valid", f not in par["invalid_fields"]), ("row_error_type", "" if par["valid"] else "intra_document"),
                         ("pair_id", par["pair_id"]), ("cell", par["cell"])):
                cols[c].append(v)
            xs.append(span_content(vals, pos))
    out = {k: np.array(v) for k, v in cols.items()}
    out["X"] = np.stack(xs)
    return out


def make_data(parents):
    return {"rows": {(r["pair_id"], r["valid"]): r for r in parents}, "eq": EQ,
            "item_keys": {d: item_keys(its, LF, EQ) for d, its in ITEMS.items()}}


def test_lm_member_spans_and_features_hand_worked(tmp_path):
    f, par = frame(tmp_path), parents_of("d")
    s = make_split(par)
    idx, rows = rs.lm_member_spans(f, s, make_data(par), require_all=True)
    assert len(idx) == 16
    i = f["index"][sid("p2", False, CH, ED)]
    spans = rows[i]   # f's rows are in file order and all are scored, so positions coincide
    assert [s["field"][j] for j in spans] == [CH, ED] and [s["value"][j] for j in spans] == ["7", "02/10/20"]
    assert set(s["pair_id"][spans].tolist()) == {"d-p2"} and not s["row_error_type"][spans[0]] == ""
    x = s["X"]
    np.testing.assert_allclose(rs.subset_features(x, [spans], "sum")[0], x[spans[0]] + x[spans[1]])
    np.testing.assert_allclose(rs.subset_features(x, [spans], "last")[0], x[spans[1]])
    np.testing.assert_allclose(rs.subset_features(x, [spans], "context")[0], x[spans[0]])
    full = rows[f["index"][sid("p2", False, CH, DESC, ED)]]
    np.testing.assert_allclose(rs.subset_features(x, [full], "context")[0], x[full[0]] + x[full[1]])
    with pytest.raises(AssertionError, match="unknown feature"):
        rs.subset_features(x, [spans], "mean")


def test_lm_member_spans_refuses_disagreements(tmp_path):
    f, par = frame(tmp_path), parents_of("d")
    s, data = make_split(par), make_data(par)
    s["value"][0] = "zz"
    with pytest.raises(AssertionError, match="row file"):
        rs.lm_member_spans(f, s, data, require_all=True)
    flipped = {**f, "rows": [{**r, "valid": not r["valid"]} if r["subset_id"] == sid("p1", True, CH, ED) else r for r in f["rows"]]}
    with pytest.raises(AssertionError, match="label rule disagrees"):
        rs.lm_member_spans(flipped, make_split(par), data, require_all=True)
    only_p1 = [p for p in par if p["pair_id"] == "d-p1"]
    with pytest.raises(AssertionError, match="not judged whole|parent is not in the run"):
        rs.lm_member_spans(f, make_split(only_p1), make_data(only_p1), require_all=True)
    idx, _ = rs.lm_member_spans(f, make_split(only_p1), make_data(only_p1), require_all=False)
    assert len(idx) == 8   # only p1's rows have their parent in the run


def test_prefix_match_known_answer(tmp_path):
    f, par = frame(tmp_path), parents_of("d")
    s = make_split(par)
    idx, rows = rs.lm_member_spans(f, s, make_data(par), require_all=True)
    ev = rs.cut(f, idx)
    feats = rs.subset_features(s["X"], rows, "sum")
    dev, exact = rs.prefix_match_dev(feats, ev)
    assert dev == 0.0 and exact == 1.0   # causal content: a hidden row's members see the twin's prefix
    s2 = make_split(par)
    j = rows[f["index"][sid("p1", False, CH, DESC)]][0]
    s2["X"][j] += 1.0
    dev, exact = rs.prefix_match_dev(rs.subset_features(s2["X"], rows, "sum"), ev)
    assert dev > 0.1 and exact == 0.0
    no_hidden = rs.cut(f, np.array([i for i in range(16) if f["kind"][i] != "hidden"]))
    with pytest.raises(AssertionError, match="no hidden row"):
        rs.prefix_match_dev(feats[:15], no_hidden)


# ------------------------------------------------------------------------ metrics
def test_row_stratum_report_known_answer():
    y = np.array([True, True, False, False, True, True])
    probs = np.array([0.9, 0.2, 0.1, 0.7, 0.8, 0.6])
    r = rs.row_stratum_report(y, probs, 0.5, np.array(["a", "a", "a", "a", "b", "b"]))
    assert (r["n_a"], r["n_b"]) == (4, 2) and r["auroc_a"] == pytest.approx(3 / 4)
    assert r["detect_a"] == 0.5 and r["valid_acc_a"] == 0.5 and r["accuracy_a"] == 0.5 and r["mean_p_valid_a"] == pytest.approx(0.475)
    assert r["valid_rate_b"] == 1.0 and r["accuracy_b"] == 1.0 and r["valid_acc_b"] == 1.0
    assert "auroc_b" not in r and "detect_b" not in r   # one class: undefined, not a default


P = dict(tac.P, lr_C=1.0, lr_max_iter=1000, lr_tol=1e-4)


def test_score_method_hand_worked(tmp_path):
    ev = rs.cut(frame(tmp_path), np.arange(16))
    probs = np.where(ev["y"], 0.9, 0.1)
    m = rs.score_method("m", ev, probs, P, 0)
    assert m["n"] == 16 and m["valid_rate"] == 11 / 16 and m["test_auroc"] == 1.0 and m["test_accuracy"] == 1.0
    assert m["test_brier"] == pytest.approx(0.01)
    assert m["test_brier_ref"] == pytest.approx(11 / 16 * (5 / 16) ** 2 + 5 / 16 * (11 / 16) ** 2)
    assert m["test_bss"] == pytest.approx(1 - 0.01 / m["test_brier_ref"])
    assert m["test_n_hidden"] == 1 and m["test_valid_acc_hidden"] == 1.0 and "test_auroc_hidden" not in m
    assert m["test_n_invalid"] == 5 and m["test_detect_invalid"] == 1.0 and m["test_n_size3"] == 4 and m["test_n_parent_clean"] == 8
    assert m["test_context_hidden_mean_p_valid"] == 0.9 and m["test_context_visible_twin_mean_p_valid"] == 0.9
    assert m["test_twin_vs_context_corrupted_member_auroc"] == 0.5   # tied: the method cannot tell a corrupted-member row from its twin
    wrong = np.where(ev["y"], 0.1, 0.9)   # perfectly wrong ranks
    assert rs.score_method("w", ev, wrong, P, 0)["test_auroc"] == 0.0


def test_score_method_needs_both_classes_and_has_no_permutation_null(tmp_path):
    ev = rs.cut(frame(tmp_path), np.arange(16))
    assert not [k for k in rs.score_method("m", ev, np.where(ev["y"], 0.9, 0.1), P, 0) if k.startswith("control_")]   # the report's shuffle is the control
    with pytest.raises(AssertionError, match="one class missing"):
        rs.score_method("m", rs.cut(ev, np.flatnonzero(ev["y"])), np.full(11, 0.5), P, 0)


def test_structure_baseline_reports_what_fields_alone_predict(tmp_path):
    tr, te = frame(tmp_path, "train", "dtr"), frame(tmp_path, "test", "d")
    ev = rs.cut(te, np.arange(16))
    m = rs.structure_baseline(tr, ev, P, 0)
    # (CH,ED) and (DESC,ED), (CH,DESC,ED) are invalid in 1 of 4 rows of each fields group, (CH,DESC) in 0 of 4: not constant
    assert 0.0 <= m["test_baseline_fields_auroc"] <= 1.0 and m["test_baseline_fields_auroc"] != 0.5


def test_doc_bootstrap_auroc_diff():
    y = np.array([False, True] * 4)
    docs = np.repeat(["a", "b", "c", "d"], 2)
    good, bad = y.astype(float), 1.0 - y.astype(float)
    assert rs.doc_bootstrap_auroc_diff(y, good, bad, docs, 50, 0.95, 0) == (1.0, 1.0)
    assert rs.doc_bootstrap_auroc_diff(y, good, good, docs, 50, 0.95, 0) == (0.0, 0.0)


# ------------------------------------------------------------------------ judges
def write_judge(root: Path, run_id: str, judge: str, ids, p_true, label, task, samples_path="x/conditional-test-subsets.json", over=None,
                doc="d", cell=None, size=None):
    d = root / run_id
    (d / "artifacts").mkdir(parents=True)
    n = len(ids)
    cols = dict(p_true=np.asarray(p_true, dtype=np.float64), label_valid=np.asarray(label), sample_id=np.array(ids),
                doc_ids=np.array([doc] * n), cell=np.array(cell if cell is not None else ["intra_document:k1"] * n),
                size=np.array(size if size is not None else [2] * n))
    if judge == "clef":
        cols.update(logit_true=np.log(cols["p_true"]), logit_false=np.log(1 - cols["p_true"]))
    np.savez_compressed(d / "artifacts" / f"{judge}_predictions.npz", **cols)
    (d / "metrics.json").write_text(json.dumps({"n_predictions": n, "n_valid": int(np.sum(label)), "n_excluded_overlong": 3,
                                                "determinism_max_abs_diff": 0.02, **(over or {})}))
    (d / "run.json").write_text(json.dumps({"status": "success", "task": task}))
    (d / "config.snapshot.yaml").write_text(yaml.safe_dump({"params": {"samples_path": samples_path}}))


def judge_cols(f, ids):
    i = [f["index"][s] for s in ids]
    return dict(label=f["y"][i], cell=f["cell"][i].tolist(), size=f["size"][i].tolist())


def test_load_judge_gates(tmp_path):
    f = frame(tmp_path)
    ids = [sid("p1", False, CH, DESC), sid("p1", False, CH, ED), sid("p2", True, DESC, ED)]
    p = [0.9, 0.2, 0.8]
    c = judge_cols(f, ids)
    write_judge(tmp_path, "clef", "clef", ids, p, c["label"], "clef_relation_conditional", cell=c["cell"], size=c["size"])
    r = rs.load_judge({"judge": "clef", "kind": "conditional", "run_id": "clef"}, tmp_path, f)
    assert r["idx"].tolist() == [f["index"][s] for s in ids] and r["probs"].tolist() == p and r["n_excluded"] == 3 and r["extra"] == {}
    write_judge(tmp_path, "jev", "jev", ids, p, c["label"], "jev_relation_conditional", cell=c["cell"], size=c["size"], over={"n_excluded_overlong": 0})
    r = rs.load_judge({"judge": "jev", "kind": "conditional", "run_id": "jev"}, tmp_path, f)
    assert r["n_excluded"] == 0 and r["extra"]["determinism_max_abs_diff"] == 0.02 and r["extra"]["n_distinct_p_true"] == 3

    def bad(run_id, judge, match, **kw):
        write_judge(tmp_path, run_id, judge, **{"ids": ids, "p_true": p, "label": c["label"], "task": f"{judge}_relation_conditional",
                                                "cell": c["cell"], "size": c["size"], **kw})
        with pytest.raises(AssertionError, match=match):
            rs.load_judge({"judge": judge, "kind": "conditional", "run_id": run_id}, tmp_path, f)

    bad("b1", "clef", "not a subsets run", samples_path="x/conditional-test.json")
    bad("b2", "clef", "no test subset's", ids=ids[:2] + ["nope"])
    bad("b3", "clef", "stored label differs", label=~c["label"])
    bad("b4", "clef", "stored size differs", size=[3, 3, 3])
    bad("b5", "jev", "not on the 0.01 grid", p_true=[0.9, 0.205, 0.8])
    bad("b6", "clef", "task", task="clef_entity_row")
    bad("b7", "clef", "n_predictions", over={"n_predictions": 5})


# ------------------------------------------------------------------------ whole analysis on synthetic runs
def write_lm_run(root: Path, run_id: str, parents, layer: int):
    s = make_split(parents)
    n = len(s["field"])
    d = root / run_id
    (d / "artifacts").mkdir(parents=True)
    fet = np.where(s["label_valid"], "", "intra_document")
    np.savez_compressed(
        d / "artifacts" / "span_representations.npz", layers=np.array([0, layer]), rep_0=np.ones((n, 4), np.float32), **{f"rep_{layer}": s["X"]},
        doc_ids=s["doc_ids"], row_ordinal=s["row_ordinal"], field=s["field"], field_position=s["field_position"], label_valid=s["label_valid"],
        field_error_type=fet, row_error_type=s["row_error_type"], row_k=np.ones(n, np.int64), line_index=np.zeros(n, np.int64),
        value=s["value"], pair_id=s["pair_id"], cell=s["cell"])
    (d / "run.json").write_text(json.dumps({"status": "success", "task": "relation_detection"}))
    nv = int(s["label_valid"].sum())
    (d / "metrics.json").write_text(json.dumps({"n_spans": n, "n_valid_spans": nv, "n_invalid_spans": n - nv, "n_spans_intra_document": n - nv}))


def test_run_end_to_end_on_synthetic_runs(tmp_path, monkeypatch):
    runs, repo = tmp_path / "runs", tmp_path / "repo"
    runs.mkdir()
    (repo / "data" / "vrdu").mkdir(parents=True)
    sub_tr, par_tr = write_files(repo / "data" / "vrdu", "train", "dtr")
    sub_te, par_te = write_files(repo / "data" / "vrdu", "test", "d")
    ptr, pte = parents_of("dtr"), parents_of("d")
    for model, layer in (("m", 3),):
        write_lm_run(runs, f"{model}-train", ptr, layer)
        write_lm_run(runs, f"{model}-test", pte, layer)
    f_te = rs.load_subsets(sub_te, par_te)
    ids = f_te["subset_id"].tolist()
    c = judge_cols(f_te, ids)
    write_judge(runs, "clef-run", "clef", ids, np.where(f_te["y"], 0.8, 0.3), f_te["y"], "clef_relation_conditional", cell=c["cell"], size=c["size"])
    write_judge(runs, "jev-run", "jev", ids, np.where(f_te["y"], 0.7, 0.2), f_te["y"], "jev_relation_conditional", cell=c["cell"], size=c["size"],
                over={"n_excluded_overlong": 0})
    monkeypatch.setattr(rs, "REPO_ROOT", repo)
    monkeypatch.setattr(rs.rr, "load_line_data", lambda run_id, runs_root, repo_root: make_data(ptr if "train" in run_id else pte))
    monkeypatch.setattr(er := rs.er, "run_rows_path", lambda run_id, runs_root, repo_root, dataset: par_tr if "train" in run_id else par_te)
    p = {**P, "train_subsets": "data/vrdu/line/train_subsets.json", "test_subsets": "data/vrdu/line/test_subsets.json",
         "train_parents": "data/vrdu/line/train.json", "test_parents": "data/vrdu/line/test.json", "ablations": ["last", "context"],
         "models": [{"label": "m", "train_run_id": "m-train", "test_run_id": "m-test", "layer": 3}],
         "judges": [{"label": "clef", "judge": "clef", "kind": "conditional", "run_id": "clef-run"},
                    {"label": "jev", "judge": "jev", "kind": "conditional", "run_id": "jev-run"}],
         "max_docs": None, "run_controls": False, "report": {"n_boot": 20}, "prefix_match_rtol": 1e-6, "n_train_label_shuffles": 3, "train_label_shuffle_auroc_tolerance": 0.5}
    monkeypatch.setattr(rs, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(rs, "report", lambda *a, **k: {})   # one document, one error type here: the report has its own tests below
    m = rs.run({"id": "r1", "seed": 0, "params": p}, runs)
    assert m["n_common_test_rows"] == 16 and all(m[f"{k}_n_dropped_rows"] == 0 for k in ("m", "clef", "jev"))
    assert m["clef_test_auroc"] == 1.0 and m["jev_test_auroc"] == 1.0 and m["clef_n_excluded_overlong"] == 3
    assert m["m_prefix_match_dev"] == 0.0 and m["m_layer"] == 3 and m["m_n_train"] == 16
    assert "m_test_auroc" in m and "m_ablation_last_test_auroc" in m
    assert "m_test_auroc_minus_context_ci_lo" in m and "test_baseline_fields_auroc" in m
    saved = np.load(runs / "r1" / "artifacts" / "clef_test_predictions.npz")
    assert saved["subset_id"].tolist() == ids and (saved["label_valid"] == f_te["y"]).all()
    assert json.loads((runs / "r1" / "metrics.json").read_text()) == m

    # a judge that scored fewer rows shrinks the common set for everyone, and the drop is reported
    keep = [i for i, s in enumerate(ids) if not s.startswith("d-p2")]
    ids2 = [ids[i] for i in keep]
    c2 = judge_cols(f_te, ids2)
    write_judge(runs, "clef-half", "clef", ids2, np.where(f_te["y"][keep], 0.8, 0.3), f_te["y"][keep], "clef_relation_conditional",
                cell=c2["cell"], size=c2["size"])
    p2 = {**p, "judges": [{"label": "clef", "judge": "clef", "kind": "conditional", "run_id": "clef-half"}]}
    m2 = rs.run({"id": "r2", "seed": 0, "params": p2}, runs)
    assert m2["n_common_test_rows"] == 8 and m2["m_n_dropped_rows"] == 8 and m2["clef_n_dropped_rows"] == 0 and m2["m_n"] == 8


# ----------------------------------------------------------------------------- report and control
def test_parent_size_and_control_mask_hand_counted(tmp_path):
    ev = rs.cut(frame(tmp_path), np.arange(16))
    assert (ev["parent_size"] == 3).all()                      # every parent shows CH, DESC, ED
    rs.check_pairs(ev)                                         # both pairs have an invalid parent and a clean twin
    keep = rs.control_mask(ev)
    # dropped: the strict subsets (size 2) of the two clean parents = 3 each; kept: 8 rows of invalid parents + 2 full clean rows
    assert int((~keep).sum()) == 6 and int(keep.sum()) == 10
    assert (ev["parent_valid"][~keep]).all() and (ev["size"][~keep] == 2).all()
    assert int((ev["parent_valid"] & keep).sum()) == 2 and (ev["size"][ev["parent_valid"] & keep] == 3).all()
    assert sorted(set(ev["kind"][keep].tolist())) == ["clean_parent", "corrupted_member", "hidden", "invalid", "visible"]


def test_control_mask_alerts_when_a_full_row_lost_its_counterpart(tmp_path):
    ev = rs.cut(frame(tmp_path), np.arange(16))
    full_bad = np.flatnonzero(~ev["parent_valid"] & (ev["size"] == 3))[0]
    with pytest.raises(AssertionError, match="not matched by pair"):
        rs.control_mask(_drop(ev, full_bad))


def _drop(ev, i):
    """``ev`` without row ``i`` (no twin needed: only column arrays are used by the control checks)."""
    keep = np.array([j for j in range(len(ev["y"])) if j != i])
    return {c: ev[c][keep] for c in rs.SUBSET_COLS}


def test_check_pairs_alerts_on_a_pair_with_one_parent(tmp_path):
    ev = rs.cut(frame(tmp_path), np.arange(16))
    only_bad = {c: ev[c][~ev["parent_valid"]] for c in rs.SUBSET_COLS}
    with pytest.raises(AssertionError, match="only one of the pair's parents"):
        rs.check_pairs(only_bad)


def _report_ev(n_docs=30, seed=0):
    """Two error types x ``n_docs`` documents; per (doc, type) one pair: a clean full row, a corrupted full row, a
    corrupted strict subset (valid for half the docs), a clean strict subset."""
    rng = np.random.default_rng(seed)
    cols = {c: [] for c in rs.SUBSET_COLS}
    for et in ("intra_document", "cross_item"):
        for d in range(n_docs):
            pid = f"{et}-d{d}"
            for parent_valid, size, y in ((True, 3, True), (False, 3, False), (False, 2, bool(d % 2)), (True, 2, True)):
                for c, v in zip(rs.SUBSET_COLS, (f"{pid}-{parent_valid}-{size}", pid, pid, f"d{d}-{et}", f"{et}:k1", et, parent_valid, size, 3,
                                                 "CH,DESC", 0 if parent_valid else 1, y, "x")):
                    cols[c].append(v)
    ev = {c: np.array(v) for c, v in cols.items()}
    ev["y"], ev["parent_valid"] = ev["y"].astype(bool), ev["parent_valid"].astype(bool)
    return ev


def test_report_writes_both_reports_and_the_control_drops_only_clean_strict_subsets(tmp_path):
    ev = _report_ev()
    rng = np.random.default_rng(1)
    probs = {l: np.clip(np.where(ev["y"], 0.7, 0.3) + rng.normal(0, 0.2, len(ev["y"])), 0.01, 0.99) for l in ("m1", "m2")}
    metrics = {}
    for l, f in probs.items():
        sc, _ = rs.cr.score_sample(ev["y"], f)
        metrics.update({f"{l}_test_{k}": v for k, v in sc.items()})
    p = {"ci_level": 0.95, "report": {"n_boot": 20, "methods": {"m1": "one", "m2": "two"},
                                       "error_type_names": {"intra_document": "Intra-document", "cross_item": "Cross-item"},
                                       "font": "DejaVu Serif", "white_frac": 0.3, "mesh_pts": 21, "alpha_floor": 0.15, "band_alpha": 0.35, "alpha_levels": 12,
                                       "shuffled_auroc_tolerance": 0.2, "figsize": [3.0, 3.0]}}
    fig_dir, out = tmp_path / "fig", tmp_path / "out"
    (out / "artifacts").mkdir(parents=True)
    fig_dir.mkdir()
    m = rs.report(ev, probs, metrics, p, 0, fig_dir, out)
    n_pairs = 2 * 30
    assert m["control_n_dropped_strict_clean_subsets"] == n_pairs and m["control_n_rows"] == 3 * n_pairs
    assert m["calibration_n_intra_document"] == 4 * 30 and m["control_calibration_n_intra_document"] == 3 * 30
    assert m["control_calibration_n_valid_cross_item"] == 30 + 15 and m["control_calibration_n_invalid_cross_item"] == 30 + 15
    for stem in ("calibration", "control_calibration"):
        for name in (f"{stem}_intra_document.pdf", f"{stem}_cross_item.pdf", f"{stem}_legend.pdf"):
            assert (fig_dir / name).stat().st_size > 0, name
        assert (out / f"{stem}_table.tex").exists() and (out / f"{stem}_results.csv").exists()
    assert not list(fig_dir.glob("*.png"))                      # calibration figures only
