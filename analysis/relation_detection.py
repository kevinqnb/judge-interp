"""Row-based relation detection on the line dataset: a validity probe on the concatenated activations of
two shown fields of one sample, scored on minimal pairs.

Usage: ``uv run --python 3.12 --extra analysis python analysis/relation_detection.py analysis/configs/<type>/<id>.yaml``

Reads the ``relation_detection`` train and test runs named in the config, then:

1. Load one layer and subtract each document's mean activation.
2. Join every judged sample to its row in the line file the run judged (by ``pair_id`` and label).
   Each invalid sample has its clean twin (same item, same ordered fields, original values).
3. Label every unordered pair of shown fields with the data's label rule (``vrdu_common.matches_item``
   against the split's line items, ``line/<split>_base.json``). A field pair of an invalid sample that
   is invalid is matched with the same field pair of the twin (valid): train and test sets are these
   matched field pairs, so each (cell, field pair) stratum is exactly 50/50. Feature ``[X[a], X[b]]``
   in canonical field order.
4. Logistic regression, train -> test, with the metrics of ``analysis/entity_detection.py``: overall
   (with a document-bootstrap interval), per error type and per cell.
5. Context diagnostic: field pairs of invalid samples that are still valid (no corrupted value, or
   one that happens to match an item) against the same pairs of the twin; AUROC ~ 0.5 means the
   probe does not just sense "this sample has a corruption somewhere".
6. Controls: structure baseline (cell x field pair one-hot; exactly 0.5 AUROC, asserted),
   shuffled-label AUROC ~ 0.5 (asserted), seed determinism (asserted).

Outputs go to ``$RUNS_ROOT/<id>/`` and ``analysis/figures/<id>/``.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from judge_interp.dataset_spec import load_dataset_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "data" / "vrdu"))
from vrdu_common import check_build, field_kinds, item_keys, make_eq, matches_keys, read_base  # noqa: E402

LINE_FIELDS = list(load_dataset_spec(REPO_ROOT / "experiments" / "dataset-configs" / "vrdu_line.py").fields)

_spec = importlib.util.spec_from_file_location("analysis_entity_detection", Path(__file__).parent / "entity_detection.py")
er = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(er)  # shared loader, metrics and plots

FIGURES_DIR = Path(__file__).parent / "figures"
# Unordered field pairs, in canonical field order (the baseline's categories).
PAIRS = list(itertools.combinations(LINE_FIELDS, 2))
PAIR_INDEX = {pr: i for i, pr in enumerate(PAIRS)}


# ------------------------------------------------------------------------ samples
def group_rows(split: dict) -> list[dict]:
    """One entry per ``(doc_id, row_ordinal)`` prompt (a judged sample):
    ``{doc, ordinal, rows (span indices in shown order), fields, pair_id, cell, valid}``."""
    by_key: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, (d, o) in enumerate(zip(split["doc_ids"].tolist(), split["row_ordinal"].tolist())):
        by_key[(d, o)].append(i)
    groups = []
    for (d, o), rows in by_key.items():
        rows = sorted(rows, key=lambda i: int(split["field_position"][i]))
        fields = [str(split["field"][i]) for i in rows]
        assert len(set(fields)) == len(fields) and set(fields) <= set(LINE_FIELDS), (d, o, fields)
        const = {c: {str(split[c][i]) for i in rows} for c in ("row_error_type", "pair_id", "cell")}
        assert all(len(v) == 1 for v in const.values()), f"{d}/{o}: {const} varies within a sample"
        groups.append({"doc": d, "ordinal": o, "rows": rows, "fields": fields, "pair_id": const["pair_id"].pop(),
                       "cell": const["cell"].pop(), "valid": const["row_error_type"].pop() == ""})
    return groups


def load_line_data(run_id: str, runs_root: Path, repo_root: Path) -> dict:
    """The line file a run judged (via its ``config.snapshot.yaml``; refused if stale), its line items
    and the label rule's equivalence: ``{rows: {(pair_id, valid): row}, item_keys: {doc: keys}, eq}``."""
    params = yaml.safe_load((runs_root / run_id / "config.snapshot.yaml").read_text())["params"]
    spec = load_dataset_spec(repo_root / params["dataset_config"])
    assert spec.name == "line" and tuple(spec.fields) == tuple(LINE_FIELDS), (run_id, spec.name)
    path = repo_root / params["data_root"] / spec.split_path.format(split=params["split"])
    build = check_build(path.parent.parent, f"{path.parent.name}/{path.name}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_key = {(r["pair_id"], r["valid"]): r for r in rows}
    assert len(by_key) == len(rows), f"{path}: (pair_id, valid) is not unique"
    bp = build["config_snapshot"]["params"]
    eq = make_eq(field_kinds(bp["cross_type"]["kinds"]), bp["dates"]["year_pivot"])
    items: dict[str, list[dict]] = defaultdict(list)
    for it in read_base(path.parent / f"{params['split']}_base.json", LINE_FIELDS, ["document_id", "line_index"]):
        items[it["document_id"]].append(it)
    return {"rows": by_key, "item_keys": {d: item_keys(its, LINE_FIELDS, eq) for d, its in items.items()}, "eq": eq}


def is_item_pair(values: dict, keys: list[dict], eq) -> bool:
    """The label rule on two shown values: are they one line item's?"""
    return matches_keys({f: eq(f, v) for f, v in values.items()}, keys)


def build_pairs(split: dict, groups: list[dict], data: dict) -> dict:
    """Matched field pairs of every judged minimal pair of samples.

    Returns arrays (one entry per matched field pair) ``inv_i, inv_j`` (the two spans in the invalid
    sample, canonical field order), ``val_i, val_j`` (the same fields in the twin), ``pair`` (field
    pair index), ``cell``, ``error_type``, ``doc``; the same for context pairs (``ctx_*``: valid field
    pairs of invalid samples, with their twins); and counts ``n_samples`` and
    ``n_invalid_samples_without_invalid_pair`` (an invalid sample all of whose field pairs match some
    item: possible, since the label rule is on the whole subset).

    Raises:
        AssertionError: a sample not in the row file, a span that disagrees with it, a pair not judged
            whole, or a twin field pair that is no item's.
    """
    by_pair: dict[str, dict[bool, dict]] = defaultdict(dict)
    for g in groups:
        key = (g["pair_id"], g["valid"])
        assert key in data["rows"], f"{g['doc']}/{g['ordinal']}: no row {key} in the row file (a run on other data?)"
        row = data["rows"][key]
        assert row["document_id"] == g["doc"] and row["cell"] == g["cell"], key
        assert row["fields"] == g["fields"], f"{key}: span fields {g['fields']} != row fields {row['fields']}"
        for i, f in zip(g["rows"], g["fields"]):
            assert row[f] == str(split["value"][i]), f"span {i}: value {split['value'][i]!r} != row file's {row[f]!r} for {key}"
            assert bool(split["label_valid"][i]) == (row["field_errors"][f] is None), (key, f, "span label")
        assert g["valid"] not in by_pair[g["pair_id"]], f"{key} judged twice"
        by_pair[g["pair_id"]][g["valid"]] = g
    incomplete = [pid for pid, m in by_pair.items() if set(m) != {True, False}]
    assert not incomplete, f"{len(incomplete)} pairs not judged whole, e.g. {incomplete[:2]}"

    out: dict[str, list] = defaultdict(list)
    n_without = 0
    for pid in sorted(by_pair):
        bad, good = by_pair[pid][False], by_pair[pid][True]
        keys = data["item_keys"][bad["doc"]]
        pos_bad, pos_good = dict(zip(bad["fields"], bad["rows"])), dict(zip(good["fields"], good["rows"]))
        n_invalid = 0
        for a, b in itertools.combinations(sorted(bad["fields"], key=LINE_FIELDS.index), 2):
            assert is_item_pair({f: str(split["value"][pos_good[f]]) for f in (a, b)}, keys, data["eq"]), (
                pid, a, b, "a twin field pair that is no item's")
            label = is_item_pair({f: str(split["value"][pos_bad[f]]) for f in (a, b)}, keys, data["eq"])
            pre = "ctx_" if label else ""
            n_invalid += not label
            out[f"{pre}inv_i"].append(pos_bad[a])
            out[f"{pre}inv_j"].append(pos_bad[b])
            out[f"{pre}val_i"].append(pos_good[a])
            out[f"{pre}val_j"].append(pos_good[b])
            out[f"{pre}pair"].append(PAIR_INDEX[(a, b)])
            out[f"{pre}cell"].append(bad["cell"])
            out[f"{pre}error_type"].append(bad["cell"].split(":")[0])
            out[f"{pre}doc"].append(bad["doc"])
        n_without += n_invalid == 0
    assert out["inv_i"], "no matched field pairs"
    res = {k: np.array(v) for k, v in out.items()}
    res["n_samples"], res["n_invalid_samples_without_invalid_pair"] = len(groups), n_without
    for pre in ("", "ctx_"):
        if f"{pre}inv_i" in res:
            assert (split["doc_ids"][res[f"{pre}inv_i"]] == split["doc_ids"][res[f"{pre}val_j"]]).all()
    return res


def stack(z: dict, pre: str = "") -> dict:
    """Both members of every matched field pair (``pre`` = ``""`` or ``"ctx_"``): span indices ``i, j``,
    ``y`` (valid: the twin's pair), and ``pair, cell, error_type, doc`` repeated for both."""
    n = len(z[f"{pre}inv_i"])
    return {"i": np.concatenate([z[f"{pre}inv_i"], z[f"{pre}val_i"]]), "j": np.concatenate([z[f"{pre}inv_j"], z[f"{pre}val_j"]]),
            "y": np.concatenate([np.zeros(n, bool), np.ones(n, bool)]),
            **{c: np.concatenate([z[f"{pre}{c}"]] * 2) for c in ("pair", "cell", "error_type", "doc")}}


def pair_features(x: np.ndarray, s: dict) -> np.ndarray:
    """``[n, 2 * hidden]``: ``[X[i], X[j]]`` per field pair."""
    f = np.concatenate([x[s["i"]], x[s["j"]]], axis=1)
    assert f.shape == (len(s["i"]), 2 * x.shape[1]) and np.isfinite(f).all()
    return f


# ----------------------------------------------------------------------------- main
def run(cfg: dict, runs_root: Path) -> dict:
    p, seed = cfg["params"], cfg["seed"]
    out = runs_root / cfg["id"]
    fig_dir = FIGURES_DIR / cfg["id"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    layer, thr = p["layer"], p["decision_threshold"]
    assert isinstance(layer, int) and layer > 0, "layer must be a positive int (layer 0 is the constant null control)"

    extra = ("line_index", "value", "pair_id", "cell")
    tr = er.load_split(p["train_run_id"], runs_root, layer, p["max_docs"], "relation_detection", extra)
    te = er.load_split(p["test_run_id"], runs_root, layer, p["max_docs"], "relation_detection", extra)
    assert not (set(tr["doc_ids"].tolist()) & set(te["doc_ids"].tolist())), "train/test documents overlap"
    for s in (tr, te):
        assert set(np.unique(s["field"]).tolist()) <= set(LINE_FIELDS), "non-line field in the relation run"
    er.center_per_document(tr["X"], tr["doc_ids"])
    er.center_per_document(te["X"], te["doc_ids"])

    z_tr = build_pairs(tr, group_rows(tr), load_line_data(p["train_run_id"], runs_root, REPO_ROOT))
    z_te = build_pairs(te, group_rows(te), load_line_data(p["test_run_id"], runs_root, REPO_ROOT))
    s_tr, s_te = stack(z_tr), stack(z_te)
    f_tr, f_te = pair_features(tr["X"], s_tr), pair_features(te["X"], s_te)
    y_tr, y_te = s_tr["y"], s_te["y"]
    ref_rate = float(y_tr.mean())
    assert ref_rate == 0.5 == float(y_te.mean()), "matched field-pair sets are not 50/50"
    types = sorted(set(z_tr["error_type"].tolist()))
    assert types == sorted(set(z_te["error_type"].tolist())), "train and test have different error types"
    assert set(types) <= set(er.ALL_ERROR_TYPES), types

    metrics: dict[str, float] = {
        "layer": layer, "n_train_pairs": int(len(z_tr["inv_i"])), "n_test_pairs": int(len(z_te["inv_i"])),
        "n_train_samples": z_tr["n_samples"], "n_test_samples": z_te["n_samples"],
        "n_train_invalid_samples_without_invalid_pair": z_tr["n_invalid_samples_without_invalid_pair"],
        "n_test_invalid_samples_without_invalid_pair": z_te["n_invalid_samples_without_invalid_pair"],
        "n_train_docs": len(set(tr["doc_ids"].tolist())), "n_test_docs": len(set(te["doc_ids"].tolist())),
    }

    clf = er.fit_logreg(f_tr, y_tr, p, seed)
    probs = clf.predict_proba(f_te)[:, 1]
    assert probs.shape == y_te.shape
    metrics.update(er.probe_metrics("test", y_te, probs, ref_rate, thr))
    metrics["train_auroc"] = float(roc_auc_score(y_tr, clf.predict_proba(f_tr)[:, 1]))
    metrics["test_auroc_ci_lo"], metrics["test_auroc_ci_hi"] = er.doc_bootstrap_auroc(
        y_te, probs, s_te["doc"], p["auroc_n_boot"], p["ci_level"], seed)

    d = er.smooth_ece(probs, y_te, p["ece_n_boot"], seed)
    metrics["test_smece"], metrics["test_smece_ci_halfwidth"] = float(d["ce"]), float(d["ce_ci_width"])
    er.plot_calibration(d, fig_dir)
    er.plot_prob_histogram(probs, p["hist_n_bins"], fig_dir)

    for col in ("error_type", "cell"):
        for k_, v in er.stratum_report(y_te, probs, thr, s_te[col]).items():
            metrics[f"test_{k_}"] = v

    # Context diagnostic: still-valid field pairs of invalid samples vs the same pairs of the twin.
    if "ctx_inv_i" in z_te:
        c = stack(z_te, "ctx_")
        p_ctx = clf.predict_proba(pair_features(te["X"], c))[:, 1]
        n_ctx = len(z_te["ctx_inv_i"])
        metrics["test_n_context_pairs"] = n_ctx
        metrics["test_context_mean_p_valid"], metrics["test_context_twin_mean_p_valid"] = (
            float(p_ctx[:n_ctx].mean()), float(p_ctx[n_ctx:].mean()))
        metrics["test_context_valid_acc"] = float((p_ctx[:n_ctx] > thr).mean())
        metrics["test_twin_vs_context_auroc"] = float(roc_auc_score(c["y"], p_ctx))

    # Structure baseline: cell x field pair, each exactly 50/50 on matched pairs -> AUROC exactly 0.5 (asserted).
    pair_ids = np.array([f"{a}__{b}" for a, b in PAIRS])
    struct_tr = np.char.add(np.char.add(s_tr["cell"], "|"), pair_ids[s_tr["pair"]])
    struct_te = np.char.add(np.char.add(s_te["cell"], "|"), pair_ids[s_te["pair"]])
    metrics.update(er.assert_structure_auroc_is_chance("cell_field_pair", y_tr, struct_tr, y_te, struct_te, p, seed))

    if p["run_controls"]:
        auc_shuf, auc_shuf_std = er.shuffled_label_auroc(f_tr, y_tr, f_te, y_te, p, seed)
        metrics["control_shuffled_auroc"], metrics["control_shuffled_auroc_std"] = auc_shuf, auc_shuf_std
        assert abs(auc_shuf - 0.5) <= p["shuffled_label_auroc_tolerance"], (
            f"mean shuffled-label AUROC {auc_shuf:.4f} (std {auc_shuf_std:.4f}, n={p['n_shuffles']}) is not ~0.5 -- something leaks"
        )
        assert np.array_equal(probs, er.fit_logreg(f_tr, y_tr, p, seed).predict_proba(f_te)[:, 1]), (
            "same-seed refit gave different probabilities"
        )
        metrics["control_seed_determinism"] = 1.0

    np.savez_compressed(
        out / "artifacts" / "test_predictions.npz",
        prob_valid=probs, label_valid=y_te, pair_doc=s_te["doc"], pair=pair_ids[s_te["pair"]], cell=s_te["cell"],
        error_type=s_te["error_type"], span_i=s_te["i"], span_j=s_te["j"], pair_id=te["pair_id"][s_te["i"]],
    )
    metrics_text = json.dumps(metrics, indent=2) + "\n"
    (out / "metrics.json").write_text(metrics_text)
    (fig_dir / "metrics.json").write_text(metrics_text)  # copy beside the figures
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    cfg_path = Path(sys.argv[1])
    cfg = yaml.safe_load(cfg_path.read_text())
    runs_root = Path(os.environ["RUNS_ROOT"])
    out = runs_root / cfg["id"]
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cfg_path, out / "config.snapshot.yaml")
    run(cfg, runs_root)


if __name__ == "__main__":
    main()
