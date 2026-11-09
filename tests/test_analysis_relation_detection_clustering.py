"""Unit tests for analysis/relation_detection_clustering.py on hand-built inputs.

Fixture (tests/fixtures/relation_detection_mini, doc-a): li0 = {7, Movie, 01/05/20, 01/10/20, $100.00},
li1 = {8, null, 01/11/20, 01/20/20, $250.00}, li2 = {8, News, 01/21/20, 01/31/20, $400.00}.
Invalid m5 prompt of li0 (end date <- li1, amount <- li2): memberships CH 7 -> {0}, DESC Movie -> {0}, SD 01/05/20 -> {0},
ED 01/20/20 -> {1}, AMT $400.00 -> {2}: groups [0, 0, 0, 1, 2]; both corrupted entities are unique, so it is eligible
(3 within pairs, 7 between pairs).
Invalid m5 cross_item prompt of li2: CH <-> AMT swapped; CH shows "$400.00" (source field sub_amount -> {2}), AMT shows "8"
(source field channel -> {1, 2}, not unique): never eligible.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analysis_relation_detection_clustering", ROOT / "analysis" / "relation_detection_clustering.py")
cl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl)
sys.path.insert(0, str(ROOT / "data" / "vrdu"))
from vrdu_common import field_kinds, make_eq  # noqa: E402

rr = cl.rr
CH, DESC, SD, ED, AMT = rr.LINE_FIELDS
KINDS = {"name": ["advertiser", "agency", "property", "tv_address", "product"],
         "date": ["flight_from", "flight_to", "program_start_date", "program_end_date"],
         "amount": ["gross_amount", "sub_amount"]}
EQ = make_eq(field_kinds(KINDS), 50)
ROWS = json.loads((ROOT / "tests/fixtures/relation_detection_mini/vrdu/line/train.json").read_text())
DOC_A = [{CH: "7", DESC: "Movie", SD: "01/05/20", ED: "01/10/20", AMT: "$100.00"},
         {CH: "8", DESC: None, SD: "01/11/20", ED: "01/20/20", AMT: "$250.00"},
         {CH: "8", DESC: "News", SD: "01/21/20", ED: "01/31/20", AMT: "$400.00"}]
ITEMS = {"doc-a": dict(enumerate(DOC_A))}


def split_of(rows):
    cols = defaultdict(list)
    ordinal = 0
    for r in rows:
        for pos, f in enumerate(r["fields"], start=1):
            err = r["field_errors"][f]
            src = err["source"] if err else None
            for c, v in (("doc_ids", r["document_id"]), ("row_ordinal", ordinal), ("field", f), ("field_position", pos), ("value", r[f]),
                         ("label_valid", err is None), ("row_error_type", r["error_type"] or ""), ("pair_id", r["pair_id"]),
                         ("cell", r["cell"]), ("line_index", r["line_index"]), ("source_field", src["field"] if src else ""),
                         ("source_line_index", src["line_index"] if src else -1)):
                cols[c].append(v)
        ordinal += 1
    return {k: np.array(v) for k, v in cols.items()}


def doc_a_invalid():
    rows = [r for r in ROWS if r["document_id"] == "doc-a" and not r["valid"] and r["fields"] == rr.LINE_FIELDS]
    assert [r["cell"] for r in rows] == ["intra_document:k2", "cross_item:k2"]
    s = split_of(rows)
    return s, rr.group_rows(s)


# -------------------------------------------------------------------- membership, fill
def test_membership_hand_worked_and_provenance_known_answer():
    s, groups = doc_a_invalid()
    ent = cl.entity_table(s, groups, ITEMS, EQ)
    r0, r1 = groups[0]["rows"], groups[1]["rows"]
    assert [ent[i]["prov"] for i in r0] == [0, 0, 0, 1, 2]   # provenance labels: native li0 x3, donors li1 and li2
    assert [ent[i]["members"] for i in r0] == [frozenset({0}), frozenset({0}), frozenset({0}), frozenset({1}), frozenset({2})]
    assert [ent[i]["corrupted"] for i in r0] == [False, False, False, True, True]
    # cross_item (li2): CH shows $400.00 (source sub_amount: li2); AMT shows 8 (source channel: li1 and li2 both have 8)
    assert [ent[i]["prov"] for i in r1] == [2, 2, 2, 2, 2]
    assert [ent[i]["members"] for i in r1] == [frozenset({2})] * 4 + [frozenset({1, 2})]


def test_a_provenance_outside_the_matches_is_refused():
    s, groups = doc_a_invalid()
    s["source_line_index"][np.where(s["field"] == ED)[0][0]] = 2   # li0's end date claimed to come from li2
    with pytest.raises(AssertionError, match="provenance"):
        cl.entity_table(s, groups, ITEMS, EQ)


def test_fill_needs_a_strictly_earlier_corruption():
    pos = np.array([1, 2, 3, 4, 5])
    corr = np.array([False, False, False, True, True])
    assert cl.fill_flags(pos, corr).tolist() == [False, False, False, False, True]   # the first corruption itself is open
    assert cl.fill_flags(pos, np.zeros(5, bool)).tolist() == [False] * 5
    assert cl.fill_flags(pos, np.array([True, False, False, False, False])).tolist() == [False, True, True, True, True]


def test_links_hand_worked_on_the_fixture_prompt():
    s, groups = doc_a_invalid()
    ent = cl.entity_table(s, groups, ITEMS, EQ)
    rows = groups[0]["rows"]   # members: CH {0}, DESC {0}, SD {0}, ED {1}, AMT {2}; corrupted: ED, AMT
    lk = cl.links([ent[i]["members"] for i in rows], np.array([ent[i]["corrupted"] for i in rows]))
    assert lk == [(0, 1, True), (0, 2, True), (1, 2, True)]   # only the three native entities of li0 relate; donors share nothing
    # shared items make edges: CH "8" is li1 and li2's, so it links to a corrupted ED (li1) with a grey (incidental) edge
    assert cl.links([frozenset({1, 2}), frozenset({1}), frozenset({2})], np.array([False, True, False])) == [(0, 1, False), (0, 2, True)]
    assert cl.links([frozenset({0}), frozenset({1})], np.array([False, False])) == []


def test_union_exclusions_drops_a_zero_entity_of_either_model():
    sc = {"a": {"all": {"p1", "p2", "p3", "p4"}, "small": {"p1"}, "zero": {"p2"}},
          "b": {"all": {"p1", "p2", "p3", "p4"}, "small": {"p1"}, "zero": {"p3"}}}
    ex = cl.union_exclusions(sc)
    assert ex["drop"] == {"p1", "p2", "p3"} and ex["zero_union"] == {"p2", "p3"} and ex["small"] == {"p1"}
    with pytest.raises(AssertionError, match="other prompts"):
        cl.union_exclusions({**sc, "b": {**sc["b"], "all": {"p1"}, "zero": set()}})
    with pytest.raises(AssertionError, match="small-group"):
        cl.union_exclusions({**sc, "b": {**sc["b"], "small": set()}})


# -------------------------------------------------------------------- distances and metric
LAB = np.array([0, 0, 0, 1, 2])


def toy(far=10.0):
    """Three natives near the origin, two donors far away."""
    return np.array([[0, 0], [0.1, 0], [0, 0.1], [far, 0], [0, far]], dtype=float)


def test_pair_sets_and_eligibility():
    iw, jw, ib, jb = cl.pair_sets(LAB)
    assert len(iw) == 3 and len(ib) == 7
    assert cl.pair_sets(np.array([0, 0, 0])) is None                       # no between pair
    assert cl.pair_sets(np.array([0, 1, 2])) is None                       # no within pair


@pytest.mark.parametrize("kind", cl.DISTANCES)
def test_known_answer_clustered_and_anti_clustered(kind):
    v = toy() + 1.0   # off the origin so cosine is defined
    d = cl.dist_matrix(v, kind)
    assert cl.win_frac(d, cl.pair_sets(LAB)) == 1.0
    anti = np.array([[0, 0], [10, 0], [0, 10], [0.1, 0], [0, 0.1]], dtype=float) + 1.0   # the "within" members are the far ones
    assert cl.win_frac(cl.dist_matrix(anti, "euclidean"), cl.pair_sets(LAB)) < 0.5


def test_constant_vectors_tie_at_exactly_half():
    for kind in cl.DISTANCES:
        v = np.tile(np.array([[1.0, 2.0, 3.0]]), (5, 1))
        s = cl.prompt_stats(v, v, LAB, kind, 3, np.random.default_rng(0))
        assert s["win_frac"] == 0.5 and s["perm_win_frac"] == 0.5 and s["contrast"] == 0.0


def test_permutation_mean_is_the_prompts_own_chance_level():
    d = cl.dist_matrix(toy() + 1.0, "euclidean")
    assert cl.win_frac(d, cl.pair_sets(LAB)) == 1.0
    perm = cl.permuted_win_frac(d, LAB, 2000, np.random.default_rng(1))
    assert 0.5 < perm < 0.9   # an outlier structure: chance is NOT 0.5 for this prompt
    # hand-worked: 5 points, labels are a random permutation of [0,0,0,1,2]; recompute exactly by enumeration
    import itertools
    labs = {p_ for p_ in itertools.permutations(LAB.tolist())}
    exact = np.mean([cl.win_frac(d, cl.pair_sets(np.array(l))) for l in labs])
    assert perm == pytest.approx(exact, abs=0.02)
    assert cl.permuted_win_frac(d, LAB, 5, np.random.default_rng(3)) == cl.permuted_win_frac(d, LAB, 5, np.random.default_rng(3))


def test_random_vectors_give_half():
    rng = np.random.default_rng(0)
    w = [cl.prompt_stats(rng.normal(size=(5, 64)), rng.normal(size=(5, 64)), LAB, "euclidean", 1, rng)["win_frac"] for _ in range(400)]
    assert abs(np.mean(w) - 0.5) < 0.03


def test_twin_contrast_hand_worked():
    v = toy() + 1.0
    twin = v.copy()
    twin[3] = [1.0, 1.0]   # in the twin, entity 3 sits on the origin cluster's side: pairs (0..2, 3) shrink
    s = cl.prompt_stats(v, twin, LAB, "euclidean", 1, np.random.default_rng(0))
    d, dt = cl.dist_matrix(v, "euclidean"), cl.dist_matrix(twin, "euclidean")
    iw, jw, ib, jb = cl.pair_sets(LAB)
    assert s["contrast"] == pytest.approx((d[ib, jb] - dt[ib, jb]).mean() - (d[iw, jw] - dt[iw, jw]).mean())
    assert s["contrast"] > 0 and s["twin_win_frac"] == pytest.approx(cl.win_frac(dt, (iw, jw, ib, jb)))


def test_win_vs_twin_is_the_paired_difference():
    v = toy() + 1.0                                  # clustered: win_frac 1.0
    twin_flat = np.tile(np.array([[1.0, 2.0]]), (5, 1))   # twin carries no structure: every distance ties, twin_win_frac 0.5
    s = cl.prompt_stats(v, twin_flat, LAB, "euclidean", 1, np.random.default_rng(0))
    assert s["win_frac"] == 1.0 and s["twin_win_frac"] == 0.5 and s["win_vs_twin"] == 0.5
    same = cl.prompt_stats(v, v, LAB, "euclidean", 1, np.random.default_rng(0))
    assert same["win_vs_twin"] == 0.0                # a twin identical to the prompt adds nothing


def test_cosine_of_a_zero_vector_is_refused():
    with pytest.raises(AssertionError, match="zero vector"):
        cl.dist_matrix(np.zeros((3, 2)), "cosine")


# -------------------------------------------------------------------- preprocessing
def test_center_field_doc_zero_mean_per_group_and_distances_between_fields_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 4)).astype(np.float32)
    docs = np.array(["a"] * 4 + ["b"] * 4)
    fields = np.array(["f", "g", "f", "g"] * 2)
    c, counts = cl.center_field_doc(x, docs, fields)
    assert counts == {(d, f): 2 for d in 'ab' for f in 'fg'}
    for d in "ab":
        for f in "fg":
            m = (docs == d) & (fields == f)
            assert np.allclose(c[m].mean(0), 0) and np.allclose(c[m], x[m] - x[m].mean(0))
    # a constant shift of every row changes nothing (the layer-0 subtraction is a no-op)
    assert np.allclose(cl.center_field_doc(x + 5.0, docs, fields)[0], c, atol=1e-5)
    # two rows are left as +-half their difference: why small groups are excluded
    assert np.allclose(c[docs == 'a'][fields[docs == 'a'] == 'f'][0], -c[docs == 'a'][fields[docs == 'a'] == 'f'][1])


def test_an_entity_identical_across_its_group_centres_to_zero_and_is_flagged():
    x = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [3.0, 1.0], [5.0, 1.0], [3.0, 3.0]], dtype=np.float32)
    docs, fields = np.array(["a"] * 6), np.array(["f"] * 3 + ["g"] * 3)
    c, _ = cl.center_field_doc(x, docs, fields)
    assert cl.zero_rows(c, 1e-6).tolist() == [True, True, True, False, False, False]   # group f is homogeneous, g is not


def test_constant_layer_zero_subtraction_leaves_euclidean_distances_unchanged():
    v = toy()
    assert np.allclose(cl.dist_matrix(v, "euclidean"), cl.dist_matrix(v - np.array([3.0, -2.0]), "euclidean"))


# -------------------------------------------------------------------- examples, PCA, bootstrap
def test_choose_examples_is_random_seeded_and_requires_every_cell():
    def P(doc, cell, o=0, m=5):
        return {"doc": doc, "cell": cell, "ordinal": o, "m": m}
    cells = ["x", "y"]
    prompts = [P("d1", "x"), P("d1", "y"), P("d2", "x"), P("d2", "y"), P("d2", "y", o=1), P("d3", "x", m=4), P("d3", "y"),
               P("d4", "x", o=7), P("d4", "x", o=2), P("d4", "y")]
    ex = cl.choose_examples(prompts, cells, 5, 2, np.random.default_rng(0))
    assert len({e["doc"] for e in ex}) == 2 and {e["doc"] for e in ex} <= {"d1", "d2", "d4"}   # d3 has no m=5 prompt in x
    for e in ex:
        assert [q["cell"] for q in e["prompts"]] == cells and all(q["doc"] == e["doc"] and q["m"] == 5 for q in e["prompts"])
    again = cl.choose_examples(list(reversed(prompts)), cells, 5, 2, np.random.default_rng(0))
    assert [(e["doc"], [q["ordinal"] for q in e["prompts"]]) for e in again] == [(e["doc"], [q["ordinal"] for q in e["prompts"]]) for e in ex]
    picked = {tuple(e["doc"] for e in cl.choose_examples(prompts, cells, 5, 2, np.random.default_rng(s_))) for s_ in range(30)}
    assert len(picked) > 1   # not a fixed rule
    with pytest.raises(AssertionError, match="only 3 documents"):
        cl.choose_examples(prompts, cells, 5, 4, np.random.default_rng(0))


def test_pca_basis_recovers_the_main_axis():
    pts = np.array([[0, 0, 0], [2, 0, 0], [4, 0.1, 0], [6, 0, 0.1]], dtype=float)
    mu, comp, var = cl.pca_basis(pts)
    assert abs(comp[0, 0]) > 0.99 and var[0] > 0.99


def test_doc_bootstrap_mean_interval_contains_the_mean_and_is_seeded():
    vals = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 0.0])
    docs = np.array(["a", "a", "b", "b", "c", "c"])
    m, lo, hi = cl.doc_bootstrap_mean(vals, docs, 200, 0.95, np.random.default_rng(0))
    assert m == pytest.approx(vals.mean()) and lo <= m <= hi
    assert (m, lo, hi) == cl.doc_bootstrap_mean(vals, docs, 200, 0.95, np.random.default_rng(0))
