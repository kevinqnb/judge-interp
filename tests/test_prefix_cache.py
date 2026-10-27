"""Prefix (KV) cache of RowDetectionLM, on the real Qwen2.5-0.5B-Instruct (cached weights, CPU, float32)
over the hand-built tests/fixtures/entity_detection_mini.

Reads taken with the document's instructions + context cached must equal reads of the whole prompt, a
cache of another document must not, and the token counts must say what was computed.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from judge_interp import entity_detection as er
from judge_interp.row_detection import RowDetectionLM
from dataset_specs import LINE

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = "tests/fixtures/entity_detection_mini"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LAYERS = [0, 8, "last"]
LINE_CELLS = ["cross_item:k2", "intra_document:k1", "intra_document:k2"]
RTOL = 1e-3   # of each layer's max|uncached|; float32 on CPU agrees far more tightly. The bf16 GPU runs use a larger value (config)


def _items():
    p = dict(data_root=FIXTURE, split="train", error_types=list(LINE.error_types),
             rows_per_stratum=dict.fromkeys(LINE_CELLS), context_token_limit=None)
    return er.build_items(LINE, p, 0, REPO_ROOT)


def _lm(**kw):
    args = dict(prefix_cache=True, prefix_cache_check_n=0, prefix_cache_rtol=RTOL) | kw
    return RowDetectionLM(model_name=MODEL, layers=LAYERS, device="cpu", dtype="float32", **args)


@pytest.fixture(scope="module")
def lm():
    return _lm()


def _judge(lm, it):
    return lm.judge_row(it["instructions"], it["context"], it["query"], it["spans"])


def _reads_equal(a, b, rtol):
    return all(np.abs(a["representations"][L] - b["representations"][L]).max() <= rtol * np.abs(b["representations"][L]).max()
               for L in a["representations"])


def test_fixture_has_a_document_with_several_rows():
    docs = [it["document_id"] for it in _items()]
    assert len(docs) == 8 and max(docs.count(d) for d in set(docs)) >= 2


def test_cached_reads_match_the_uncached_forward_for_every_row(lm):
    items = _items()
    lm._kv = lm._prefix_ids = None
    cached = [_judge(lm, it) for it in items]
    lm.prefix_cache = False
    try:
        plain = [_judge(lm, it) for it in items]
    finally:
        lm.prefix_cache = True
    for it, c, u in zip(items, cached, plain):
        assert c["prompt_n_tokens"] == u["prompt_n_tokens"]
        assert np.array_equal(c["token_index"], u["token_index"])   # indices are into the whole prompt either way
        assert _reads_equal(c, u, RTOL), (it["document_id"], it["row_ordinal"])


def test_token_accounting(lm):
    items = _items()
    lm._kv = lm._prefix_ids = None
    seen = set()
    for it in items:
        e = _judge(lm, it)["efficiency"]
        n = e["prompt_n_tokens"]
        assert e["reused_n_tokens"] + e["computed_n_tokens"] == n
        if it["document_id"] in seen:
            ids, offsets, shift = lm._build_prompt_with_offsets(it["instructions"], it["context"], it["query"])
            # known answer: the reused prefix is exactly the tokens before the record
            assert e["reused_n_tokens"] == sum(1 for _, end in offsets if end <= shift) > 0
            assert e["computed_n_tokens"] == n - e["reused_n_tokens"] < n
        else:
            assert e["reused_n_tokens"] == 0 and e["computed_n_tokens"] == n   # the first row computes its prefix too
        seen.add(it["document_id"])
    assert len(seen) >= 2   # a new document does not reuse the previous document's prefix


def test_cache_off_reuses_nothing():
    off = _lm(prefix_cache=False)
    e = _judge(off, _items()[0])["efficiency"]
    assert e["reused_n_tokens"] == 0 and e["computed_n_tokens"] == e["prompt_n_tokens"]
    assert off._kv is None


def test_cache_is_restored_after_every_row(lm):
    it = _items()[0]
    lm._kv = lm._prefix_ids = None
    _judge(lm, it)
    n = len(lm._prefix_ids)
    assert lm._kv.get_seq_length() == n
    _judge(lm, it)
    assert lm._kv.get_seq_length() == n


def test_a_corrupted_prefix_cache_is_caught_by_the_check():
    """Negative control: with the right ids but wrong KV the reads move, and the gate must fail."""
    lm = _lm(prefix_cache_check_n=1)
    it = _items()[0]
    _judge(lm, it)                         # builds the correct cache; the check passes
    assert lm.n_prefix_checked == 1 and lm.prefix_cache_max_rel_diff <= RTOL
    for layer in lm._kv.layers:
        layer.values = torch.zeros_like(layer.values)
    lm.n_prefix_checked = 0
    with pytest.raises(AssertionError, match="prefix cache changes layer"):
        _judge(lm, it)


def test_check_records_the_largest_difference():
    lm = _lm(prefix_cache_check_n=2)
    for it in _items()[:3]:
        _judge(lm, it)
    assert lm.n_prefix_checked == 2
    assert 0 <= lm.prefix_cache_max_rel_diff <= RTOL


@pytest.mark.parametrize("kw", [
    dict(prefix_cache=1), dict(prefix_cache_check_n=-1), dict(prefix_cache_check_n=True), dict(prefix_cache_rtol=-1.0),
    dict(prefix_cache=False, prefix_cache_check_n=1),
])
def test_bad_params_are_hard_errors(kw):
    with pytest.raises(ValueError):
        RowDetectionLM.__init__(RowDetectionLM.__new__(RowDetectionLM), model_name=MODEL, layers=LAYERS, device="cpu",
                                dtype="float32", **(dict(prefix_cache=True, prefix_cache_check_n=0, prefix_cache_rtol=RTOL) | kw))


def test_the_fingerprint_depends_on_the_cache_flag():
    a = RowDetectionLM.__new__(RowDetectionLM)
    a.model_name = MODEL
    b = RowDetectionLM.__new__(RowDetectionLM)
    b.model_name = MODEL
    a.prefix_cache, b.prefix_cache = True, False
    it = _items()[0]
    assert a._item_fingerprint(it) != b._item_fingerprint(it)
