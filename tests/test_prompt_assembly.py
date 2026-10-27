"""Tokenizer-level tests of the assembled prompt: which token is read at each entity.

Uses the cached Qwen2.5 and Llama-3.1 tokenizers (no weights) over the hand-built
tests/fixtures/entity_detection_mini. Every entity must be read at the same token -- the pad
space, closing quote and newline merged into ' "\\n' -- including the last entity of a row, whose
newline Llama's chat template would strip if the content were passed through it.
"""
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from judge_interp import entity_detection as er
from judge_interp.representationlm import RepresentationLM
from judge_interp.row_detection import RowDetectionLM
from dataset_specs import LINE, MAIN

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = "tests/fixtures/entity_detection_mini"
READ_TOKEN = ' "\n'
LINE_CELLS = ["cross_item:k2", "intra_document:k1", "intra_document:k2"]
END_OF_TURN = {"Qwen/Qwen2.5-0.5B-Instruct": "<|im_end|>", "meta-llama/Llama-3.1-8B-Instruct": "<|eot_id|>"}


def _items(spec, strata):
    p = dict(data_root=FIXTURE, split="train", error_types=list(spec.error_types),
             rows_per_stratum=strata, context_token_limit=None)
    return er.build_items(spec, p, 0, REPO_ROOT)


@pytest.fixture(scope="module", params=list(END_OF_TURN))
def lm(request):
    obj = RowDetectionLM.__new__(RowDetectionLM)
    obj.tokenizer = AutoTokenizer.from_pretrained(request.param)
    obj.model_name = request.param
    return obj


def test_every_entity_is_read_at_the_same_token_including_the_last_of_a_row(lm):
    items = (_items(MAIN, {"valid": None, "inter_document": None, "cross_type": None})
             + _items(LINE, dict.fromkeys(LINE_CELLS)))
    assert len(items) == 4 + 8
    for it in items:
        ids, offsets, shift = lm._build_prompt_with_offsets(it["instructions"], it["context"], it["query"])
        reads = []
        for sp in it["spans"]:
            idx = lm._last_token_covering(offsets, shift + sp["char_start"])
            reads.append(idx)
            assert lm.tokenizer.decode([ids[idx]]) == READ_TOKEN, (it["document_id"], sp["field"])
        # the last entity's token is followed directly by the end-of-turn marker, not by more query text
        assert lm.tokenizer.decode([ids[reads[-1] + 1]]) == END_OF_TURN[lm.model_name]


def test_chat_text_changes_nothing_but_the_template_wrapper(lm):
    content = "## A:\nx \n\n## B:\ny \"\n"  # leading/trailing whitespace must survive
    text = lm._chat_text(content)
    assert text.count(content) == 1
    assert "@@judge-interp-content@@" not in text
    if "Qwen" in lm.model_name:  # Qwen's template passes content through: splicing equals the plain template
        plain = lm.tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                 add_generation_prompt=True)
        assert text == plain
    with pytest.raises(AssertionError):
        lm._chat_text("contains @@judge-interp-content@@")
