"""Build and render per-(document, field) entity lists for the entity-detection
and relation-detection judge tasks.

Pure text / list work only -- no torch, no nnsight -- so the unit tests here
run on a login node without a GPU. Anything that touches the model lives in
``representationlm.py``.

Ground truth is never synthesized here: it is read off the existing
``data/vrdu/{main,line}/{train,test}.json`` rows (the same corpus
``build_vrdu_invalids.py`` already built and ``prompts.render_query`` already
consumes for the whole-record judge). A field's *valid* entities are the
non-null values on that document's ``valid: true`` row(s). Its *invalid*
entities (decoys) are non-null corrupted values drawn from *inter_document*
corruption (and, for ``main``, any corruption at all -- ``main`` has one row
per document, so every corruption is inherently cross-document) -- genuinely
foreign values, never ``intra_document`` swaps, which are real values of the
*same* document just placed on the wrong row (that's the relation-detection
task's material, not entity-detection's).
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from judge_interp.prompts import LINE_FIELDS, MAIN_FIELDS, load_split

FIELDS_BY_DATASET: dict[str, list[str]] = {"main": MAIN_FIELDS, "line": LINE_FIELDS}

_CUE_TEXT = "valid?"
_SEP = " — "  # em dash, matches the rendered " — valid?" cue


def _fields_for(dataset: str) -> list[str]:
    if dataset not in FIELDS_BY_DATASET:
        raise ValueError(f"dataset must be one of {sorted(FIELDS_BY_DATASET)}, got {dataset!r}")
    return FIELDS_BY_DATASET[dataset]


def _sub_seed(seed: int, *parts: str) -> int:
    """Deterministic per-key seed, stable across processes and Python versions.

    ``random.Random`` accepts an int seed directly; deriving one from
    ``hash((seed, *parts))`` would not be reproducible across runs, since
    Python randomizes ``str`` hashing per-process by default.
    """
    key = "|".join([str(seed), *parts]).encode()
    return int(hashlib.sha256(key).hexdigest(), 16) % (2**63)


def build_entity_lists(
    data_root: str | Path,
    dataset: str,
    split: str,
    seed: int,
    min_valid_per_list: int,
    min_invalid_per_list: int,
    max_invalid_per_list: int,
) -> dict:
    """Aggregate every dataset row into per-(document, field) entity lists.

    Args:
        min_valid_per_list / min_invalid_per_list: a list with fewer valid or
            invalid entities than this is excluded (logged in ``skipped``, not
            a hard error -- a null-heavy field or a document with few
            corruptions is expected corpus structure, not a pipeline bug).
        max_invalid_per_list: a list with more invalid entities than this is
            capped by deterministic seeded sampling.

    Returns:
        ``{"lists": {(document_id, field): [(value, label_valid), ...]}, ...}``
        -- each list already in final, randomized presentation order (see
        below) -- plus ``"skipped": [{"document_id", "field", "n_valid",
        "n_invalid"}, ...]`` for every excluded (document, field) pair.

    Raises:
        ValueError: unknown ``dataset``, or ``min_valid_per_list`` /
            ``min_invalid_per_list`` / ``max_invalid_per_list`` not
            non-negative ints, or ``max_invalid_per_list < min_invalid_per_list``.

    Ordering: entities within a list are always seeded-shuffled (valid and
    invalid values interleaved), never valid-then-decoy -- a fixed ordering
    would let the model, and any downstream geometric analysis, trivially
    recover the label (or list membership) from position alone.
    """
    fields = _fields_for(dataset)
    for name, value in (
        ("min_valid_per_list", min_valid_per_list),
        ("min_invalid_per_list", min_invalid_per_list),
        ("max_invalid_per_list", max_invalid_per_list),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative int, got {value!r}")
    if max_invalid_per_list < min_invalid_per_list:
        raise ValueError(
            f"max_invalid_per_list ({max_invalid_per_list}) < "
            f"min_invalid_per_list ({min_invalid_per_list})"
        )

    rows = load_split(data_root, dataset, split)
    by_doc: dict[str, list[dict]] = {}
    for r in rows:
        by_doc.setdefault(r["document_id"], []).append(r)

    lists: dict[tuple[str, str], list[tuple[str, bool]]] = {}
    skipped: list[dict] = []

    for doc_id in sorted(by_doc):
        doc_rows = by_doc[doc_id]
        base_rows = [r for r in doc_rows if r["valid"]]
        for field in fields:
            valid_vals = sorted({r[field] for r in base_rows if r[field] is not None})

            decoy_vals = sorted(
                {
                    r[field]
                    for r in doc_rows
                    if not r["valid"]
                    and field in r["invalid_fields"]
                    and (dataset == "main" or r["error_type"] != "intra_document")
                    and r[field] is not None
                    and r[field] not in valid_vals
                }
            )
            # Boundary check: the filter above should make this impossible by
            # construction. Assert it rather than silently trusting the filter.
            assert not (set(valid_vals) & set(decoy_vals)), (
                f"{doc_id!r}/{field!r}: value(s) in both valid_vals and decoy_vals "
                f"after filtering: {set(valid_vals) & set(decoy_vals)}"
            )

            n_valid, n_invalid = len(valid_vals), len(decoy_vals)
            if n_valid < min_valid_per_list or n_invalid < min_invalid_per_list:
                skipped.append(
                    {"document_id": doc_id, "field": field, "n_valid": n_valid, "n_invalid": n_invalid}
                )
                continue

            rng = random.Random(_sub_seed(seed, doc_id, field))
            if n_invalid > max_invalid_per_list:
                decoy_vals = sorted(rng.sample(decoy_vals, max_invalid_per_list))

            entities = [(v, True) for v in valid_vals] + [(v, False) for v in decoy_vals]
            rng.shuffle(entities)
            lists[(doc_id, field)] = entities

    return {"lists": lists, "skipped": skipped}


def render_entity_list(entities: list[tuple[str, bool]]) -> dict:
    """Render a (already-ordered) entity list as the judge's ``## QUERY:`` block.

    Each entity is rendered ``<n>. <json-quoted value> — valid?`` on its own
    line. Character spans for both the entity's own content and its verdict
    cue are computed *during* rendering (never recovered by substring search
    afterward), so a value that recurs verbatim in the list never causes span
    ambiguity.

    Returns:
        ``{"text": str, "spans": [{"index", "value", "label_valid",
        "content_start", "content_end", "cue_start", "cue_end"}, ...]}``.
        Offsets are into ``text``. ``content_start:content_end`` covers the
        JSON-quoted value (including its quotes); ``cue_start:cue_end`` covers
        the literal ``"valid?"`` immediately after it -- the position whose
        next-token logits are read as that entity's verdict.

    Raises:
        ValueError: ``entities`` is empty.
    """
    if not entities:
        raise ValueError("entities is empty")

    lines: list[str] = []
    spans: list[dict] = []
    offset = 0
    for i, (value, label_valid) in enumerate(entities, start=1):
        prefix = f"{i}. "
        quoted = json.dumps(value)
        line = f"{prefix}{quoted}{_SEP}{_CUE_TEXT}"

        content_start = offset + len(prefix)
        content_end = content_start + len(quoted)
        cue_start = content_end + len(_SEP)
        cue_end = cue_start + len(_CUE_TEXT)
        spans.append(
            {
                "index": i,
                "value": value,
                "label_valid": label_valid,
                "content_start": content_start,
                "content_end": content_end,
                "cue_start": cue_start,
                "cue_end": cue_end,
            }
        )
        lines.append(line)
        offset += len(line) + 1  # +1 for the "\n" this line joins on

    return {"text": "\n".join(lines), "spans": spans}
