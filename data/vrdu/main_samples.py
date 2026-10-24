"""Entity-level evaluation pairs of the VRDU main dataset: each corrupted span with its clean twin span.

A span is one shown field of one main row, keyed ``(document_id, error_type, k, field)`` (``error_type``
None and ``k`` 0 for the valid row). For each stratum ``(error_type, k)``:

  invalid pairs  every corrupted span, paired with the same field in a clean row. The twin is the
                 document's own clean row, except for a null-fill span (the field is null there): its twin
                 is the same field in another document's clean row with that field non-null, drawn with
                 ``seed`` (``twin = "other_document"``). Such a twin may also be a twin elsewhere in the
                 stratum: at high k nearly every clean span already is.
  kept pairs     every uncorrupted shown span of a corrupted row, paired with the same field in the
                 document's clean row. Both are valid: a diagnostic for a probe that only senses "this row
                 has a corruption somewhere".

So every stratum has as many valid twin spans as corrupted spans, matched on field (and, except for
null-fill, on document). Drop ``null_fill`` pairs for the analysis without filled nulls.
"""

from __future__ import annotations

import random

from vrdu_common import MAIN_FIELDS

ERROR_TYPES = ("inter_document", "cross_type")


def span(document_id, error_type, k, field):
    """A span key."""
    return {"document_id": document_id, "error_type": error_type, "k": k, "field": field}


def _clean_rows(rows):
    """``{document_id: valid row}``; asserts exactly one valid row per document."""
    clean = {}
    for r in rows:
        if r["valid"]:
            assert r["document_id"] not in clean, (r["document_id"], "two valid rows")
            assert r["error_type"] is None and r["k"] == 0, r["document_id"]
            clean[r["document_id"]] = r
    return clean


def _strata(rows):
    """``{(error_type, k): [invalid rows sorted by document]}``; asserts one row per document per stratum."""
    out = {}
    for r in rows:
        if r["valid"]:
            continue
        assert r["error_type"] in ERROR_TYPES and r["k"] == len(r["invalid_fields"]) >= 1, (r["document_id"], r["error_type"], r["k"])
        out.setdefault((r["error_type"], r["k"]), []).append(r)
    for key, rs in out.items():
        docs = [r["document_id"] for r in rs]
        assert len(docs) == len(set(docs)), (key, "a document has two rows in one stratum")
        rs.sort(key=lambda r: r["document_id"])
    return dict(sorted(out.items()))


def main_entity_pairs(rows, seed):
    """Invalid spans paired with clean twin spans, by stratum.

    Args:
        rows: one split's ``main/{split}.json`` rows.
        seed: seeds the other-document twins of null-fill spans.

    Returns:
        One dict per corrupted span: ``{error_type, k, field, null_fill, twin, invalid: span, valid: span}``,
        ``twin`` being ``"same_document"`` or ``"other_document"``.

    Raises:
        AssertionError: a malformed row set, or a null-fill span whose field no other document has.
    """
    clean = _clean_rows(rows)
    out = []
    for (et, k), rs in _strata(rows).items():
        assert {r["document_id"] for r in rs} <= set(clean), (et, k, "invalid row without a clean row")
        pairs = []
        for r in rs:
            doc = r["document_id"]
            for f in r["invalid_fields"]:
                null_fill = r["field_errors"][f]["source"]["null_fill"]
                assert null_fill == (clean[doc][f] is None), (doc, et, k, f, "null_fill disagrees with the clean row")
                if null_fill:
                    cands = [d for d in sorted(clean) if d != doc and clean[d][f] is not None]
                    assert cands, (doc, et, k, f, "no other document has this field for a null-fill twin")
                    twin_doc = random.Random(f"{seed}:{doc}:{et}:{k}:{f}").choice(cands)
                else:
                    twin_doc = doc
                pairs.append({"error_type": et, "k": k, "field": f, "null_fill": null_fill,
                              "twin": "other_document" if null_fill else "same_document",
                              "invalid": span(doc, et, k, f), "valid": span(twin_doc, None, 0, f)})
        same = [(p["valid"]["document_id"], p["field"]) for p in pairs if not p["null_fill"]]
        assert len(same) == len(set(same)), (et, k, "a same-document twin occurs twice in one stratum")
        out += pairs
    return out


def kept_span_pairs(rows):
    """Uncorrupted shown spans of corrupted rows, each paired with the same field in the clean row.

    Args:
        rows: one split's ``main/{split}.json`` rows.

    Returns:
        One dict per kept span: ``{error_type, k, field, kept: span, clean: span}``.
    """
    clean = _clean_rows(rows)
    out = []
    for (et, k), rs in _strata(rows).items():
        for r in rs:
            doc = r["document_id"]
            for f in MAIN_FIELDS:
                if r[f] is None or f in r["invalid_fields"]:
                    continue
                assert r[f] == clean[doc][f], (doc, et, k, f, "an unmarked field differs from the clean row")
                out.append({"error_type": et, "k": k, "field": f, "kept": span(doc, et, k, f), "clean": span(doc, None, 0, f)})
    return out
