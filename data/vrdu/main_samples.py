"""Entity-level evaluation pairs of the VRDU main dataset: each corrupted span with its patched counterpart.

A span is one shown field of one main row, keyed ``(document_id, error_type, k, field)`` (``error_type``
None and ``k`` 0 for the valid row), plus ``patch_target`` for a patched row. Per corrupted row:

  patched pairs  every corrupted span whose original value is not null, paired with the same field in the
                 PATCHED row: the corrupted row with only that field restored (``patched_rows``). Nothing else
                 in the context differs, at any k. For k = 1 the patched row is the document's clean row.
                 A null-fill span has no patched counterpart (the original shows nothing) and is dropped.
  kept pairs     every uncorrupted shown span of a corrupted row, paired with the same field in the
                 document's clean row. Both are valid: a diagnostic for a probe that only senses "this row
                 has a corruption somewhere".

Each (error type, k, field) stratum is exactly 50/50 and every pair shares its document.
"""

from __future__ import annotations

import copy

from vrdu_common import MAIN_FIELDS

ERROR_TYPES = ("inter_document", "cross_type")


def span(document_id, error_type, k, field, patch_target=None):
    """A span key. ``patch_target``: the restored field of a patched row (None for an ordinary row)."""
    return {"document_id": document_id, "error_type": error_type, "k": k, "field": field, "patch_target": patch_target}


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


def patched_rows(rows):
    """One patched row per non-null-fill corrupted span of every row with k >= 2.

    A patched row is its corrupted parent with ONLY the target field restored to the clean value; the other
    k - 1 corruptions stay. Keys: the parent's, with ``k`` / ``num_invalid_fields`` / ``invalid_fields`` /
    ``field_errors`` updated, plus ``patch_target`` (the restored field) and ``parent_k`` (the parent's k). Its row
    identity is ``(document_id, error_type, k, patch_target)``. A k = 1 parent needs no patched row (its patch
    is the document's clean row), and a null-fill span has none (the clean value is null: nothing to show).

    Args:
        rows: one split's ``main/{split}.json`` rows.

    Returns:
        The patched rows, sorted by (document, error type, parent k, target field order).

    Raises:
        AssertionError: a malformed row set, or a patched row that differs from its parent elsewhere.
    """
    clean = _clean_rows(rows)
    out = []
    for (et, k), rs in _strata(rows).items():
        assert {r["document_id"] for r in rs} <= set(clean), (et, k, "invalid row without a clean row")
        if k == 1:
            continue
        for r in rs:
            doc = r["document_id"]
            for f in r["invalid_fields"]:
                null_fill = r["field_errors"][f]["source"]["null_fill"]
                assert null_fill == (clean[doc][f] is None), (doc, et, k, f, "null_fill disagrees with the clean row")
                if null_fill:
                    continue
                q = copy.deepcopy(r)
                q[f] = clean[doc][f]
                q["field_errors"][f] = None
                q["invalid_fields"] = [g for g in r["invalid_fields"] if g != f]
                q["k"] = q["num_invalid_fields"] = k - 1
                q["patch_target"], q["parent_k"] = f, k
                assert q["invalid_fields"] and not q["valid"]
                assert all(q[g] == r[g] for g in MAIN_FIELDS if g != f), (doc, et, k, f, "patch changed another field")
                assert q[f] != r[f], (doc, et, k, f, "patch left the value unchanged")
                out.append(q)
    return out


def eval_rows(rows):
    """The evaluation row file of a split: every row with ``patch_target`` None and ``parent_k`` its own ``k``,
    then the ``patched_rows``. Every row carries both keys."""
    base = [{**r, "patch_target": None, "parent_k": r["k"]} for r in rows]
    return base + patched_rows(rows)


def split_eval_rows(rows):
    """``(original rows, patched rows)`` of an ``eval_rows`` file, the originals without the two extra keys.

    Raises:
        AssertionError: a row without ``patch_target`` / ``parent_k``, or an original whose ``parent_k`` != ``k``.
    """
    assert all("patch_target" in r and "parent_k" in r for r in rows), "not an eval row file (patch_target / parent_k)"
    orig = [r for r in rows if r["patch_target"] is None]
    assert all(r["parent_k"] == r["k"] for r in orig)
    return [{k: v for k, v in r.items() if k not in ("patch_target", "parent_k")} for r in orig], [r for r in rows if r["patch_target"] is not None]


def patched_entity_pairs(rows, patched):
    """Each non-null-fill corrupted span paired with the SAME span in its patched row.

    The members differ only in the target's value (and, for a causal model, nothing before the target changes).
    For k = 1 the patched row is the document's clean row.

    Args:
        rows: one split's ``main/{split}.json`` rows (the whole file or any subset closed under documents).
        patched: ``patched_rows`` of those rows (or of a superset: rows of absent parents are ignored).

    Returns:
        One dict per pair: ``{error_type, k, field, invalid: span, valid: span}`` (``k`` the corrupted row's).
        Spans are keyed ``(document_id, error_type, k, field)`` plus ``patch_target`` (None unless a patched row).

    Raises:
        AssertionError: a malformed row set or a missing patched row.
    """
    clean = _clean_rows(rows)
    by_patch = {(q["document_id"], q["error_type"], q["parent_k"], q["patch_target"]): q for q in patched}
    assert len(by_patch) == len(patched), "patched rows are not unique"
    out, used = [], set()
    for (et, k), rs in _strata(rows).items():
        assert {r["document_id"] for r in rs} <= set(clean), (et, k, "invalid row without a clean row")
        for r in rs:
            doc = r["document_id"]
            for f in r["invalid_fields"]:
                if r["field_errors"][f]["source"]["null_fill"]:
                    continue
                if k == 1:
                    assert clean[doc][f] is not None and clean[doc][f] != r[f], (doc, et, f)
                    valid = span(doc, None, 0, f)
                else:
                    q = by_patch[(doc, et, k, f)]
                    assert q[f] == clean[doc][f], (doc, et, k, f, "patched target is not the clean value")
                    used.add((doc, et, k, f))
                    valid = span(doc, et, k - 1, f, patch_target=f)
                out.append({"error_type": et, "k": k, "field": f, "invalid": span(doc, et, k, f), "valid": valid})
    parents = {(r["document_id"], r["error_type"], r["k"]) for r in rows if not r["valid"]}
    assert used == {key for key in by_patch if key[:3] in parents}, "a patched row of these rows was not used"
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
