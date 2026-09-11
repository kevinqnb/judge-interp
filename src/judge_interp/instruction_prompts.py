"""Instruction strings for the extraction-validation judge.

A judge instance is given three blocks — the instructions built here, the full
OCR ``## CONTEXT:``, and a single extracted record in ``## QUERY:`` — and must
answer whether the record is a valid extraction from that context.

``build_instructions`` takes an explicit ``include_cohesion`` flag rather than
defaulting it: the cohesion criterion is the one that discriminates the ``line``
invalids (values swapped for other values that *do* appear on the same page), so
dropping it is the ablation-direction control, and an ablation you can turn off
by forgetting a keyword argument is not one.
"""
from __future__ import annotations

# The judge's fixed role and the mechanics of the verdict. `{criteria}` is filled
# with the numbered list built below.
_HEADER = """You are validating a single structured record extracted from a document by an information-extraction pipeline.

You are given:
1) In ## CONTEXT: the full OCR text of the source document.
2) In ## QUERY: one extracted record, as `field: value` lines. A value of `null` means the pipeline left that field empty.

Decide whether the record is a VALID extraction. A record is valid only if EVERY field satisfies all of the criteria below. If even one field fails any criterion, the whole record is INVALID.

Criteria:
{criteria}
Judge only against the context. Do not use outside knowledge about the entities involved, and do not guess when the evidence is unclear — unclear evidence is a failed criterion.

Respond with exactly one token, lowercase, no punctuation: `true` if the record is valid, `false` otherwise."""

# Always present.
_EXPLICIT_CRITERION = """({n}) Explicit support. Every non-null value must appear explicitly in the context. A value may differ from the context only in trivial surface formatting (whitespace, line breaks, letter case, and equivalent number forms such as 10 vs 10.0 or 1,000 vs 1000). A value produced by paraphrasing, summarising, inferring, rounding, unit conversion, or any other transformation of the text is NOT explicit support. A `null` value always satisfies this criterion."""

# Included only when include_cohesion=True.
_COHESION_CRITERION = """({n}) Cohesion. When the context contains more than one candidate value for a field, the values chosen across the record's fields must all describe the same real-world entity. For example, if the context contains two addresses, "63 Chestnut St. Boston, MA" and "44 Milton St. Albany, NY", then the record {{street: 44 Milton St., city: Boston, state: NY}} is invalid: the street is drawn from the second address, so a cohesive record must use city Albany and state NY. A value that is individually present in the context but belongs to a different entity than the record's other values fails this criterion."""


def build_instructions(include_cohesion: bool) -> str:
    """Assemble the judge instruction block.

    Args:
        include_cohesion: When True, the cohesion criterion is included. When
            False, only the explicit-support criterion is stated — the record is
            judged on whether each value appears somewhere in the context,
            regardless of which entity it belongs to.

    Returns:
        The instruction string, ready to pass as the ``instructions`` block.
    """
    if not isinstance(include_cohesion, bool):
        raise TypeError(f"include_cohesion must be a bool, got {type(include_cohesion)!r}")

    criteria = [_EXPLICIT_CRITERION.format(n=1)]
    if include_cohesion:
        criteria.append(_COHESION_CRITERION.format(n=2))
    criteria_block = "\n".join(criteria)
    return _HEADER.format(criteria=criteria_block)
