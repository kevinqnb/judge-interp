"""Dataset specification: everything the judge tasks need to know about one dataset.

The task modules (``entity_detection``, ``relation_detection``, ``clef``) contain only
dataset-independent templates and logic. A ``DatasetSpec`` supplies the vocabulary that
fills the templates (field names and descriptions, how a record and its context are described to the judge)
and the layout of the data on disk. One spec
describes one row-set (for VRDU, ``main`` and ``line`` are two specs).

Specs live in ``experiments/dataset-configs/<name>.py`` as a module-level ``DATASET`` and
are loaded with ``load_dataset_spec``. Every field is required and validated; there are no
defaults.

No torch or nnsight is imported, so this module is testable on a login node.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

SPEC_ATTRIBUTE = "DATASET"


@dataclass(frozen=True)
class DatasetSpec:
    """Vocabulary and on-disk layout of one dataset.

    The prompts are dataset-independent; the only dataset wording they receive is the
    field list, the two descriptive phrases and (optionally) one example.

    Attributes:
        name: short identifier ("main", "line"). Written to the ``dataset`` span column and
            used to derive per-stratum draw seeds, so changing it changes which rows are drawn.
        fields: the entity fields shown to the judge, in rendering order.
        field_descriptions: field -> what it denotes, shown in the instructions; keys must
            equal ``fields``.
        error_types: invalid-row error types the split file may contain.
        record_description: what one row is, as a noun phrase without article
            ("a line item of the document's order").
        context_description: what the CONTEXT block is, without article ("OCR text of the
            source document").
        type_confusion_hint: optional example of a value that is in the document but fails
            the type criterion, following "For example, " and without a final full stop;
            ``None`` to give no example.
        row_index_key: row key holding the row's index within its document, or ``None`` if
            a document has one row.
        split_path: path of a split file relative to ``data_root``; must contain ``{split}``.
        context_path: path of a document's context file relative to ``data_root``; must
            contain ``{document_id}``.
        supports_relation: whether rows are one of several similar items per document, so
            that "do these entities belong to the same item?" is a meaningful question.
            The relation tasks refuse a dataset where this is false.
        ordered_fields: whether each row lists its shown fields, in the order shown, under
            ``fields`` (exactly its non-null fields). If false, the non-null fields are shown
            in the order of ``fields`` above.
        pair_key: row key holding the id of the row's minimal pair (one invalid row and its
            clean twin), or ``None`` if rows are not paired. Paired rows also carry ``cell``,
            their stratum, and are drawn pair by pair.
    """

    name: str
    fields: tuple[str, ...]
    field_descriptions: dict[str, str]
    error_types: tuple[str, ...]
    record_description: str
    context_description: str
    type_confusion_hint: str | None
    row_index_key: str | None
    split_path: str
    context_path: str
    supports_relation: bool
    ordered_fields: bool
    pair_key: str | None

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        for name in ("fields", "error_types"):
            v = getattr(self, name)
            if not isinstance(v, tuple) or not v or not all(isinstance(x, str) and x for x in v):
                raise ValueError(f"{name} must be a non-empty tuple of non-empty str, got {v!r}")
            if len(set(v)) != len(v):
                raise ValueError(f"{name} has duplicates: {v!r}")
        if set(self.field_descriptions) != set(self.fields):
            raise ValueError(
                f"field_descriptions keys {sorted(self.field_descriptions)} != fields {sorted(self.fields)}"
            )
        for f, d in self.field_descriptions.items():
            _require_text(d, f"field_descriptions[{f!r}]")
        for name in ("record_description", "context_description"):
            _require_text(getattr(self, name), name)
        for name in ("type_confusion_hint", "row_index_key", "pair_key"):
            if getattr(self, name) is not None:
                _require_text(getattr(self, name), name)
        if "{split}" not in self.split_path:
            raise ValueError(f"split_path must contain '{{split}}', got {self.split_path!r}")
        if "{document_id}" not in self.context_path:
            raise ValueError(f"context_path must contain '{{document_id}}', got {self.context_path!r}")
        for name in ("supports_relation", "ordered_fields"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool, got {getattr(self, name)!r}")
        if self.supports_relation and len(self.fields) < 2:
            raise ValueError("a dataset with a relation task needs at least two fields")

    def require_relation(self) -> None:
        """Raise if the dataset does not support the relation tasks.

        Raises:
            ValueError: ``supports_relation`` is false.
        """
        if not self.supports_relation:
            raise ValueError(f"dataset {self.name!r} does not support the relation tasks")


def _require_text(value, where: str) -> None:
    """Raise ``ValueError`` unless ``value`` is a non-empty ``str``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty str, got {value!r}")


def load_dataset_spec(path: str | Path) -> DatasetSpec:
    """Load the ``DATASET`` object of a dataset-config module.

    The config directory name contains a hyphen, so the file is imported by path rather
    than as a package.

    Args:
        path: a ``.py`` file defining ``DATASET``, a ``DatasetSpec``.

    Returns:
        The spec.

    Raises:
        FileNotFoundError: no such file.
        AttributeError: the module has no ``DATASET``.
        TypeError: ``DATASET`` is not a ``DatasetSpec``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such dataset config: {path}")
    module_spec = importlib.util.spec_from_file_location(f"dataset_config_{path.stem}", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    if not hasattr(module, SPEC_ATTRIBUTE):
        raise AttributeError(f"{path}: module defines no {SPEC_ATTRIBUTE!r}")
    spec = getattr(module, SPEC_ATTRIBUTE)
    if not isinstance(spec, DatasetSpec):
        raise TypeError(f"{path}: {SPEC_ATTRIBUTE} must be a DatasetSpec, got {type(spec).__name__}")
    return spec
