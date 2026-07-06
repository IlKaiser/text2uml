"""L3 description signal-to-noise ratio.

For each case's real (L3) description, measures how much of the description's
content maps to something that actually appears in the gold ``plantuml.txt``
diagram ("signal") versus narrative/business-rule elaboration that never
surfaces in the diagram ("noise"). See
``docs/superpowers/specs/2026-07-06-l3-signal-noise-ratio-design.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from text.metrics.base import parse as spacy_parse

from .config import DEFAULT_LEVELS_CONFIG, LevelsConfig
from .evaluate import load_evaluator

logger = logging.getLogger(__name__)


def _eval_module(cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG):
    """Indirection point so tests can stub out src/eval.py's parser."""
    return load_evaluator(cfg)


@dataclass(frozen=True)
class GoldComponents:
    """Named UML components extracted from a gold plantuml.txt."""

    classes: Tuple[str, ...]
    attributes: Tuple[str, ...]
    associations: Tuple[str, ...]
    inheritance: Tuple[str, ...]

    def all_names(self) -> Tuple[str, ...]:
        return self.classes + self.attributes


def _class_name(raw: str) -> str:
    """Strip an embedded role name (e.g. "Pump, refillingPump" -> "Pump")."""
    return raw.split(",")[0].strip()


def gold_components(gold_path: Path, parser) -> GoldComponents:
    """Extract classes, attributes, associations, and inheritance edges from a gold plantuml.txt."""
    ev = _eval_module()
    classes, relationships, attributes, inheritance = ev.parse_path(str(gold_path), parser)

    attrs = tuple(f"{cls}.{attr}" for cls, attr in attributes)
    assocs = []
    for rel in relationships:
        names = [_class_name(k) for k in rel.keys()]
        assocs.append(" -- ".join(names))
    inh = tuple(f"{child} <|-- {parent}" for parent, child in inheritance)

    return GoldComponents(
        classes=tuple(classes),
        attributes=attrs,
        associations=tuple(assocs),
        inheritance=inh,
    )


@dataclass(frozen=True)
class Sentence:
    """One sentence from a description, with its spaCy token count."""

    text: str
    n_tokens: int


def split_sentences(text: str) -> List[Sentence]:
    """Split text into sentences via spaCy, dropping empty ones."""
    doc = spacy_parse(text)
    sentences: List[Sentence] = []
    for span in doc.sents:
        stripped = span.text.strip()
        if not stripped:
            continue
        sentences.append(Sentence(text=stripped, n_tokens=len(span)))
    return sentences
