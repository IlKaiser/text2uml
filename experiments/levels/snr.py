"""L3 description signal-to-noise ratio.

For each case's real (L3) description, measures how much of the description's
content maps to something that actually appears in the gold ``plantuml.txt``
diagram ("signal") versus narrative/business-rule elaboration that never
surfaces in the diagram ("noise"). See
``docs/superpowers/specs/2026-07-06-l3-signal-noise-ratio-design.md``.
"""

from __future__ import annotations

import logging
import re
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


_LABEL_LINE_RE = re.compile(r"^\s*(\d+)\s*:\s*(SIGNAL|NOISE)\s*$", re.IGNORECASE)

_CLASSIFICATION_PROMPT = """You will be given a numbered list of sentences from a software specification, \
and four lists of UML diagram components (classes, attributes, associations, inheritance edges) that were \
manually modeled from that same specification.

For each sentence, decide:
- SIGNAL: the sentence introduces, describes, or gives a cardinality/relationship for at least one of the \
listed components.
- NOISE: the sentence is narrative elaboration, an example, a business rule, or a process description that \
does not correspond to any listed component.

Output exactly one line per sentence, in the form "N: SIGNAL" or "N: NOISE", where N is the sentence number. \
Output nothing else.

##############

Classes: {classes}

Attributes: {attributes}

Associations: {associations}

Inheritance: {inheritance}

Sentences:
{sentences}

##############

Labels:
"""


def _invoke_classification_chain(sentences: List[Sentence], gold: GoldComponents) -> str:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    numbered = "\n".join(f"{i}. {s.text}" for i, s in enumerate(sentences, start=1))
    prompt = PromptTemplate.from_template(_CLASSIFICATION_PROMPT)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | model | StrOutputParser()
    return chain.invoke({
        "classes": list(gold.classes),
        "attributes": list(gold.attributes),
        "associations": list(gold.associations),
        "inheritance": list(gold.inheritance),
        "sentences": numbered,
    })


def _heuristic_labels(sentences: List[Sentence], gold: GoldComponents) -> List[str]:
    names = [n.lower() for n in gold.all_names()]
    labels = []
    for s in sentences:
        low = s.text.lower()
        labels.append("SIGNAL" if any(n in low for n in names) else "NOISE")
    return labels


def classify_sentences(sentences: List[Sentence], gold: GoldComponents) -> List[str]:
    if not sentences:
        return []
    try:
        response = _invoke_classification_chain(sentences, gold)
        parsed: dict[int, str] = {}
        for line in response.splitlines():
            m = _LABEL_LINE_RE.match(line)
            if m:
                parsed[int(m.group(1))] = m.group(2).upper()
        if not parsed:
            raise ValueError("no parseable SIGNAL/NOISE lines in LLM response")
        return [parsed.get(i, "NOISE") for i in range(1, len(sentences) + 1)]
    except Exception as exc:  # noqa: BLE001 - any LLM/parsing failure falls back
        logger.warning("Sentence classification failed (%s); using substring heuristic.", exc)
        return _heuristic_labels(sentences, gold)
