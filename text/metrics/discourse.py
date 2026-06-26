"""Discourse / pragmatic complexity proxies.

The paper introduces three constructs that have no single observable definition:
Rhetorical Structure Theory depth (RST), the Context Dependence Ratio (CDR), and
the Inferential Load Index (ILI). Faithful computation would require a discourse
parser or an LLM. This module provides **deterministic surface proxies** so the
whole metric suite runs offline and reproducibly. Each class docstring states
exactly which surface signal stands in for the theoretical construct.
"""

from __future__ import annotations

import logging
from typing import Set

from .base import Metric

logger = logging.getLogger(__name__)

# Cue phrases grouped by the rhetorical relation they typically signal.
_RST_CUES: Set[str] = {
    # Elaboration / Addition
    "moreover", "furthermore", "additionally", "in addition", "also", "besides",
    # Contrast
    "but", "however", "whereas", "while", "yet", "nevertheless", "nonetheless",
    "on the other hand", "in contrast", "instead",
    # Concession
    "although", "though", "even though", "despite", "in spite of", "regardless",
    # Cause / Motivation
    "because", "since", "therefore", "thus", "hence", "so that", "as a result",
    "consequently", "due to",
    # Condition
    "if", "unless", "provided that", "in case", "otherwise",
}

# Connectives that signal an inference the reader must perform.
_INFERENTIAL_CONNECTIVES: Set[str] = {
    "therefore", "thus", "hence", "consequently", "so", "because", "since",
    "as a result", "which means", "implies", "suggests", "indicates",
    "in order to", "so that",
}


def _text_lower(doc) -> str:
    return doc.text.lower()


def _count_cues(text: str, cues: Set[str]) -> int:
    """Count occurrences of cue phrases (substring match for multiword cues)."""
    return sum(text.count(cue) for cue in cues)


class RSTDepthProxy(Metric):
    """Proxy for RST rhetorical-tree richness.

    Surface signal: density of rhetorical cue phrases (Elaboration, Contrast,
    Concession, Cause, Condition markers) per sentence. More relational cues imply
    a deeper / busier discourse tree connecting nuclei and satellites.
    """

    name = "rst_depth_proxy"

    def compute(self, doc) -> float:
        sentences = sum(1 for _ in doc.sents)
        if sentences == 0:
            return float("nan")
        cues = _count_cues(_text_lower(doc), _RST_CUES)
        return cues / sentences


class ContextDependenceProxy(Metric):
    """Proxy for the Context Dependence Ratio, CDR = 1 - Iexplicit / Ireconstructed.

    Surface signal: the share of referring expressions whose referent is not made
    explicit in the immediately preceding local context. We approximate it as the
    fraction of pronouns and definite noun phrases that have no resolvable local
    antecedent (no matching head lemma earlier in the document), i.e. information
    the reader must reconstruct rather than read off the text.
    """

    name = "context_dependence_proxy"

    _PRONOUN_TAGS = {"PRP", "PRP$", "DT"}

    def compute(self, doc) -> float:
        referring = 0
        unresolved = 0
        seen_lemmas: Set[str] = set()

        for token in doc:
            is_pronoun = token.pos_ == "PRON"
            is_definite_np = (
                token.pos_ in {"NOUN", "PROPN"}
                and any(c.lemma_.lower() == "the" for c in token.children)
            )
            if is_pronoun:
                referring += 1
                # Personal/possessive pronouns nearly always point outside the NP.
                if token.lemma_.lower() in {"it", "they", "he", "she", "this", "that", "these", "those"}:
                    unresolved += 1
            elif is_definite_np:
                referring += 1
                if token.lemma_.lower() not in seen_lemmas:
                    unresolved += 1

            if token.pos_ in {"NOUN", "PROPN"}:
                seen_lemmas.add(token.lemma_.lower())

        if referring == 0:
            return 0.0
        return unresolved / referring


class InferentialLoadProxy(Metric):
    """Proxy for the Inferential Load Index, ILI = sum_i I(c_i).

    Surface signal: per-sentence count of inference triggers - inferential
    connectives, modal/conditional verbs, and coordination/ellipsis gaps - each
    standing for a reasoning step the reader must take to reconstruct meaning.
    """

    name = "inferential_load_proxy"

    def compute(self, doc) -> float:
        sentences = sum(1 for _ in doc.sents)
        if sentences == 0:
            return float("nan")

        load = _count_cues(_text_lower(doc), _INFERENTIAL_CONNECTIVES)
        for token in doc:
            # Modals and conditional auxiliaries add interpretive steps.
            if token.tag_ == "MD":
                load += 1
            # Coordinated clauses sharing an elided subject/verb add a gap.
            if token.dep_ == "conj" and token.pos_ in {"VERB", "AUX"}:
                load += 1
        return load / sentences
