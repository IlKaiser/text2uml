"""Extract candidate entity/attribute names directly from a description's
source text (no gold PlantUML) -- the fair, deployable counterpart to
``text.rewrite.gold_entities`` for the level-minus-two objective.

Level minus-one (``text.rewrite.recall_prompts``) optimizes recall_density
against gold PlantUML entity names, which a real pipeline would never have
in advance -- the gold UML is exactly what generation is trying to produce,
so scoring the rewrite against it is leaky. Level minus-two tests the same
recall_density objective using only information available from the
description itself.
"""

from __future__ import annotations

from collections import Counter
from typing import List

from ..metrics.base import parse


def extract_source_entities(text: str, model_name: str = "en_core_web_sm", min_count: int = 2) -> List[str]:
    """Repeated noun-chunk head lemmas in ``text`` -- a source-only proxy for
    the entity/attribute vocabulary a UML model would be built from.

    Frequency-filtered (``min_count``) rather than NER-based: these
    descriptions name domain-specific classes ("Pump", "SeatCategory") that
    a general-purpose NER model won't recognize as entities, but which
    recur every time the text refers back to that concept -- repetition is
    the available signal, not proper-noun capitalization (many of this
    corpus's descriptions render domain terms as lowercase common nouns).
    """
    doc = parse(text, model_name)
    counts: Counter = Counter()
    for chunk in doc.noun_chunks:
        head = chunk.root
        if head.pos_ in ("NOUN", "PROPN") and head.is_alpha:
            counts[head.lemma_.lower()] += 1
    return sorted(name for name, count in counts.items() if count >= min_count)
