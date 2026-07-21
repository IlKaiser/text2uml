"""Extract entity names AND relationship pairs from a description's source
text (no gold PlantUML) -- the level-minus-four counterpart to
``text.rewrite.llm_source_entities``.

Level minus-two (frequency-based) and level minus-three (LLM-extracted)
both plateaued ~5% *below* the L0 baseline despite near-100% recall of
their own target entity list -- entity-name recall alone gives the rewrite
no signal to protect relationship/cardinality facts, and compression
pressure seems to trade those away. This extracts relationship pairs too,
so the recall_density objective can credit keeping *both* endpoints of a
relationship in the same line, not just naming entities somewhere.
"""

from __future__ import annotations

from typing import List, Tuple

from .client import rewrite_once
from .config import RewriteConfig


def _extraction_system_prompt() -> str:
    return (
        "You are a UML modeling analyst. Given a software system "
        "description, extract:\n"
        "1. Every candidate CLASS name and ATTRIBUTE name a UML class "
        "diagram would need.\n"
        "2. Every RELATIONSHIP between two classes (which class connects "
        "to which -- ignore multiplicity/direction, just the pair).\n\n"
        "Output in exactly this format, nothing else:\n"
        "ENTITIES:\n"
        "<one name per line>\n\n"
        "RELATIONS:\n"
        "<ClassA> -> <ClassB>\n"
        "<one pair per line, using the same names as in ENTITIES>\n\n"
        "No numbering, no bullets, no explanation. Use the exact word or "
        "short phrase as it appears in the source."
    )


def _build_extraction_user_prompt(source_text: str) -> str:
    return (
        "SOURCE DESCRIPTION:\n"
        f"<source>\n{source_text}\n</source>\n\n"
        "Extract every candidate class/attribute name and every "
        "relationship pair, in the ENTITIES:/RELATIONS: format."
    )


def extract_entities_and_relations(
    client, cfg: RewriteConfig, source_text: str
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """(entity names, relationship pairs) extracted from ``source_text``
    alone via one LLM call -- no gold PlantUML involved."""
    raw = rewrite_once(client, cfg, _extraction_system_prompt(), _build_extraction_user_prompt(source_text))

    entities: List[str] = []
    relations: List[Tuple[str, str]] = []
    section = None
    for line in raw.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("ENTITIES"):
            section = "entities"
            continue
        if upper.startswith("RELATIONS"):
            section = "relations"
            continue
        cleaned = stripped.lstrip("-*•0123456789. ").strip()
        if not cleaned:
            continue
        if section == "entities":
            entities.append(cleaned.lower())
        elif section == "relations" and "->" in cleaned:
            a, b = cleaned.split("->", 1)
            a, b = a.strip().lower(), b.strip().lower()
            if a and b:
                relations.append((a, b))

    return sorted(set(entities)), sorted(set(relations))
