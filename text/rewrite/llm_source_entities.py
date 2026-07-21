"""Extract candidate entity/attribute names from a description via one LLM
call over the source text alone -- the level-minus-three counterpart to
``text.rewrite.source_entities`` (frequency-based) and ``gold_entities``
(gold PlantUML-based, leaky).

Level minus-two's frequency heuristic (repeated noun-chunk head lemmas)
can't distinguish a real class/attribute name from any other word that
happens to repeat, and it fell *below* the L0 baseline (F1 0.405 vs 0.425).
This module leans on the model's own UML-modeling judgment instead --
approximating the same reasoning a human modeler applies when building the
gold diagram -- entirely from the source description, no gold access.
"""

from __future__ import annotations

from typing import List

from .client import rewrite_once
from .config import RewriteConfig


def _extraction_system_prompt() -> str:
    return (
        "You are a UML modeling analyst. Given a software system "
        "description, extract every candidate CLASS name and ATTRIBUTE "
        "name that a UML class diagram would need to represent this "
        "system -- the nouns that would become class names or field names, "
        "not incidental descriptive words.\n\n"
        "Output ONLY a plain list, one name per line, no numbering, no "
        "bullets, no headers, no explanation. Use the exact word or short "
        "phrase as it appears in the source (singular form is fine)."
    )


def _build_extraction_user_prompt(source_text: str) -> str:
    return (
        "SOURCE DESCRIPTION:\n"
        f"<source>\n{source_text}\n</source>\n\n"
        "List every candidate class and attribute name from this "
        "description, one per line."
    )


def extract_llm_source_entities(client, cfg: RewriteConfig, source_text: str) -> List[str]:
    """One-shot LLM extraction of candidate entity/attribute names from
    ``source_text`` alone -- no gold PlantUML involved.

    Reuses ``rewrite_once`` for the call since it's already a generic
    (system, user) -> text function; this just isn't a *rewrite* call, it's
    an extraction call, so ``cfg.max_iterations``/the recall_density loop
    don't apply here -- callers run this once, before the loop.
    """
    raw = rewrite_once(client, cfg, _extraction_system_prompt(), _build_extraction_user_prompt(source_text))
    names = []
    for line in raw.splitlines():
        name = line.strip().lstrip("-*•0123456789. ").strip().lower()
        if name:
            names.append(name)
    return sorted(set(names))
