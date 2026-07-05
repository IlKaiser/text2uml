"""Prompt construction for the level-zero "structured UML notes" genre.

Unlike level one/two (linguistic simplification of the narrative), level zero
is a different genre entirely: a compact "Classes / Relationships" analyst
fact-sheet, modelled on the existing GasStation_KUL / GasStation_TUW
references. That genre naturally produces the target shape for three
metrics — high mdd (parenthetical/appositive density), low
subordination_index (no subordinating conjunctions), high
context_dependence_proxy (pronoun back-reference instead of repeated names)
— so the prompt encodes those stylistic rules directly instead of chasing an
aggregate complexity number.
"""

from __future__ import annotations

from typing import Dict, Optional

STRUCTURAL_METRIC_GUIDANCE: Dict[str, str] = {
    "mdd": "pack qualifiers onto each entity mainly via commas and "
    "parentheses (e.g. 'many Pumps (1 -> 0..*)') rather than participial "
    "clauses — parenthetical/comma packing raises dependency distance "
    "without adding a subordinate clause",
    "subordination_index": "avoid not just subordinating conjunctions (when, "
    "because, although, since, if) but also participial/appositive clause "
    "modifiers (e.g. '...Pumps, each belonging to one station') — the "
    "parser counts those as subordination too. Use at most ONE such "
    "participial phrase in the entire document (for the single most "
    "important relationship); pack every other qualifier via parentheses "
    "or commas instead",
    "context_dependence_proxy": "after an entity's first mention, refer back "
    "to it with a pronoun (it, its, they) instead of repeating its full name",
}


def structural_system_prompt() -> str:
    """The level-zero persona: compact UML analyst notes, not simplified prose."""
    return (
        "You are a UML analyst producing a compact fact-sheet from a software "
        "system description. Convert the description into two sections:\n\n"
        "Classes\n"
        "One line per class: Name — short attribute/role list.\n\n"
        "Relationships\n"
        "One line per relationship: state the two classes, the multiplicity "
        "(e.g. '1 -> 0..*'), and any qualifying condition, packed into a "
        "single sentence via commas/parentheses/participial clauses.\n\n"
        "Style rules (these produce the target notes genre):\n"
        "- Pack qualifiers onto each entity mainly via commas and "
        "parentheses (e.g. 'many Pumps (1 -> 0..*)') instead of a separate "
        "sentence — this raises dependency distance without adding a "
        "subordinate clause.\n"
        "- Never use subordinating conjunctions (when, because, although, "
        "since, if). Also avoid participial/appositive clause modifiers "
        "(e.g. '...Pumps, each belonging to one station') — a parser counts "
        "those as subordination too, exactly like a 'because' clause. Use "
        "at most ONE such participial phrase in the entire document, for "
        "the single most important relationship; pack every other "
        "qualifier via parentheses or commas instead.\n"
        "- After an entity's first mention, refer back to it with a pronoun "
        "(it, its, they) instead of repeating its full name.\n\n"
        "Hard rules:\n"
        "- Never add facts, entities, relationships, or constraints not in "
        "the source.\n"
        "- Never remove or merge any entity, attribute, relationship, "
        "multiplicity, action, or constraint from the source.\n"
        "- Output ONLY the rewritten notes in Markdown. No preamble, no "
        "explanation, no commentary."
    )


def build_structural_user_prompt(
    original: str, current_text: str, feedback: Optional[str]
) -> str:
    """Assemble the per-iteration structural-rewrite request."""
    feedback_block = f"\nFEEDBACK ON YOUR LAST ATTEMPT:\n{feedback}\n" if feedback else ""
    return (
        "TASK: rewrite the description as compact 'Classes / Relationships' "
        "UML analyst notes, per the style rules in the system prompt.\n"
        f"{feedback_block}\n"
        "SOURCE OF TRUTH — preserve every fact, entity, attribute, "
        "relationship, action, and constraint from here:\n"
        f"<source>\n{original}\n</source>\n\n"
        "VERSION TO IMPROVE — rewrite this, keeping all meaning from the "
        "source:\n"
        f"<current>\n{current_text}\n</current>\n\n"
        "Return only the rewritten Markdown notes."
    )
