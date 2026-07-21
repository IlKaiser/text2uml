"""Prompt construction for the level-six "flat prose, min-gated recall" genre.

Combines two genres that previously stayed separate:

* level four's flattening rules (``text.rewrite.flat_prompts``) -- minimize
  parse_tree_depth by writing one clause per sentence, no relative clauses,
  no stacked modifiers, no subordination/coordination joining two clauses.
* the recall-density family's exact-naming and same-sentence relationship
  rules (``text.rewrite.recall_prompts``) -- but NOT that family's compact
  "Classes:"/"Relationships:" notes format. Level six's output must read as
  ordinary human prose (short paragraphs, full sentences), the same
  register as level four's output, not a machine-readable schema -- entity
  and relationship recall are scored against that prose directly (see
  ``text.rewrite.ptd_recall_loop``).
"""

from __future__ import annotations

from typing import List, Optional


def ptd_recall_system_prompt() -> str:
    """The level-six persona: flatten syntax AND protect every entity name
    and relationship pair, while staying in ordinary prose."""
    return (
        "You are an expert technical editor. You rewrite software system "
        "descriptions into plain, human-readable prose that is as shallow "
        "and easy to parse as possible -- for a human reader and for a "
        "dependency parser alike -- WITHOUT changing their meaning. The "
        "result must read like a normal paragraph-based specification, "
        "never like compact notes, bullet lists, or a machine-readable "
        "schema.\n\n"
        "These descriptions are used to automatically generate UML models, "
        "so every piece of modelling-relevant information must be "
        "preserved, using the SOURCE'S OWN WORDING for each name wherever "
        "possible (exact matches are what get credit):\n"
        "- every actor, entity, and class, by name\n"
        "- every attribute and piece of data each entity holds\n"
        "- every relationship and association between entities, including "
        "multiplicity and direction -- state BOTH sides of a relationship "
        "in the SAME sentence, never split across two sentences\n"
        "- every action, operation, and use case, and which actor performs it\n"
        "- every business rule, constraint, and condition\n\n"
        "Hard rules:\n"
        "- Never add facts, entities, relationships, or constraints not in "
        "the source.\n"
        "- Never remove or merge any of the information listed above.\n"
        "- Change only the linguistic FORM, never the semantic content.\n"
        "- Keep English, and keep roughly the same order of information.\n"
        "- Write full sentences grouped into short paragraphs, like a "
        "normal specification document -- NOT bullet points, NOT a "
        "'Classes:' / 'Relationships:' notes format.\n"
        "- Output ONLY the rewritten description in Markdown prose. No "
        "preamble, no explanation, no commentary.\n\n"
        "To minimize how deeply nested each sentence's grammar is (the "
        "other goal, alongside keeping every name and relationship above):\n"
        "- One clause per sentence. Never use relative clauses ('which', "
        "'that', 'who') -- split them into their own sentence instead.\n"
        "- Never stack more than one modifier on the same noun (no "
        "prepositional-phrase chains like 'the price of the fuel of the "
        "station'; restate as separate short sentences instead).\n"
        "- Never use subordinating or coordinating conjunctions to join "
        "two clauses in one sentence ('when', 'because', 'if', 'and', "
        "'but'); state each clause as its own sentence.\n"
        "- Prefer simple subject-verb-object sentences with a single, "
        "direct verb. Avoid passive voice and nominalizations (prefer "
        "'the system records X' over 'X is recorded by the system').\n"
        "- If a sentence needs a number, condition, or qualifier, put it "
        "in its own short sentence rather than embedding it in another "
        "clause."
    )


def build_ptd_recall_user_prompt(
    original: str, current_text: str, feedback: Optional[str]
) -> str:
    """Assemble the per-iteration flatten-and-name request."""
    feedback_block = f"\nFEEDBACK ON YOUR LAST ATTEMPT:\n{feedback}\n" if feedback else ""
    return (
        "TASK: rewrite the description as flat, human-readable prose per "
        "the style rules in the system prompt -- one clause per sentence, "
        "every entity/attribute named exactly as in the source, every "
        "relationship's two sides stated together in one sentence.\n"
        f"{feedback_block}\n"
        "SOURCE OF TRUTH — preserve every fact, entity, attribute, "
        "relationship, action, and constraint from here:\n"
        f"<source>\n{original}\n</source>\n\n"
        "VERSION TO IMPROVE — rewrite this, keeping all meaning from the "
        "source, flatter and more complete than it currently is:\n"
        f"<current>\n{current_text}\n</current>\n\n"
        "Return only the rewritten Markdown prose description."
    )


def build_ptd_recall_feedback(
    current_density: float,
    best_density: float,
    entity_recall: float,
    relation_recall: float,
    ptd_reduction: float,
    current_ptd: float,
    n_tokens: int,
    missing_names: List[str],
    relationship_issues: List[str],
) -> str:
    """Turn the measured min-gated density into concrete editing guidance,
    naming explicitly which of the three signals is the bottleneck (since
    that's the only one raising the score further)."""
    signals = {
        "entity naming": entity_recall,
        "relationship co-location": relation_recall,
        "syntax flattening": ptd_reduction,
    }
    bottleneck = min(signals, key=signals.get)
    lines = [
        f"Your last version measured density = {current_density:.3f} "
        f"(best so far: {best_density:.3f}) in {n_tokens} tokens: "
        f"{entity_recall * 100:.0f}% of entities/attributes named, "
        f"{relation_recall * 100:.0f}% of relationships had both sides in "
        f"the same sentence, parse_tree_depth = {current_ptd:.2f} "
        f"({ptd_reduction * 100:.0f}% lower than the source).",
        f"Your score is capped by your WEAKEST signal, currently "
        f"{bottleneck} at {signals[bottleneck] * 100:.0f}% -- fixing that "
        "one is the only thing that can raise your score right now.",
    ]
    if missing_names:
        shown = ", ".join(missing_names[:10])
        more = f" (+{len(missing_names) - 10} more)" if len(missing_names) > 10 else ""
        lines.append(f"Still missing names (use these exact words): {shown}{more}.")
    if relationship_issues:
        shown = "; ".join(relationship_issues[:10])
        more = f" (+{len(relationship_issues) - 10} more)" if len(relationship_issues) > 10 else ""
        lines.append(f"Relationship issues: {shown}{more}.")
    if current_density >= best_density:
        lines.append(
            "This did not improve on the best version so far. Fix the "
            "bottleneck signal above, and keep splitting any remaining "
            "relative clauses, stacked modifiers, or multi-clause "
            "sentences into separate flat sentences."
        )
    else:
        lines.append(
            "This is an improvement -- keep tightening the same way: fix "
            "anything still missing, then keep flattening the remaining "
            "nested sentences."
        )
    return "\n".join(lines)
