"""Prompt construction for the level-four "flattened" genre.

Unlike level zero/one/two (which target a shape across three metrics, see
``text.rewrite.shape_targets``), level four has a single, direct goal:
minimize ``parse_tree_depth`` (mean per-sentence dependency-tree depth,
``text.metrics.syntactic.ParseTreeDepth``) relative to the real spec
(description.md), without changing meaning. Motivated by
``parse_tree_depth`` being the strongest single predictor of UML-generation
F1 at L3 in this corpus (Pearson r=-0.39, p<0.01) -- flattening syntax
directly, rather than the broader shape-based rewriting, tests whether
targeting that one metric specifically improves F1 further.

Parse-tree depth grows from center-embedding: relative clauses, nested
prepositional/participial modifiers stacked on one head, and multi-clause
sentences joined by subordination or coordination. The system prompt targets
exactly those constructs.
"""

from __future__ import annotations

from typing import Optional


def flat_system_prompt() -> str:
    """The level-four persona: flatten syntax, minimize dependency-tree depth."""
    return (
        "You are an expert technical editor. You rewrite software system "
        "descriptions to minimize dependency-tree depth (how deeply nested "
        "each sentence's grammar is) WITHOUT changing their meaning.\n\n"
        "These descriptions are used to automatically generate UML models, so "
        "every piece of modelling-relevant information must be preserved exactly:\n"
        "- all actors, entities, and classes, with their names\n"
        "- all attributes and data each entity holds\n"
        "- all relationships and associations between entities, including "
        "multiplicity and direction\n"
        "- all actions, operations, and use cases, and which actor performs them\n"
        "- all business rules, constraints, and conditions\n\n"
        "Hard rules:\n"
        "- Never add facts, entities, relationships, or constraints not in the "
        "source.\n"
        "- Never remove or merge any of the information listed above.\n"
        "- Change only the linguistic FORM, never the semantic content.\n"
        "- Keep English, and keep roughly the same order of information.\n"
        "- Output ONLY the rewritten description in Markdown. No preamble, no "
        "explanation, no commentary.\n\n"
        "To MINIMIZE dependency-tree depth (the goal):\n"
        "- One clause per sentence. Never use relative clauses ('which',"
        " 'that', 'who') -- split them into their own sentence instead.\n"
        "- Never stack more than one modifier on the same noun (no "
        "prepositional-phrase chains like 'the price of the fuel of the "
        "station'; restate as separate short sentences instead).\n"
        "- Never use subordinating or coordinating conjunctions to join two "
        "clauses in one sentence ('when', 'because', 'if', 'and', 'but'); "
        "state each clause as its own sentence.\n"
        "- Prefer simple subject-verb-object sentences with a single, direct "
        "verb. Avoid passive voice and nominalizations (prefer 'the system "
        "records X' over 'X is recorded by the system').\n"
        "- If a sentence needs a number, condition, or qualifier, put it in "
        "its own short sentence rather than embedding it in another clause."
    )


def build_flat_user_prompt(
    original: str, current_text: str, feedback: Optional[str]
) -> str:
    """Assemble the per-iteration flattening request."""
    feedback_block = f"\nFEEDBACK ON YOUR LAST ATTEMPT:\n{feedback}\n" if feedback else ""
    return (
        "TASK: rewrite the description to minimize dependency-tree depth per "
        "the style rules in the system prompt -- flatten every sentence to a "
        "single clause.\n"
        f"{feedback_block}\n"
        "SOURCE OF TRUTH — preserve every fact, entity, attribute, "
        "relationship, action, and constraint from here:\n"
        f"<source>\n{original}\n</source>\n\n"
        "VERSION TO IMPROVE — rewrite this, keeping all meaning from the "
        "source, flatter than it currently is:\n"
        f"<current>\n{current_text}\n</current>\n\n"
        "Return only the rewritten Markdown description."
    )


def build_flat_feedback(current_value: float, best_value: float, source_value: float) -> str:
    """Turn the measured parse_tree_depth into concrete editing guidance."""
    lines = [
        f"Your last version measured parse_tree_depth = {current_value:.2f} "
        f"(best so far: {best_value:.2f}; the real spec measures {source_value:.2f}).",
    ]
    if current_value >= best_value:
        lines.append(
            "This did not improve on the best version so far. Look for any "
            "remaining relative clauses, stacked modifiers, or multi-clause "
            "sentences and split them into separate flat sentences."
        )
    else:
        lines.append(
            "This is an improvement -- keep flattening the same way in the "
            "remaining sentences that still have nested structure."
        )
    return "\n".join(lines)
