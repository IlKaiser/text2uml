"""Prompt construction for the level-minus-one "recall-density" genre.

Unlike level zero (which targets a three-metric shape -- mdd,
subordination_index, context_dependence_proxy -- see
``text.rewrite.shape_targets``) or level four (which minimizes
parse_tree_depth), level minus-one optimizes directly for the metric this
project's own analysis found actually predicts downstream UML-generation F1:
``recall_density`` = (fraction of gold class/attribute names literally
present in the text) / (token count in thousands). Per-case correlation with
F1 across 213 (case, rewriter) pairs: Pearson r=0.56-0.60, p<0.0001 -- far
stronger and more robust than any of level zero's three shape metrics, which
did not survive per-case scrutiny (mdd_ratio: r=-0.12, not significant).

This means the rewrite is explicitly steered using the gold PlantUML's
entity names, unlike every other level here (which only ever see the prose
description). That is a deliberate departure: this level exists to test the
ceiling of "what if the L0 rewrite step targeted the metric that actually
matters" rather than a metric merely correlated with the target genre.
"""

from __future__ import annotations

from typing import List, Optional


def recall_density_system_prompt() -> str:
    """The level-minus-one persona: name every entity, as compactly as possible."""
    return (
        "You are a UML analyst producing the most information-dense possible "
        "summary of a software system description. Your only goal: maximize "
        "the fraction of entity and attribute names from the source that "
        "appear verbatim (or as an obvious close variant) in your output, "
        "while using as few tokens as possible. Every token that isn't "
        "naming an entity, attribute, or relationship is wasted.\n\n"
        "Format: two sections, 'Classes' and 'Relationships', one compact "
        "line per item. No prose, no restating the same entity's name twice, "
        "no filler adjectives or explanations.\n\n"
        "Hard rules:\n"
        "- Never add facts, entities, relationships, or constraints not in "
        "the source.\n"
        "- Never omit an entity, attribute, or relationship that the source "
        "names.\n"
        "- Prefer the source's exact wording for each name over a paraphrase "
        "-- exact string matches are what get credit.\n"
        "- Output ONLY the notes in Markdown. No preamble, no explanation, "
        "no commentary."
    )


def build_recall_density_user_prompt(
    original: str, current_text: str, feedback: Optional[str]
) -> str:
    """Assemble the per-iteration recall-density rewrite request."""
    feedback_block = f"\nFEEDBACK ON YOUR LAST ATTEMPT:\n{feedback}\n" if feedback else ""
    return (
        "TASK: rewrite the description as compact notes that name every "
        "entity/attribute/relationship using as few tokens as possible, per "
        "the style rules in the system prompt.\n"
        f"{feedback_block}\n"
        "SOURCE OF TRUTH — preserve every fact, entity, attribute, "
        "relationship, action, and constraint from here:\n"
        f"<source>\n{original}\n</source>\n\n"
        "VERSION TO IMPROVE — rewrite this, keeping all meaning from the "
        "source, denser than it currently is:\n"
        f"<current>\n{current_text}\n</current>\n\n"
        "Return only the rewritten Markdown notes."
    )


def build_recall_density_feedback(
    current_density: float,
    best_density: float,
    current_recall: float,
    n_tokens: int,
    missing_names: List[str],
) -> str:
    """Turn the measured recall_density into concrete, name-level editing guidance."""
    lines = [
        f"Your last version measured recall_density = {current_density:.3f} "
        f"(best so far: {best_density:.3f}); it named {current_recall * 100:.0f}% "
        f"of expected entities/attributes in {n_tokens} tokens.",
    ]
    if missing_names:
        shown = ", ".join(missing_names[:15])
        more = f" (+{len(missing_names) - 15} more)" if len(missing_names) > 15 else ""
        lines.append(f"Still missing (use these exact names): {shown}{more}.")
    if current_density >= best_density:
        lines.append(
            "This did not improve on the best version so far. Add the "
            "missing names above and cut any token that isn't an entity, "
            "attribute, or relationship name."
        )
    else:
        lines.append(
            "This is an improvement -- keep tightening the same way: add "
            "any names still missing, then trim remaining filler."
        )
    return "\n".join(lines)


def combined_recall_system_prompt() -> str:
    """The level-minus-four persona: name every entity AND keep every
    relationship's two endpoints stated together, as compactly as possible.

    Unlike ``recall_density_system_prompt`` (entity names only), this
    explicitly protects relationship pairs -- level minus-two/minus-three
    both plateaued below the L0 baseline despite near-100% entity recall,
    suggesting compression was trading away relationship facts that
    entity-name recall alone can't see or credit.
    """
    return (
        "You are a UML analyst producing the most information-dense "
        "possible summary of a software system description. Your goal: "
        "maximize two things per token used -- (1) the fraction of entity "
        "and attribute names from the source that appear verbatim (or an "
        "obvious close variant), and (2) for every relationship between "
        "two classes, BOTH class names appearing together in the SAME "
        "line. A relationship split across two lines, or with one "
        "endpoint's name changed/dropped, does not count.\n\n"
        "Format: two sections, 'Classes' and 'Relationships', one compact "
        "line per item. Every relationship line must name both classes it "
        "connects. No prose, no restating a name twice, no filler "
        "adjectives or explanations.\n\n"
        "Hard rules:\n"
        "- Never add facts, entities, relationships, or constraints not in "
        "the source.\n"
        "- Never omit an entity, attribute, or relationship that the "
        "source names.\n"
        "- Prefer the source's exact wording for each name over a "
        "paraphrase -- exact string matches are what get credit.\n"
        "- Output ONLY the notes in Markdown. No preamble, no explanation, "
        "no commentary."
    )


def build_combined_recall_feedback(
    current_density: float,
    best_density: float,
    entity_recall: float,
    relation_recall: float,
    n_tokens: int,
    missing_names: List[str],
    missing_relations: List[str],
) -> str:
    """Turn the measured combined recall_density into concrete, name- and
    relationship-level editing guidance."""
    lines = [
        f"Your last version measured recall_density = {current_density:.3f} "
        f"(best so far: {best_density:.3f}) in {n_tokens} tokens: "
        f"{entity_recall * 100:.0f}% of entities/attributes named, "
        f"{relation_recall * 100:.0f}% of relationships had both classes "
        "stated together on the same line.",
    ]
    if missing_names:
        shown = ", ".join(missing_names[:10])
        more = f" (+{len(missing_names) - 10} more)" if len(missing_names) > 10 else ""
        lines.append(f"Still missing names (use these exact words): {shown}{more}.")
    if missing_relations:
        shown = ", ".join(missing_relations[:10])
        more = f" (+{len(missing_relations) - 10} more)" if len(missing_relations) > 10 else ""
        lines.append(
            f"Still missing relationships (put both class names on one "
            f"line for each): {shown}{more}."
        )
    if current_density >= best_density:
        lines.append(
            "This did not improve on the best version so far. Fix the "
            "missing names/relationships above and cut any token that "
            "isn't an entity, attribute, or relationship name."
        )
    else:
        lines.append(
            "This is an improvement -- keep tightening the same way: fix "
            "anything still missing, then trim remaining filler."
        )
    return "\n".join(lines)


def cardinality_recall_system_prompt() -> str:
    """The level-minus-five persona: name every entity, keep every
    relationship's two endpoints stated together, AND state each
    relationship's multiplicity/cardinality.

    Extends ``combined_recall_system_prompt`` (entities + relationships)
    with a third protected signal -- level minus-four's relationship check
    had zero visibility into cardinality, its own full F1 scoring category.
    """
    return (
        "You are a UML analyst producing the most information-dense "
        "possible summary of a software system description. Your goal: "
        "maximize three things per token used -- (1) the fraction of "
        "entity and attribute names from the source that appear verbatim "
        "(or an obvious close variant), (2) for every relationship "
        "between two classes, BOTH class names appearing together in the "
        "SAME line, and (3) that same line ALSO stating the multiplicity/"
        "cardinality of the relationship (a number, a range like '0..*' "
        "or '1..*', or a quantity word like 'many'/'several'/'one'/"
        "'each').\n\n"
        "Format: two sections, 'Classes' and 'Relationships', one compact "
        "line per item. Every relationship line must name both classes it "
        "connects AND state the multiplicity on each side. No prose, no "
        "restating a name twice, no filler adjectives or explanations.\n\n"
        "Hard rules:\n"
        "- Never add facts, entities, relationships, or constraints not in "
        "the source.\n"
        "- Never omit an entity, attribute, relationship, or multiplicity "
        "that the source states.\n"
        "- Prefer the source's exact wording for each name over a "
        "paraphrase -- exact string matches are what get credit.\n"
        "- Output ONLY the notes in Markdown. No preamble, no explanation, "
        "no commentary."
    )


def complete_recall_system_prompt() -> str:
    """The level-minus-six persona: entities + relationships + cardinality,
    same three goals as level minus-five, but scored so none of the three
    can be cheaply traded away for the others.

    Level minus-five's naive average((entity, relation, cardinality) recall)
    let the rewrite drop multiplicity words under compression pressure --
    stating a relationship without its multiplicity only cost 1/3 of a
    point, cheaper than the 2/3 a fully-dropped relationship would cost, so
    the optimizer preferred trimming cardinality first. This level's score
    is gated on the WEAKEST of the three signals (see ``complete_loop.
    _scored_recall``), so there's no way to gain score by sacrificing one to
    boost the others -- the only way up is to raise whichever is currently
    lowest, almost always cardinality.
    """
    return (
        "You are a UML analyst producing the most information-dense "
        "possible summary of a software system description. Your score is "
        "the WEAKEST of three signals, not their average -- so you cannot "
        "trade one off against the others. All three must be maximized "
        "together: (1) the fraction of entity and attribute names from the "
        "source that appear verbatim (or an obvious close variant), (2) "
        "for every relationship between two classes, BOTH class names "
        "appearing together in the SAME line, and (3) that same line ALSO "
        "stating the multiplicity/cardinality of the relationship (a "
        "number, a range like '0..*' or '1..*', or a quantity word like "
        "'many'/'several'/'one'/'each'). A relationship stated without its "
        "multiplicity scores exactly as badly as a relationship you never "
        "wrote down -- dropping the multiplicity to save tokens gains you "
        "nothing.\n\n"
        "Format: two sections, 'Classes' and 'Relationships', one compact "
        "line per item. Every relationship line must name both classes it "
        "connects AND state the multiplicity on each side. No prose, no "
        "restating a name twice, no filler adjectives or explanations.\n\n"
        "Hard rules:\n"
        "- Never add facts, entities, relationships, or constraints not in "
        "the source.\n"
        "- Never omit an entity, attribute, relationship, or multiplicity "
        "that the source states.\n"
        "- Prefer the source's exact wording for each name over a "
        "paraphrase -- exact string matches are what get credit.\n"
        "- Output ONLY the notes in Markdown. No preamble, no explanation, "
        "no commentary."
    )


def build_complete_recall_feedback(
    current_density: float,
    best_density: float,
    entity_recall: float,
    relation_recall: float,
    cardinality_recall: float,
    n_tokens: int,
    missing_names: List[str],
    relationship_issues: List[str],
) -> str:
    """Turn the measured min-gated recall_density into concrete editing
    guidance, naming explicitly which of the three signals is the
    bottleneck (since that's the only one raising the score further)."""
    signals = {
        "entity": entity_recall, "relationship": relation_recall, "cardinality": cardinality_recall,
    }
    bottleneck = min(signals, key=signals.get)
    lines = [
        f"Your last version measured recall_density = {current_density:.3f} "
        f"(best so far: {best_density:.3f}) in {n_tokens} tokens: "
        f"{entity_recall * 100:.0f}% of entities/attributes named, "
        f"{relation_recall * 100:.0f}% of relationships had both classes "
        f"stated together, {cardinality_recall * 100:.0f}% of those also "
        "stated a multiplicity.",
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
            "bottleneck signal above first, then cut any token that isn't "
            "an entity, attribute, relationship name, or multiplicity."
        )
    else:
        lines.append(
            "This is an improvement -- keep tightening the same way: fix "
            "the current bottleneck, then trim remaining filler."
        )
    return "\n".join(lines)


def build_cardinality_recall_feedback(
    current_density: float,
    best_density: float,
    entity_recall: float,
    relation_recall: float,
    cardinality_recall: float,
    n_tokens: int,
    missing_names: List[str],
    relationship_issues: List[str],
) -> str:
    """Turn the measured entity+relation+cardinality recall_density into
    concrete editing guidance."""
    lines = [
        f"Your last version measured recall_density = {current_density:.3f} "
        f"(best so far: {best_density:.3f}) in {n_tokens} tokens: "
        f"{entity_recall * 100:.0f}% of entities/attributes named, "
        f"{relation_recall * 100:.0f}% of relationships had both classes "
        f"stated together, {cardinality_recall * 100:.0f}% of those also "
        "stated a multiplicity.",
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
            "issues above and cut any token that isn't an entity, "
            "attribute, relationship name, or multiplicity."
        )
    else:
        lines.append(
            "This is an improvement -- keep tightening the same way: fix "
            "anything still missing, then trim remaining filler."
        )
    return "\n".join(lines)
