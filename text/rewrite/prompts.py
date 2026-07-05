"""Prompt construction for the Claude-powered rewrite loop.

The prompts steer Claude toward a target ``z_index`` while preserving every
piece of information a UML model would capture. Feedback each iteration turns
the measured metric breakdown into concrete, human-readable editing guidance.
"""

from __future__ import annotations

from typing import Dict, Optional

from .scorer import ScoreResult

# Plain-language guidance for reducing each metric (what makes it high).
METRIC_GUIDANCE: Dict[str, str] = {
    "mdd": "shorten sentences and keep related words close together "
    "(reduces mean dependency distance)",
    "subordination_index": "split subordinate clauses into separate, "
    "standalone sentences",
    "parse_tree_depth": "flatten nested phrasing; break complex sentences apart",
    "parse_tree_depth_max": "eliminate the single most deeply-nested sentence; "
    "split it into several",
    "flesch_reading_ease": "use shorter, more common words and shorter sentences "
    "(raises readability)",
    "rst_depth_proxy": "reduce nested discourse relations; state points in a flat "
    "sequence",
    "context_dependence_proxy": "make references explicit; avoid pronouns and "
    "anaphora that depend on prior context",
    "inferential_load_proxy": "state facts directly instead of leaving them to be "
    "inferred",
}


def system_prompt() -> str:
    """The rewrite persona and the meaning-preservation contract."""
    return (
        "You are an expert technical editor. You rewrite software system "
        "descriptions to a target reading complexity WITHOUT changing their "
        "meaning.\n\n"
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
        "To REDUCE complexity: use short sentences (one idea each), split long or "
        "subordinate clauses into separate sentences, prefer common everyday "
        "words, use active voice and concrete verbs instead of nominalisations, "
        "and avoid deep nesting.\n"
        "To INCREASE complexity (only when asked to move back up toward a "
        "target): combine closely related short sentences and use richer "
        "connectives and subordinate clauses — but never change the facts."
    )


def _direction(current: float, target: float, tolerance: float) -> str:
    if current > target + tolerance:
        return "make it SIMPLER (bring the z_index DOWN)"
    if current < target - tolerance:
        return "make it slightly MORE detailed and complex (bring the z_index UP), "\
            "without changing any facts"
    return "keep it about the same complexity"


def build_feedback(score: ScoreResult, target: float, tolerance: float) -> str:
    """Turn the measured metric breakdown into editing guidance."""
    top = sorted(score.oriented_z.items(), key=lambda kv: kv[1], reverse=True)[:3]
    lines = [
        f"Your last version measured z_index = {score.z_index:.2f} "
        f"(target {target:.2f}). You need to {_direction(score.z_index, target, tolerance)}.",
    ]
    if score.z_index > target:
        lines.append("The metrics contributing MOST to the excess complexity are:")
        for name, val in top:
            guidance = METRIC_GUIDANCE.get(name, "simplify the phrasing")
            lines.append(f"  - {name} (+{val:.2f} above corpus average): {guidance}.")
    else:
        lines.append(
            "It is now simpler than the target. Restore some detail and "
            "connective structure (without adding or removing facts) to raise "
            "the complexity back toward the target."
        )
    return "\n".join(lines)


def build_user_prompt(
    original: str,
    current_text: str,
    target: float,
    level_name: str,
    measured_z: float,
    feedback: str | None,
) -> str:
    """Assemble the per-iteration rewrite request."""
    feedback_block = f"\nFEEDBACK ON YOUR LAST ATTEMPT:\n{feedback}\n" if feedback else ""
    return (
        f"TASK: rewrite the description so its linguistic-complexity index "
        f"(z_index) is {target:.2f}.\n"
        f"The z_index runs from 0.00 (the simplest, clearest description in our "
        f"corpus) to 1.00 (the most complex). Target level: {level_name}.\n\n"
        f"The version you must improve currently measures z_index = "
        f"{measured_z:.2f}.\n"
        f"{feedback_block}\n"
        f"SOURCE OF TRUTH — preserve every fact, entity, attribute, relationship, "
        f"action, and constraint from here:\n"
        f"<source>\n{original}\n</source>\n\n"
        f"VERSION TO IMPROVE — rewrite this, keeping all meaning from the source:\n"
        f"<current>\n{current_text}\n</current>\n\n"
        f"Return only the rewritten Markdown description."
    )


def build_shape_user_prompt(
    original: str, current_text: str, level_label: str, feedback: Optional[str]
) -> str:
    """Per-iteration narrative-rewrite request, framed by level label instead
    of a numeric z_index target (acceptance is shape-based, see
    ``text.rewrite.shape_targets``)."""
    feedback_block = f"\nFEEDBACK ON YOUR LAST ATTEMPT:\n{feedback}\n" if feedback else ""
    return (
        f"TASK: rewrite the description as {level_label}.\n"
        f"{feedback_block}\n"
        f"SOURCE OF TRUTH — preserve every fact, entity, attribute, "
        f"relationship, action, and constraint from here:\n"
        f"<source>\n{original}\n</source>\n\n"
        f"VERSION TO IMPROVE — rewrite this, keeping all meaning from the "
        f"source:\n"
        f"<current>\n{current_text}\n</current>\n\n"
        f"Return only the rewritten Markdown description."
    )
