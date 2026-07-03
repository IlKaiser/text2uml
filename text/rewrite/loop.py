"""The feedback loop that drives one description to one target complexity."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import TextConfig
from .client import rewrite_once
from .config import RewriteConfig
from .prompts import build_feedback, build_user_prompt, system_prompt
from .scorer import ComplexityReference, ScoreResult, score_text
from .verifier import verify_meaning

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LevelResult:
    """Outcome of driving one description toward one target level."""

    level_name: str
    target: float
    final_z: float
    reached: bool
    iterations: int
    text: str


def rewrite_to_target(
    client,
    cfg: RewriteConfig,
    original: str,
    original_score: ScoreResult,
    reference: ComplexityReference,
    level_name: str,
    target: float,
) -> LevelResult:
    """Iteratively rewrite ``original`` until its z_index reaches ``target``.

    Keeps the best-scoring candidate seen; a candidate that collapses the token
    count below ``min_token_ratio`` of the source is never accepted as best
    (guards against over-shortening that would drop content).
    """
    tconf: TextConfig = cfg.text_config
    min_tokens = cfg.min_token_ratio * max(1, original_score.n_tokens)

    # Already within tolerance — no API call needed.
    if abs(original_score.z_index - target) <= cfg.tolerance:
        return LevelResult(level_name, target, original_score.z_index, True, 0, original)

    best_text = original
    best_dist = abs(original_score.z_index - target)
    best_z = original_score.z_index
    candidate_text = original
    latest_score = original_score
    feedback: str | None = None
    reached = False
    iterations = 0

    for i in range(cfg.max_iterations):
        iterations = i + 1
        user = build_user_prompt(
            original, candidate_text, target, level_name, latest_score.z_index, feedback
        )
        try:
            candidate_text = rewrite_once(client, cfg, system_prompt(), user)
        except Exception as exc:  # noqa: BLE001 - report and stop iterating
            logger.error("Rewrite call failed on '%s' iter %d: %s", level_name, i, exc)
            break

        try:
            latest_score = score_text(candidate_text, reference, tconf)
        except ValueError as exc:
            logger.warning("Could not score candidate (%s); retrying with feedback.", exc)
            feedback = "The previous output could not be parsed. Return a normal, "\
                "complete Markdown description."
            continue

        dist = abs(latest_score.z_index - target)
        keeps_content = latest_score.n_tokens >= min_tokens

        # Complexity target met and enough content retained: accept, but gate on
        # the semantic-equivalence check when enabled.
        if dist <= cfg.tolerance and keeps_content:
            if not cfg.verify_meaning:
                best_dist, best_text, best_z = dist, candidate_text, latest_score.z_index
                reached = True
                break
            check = verify_meaning(client, cfg, original, candidate_text)
            if check.equivalent:
                best_dist, best_text, best_z = dist, candidate_text, latest_score.z_index
                reached = True
                break
            # Complexity is on target but meaning drifted — hold the target and fix.
            feedback = "Your complexity is on target — keep it there. " + check.feedback()
            continue

        # Off target: track the closest content-keeping candidate and steer.
        if dist < best_dist and keeps_content:
            best_dist, best_text, best_z = dist, candidate_text, latest_score.z_index
        feedback = build_feedback(latest_score, target, cfg.tolerance)
        if not keeps_content:
            feedback += (
                "\nWARNING: your version is much shorter than the source — you may "
                "have dropped information. Restore every entity, attribute, "
                "relationship, action, and constraint from the source."
            )

    return LevelResult(
        level_name=level_name,
        target=target,
        final_z=best_z,
        reached=reached,
        iterations=iterations,
        text=best_text,
    )
