"""The feedback loop that drives one description to one target complexity."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from ..config import TextConfig
from .client import rewrite_once
from .config import RewriteConfig
from .prompts import build_feedback, build_user_prompt, system_prompt
from .scorer import ComplexityReference, ScoreResult, score_text
from .shape_targets import check_shape, format_feedback, shape_ok
from .verifier import verify_meaning

# Feedback builder signature shared by rewrite_to_minimize's caller (e.g.
# text.rewrite.flat_prompts.build_flat_feedback): (current_value, best_value,
# source_value) -> str.
MinimizeFeedbackFn = Callable[[float, float, float], str]

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
            logger.info("  %s: Starting iteration %d/%d...", level_name, iterations, cfg.max_iterations)
            candidate_text = rewrite_once(client, cfg, system_prompt(), user)
            logger.info("  %s: Received rewrite candidate. Scoring...", level_name)
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


@dataclass(frozen=True)
class ShapeLevelResult:
    """Outcome of driving one description toward one level's target shape."""

    level_name: str
    final_z: float
    shape_checks: Tuple  # Tuple[ShapeCheck, ...]
    reached: bool
    iterations: int
    text: str


def rewrite_to_shape(
    client,
    cfg: RewriteConfig,
    original: str,
    original_score: ScoreResult,
    reference: ComplexityReference,
    level_name: str,
    l3_values: Dict[str, float],
    system_prompt: str,
    user_prompt_fn: Callable[[str, str, Optional[str]], str],
    metric_guidance: Dict[str, str],
) -> ShapeLevelResult:
    """Iteratively rewrite ``original`` until its shape (rank + band) matches.

    Generalizes ``rewrite_to_target``: instead of chasing a single aggregate
    z_index, checks ``mdd`` / ``subordination_index`` / ``context_dependence_proxy``
    against ``l3_values`` (that case's own, untouched real-spec metrics) via
    ``text.rewrite.shape_targets``. ``system_prompt`` / ``user_prompt_fn`` are
    supplied by the caller so this one loop drives both the structural
    (level zero) and narrative (level one/two) genres.
    """
    tconf: TextConfig = cfg.text_config
    min_tokens = cfg.min_token_ratio * max(1, original_score.n_tokens)

    def _fail_count(checks) -> int:
        """Number of checks failing (non-degenerate and not both rank+band ok)."""
        return sum(1 for c in checks if not (c.degenerate or (c.rank_ok and c.band_ok)))

    best_text = original
    best_checks = check_shape(level_name, original_score.values, l3_values)
    best_fail_count = _fail_count(best_checks)
    candidate_text = original
    feedback: Optional[str] = None
    reached = False
    iterations = 0
    final_z = original_score.z_index

    for i in range(cfg.max_iterations):
        iterations = i + 1
        user = user_prompt_fn(original, candidate_text, feedback)
        try:
            logger.info("  %s: Starting iteration %d/%d...", level_name, iterations, cfg.max_iterations)
            candidate_text = rewrite_once(client, cfg, system_prompt, user)
            logger.info("  %s: Received rewrite candidate. Scoring...", level_name)
        except Exception as exc:  # noqa: BLE001 - report and stop iterating
            logger.error("Shape rewrite call failed on '%s' iter %d: %s", level_name, i, exc)
            break

        try:
            latest_score = score_text(candidate_text, reference, tconf)
        except ValueError as exc:
            logger.warning("Could not score shape candidate (%s); retrying.", exc)
            feedback = "The previous output could not be parsed. Return a normal, complete Markdown document."
            continue

        checks = check_shape(level_name, latest_score.values, l3_values)
        ok = shape_ok(checks)
        keeps_content = latest_score.n_tokens >= min_tokens

        if ok and keeps_content:
            if not cfg.verify_meaning:
                best_text, best_checks, best_fail_count, final_z = candidate_text, checks, 0, latest_score.z_index
                reached = True
                break
            check = verify_meaning(client, cfg, original, candidate_text)
            if check.equivalent:
                best_text, best_checks, best_fail_count, final_z = candidate_text, checks, 0, latest_score.z_index
                reached = True
                break
            feedback = "Your shape is on target — keep it there. " + check.feedback()
            continue

        if ok and not keeps_content:
            feedback = format_feedback(checks, metric_guidance) or "Restore every entity, attribute, relationship, action, and constraint from the source; your version is too short."
            continue

        fail_count = _fail_count(checks)
        if keeps_content and fail_count < best_fail_count:
            best_text, best_checks, best_fail_count, final_z = candidate_text, checks, fail_count, latest_score.z_index
        feedback = format_feedback(checks, metric_guidance)

    return ShapeLevelResult(
        level_name=level_name,
        final_z=final_z,
        shape_checks=tuple(best_checks),
        reached=reached,
        iterations=iterations,
        text=best_text,
    )


@dataclass(frozen=True)
class MinimizeResult:
    """Outcome of iteratively rewriting ``original`` to minimize one metric."""

    metric_name: str
    source_value: float  # the metric's value on the untouched original (L3)
    best_value: float  # the lowest value achieved (== source_value if never improved)
    improved: bool  # whether any accepted candidate beat source_value
    iterations: int
    text: str


def rewrite_to_minimize(
    client,
    cfg: RewriteConfig,
    original: str,
    original_score: ScoreResult,
    reference: ComplexityReference,
    metric_name: str,
    system_prompt: str,
    user_prompt_fn: Callable[[str, str, Optional[str]], str],
    feedback_fn: MinimizeFeedbackFn,
) -> MinimizeResult:
    """Iteratively rewrite ``original`` to minimize ``metric_name``, keeping
    meaning and content. Unlike ``rewrite_to_shape``/``rewrite_to_target``
    (which stop once a target/shape is reached), this always runs the full
    ``cfg.max_iterations`` budget and keeps the single best (lowest-value,
    meaning-preserving, content-preserving) candidate seen across all of
    them -- there is no target to "reach", only "as low as possible."
    """
    tconf: TextConfig = cfg.text_config
    min_tokens = cfg.min_token_ratio * max(1, original_score.n_tokens)
    source_value = original_score.values[metric_name]

    best_text = original
    best_value = source_value
    improved = False
    candidate_text = original
    feedback: Optional[str] = None
    iterations = 0

    for i in range(cfg.max_iterations):
        iterations = i + 1
        user = user_prompt_fn(original, candidate_text, feedback)
        try:
            logger.info("  %s: Starting iteration %d/%d...", metric_name, iterations, cfg.max_iterations)
            candidate_text = rewrite_once(client, cfg, system_prompt, user)
            logger.info("  %s: Received rewrite candidate. Scoring...", metric_name)
        except Exception as exc:  # noqa: BLE001 - report and stop iterating
            logger.error("Minimize rewrite call failed on '%s' iter %d: %s", metric_name, i, exc)
            break

        try:
            latest_score = score_text(candidate_text, reference, tconf)
        except ValueError as exc:
            logger.warning("Could not score minimize candidate (%s); retrying.", exc)
            feedback = "The previous output could not be parsed. Return a normal, complete Markdown document."
            continue

        current_value = latest_score.values[metric_name]
        keeps_content = latest_score.n_tokens >= min_tokens

        if keeps_content and current_value < best_value:
            if cfg.verify_meaning:
                check = verify_meaning(client, cfg, original, candidate_text)
                if not check.equivalent:
                    feedback = (
                        f"Your {metric_name} improved to {current_value:.2f}, but the meaning "
                        "check failed -- fix these while keeping it flat: " + check.feedback()
                    )
                    continue
            best_text, best_value, improved = candidate_text, current_value, True

        feedback = feedback_fn(current_value, best_value, source_value)
        if not keeps_content:
            feedback += (
                "\nWARNING: your version is much shorter than the source — you may "
                "have dropped information. Restore every entity, attribute, "
                "relationship, action, and constraint from the source."
            )

    return MinimizeResult(
        metric_name=metric_name,
        source_value=source_value,
        best_value=best_value,
        improved=improved,
        iterations=iterations,
        text=best_text,
    )
