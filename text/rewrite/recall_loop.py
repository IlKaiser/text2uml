"""Feedback loop for level-minus-one: directly maximize recall_density.

Separate from ``loop.py`` (which scores every candidate on linguistic
metrics via ``text.rewrite.scorer``) because this objective scores against a
case's gold PlantUML entity names instead -- a fundamentally different
signal that has nothing to do with sentence-level complexity, so it has no
use for ``scorer.ComplexityReference``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from .client import rewrite_once
from .config import RewriteConfig

logger = logging.getLogger(__name__)

# (density, best_density, recall, n_tokens, missing_names) -> feedback text
FeedbackFn = Callable[[float, float, float, int, List[str]], str]


@dataclass(frozen=True)
class RecallDensityResult:
    """Outcome of driving one description toward maximum recall_density."""

    source_recall: float
    source_density: float
    best_recall: float
    best_density: float
    best_n_tokens: int
    improved: bool
    iterations: int
    text: str


def _entity_recall(text: str, names: Sequence[str]) -> Tuple[float, List[str]]:
    """(recall fraction, sorted missing names) for ``names`` in ``text``.

    Case-insensitive substring match -- the same criterion used throughout
    this project's recall analysis (see ``experiments/levels`` correlation
    work), not a strict token-boundary match, since rewrites often inflect
    or compound a name (e.g. "Pumps" for gold "Pump").
    """
    t = text.lower()
    uniq = sorted({n for n in names if n})
    missing = [n for n in uniq if n.lower() not in t]
    recall = (len(uniq) - len(missing)) / len(uniq) if uniq else 0.0
    return recall, missing


def rewrite_to_maximize_recall_density(
    client,
    cfg: RewriteConfig,
    original: str,
    gold_names: Sequence[str],
    n_tokens_fn: Callable[[str], int],
    system_prompt: str,
    user_prompt_fn: Callable[[str, str, Optional[str]], str],
    feedback_fn: FeedbackFn,
) -> RecallDensityResult:
    """Iteratively rewrite ``original`` to maximize entity-recall-per-token.

    ``n_tokens_fn`` is injected rather than baked in so this module stays
    free of the scorer's spaCy dependency for a metric that has nothing to
    do with linguistic complexity -- callers already have a tokenizer handy
    (``text.metrics.compute_all``) from scoring the shape-matching levels.

    Unlike ``rewrite_to_shape``/``rewrite_to_minimize`` there is no
    "reached" target -- recall_density has no natural ceiling to stop at, so
    this always runs the full ``cfg.max_iterations`` budget and keeps
    whichever candidate scored highest, mirroring ``rewrite_to_minimize``'s
    "no target, just best-seen" pattern.
    """
    source_recall, _ = _entity_recall(original, gold_names)
    source_tokens = max(1, n_tokens_fn(original))
    source_density = source_recall / (source_tokens / 1000.0)

    best_text = original
    best_recall, best_density, best_tokens = source_recall, source_density, source_tokens
    improved = False
    candidate_text = original
    feedback: Optional[str] = None
    iterations = 0

    for i in range(cfg.max_iterations):
        iterations = i + 1
        user = user_prompt_fn(original, candidate_text, feedback)
        try:
            candidate_text = rewrite_once(client, cfg, system_prompt, user)
        except Exception as exc:  # noqa: BLE001 - report and stop iterating
            logger.error("Recall-density rewrite call failed on iter %d: %s", i, exc)
            break

        recall, missing = _entity_recall(candidate_text, gold_names)
        tokens = max(1, n_tokens_fn(candidate_text))
        density = recall / (tokens / 1000.0)

        if density > best_density:
            best_text, best_recall, best_density, best_tokens, improved = (
                candidate_text, recall, density, tokens, True,
            )

        feedback = feedback_fn(density, best_density, recall, tokens, missing)

    return RecallDensityResult(
        source_recall=source_recall,
        source_density=source_density,
        best_recall=best_recall,
        best_density=best_density,
        best_n_tokens=best_tokens,
        improved=improved,
        iterations=iterations,
        text=best_text,
    )
