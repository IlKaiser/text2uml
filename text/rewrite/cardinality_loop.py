"""Feedback loop for level-minus-five: relation_recall_density
(``text.rewrite.relation_loop``) extended to also credit multiplicity/
cardinality notation on each relationship line.

Level minus-four's relationship check only verifies both endpoint names
co-occur, with zero signal for cardinality -- its own full F1 scoring
category (``f1_cardinality``), and likely most of the remaining gap to
level minus-one's gold-access ceiling (0.52 vs. L-4's 0.42). Multiplicity is
expressed inconsistently in prose ("several", "many", "0 or more",
"1..*"), so this uses a lenient token-presence check (any number-like or
quantity word on the relationship's line) rather than exact matching
against an extracted target value. Reuses ``relation_entities.
extract_entities_and_relations`` unchanged -- no new extraction needed,
since the check is presence-based, not a comparison against an extracted
"correct" cardinality string.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from .client import rewrite_once
from .config import RewriteConfig

logger = logging.getLogger(__name__)

# (density, best_density, entity_recall, relation_recall, cardinality_recall,
#  n_tokens, missing_entities, relationship_issues) -> feedback text
FeedbackFn = Callable[[float, float, float, float, float, int, List[str], List[str]], str]

_MULTIPLICITY_RE = re.compile(
    r"\b(\d+(\.\.(\d+|\*))?|\*|many|several|one|all|each|multiple|few)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CardinalityRecallResult:
    """Outcome of driving one description toward maximum entity + relation +
    cardinality recall_density."""

    source_entity_recall: float
    source_relation_recall: float
    source_cardinality_recall: float
    source_density: float
    best_entity_recall: float
    best_relation_recall: float
    best_cardinality_recall: float
    best_density: float
    best_n_tokens: int
    improved: bool
    iterations: int
    text: str


def _scored_recall(
    text: str, entity_names: Sequence[str], relation_pairs: Sequence[Tuple[str, str]]
) -> Tuple[float, float, float, float, List[str], List[str]]:
    """(combined, entity_recall, relation_recall, cardinality_recall,
    missing_entities, relationship_issues)."""
    t = text.lower()
    uniq_entities = sorted({n for n in entity_names if n})
    missing_entities = [n for n in uniq_entities if n.lower() not in t]
    entity_recall = (
        (len(uniq_entities) - len(missing_entities)) / len(uniq_entities) if uniq_entities else 0.0
    )

    lines = [line.lower() for line in text.splitlines() if line.strip()]
    relation_hits = 0
    cardinality_hits = 0
    issues: List[str] = []
    for a, b in relation_pairs:
        matched_line = next((line for line in lines if a in line and b in line), None)
        if matched_line is None:
            issues.append(f"{a}-{b} (relationship not stated together)")
            continue
        relation_hits += 1
        if _MULTIPLICITY_RE.search(matched_line):
            cardinality_hits += 1
        else:
            issues.append(f"{a}-{b} (stated, but no multiplicity given)")

    n_rel = len(relation_pairs)
    relation_recall = relation_hits / n_rel if n_rel else 1.0
    cardinality_recall = cardinality_hits / n_rel if n_rel else 1.0

    combined = (entity_recall + relation_recall + cardinality_recall) / 3
    return combined, entity_recall, relation_recall, cardinality_recall, missing_entities, issues


def rewrite_to_maximize_cardinality_recall_density(
    client,
    cfg: RewriteConfig,
    original: str,
    entity_names: Sequence[str],
    relation_pairs: Sequence[Tuple[str, str]],
    n_tokens_fn: Callable[[str], int],
    system_prompt: str,
    user_prompt_fn: Callable[[str, str, Optional[str]], str],
    feedback_fn: FeedbackFn,
) -> CardinalityRecallResult:
    """Iteratively rewrite ``original`` to maximize (entity recall +
    relation-pair recall + cardinality-presence recall) per token. Mirrors
    ``relation_loop``'s "no target, keep best-seen" pattern.
    """

    def score(text: str) -> Tuple[float, float, float, float, int, List[str], List[str]]:
        combined, er, rr, cr, missing_e, issues = _scored_recall(text, entity_names, relation_pairs)
        tokens = max(1, n_tokens_fn(text))
        density = combined / (tokens / 1000.0)
        return density, er, rr, cr, tokens, missing_e, issues

    source_density, source_er, source_rr, source_cr, source_tokens, _, _ = score(original)

    best_text = original
    best_density = source_density
    best_er, best_rr, best_cr, best_tokens = source_er, source_rr, source_cr, source_tokens
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
            logger.error("Cardinality-recall rewrite call failed on iter %d: %s", i, exc)
            break

        density, er, rr, cr, tokens, missing_e, issues = score(candidate_text)

        if density > best_density:
            best_text, best_density, improved = candidate_text, density, True
            best_er, best_rr, best_cr, best_tokens = er, rr, cr, tokens

        feedback = feedback_fn(density, best_density, er, rr, cr, tokens, missing_e, issues)

    return CardinalityRecallResult(
        source_entity_recall=source_er,
        source_relation_recall=source_rr,
        source_cardinality_recall=source_cr,
        source_density=source_density,
        best_entity_recall=best_er,
        best_relation_recall=best_rr,
        best_cardinality_recall=best_cr,
        best_density=best_density,
        best_n_tokens=best_tokens,
        improved=improved,
        iterations=iterations,
        text=best_text,
    )
