"""Feedback loop for level-minus-four: recall_density extended to credit
relationship pairs, not just entity names.

Separate from ``recall_loop.py`` (entity-name recall only) because this
scores two signals -- entity presence and whether a relationship's two
endpoint names co-occur in the same line of the candidate text (a proxy for
"the relationship is stated," matching how these compact notes format
relationships: one bullet/line per relationship, e.g. "Aircraft performs
several Flights").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from .client import rewrite_once
from .config import RewriteConfig

logger = logging.getLogger(__name__)

# (density, best_density, entity_recall, relation_recall, n_tokens,
#  missing_entities, missing_relations) -> feedback text
FeedbackFn = Callable[[float, float, float, float, int, List[str], List[str]], str]


@dataclass(frozen=True)
class RelationRecallResult:
    """Outcome of driving one description toward maximum combined recall_density."""

    source_entity_recall: float
    source_relation_recall: float
    source_density: float
    best_entity_recall: float
    best_relation_recall: float
    best_density: float
    best_n_tokens: int
    improved: bool
    iterations: int
    text: str


def _combined_recall(
    text: str, entity_names: Sequence[str], relation_pairs: Sequence[Tuple[str, str]]
) -> Tuple[float, float, List[str], List[str]]:
    """(combined recall, entity recall, missing entities, missing relations).

    Combined recall is the equal-weight average of entity recall and
    relation recall -- deliberately simple/symmetric to test the direct
    hypothesis that adding relationship credit (vs. entity-name recall
    alone) is what was missing, without also introducing a weighting
    parameter to tune.
    """
    t = text.lower()
    uniq_entities = sorted({n for n in entity_names if n})
    missing_entities = [n for n in uniq_entities if n.lower() not in t]
    entity_recall = (
        (len(uniq_entities) - len(missing_entities)) / len(uniq_entities) if uniq_entities else 0.0
    )

    lines = [line.lower() for line in text.splitlines() if line.strip()]
    missing_relations: List[str] = []
    hits = 0
    for a, b in relation_pairs:
        if any(a in line and b in line for line in lines):
            hits += 1
        else:
            missing_relations.append(f"{a}-{b}")
    relation_recall = hits / len(relation_pairs) if relation_pairs else 1.0

    combined = (entity_recall + relation_recall) / 2
    return combined, entity_recall, relation_recall, missing_entities, missing_relations


def rewrite_to_maximize_relation_recall_density(
    client,
    cfg: RewriteConfig,
    original: str,
    entity_names: Sequence[str],
    relation_pairs: Sequence[Tuple[str, str]],
    n_tokens_fn: Callable[[str], int],
    system_prompt: str,
    user_prompt_fn: Callable[[str, str, Optional[str]], str],
    feedback_fn: FeedbackFn,
) -> RelationRecallResult:
    """Iteratively rewrite ``original`` to maximize (entity recall +
    relationship-pair recall) per token. Mirrors ``recall_loop.
    rewrite_to_maximize_recall_density``'s "no target, keep best-seen"
    pattern -- there's no natural ceiling to stop at.
    """

    def score(text: str) -> Tuple[float, float, float, int, List[str], List[str]]:
        combined, entity_recall, relation_recall, missing_e, missing_r = _combined_recall(
            text, entity_names, relation_pairs
        )
        tokens = max(1, n_tokens_fn(text))
        density = combined / (tokens / 1000.0)
        return density, entity_recall, relation_recall, tokens, missing_e, missing_r

    source_density, source_entity_recall, source_relation_recall, source_tokens, _, _ = score(original)

    best_text = original
    best_density = source_density
    best_entity_recall, best_relation_recall, best_tokens = source_entity_recall, source_relation_recall, source_tokens
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
            logger.error("Relation-recall rewrite call failed on iter %d: %s", i, exc)
            break

        density, entity_recall, relation_recall, tokens, missing_e, missing_r = score(candidate_text)

        if density > best_density:
            best_text, best_density, improved = candidate_text, density, True
            best_entity_recall, best_relation_recall, best_tokens = entity_recall, relation_recall, tokens

        feedback = feedback_fn(density, best_density, entity_recall, relation_recall, tokens, missing_e, missing_r)

    return RelationRecallResult(
        source_entity_recall=source_entity_recall,
        source_relation_recall=source_relation_recall,
        source_density=source_density,
        best_entity_recall=best_entity_recall,
        best_relation_recall=best_relation_recall,
        best_density=best_density,
        best_n_tokens=best_tokens,
        improved=improved,
        iterations=iterations,
        text=best_text,
    )
