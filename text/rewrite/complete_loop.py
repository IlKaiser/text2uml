"""Feedback loop for level-minus-six: entity + relation + cardinality
recall_density (``text.rewrite.cardinality_loop``), but combined via
min(...) instead of average(...).

Level minus-five's ``(entity_recall + relation_recall + cardinality_recall)
/ 3`` let the rewrite trade cardinality away under compression pressure: a
relationship stated without its multiplicity only cost 1/3 of a point in
the average, cheaper than the 2/3 a fully-dropped relationship would cost,
so density-maximizing pressure always found it cheapest to cut cardinality
first. Level minus-five landed *below* level minus-four on both global F1
(0.4233 vs 0.4249) and cardinality F1 (0.344 vs 0.363) as a direct result.

Using min(entity_recall, relation_recall, cardinality_recall) as the
combined signal removes that trade -- the score is bottlenecked by
whichever of the three is currently weakest, so there is no way to raise it
except by improving that exact signal. Still no gold access; reuses the
same extraction as level minus-four/minus-five
(``text.rewrite.relation_entities.extract_entities_and_relations``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from .cardinality_loop import _MULTIPLICITY_RE
from .client import rewrite_once
from .config import RewriteConfig

logger = logging.getLogger(__name__)

# (density, best_density, entity_recall, relation_recall, cardinality_recall,
#  n_tokens, missing_entities, relationship_issues) -> feedback text
FeedbackFn = Callable[[float, float, float, float, float, int, List[str], List[str]], str]


@dataclass(frozen=True)
class CompleteRecallResult:
    """Outcome of driving one description toward maximum min-gated entity +
    relation + cardinality recall_density."""

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
    missing_entities, relationship_issues).

    ``combined`` is the min of the three sub-recalls, not their average --
    see module docstring for why min() closes level minus-five's exploit.
    """
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

    combined = min(entity_recall, relation_recall, cardinality_recall)
    return combined, entity_recall, relation_recall, cardinality_recall, missing_entities, issues


def rewrite_to_maximize_complete_recall_density(
    client,
    cfg: RewriteConfig,
    original: str,
    entity_names: Sequence[str],
    relation_pairs: Sequence[Tuple[str, str]],
    n_tokens_fn: Callable[[str], int],
    system_prompt: str,
    user_prompt_fn: Callable[[str, str, Optional[str]], str],
    feedback_fn: FeedbackFn,
) -> CompleteRecallResult:
    """Iteratively rewrite ``original`` to maximize
    min(entity recall, relation-pair recall, cardinality-presence recall)
    per token. Mirrors ``cardinality_loop``'s "no target, keep best-seen"
    pattern; only the combine function in ``_scored_recall`` differs.
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
            logger.error("Complete-recall rewrite call failed on iter %d: %s", i, exc)
            break

        density, er, rr, cr, tokens, missing_e, issues = score(candidate_text)

        if density > best_density:
            best_text, best_density, improved = candidate_text, density, True
            best_er, best_rr, best_cr, best_tokens = er, rr, cr, tokens

        feedback = feedback_fn(density, best_density, er, rr, cr, tokens, missing_e, issues)

    return CompleteRecallResult(
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
