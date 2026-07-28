
"""Feedback loop for level six: entity + relation recall_density
(``text.rewrite.relation_loop``'s signals) extended with a third,
min-gated signal -- parse_tree_depth reduction relative to the source.

Level four already minimizes parse_tree_depth directly
(``text.rewrite.flat_prompts`` + ``loop.rewrite_to_minimize``), but its only
content guard is a token-count ratio -- nothing scores whether specific
entities/relationships survived the flattening. Level six instead reuses
level minus-six's min-gated recall-density architecture
(``text.rewrite.complete_loop``): entity naming, relationship co-location,
and parse-tree-depth reduction are combined via ``min(...)``, not an
average, so the rewrite cannot buy a higher score by trading completeness
for flatness (or vice versa) -- whichever signal is currently weakest is the
only one that can raise the score. Reuses the same entity/relation
extraction as level minus-four/five/six (``text.rewrite.relation_entities.
extract_entities_and_relations``). No gold access.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from .client import rewrite_once
from .config import RewriteConfig

logger = logging.getLogger(__name__)

# (density, best_density, entity_recall, relation_recall, ptd_reduction,
#  current_ptd, n_tokens, missing_entities, relationship_issues) -> feedback text
FeedbackFn = Callable[[float, float, float, float, float, float, int, List[str], List[str]], str]


@dataclass(frozen=True)
class PtdRecallResult:
    """Outcome of driving one description toward maximum min-gated entity +
    relation + parse-tree-depth-reduction recall_density."""

    source_entity_recall: float
    source_relation_recall: float
    source_ptd_reduction: float
    source_ptd: float
    source_density: float
    best_entity_recall: float
    best_relation_recall: float
    best_ptd_reduction: float
    best_ptd: float
    best_density: float
    best_n_tokens: int
    improved: bool
    iterations: int
    text: str


def _ptd_reduction(current_ptd: float, source_ptd: float) -> float:
    """Fraction by which ``current_ptd`` is lower than ``source_ptd``, clamped
    to [0, 1] -- 0 if parse_tree_depth didn't improve (or got worse), so it
    stays on the same scale as the entity/relation recall signals it's
    min-gated against."""
    if source_ptd <= 0 or not math.isfinite(current_ptd):
        return 0.0
    return max(0.0, min(1.0, (source_ptd - current_ptd) / source_ptd))


def _scored_recall(
    text: str,
    entity_names: Sequence[str],
    relation_pairs: Sequence[Tuple[str, str]],
    ptd_reduction: float,
) -> Tuple[float, float, float, List[str], List[str]]:
    """(combined, entity_recall, relation_recall, missing_entities,
    relationship_issues).

    ``combined`` is the min of the three sub-signals (entity recall,
    relation recall, ``ptd_reduction``), not their average -- see the module
    docstring for why min() prevents trading one signal for another.
    """
    t = text.lower()
    uniq_entities = sorted({n for n in entity_names if n})
    missing_entities = [n for n in uniq_entities if n.lower() not in t]
    entity_recall = (
        (len(uniq_entities) - len(missing_entities)) / len(uniq_entities) if uniq_entities else 0.0
    )

    lines = [line.lower() for line in text.splitlines() if line.strip()]
    issues: List[str] = []
    relation_hits = 0
    for a, b in relation_pairs:
        if any(a in line and b in line for line in lines):
            relation_hits += 1
        else:
            issues.append(f"{a}-{b} (relationship not stated together)")
    n_rel = len(relation_pairs)
    relation_recall = relation_hits / n_rel if n_rel else 1.0

    combined = min(entity_recall, relation_recall, ptd_reduction)
    return combined, entity_recall, relation_recall, missing_entities, issues


def rewrite_to_maximize_ptd_recall_density(
    client,
    cfg: RewriteConfig,
    original: str,
    entity_names: Sequence[str],
    relation_pairs: Sequence[Tuple[str, str]],
    metrics_fn: Callable[[str], Tuple[int, float]],
    system_prompt: str,
    user_prompt_fn: Callable[[str, str, Optional[str]], str],
    feedback_fn: FeedbackFn,
) -> PtdRecallResult:
    """Iteratively rewrite ``original`` to maximize
    min(entity recall, relation-pair recall, parse-tree-depth reduction)
    per token. Mirrors ``complete_loop``'s "no target, keep best-seen"
    pattern; only the third signal (parse-tree-depth reduction instead of
    cardinality-presence recall) and the ``metrics_fn`` dependency (one
    combined n_tokens + parse_tree_depth measurement per candidate, instead
    of a separate token-counting call) differ.

    Args:
        metrics_fn: ``text -> (n_tokens, parse_tree_depth)``, computed
            together from one parse pass (avoids parsing each candidate
            twice).
    """
    source_tokens, source_ptd = metrics_fn(original)
    source_tokens = max(1, source_tokens)
    source_ptd_reduction = _ptd_reduction(source_ptd, source_ptd)
    source_combined, source_er, source_rr, _, _ = _scored_recall(
        original, entity_names, relation_pairs, source_ptd_reduction
    )
    source_density = source_combined / (source_tokens / 1000.0)

    def score(text: str) -> Tuple[float, float, float, float, float, int, List[str], List[str]]:
        tokens, ptd = metrics_fn(text)
        tokens = max(1, tokens)
        ptd_reduction = _ptd_reduction(ptd, source_ptd)
        combined, er, rr, missing_e, issues = _scored_recall(text, entity_names, relation_pairs, ptd_reduction)
        density = combined / (tokens / 1000.0)
        return density, er, rr, ptd_reduction, ptd, tokens, missing_e, issues

    best_text = original
    best_density = source_density
    best_er, best_rr = source_er, source_rr
    best_ptd_reduction, best_ptd, best_tokens = source_ptd_reduction, source_ptd, source_tokens
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
            logger.error("PTD-recall rewrite call failed on iter %d: %s", i, exc)
            break

        density, er, rr, ptd_reduction, ptd, tokens, missing_e, issues = score(candidate_text)

        if density > best_density:
            best_text, best_density, improved = candidate_text, density, True
            best_er, best_rr, best_tokens = er, rr, tokens
            best_ptd_reduction, best_ptd = ptd_reduction, ptd

        feedback = feedback_fn(density, best_density, er, rr, ptd_reduction, ptd, tokens, missing_e, issues)

    return PtdRecallResult(
        source_entity_recall=source_er,
        source_relation_recall=source_rr,
        source_ptd_reduction=source_ptd_reduction,
        source_ptd=source_ptd,
        source_density=source_density,
        best_entity_recall=best_er,
        best_relation_recall=best_rr,
        best_ptd_reduction=best_ptd_reduction,
        best_ptd=best_ptd,
        best_density=best_density,
        best_n_tokens=best_tokens,
        improved=improved,
        iterations=iterations,
        text=best_text,
    )
