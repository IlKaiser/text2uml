"""Metric registry and document-level computation.

All metrics register themselves here so the pipeline stays config-driven: adding
a metric is a one-line registration, not a pipeline edit.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from .base import Metric, MetricResult, get_nlp, parse
from .discourse import (
    ContextDependenceProxy,
    InferentialLoadProxy,
    RSTDepthProxy,
)
from .readability import FleschReadingEase
from .syntactic import (
    MeanDependencyDistance,
    ParseTreeDepth,
    ParseTreeDepthMax,
    SubordinationIndex,
)

logger = logging.getLogger(__name__)

# Ordered registry: insertion order is the column order in the output CSV.
METRIC_REGISTRY: Dict[str, Metric] = {}


def register_metric(metric: Metric) -> Metric:
    """Add a metric instance to the global registry, keyed by its ``name``."""
    if metric.name in METRIC_REGISTRY:
        logger.warning("Overwriting already-registered metric '%s'", metric.name)
    METRIC_REGISTRY[metric.name] = metric
    return metric


# Register the paper's metrics in reporting order.
for _metric in (
    MeanDependencyDistance(),
    SubordinationIndex(),
    ParseTreeDepth(),
    ParseTreeDepthMax(),
    FleschReadingEase(),
    RSTDepthProxy(),
    ContextDependenceProxy(),
    InferentialLoadProxy(),
):
    register_metric(_metric)


def metric_names() -> List[str]:
    """Return registered metric names in reporting order."""
    return list(METRIC_REGISTRY.keys())


def compute_all(text: str, sample_id: str, model_name: str = "en_core_web_sm") -> MetricResult:
    """Parse ``text`` once and compute every registered metric.

    Args:
        text: Raw description text.
        sample_id: Identifier for the sample (e.g. dataset folder name).
        model_name: spaCy model to use.

    Returns:
        A ``MetricResult`` with one value per registered metric. On empty input
        or a parse failure, every value is ``nan`` and ``error`` is set.
    """
    text = (text or "").strip()
    if not text:
        return MetricResult(
            sample_id=sample_id,
            values={name: float("nan") for name in metric_names()},
            n_tokens=0,
            n_sentences=0,
            error="empty text",
        )

    try:
        doc = parse(text, model_name)
    except Exception as exc:  # noqa: BLE001 - report and continue the batch
        logger.error("Failed to parse sample '%s': %s", sample_id, exc)
        return MetricResult(
            sample_id=sample_id,
            values={name: float("nan") for name in metric_names()},
            n_tokens=0,
            n_sentences=0,
            error=str(exc),
        )

    values: Dict[str, float] = {}
    for name, metric in METRIC_REGISTRY.items():
        try:
            values[name] = float(metric.compute(doc))
        except Exception as exc:  # noqa: BLE001 - isolate a single bad metric
            logger.error("Metric '%s' failed on '%s': %s", name, sample_id, exc)
            values[name] = float("nan")

    return MetricResult(
        sample_id=sample_id,
        values=values,
        n_tokens=len(doc),
        n_sentences=sum(1 for _ in doc.sents),
        error=None,
    )


__all__ = [
    "METRIC_REGISTRY",
    "register_metric",
    "metric_names",
    "compute_all",
    "MetricResult",
    "get_nlp",
]
