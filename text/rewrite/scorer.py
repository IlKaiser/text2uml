"""Reference-based complexity scoring for candidate descriptions.

Reuses the project's metric suite to score arbitrary text on the *same*
normalized ``z_index`` scale produced by ``text.plots.complexity_index``:

* Each metric is z-scored against the original corpus (mean / std from
  ``complexity_metrics.csv``).
* Flesch Reading Ease is inverted so every oriented metric grows with
  complexity; their mean is the raw complexity index.
* The raw index is min-max scaled with the corpus min / max, so ``0.0`` is the
  simplest original dataset and ``1.0`` the most complex.

This lets the feedback loop measure a freshly-rewritten description on a scale
that is directly comparable to the datasets it came from.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from ..config import DEFAULT_CONFIG, TextConfig
from ..metrics import compute_all, metric_names

logger = logging.getLogger(__name__)

# Metrics where a higher raw value means *simpler* text; flip their sign so the
# oriented value grows with complexity like every other metric.
_INVERTED = {"flesch_reading_ease"}


def _oriented_z(
    values: Dict[str, float],
    means: Dict[str, float],
    stds: Dict[str, float],
    metrics: List[str],
) -> Dict[str, float]:
    """Per-metric oriented z-score (positive = adds complexity)."""
    out: Dict[str, float] = {}
    for name in metrics:
        v = values.get(name)
        mu = means.get(name)
        sd = stds.get(name)
        if v is None or mu is None or sd is None:
            continue
        if not (math.isfinite(v) and math.isfinite(sd)) or sd <= 0:
            continue
        z = (v - mu) / sd
        out[name] = -z if name in _INVERTED else z
    return out


def _raw_complexity(oriented: Dict[str, float]) -> float:
    """Mean oriented z-score across the metrics that could be computed."""
    if not oriented:
        return 0.0
    return sum(oriented.values()) / len(oriented)


@dataclass(frozen=True)
class ComplexityReference:
    """Corpus statistics used to score new text on the ``z_index`` scale."""

    metrics: List[str]
    means: Dict[str, float]
    stds: Dict[str, float]
    raw_min: float
    raw_max: float

    def oriented_z(self, values: Dict[str, float]) -> Dict[str, float]:
        return _oriented_z(values, self.means, self.stds, self.metrics)

    def z_index(self, values: Dict[str, float]) -> float:
        raw = _raw_complexity(self.oriented_z(values))
        span = self.raw_max - self.raw_min
        if not math.isfinite(span) or span <= 0:
            return 0.5
        return (raw - self.raw_min) / span


@dataclass(frozen=True)
class ScoreResult:
    """The complexity measurement of a single piece of text."""

    z_index: float
    oriented_z: Dict[str, float]
    values: Dict[str, float]
    n_tokens: int


def build_reference(cfg: TextConfig = DEFAULT_CONFIG) -> ComplexityReference:
    """Build the scoring reference from the computed metrics CSV.

    Raises:
        FileNotFoundError: When the metrics CSV does not exist yet.
    """
    if not cfg.metrics_csv.is_file():
        raise FileNotFoundError(
            f"Metrics CSV not found: {cfg.metrics_csv}. Run 'python -m text.run' first."
        )
    df = pd.read_csv(cfg.metrics_csv)
    metrics = [m for m in metric_names() if m in df.columns]
    data = df.set_index("sub_folder_name")[metrics].astype(float)
    means = data.mean().to_dict()
    stds = data.std(ddof=0).to_dict()

    raws = [
        _raw_complexity(_oriented_z(row.to_dict(), means, stds, metrics))
        for _, row in data.iterrows()
    ]
    return ComplexityReference(
        metrics=metrics,
        means=means,
        stds=stds,
        raw_min=min(raws),
        raw_max=max(raws),
    )


def score_text(
    text: str, reference: ComplexityReference, cfg: TextConfig = DEFAULT_CONFIG
) -> ScoreResult:
    """Compute the metric suite on ``text`` and map it to the ``z_index`` scale.

    Raises:
        ValueError: When the metrics could not be computed (empty text / parse
            failure).
    """
    result = compute_all(text, sample_id="candidate", model_name=cfg.spacy_model)
    if result.error:
        raise ValueError(f"Could not score candidate text: {result.error}")
    return ScoreResult(
        z_index=reference.z_index(result.values),
        oriented_z=reference.oriented_z(result.values),
        values=result.values,
        n_tokens=result.n_tokens,
    )
