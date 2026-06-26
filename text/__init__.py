"""Linguistic-complexity metrics for the Text2UML dataset descriptions.

Implements the metrics from the ICSOC2026 paper and correlates them with the
LLM evaluation results. See ``text/README.md`` for usage.
"""

from __future__ import annotations

from .config import DEFAULT_CONFIG, TextConfig
from .metrics import compute_all, metric_names
from .pipeline import compute_dataset_metrics, run_pipeline
from .plots import generate_all_plots

__all__ = [
    "DEFAULT_CONFIG",
    "TextConfig",
    "compute_all",
    "metric_names",
    "compute_dataset_metrics",
    "run_pipeline",
    "generate_all_plots",
]
