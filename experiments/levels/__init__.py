"""Complexity-levels vs two-shot F1 experiment.

Compares two-shot UML generation across three complexity variants of each
project's description (level one / two / real), then evaluates and plots the
global and per-category (class, association, attribute, cardinality) F1.

Public API:
    LevelsConfig / DEFAULT_LEVELS_CONFIG
    generate            -- run two-shot generation per level
    evaluate_all / write_f1_csv
    generate_all_plots
"""

from __future__ import annotations

from .complexity import (
    compute_level_metrics,
    generate_all as generate_complexity_matrix,
    plot_complexity_matrix,
    plot_zindex_matrix,
)
from .config import CATEGORIES, DEFAULT_LEVELS_CONFIG, LevelsConfig
from .correlation import (
    generate_all as generate_correlation,
    load_merged,
    plot_f1_per_case,
    plot_f1_vs_complexity,
)
from .evaluate import evaluate_all, write_f1_csv
from .generate import generate
from .plots import generate_all_plots

__all__ = [
    "LevelsConfig",
    "DEFAULT_LEVELS_CONFIG",
    "CATEGORIES",
    "generate",
    "evaluate_all",
    "write_f1_csv",
    "generate_all_plots",
    "compute_level_metrics",
    "generate_complexity_matrix",
    "plot_complexity_matrix",
    "plot_zindex_matrix",
    "generate_correlation",
    "load_merged",
    "plot_f1_vs_complexity",
    "plot_f1_per_case",
]
