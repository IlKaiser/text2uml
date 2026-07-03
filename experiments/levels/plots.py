"""Visualize how F1 varies across the three complexity levels.

Figures (written to ``experiments/levels/output`` in every configured format):

1. ``levels_global_f1``       - global F1 vs level: per-project lines + mean.
2. ``levels_category_f1``     - mean F1 per level, one panel per category.
3. ``levels_category_bars``   - grouped bars: mean F1 per category per level.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")  # headless / reproducible rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CATEGORIES, DEFAULT_LEVELS_CONFIG, LevelsConfig

logger = logging.getLogger(__name__)

_LEVEL_ORDER_COL = "level_rank"


def _ordered_levels(df: pd.DataFrame):
    """Return (labels, ranks) sorted by rank for a consistent x-axis."""
    seen = df[[_LEVEL_ORDER_COL, "level_label"]].drop_duplicates().sort_values(_LEVEL_ORDER_COL)
    return list(seen["level_label"]), list(seen[_LEVEL_ORDER_COL])


def _save(fig, cfg: LevelsConfig, stem: str) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in cfg.figure_formats:
        path = cfg.output_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=cfg.dpi)
        logger.info("Saved %s", path)
    plt.close(fig)


def plot_global_f1(df: pd.DataFrame, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> None:
    """Global F1 vs complexity level: faint per-project lines + bold mean."""
    labels, ranks = _ordered_levels(df)
    fig, ax = plt.subplots(figsize=(7, 5))

    # One faint line per project (only projects present at every level).
    pivot = df.pivot_table(index="sub_folder_name", columns=_LEVEL_ORDER_COL, values="f1_global")
    pivot = pivot.reindex(columns=ranks)
    complete = pivot.dropna()
    for _name, row in complete.iterrows():
        ax.plot(ranks, row.values, color="#9db4d4", alpha=0.5, linewidth=0.9, zorder=1)

    # Mean across all available projects at each level (uses partial coverage).
    mean_by_level = df.groupby(_LEVEL_ORDER_COL)["f1_global"].mean().reindex(ranks)
    sem_by_level = df.groupby(_LEVEL_ORDER_COL)["f1_global"].sem().reindex(ranks)
    ax.errorbar(
        ranks, mean_by_level.values, yerr=sem_by_level.values,
        color="#c0392b", linewidth=2.5, marker="o", markersize=7,
        capsize=4, zorder=3, label="mean ± SEM",
    )
    for x, y in zip(ranks, mean_by_level.values):
        if np.isfinite(y):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9, color="#c0392b")

    ax.set_xticks(ranks)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("description complexity level")
    ax.set_ylabel("global F1 (mean of 4 categories)")
    n_complete = len(complete)
    ax.set_title(
        f"Two-shot UML global F1 vs description complexity\n"
        f"(faint lines: {n_complete} projects present at all levels)"
    )
    ax.legend(loc="best")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    _save(fig, cfg, "levels_global_f1")


def plot_category_lines(df: pd.DataFrame, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> None:
    """One panel per category: mean F1 (± SEM) vs level."""
    labels, ranks = _ordered_levels(df)
    n = len(CATEGORIES)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
    axes = np.array(axes).reshape(-1)

    for ax, cat in zip(axes, CATEGORIES):
        col = f"f1_{cat}"
        mean = df.groupby(_LEVEL_ORDER_COL)[col].mean().reindex(ranks)
        sem = df.groupby(_LEVEL_ORDER_COL)[col].sem().reindex(ranks)
        ax.errorbar(ranks, mean.values, yerr=sem.values, color="#3b6fb0",
                    linewidth=2, marker="o", markersize=6, capsize=4)
        for x, y in zip(ranks, mean.values):
            if np.isfinite(y):
                ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8)
        ax.set_title(cat, fontsize=11)
        ax.set_xticks(ranks)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    for ax in axes[n:]:
        ax.axis("off")
    axes[0].set_ylabel("mean F1")
    if nrows > 1:
        axes[ncols].set_ylabel("mean F1")
    fig.suptitle("Two-shot UML F1 by category vs description complexity", y=1.0, fontsize=13)
    fig.tight_layout()
    _save(fig, cfg, "levels_category_f1")


def plot_category_bars(df: pd.DataFrame, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> None:
    """Grouped bar chart: mean F1 per category, grouped by level."""
    labels, ranks = _ordered_levels(df)
    means = {
        cat: df.groupby(_LEVEL_ORDER_COL)[f"f1_{cat}"].mean().reindex(ranks).values
        for cat in CATEGORIES
    }
    x = np.arange(len(CATEGORIES))
    width = 0.8 / max(1, len(ranks))
    colors = ["#4c78a8", "#f58518", "#54a24b"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, rank) in enumerate(zip(labels, ranks)):
        vals = [means[cat][i] for cat in CATEGORIES]
        offset = (i - (len(ranks) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=label,
                      color=colors[i % len(colors)], edgecolor="white")
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean F1")
    ax.set_xlabel("category")
    ax.set_title("Two-shot UML mean F1 by category and complexity level")
    ax.legend(title="level")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    _save(fig, cfg, "levels_category_bars")


_CASE_BAR_PALETTE = ["#9d4edd", "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"]


def plot_case_bars(case: str, df: pd.DataFrame, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> None:
    """Grouped bar chart for one case: global + per-category F1, grouped by level.

    Unlike ``plot_category_bars`` (which averages across every dataset), this
    plots a single case's own scores and saves to a case-specific filename so
    it never clobbers the corpus-wide plot. Supports any number of levels.
    """
    sub = df[df["sub_folder_name"] == case]
    if sub.empty:
        raise ValueError(f"No F1 rows for case {case!r}.")
    labels, ranks = _ordered_levels(sub)
    series = ["global"] + list(CATEGORIES)

    x = np.arange(len(series))
    width = 0.8 / max(1, len(ranks))
    colors = [_CASE_BAR_PALETTE[i % len(_CASE_BAR_PALETTE)] for i in range(len(ranks))]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, rank) in enumerate(zip(labels, ranks)):
        row = sub[sub[_LEVEL_ORDER_COL] == rank].iloc[0]
        vals = [row[f"f1_{s}"] for s in series]
        offset = (i - (len(ranks) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=colors[i], edgecolor="white")
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(series)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_xlabel("category")
    model = sub["model"].iloc[0] if "model" in sub.columns else ""
    ax.set_title(f"{case}: F1 by category and complexity level (model: {model})")
    ax.legend(title="level")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    _save(fig, cfg, f"levels_bars_{case}")


def generate_all_plots(df: pd.DataFrame, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> None:
    if df.empty:
        logger.warning("No F1 data; skipping plots.")
        return
    plot_global_f1(df, cfg)
    plot_category_lines(df, cfg)
    plot_category_bars(df, cfg)
