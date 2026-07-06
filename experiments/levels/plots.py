"""Visualize how F1 varies across the three complexity levels.

Figures (written to ``experiments/levels/output`` in every configured format):

1. ``levels_global_f1``       - global F1 vs level: per-project lines + mean.
2. ``levels_category_f1``     - mean F1 per level, one panel per category.
3. ``levels_category_bars``   - grouped bars: mean F1 per category per level.
"""

from __future__ import annotations

import logging
from typing import Optional

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


def _save(fig, cfg: LevelsConfig, stem: str, subdir: str = "corpus") -> None:
    out_dir = cfg.figure_dir(subdir)
    for fmt in cfg.figure_formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=cfg.dpi)
        logger.info("Saved %s", path)
    plt.close(fig)


def _weighted_mean(group: pd.DataFrame, value_col: str, weight_col: Optional[str]) -> float:
    """Plain mean, or weight_col-weighted mean when weight_col is given."""
    if weight_col is None:
        return group[value_col].mean()
    w = group[weight_col]
    if w.sum() == 0:
        return float("nan")
    return np.average(group[value_col], weights=w)


def _weighted_sem(group: pd.DataFrame, value_col: str, weight_col: Optional[str]) -> float:
    """Plain SEM, or a weighted-variance-based SEM when weight_col is given."""
    if weight_col is None:
        return group[value_col].sem()
    w = group[weight_col].to_numpy()
    v = group[value_col].to_numpy()
    if w.sum() == 0 or len(v) < 2:
        return float("nan")
    mean = np.average(v, weights=w)
    variance = np.average((v - mean) ** 2, weights=w)
    # Effective sample size for weighted data (Kish's approximation).
    n_eff = (w.sum() ** 2) / (w ** 2).sum()
    return float(np.sqrt(variance / max(n_eff, 1.0)))


def plot_global_f1(
    df: pd.DataFrame, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG, weight_col: Optional[str] = None
) -> None:
    """Global F1 vs complexity level: faint per-project lines + bold mean.

    ``weight_col`` (e.g. gold-model size from ``model_complexity.py``, merged
    into ``df`` by the caller) switches the bold line from a plain mean/SEM
    to a weight_col-weighted mean/SEM, and saves to a ``_weighted`` filename
    instead of overwriting the unweighted plot.
    """
    labels, ranks = _ordered_levels(df)
    fig, ax = plt.subplots(figsize=(7, 5))

    # One faint line per project (only projects present at every level).
    pivot = df.pivot_table(index="sub_folder_name", columns=_LEVEL_ORDER_COL, values="f1_global")
    pivot = pivot.reindex(columns=ranks)
    complete = pivot.dropna()
    for _name, row in complete.iterrows():
        ax.plot(ranks, row.values, color="#9db4d4", alpha=0.5, linewidth=0.9, zorder=1)

    # Mean across all available projects at each level (uses partial coverage).
    grouped = df.groupby(_LEVEL_ORDER_COL)
    mean_by_level = pd.Series(
        {rank: _weighted_mean(g, "f1_global", weight_col) for rank, g in grouped}
    ).reindex(ranks)
    sem_by_level = pd.Series(
        {rank: _weighted_sem(g, "f1_global", weight_col) for rank, g in grouped}
    ).reindex(ranks)
    label = "weighted mean ± weighted SEM" if weight_col else "mean ± SEM"
    ax.errorbar(
        ranks, mean_by_level.values, yerr=sem_by_level.values,
        color="#c0392b", linewidth=2.5, marker="o", markersize=7,
        capsize=4, zorder=3, label=label,
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
    weight_note = f"\n(weighted by {weight_col})" if weight_col else ""
    ax.set_title(
        f"Two-shot UML global F1 vs description complexity\n"
        f"(faint lines: {n_complete} projects present at all levels){weight_note}"
    )
    ax.legend(loc="best")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    _save(fig, cfg, "levels_global_f1_weighted" if weight_col else "levels_global_f1")


def plot_category_lines(
    df: pd.DataFrame, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG, weight_col: Optional[str] = None
) -> None:
    """One panel per category: mean F1 (± SEM) vs level.

    See ``plot_global_f1`` for ``weight_col`` semantics.
    """
    labels, ranks = _ordered_levels(df)
    n = len(CATEGORIES)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
    axes = np.array(axes).reshape(-1)

    grouped = df.groupby(_LEVEL_ORDER_COL)
    for ax, cat in zip(axes, CATEGORIES):
        col = f"f1_{cat}"
        mean = pd.Series({rank: _weighted_mean(g, col, weight_col) for rank, g in grouped}).reindex(ranks)
        sem = pd.Series({rank: _weighted_sem(g, col, weight_col) for rank, g in grouped}).reindex(ranks)
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
    weight_note = f" (weighted by {weight_col})" if weight_col else ""
    fig.suptitle(f"Two-shot UML F1 by category vs description complexity{weight_note}", y=1.0, fontsize=13)
    fig.tight_layout()
    _save(fig, cfg, "levels_category_f1_weighted" if weight_col else "levels_category_f1")


def plot_category_bars(
    df: pd.DataFrame, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG, weight_col: Optional[str] = None
) -> None:
    """Grouped bar chart: mean F1 per category, grouped by level.

    See ``plot_global_f1`` for ``weight_col`` semantics.
    """
    labels, ranks = _ordered_levels(df)
    grouped = df.groupby(_LEVEL_ORDER_COL)
    means = {
        cat: pd.Series({rank: _weighted_mean(g, f"f1_{cat}", weight_col) for rank, g in grouped}).reindex(ranks).values
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
    weight_note = f" (weighted by {weight_col})" if weight_col else ""
    ax.set_title(f"Two-shot UML mean F1 by category and complexity level{weight_note}")
    ax.legend(title="level")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    _save(fig, cfg, "levels_category_bars_weighted" if weight_col else "levels_category_bars")


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
    _save(fig, cfg, case, subdir="levels_bars")


def generate_all_plots(df: pd.DataFrame, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> None:
    if df.empty:
        logger.warning("No F1 data; skipping plots.")
        return
    plot_global_f1(df, cfg)
    plot_category_lines(df, cfg)
    plot_category_bars(df, cfg)


def generate_all_weighted_plots(
    df: pd.DataFrame,
    complexity_df: pd.DataFrame,
    cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG,
    weight_col: str = "total_size",
) -> None:
    """Weighted counterparts of ``generate_all_plots``, using gold-model size
    (from ``model_complexity.compute_gold_complexity``) as the weight."""
    if df.empty:
        logger.warning("No F1 data; skipping weighted plots.")
        return
    merged = df.merge(complexity_df[["sub_folder_name", weight_col]], on="sub_folder_name", how="inner")
    if merged.empty:
        logger.warning("No overlap between F1 data and complexity data; skipping weighted plots.")
        return
    plot_global_f1(merged, cfg, weight_col=weight_col)
    plot_category_lines(merged, cfg, weight_col=weight_col)
    plot_category_bars(merged, cfg, weight_col=weight_col)
