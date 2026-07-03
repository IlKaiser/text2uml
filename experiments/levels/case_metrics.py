"""Per-case text-metric trajectory across the description complexity levels.

For a given project, plots every linguistic metric at each configured level
(``LevelsConfig.levels``) and identifies which metric moved the most from the
first to the last level. Because the metrics live on different scales, "moved
the most" is measured in corpus standard-deviation units (the same
normalization behind ``z_index``), so the ranking is fair; raw values are
shown too.

Reads the precomputed ``levels_complexity.csv`` (no spaCy needed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless / reproducible rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from text.config import DEFAULT_CONFIG, TextConfig
from text.metrics import metric_names
from text.rewrite.scorer import build_reference

from .config import DEFAULT_LEVELS_CONFIG, LevelsConfig

logger = logging.getLogger(__name__)

_COMPLEXITY_CSV = "levels_complexity.csv"


@dataclass(frozen=True)
class MetricChange:
    """Per-level value trajectory for one metric, plus first->last level change."""

    metric: str
    values: Tuple[float, ...]  # one value per level, in rank order
    labels: Tuple[str, ...]  # short level labels ("L0", "L1", ...), same order
    raw_delta: float  # last level - first level
    std_delta: float  # (last - first) / corpus_std  (signed; oriented toward complexity)


def _load(cfg: LevelsConfig) -> pd.DataFrame:
    path = cfg.output_dir / _COMPLEXITY_CSV
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run the complexity stage first "
            f"(python -m experiments.levels.complexity)."
        )
    return pd.read_csv(path)


def analyze_case(
    case: str,
    levels_cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG,
    text_cfg: TextConfig = DEFAULT_CONFIG,
) -> List[MetricChange]:
    """Rank metrics by |first->last level| change in corpus-std units for one case.

    Works over however many levels are present for the case (sorted by rank),
    so it keeps working as levels are added or removed from ``LevelsConfig``.
    Flesch Reading Ease is sign-flipped so a positive ``std_delta`` always
    means "moved toward more complex", matching the z_index orientation.
    """
    df = _load(levels_cfg)
    sub = df[df["sub_folder_name"] == case]
    if sub.empty:
        raise ValueError(f"No rows for case {case!r} in {_COMPLEXITY_CSV}.")

    ref = build_reference(text_cfg)
    metrics = [m for m in metric_names() if m in sub.columns]
    by_rank = sub.sort_values("level_rank").set_index("level_rank")
    ranks = list(by_rank.index)
    labels = tuple(f"L{r}" for r in ranks)

    changes: List[MetricChange] = []
    for m in metrics:
        values = tuple(float(by_rank.loc[r, m]) for r in ranks)
        sd = ref.stds.get(m) or float("nan")
        raw = values[-1] - values[0]
        std_d = raw / sd if sd and np.isfinite(sd) and sd > 0 else float("nan")
        if m == "flesch_reading_ease":
            std_d = -std_d  # orient toward complexity
        changes.append(MetricChange(m, values, labels, raw, std_d))

    changes.sort(key=lambda c: abs(c.std_delta) if np.isfinite(c.std_delta) else -1, reverse=True)
    return changes


def plot_case(
    case: str,
    levels_cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG,
    text_cfg: TextConfig = DEFAULT_CONFIG,
) -> MetricChange:
    """Plot every metric across the three levels for ``case``.

    Left grid: raw value per metric vs level (the biggest mover is titled in
    red). Right panel: |L1->L3| change in corpus-std units, sorted, so the
    dominant metric is obvious. Returns the top mover.
    """
    changes = analyze_case(case, levels_cfg, text_cfg)
    top = changes[0]
    metrics = [c.metric for c in changes]
    labels = list(top.labels)
    xs = list(range(1, len(labels) + 1))

    n = len(metrics)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(5 * ncols + 4, 3.2 * nrows))
    gs = fig.add_gridspec(nrows, ncols + 1, width_ratios=[1] * ncols + [1.1], wspace=0.35, hspace=0.55)

    # One small-multiple per metric (ordered by importance).
    for i, ch in enumerate(changes):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c])
        vals = list(ch.values)
        color = "#c0392b" if ch.metric == top.metric else "#3b6fb0"
        ax.plot(xs, vals, marker="o", color=color, linewidth=2)
        for x, y in zip(xs, vals):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7)
        title = ch.metric + ("  ★" if ch.metric == top.metric else "")
        ax.set_title(f"{title}\nΔstd={ch.std_delta:+.2f}", fontsize=9,
                     color=color, fontweight="bold" if ch.metric == top.metric else "normal")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(linestyle=":", alpha=0.4)

    # Right column (full height): ranked |Δstd| bar chart.
    ax_bar = fig.add_subplot(gs[:, ncols])
    ordered = changes  # already sorted by |std_delta| desc
    names = [c.metric for c in ordered][::-1]
    std_deltas = [c.std_delta for c in ordered][::-1]
    colors = ["#c0392b" if c.metric == top.metric else "#7f8ca8" for c in ordered][::-1]
    ax_bar.barh(names, std_deltas, color=colors, edgecolor="white")
    ax_bar.axvline(0, color="black", linewidth=0.8)
    for y, v in enumerate(std_deltas):
        ax_bar.annotate(f"{v:+.2f}", (v, y), textcoords="offset points",
                        xytext=(6 if v >= 0 else -6, 0),
                        va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax_bar.set_title("L1→L3 change\n(corpus-std units)", fontsize=10)
    ax_bar.set_xlabel("Δstd (+ = more complex)")
    ax_bar.grid(axis="x", linestyle=":", alpha=0.4)

    fig.suptitle(
        f"{case}: text metrics across complexity levels — "
        f"biggest mover: {top.metric} (Δstd={top.std_delta:+.2f}, "
        f"{top.values[0]:.2f}→{top.values[-1]:.2f})",
        fontsize=13, y=1.0,
    )
    levels_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in levels_cfg.figure_formats:
        path = levels_cfg.output_dir / f"case_metrics_{case}.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=levels_cfg.dpi)
        logger.info("Saved %s", path)
    plt.close(fig)
    return top


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot per-case text metrics across the three levels and find the biggest mover."
    )
    parser.add_argument(
        "cases", nargs="+",
        help="Dataset folder name(s). Prefix match expands (e.g. 'GasStation').",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    cfg = DEFAULT_LEVELS_CONFIG
    df = _load(cfg)
    available = set(df["sub_folder_name"].unique())
    targets: List[str] = []
    for q in args.cases:
        exact = [c for c in available if c == q]
        pref = sorted(c for c in available if c.startswith(q))
        targets.extend(exact or pref)
    targets = list(dict.fromkeys(targets))
    if not targets:
        logger.error("No matching cases for %s. Available e.g.: %s",
                     args.cases, sorted(available)[:5])
        return 1

    for case in targets:
        top = plot_case(case, cfg)
        changes = analyze_case(case, cfg)
        logger.info("%s — metrics ranked by |first->last level| change (std units):", case)
        for c in changes:
            val_str = "  ".join(f"{lab}={v:7.3f}" for lab, v in zip(c.labels, c.values))
            logger.info("  %-26s %s  Δstd=%+.2f", c.metric, val_str, c.std_delta)
        logger.info("  >> biggest mover: %s (Δstd=%+.2f)", top.metric, top.std_delta)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
