"""Correlate per-case F1 against description complexity.

Joins the per-(case, level) F1 scores (``levels_f1.csv``) with the per-(case,
level) complexity (``levels_complexity.csv``) on (sub_folder_name, level), then:

1. ``levels_f1_vs_complexity`` - scatter of F1 vs z_index, one point per
   case x level, for the global score and each category, with a trend line and
   Pearson / Spearman correlation. This is the score-vs-complexity view.
2. ``levels_f1_per_case``      - per-case breakdown: a heatmap of global F1
   (rows = cases, cols = levels) so you can read each case's trajectory.

Each figure is written under ``experiments/levels/output`` in every format.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless / reproducible rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import CATEGORIES, DEFAULT_LEVELS_CONFIG, LevelsConfig

logger = logging.getLogger(__name__)

_COMPLEXITY_CSV = "levels_complexity.csv"
# Series correlated against complexity: global + the four categories.
_SERIES: List[Tuple[str, str]] = [("f1_global", "global")] + [
    (f"f1_{c}", c) for c in CATEGORIES
]
_LEVEL_COLORS = {1: "#4c78a8", 2: "#f58518", 3: "#54a24b"}


def load_merged(cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> pd.DataFrame:
    """Join F1 and complexity on (sub_folder_name, level).

    Raises:
        FileNotFoundError: When either input CSV is missing.
    """
    f1_path = cfg.f1_csv
    cx_path = cfg.output_dir / _COMPLEXITY_CSV
    for p in (f1_path, cx_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing {p.name}. Run the evaluate stage and the complexity "
                f"stage first."
            )
    f1 = pd.read_csv(f1_path)
    cx = pd.read_csv(cx_path)[["sub_folder_name", "level", "z_index", "n_tokens"]]
    merged = f1.merge(cx, on=["sub_folder_name", "level"], how="inner")
    if merged.empty:
        logger.warning("F1 and complexity CSVs share no (case, level) rows.")
    return merged.sort_values(["sub_folder_name", "level_rank"]).reset_index(drop=True)


def _corr(x: pd.Series, y: pd.Series):
    """(pearson_r, pearson_p, spearman_r, spearman_p); nan when undefined."""
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return (float("nan"),) * 4
    from scipy import stats

    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return pr, pp, sr, sp


def _save(fig, cfg: LevelsConfig, stem: str) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in cfg.figure_formats:
        path = cfg.output_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=cfg.dpi)
        logger.info("Saved %s", path)
    plt.close(fig)


def plot_f1_vs_complexity(
    merged: pd.DataFrame, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG
) -> pd.DataFrame:
    """Scatter grid: F1 (global + categories) vs z_index, colored by level.

    Returns a correlation table (one row per series).
    """
    n = len(_SERIES)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    records = []
    x = merged["z_index"].astype(float)
    for ax, (col, label) in zip(axes, _SERIES):
        y = merged[col].astype(float)
        pr, pp, sr, sp = _corr(x, y)
        records.append(
            {"series": label, "pearson_r": pr, "pearson_p": pp,
             "spearman_r": sr, "spearman_p": sp, "n": int((x.notna() & y.notna()).sum())}
        )

        for rank, color in _LEVEL_COLORS.items():
            sub = merged[merged["level_rank"] == rank]
            if not sub.empty:
                ax.scatter(sub["z_index"], sub[col], color=color, alpha=0.75,
                           edgecolor="white", linewidth=0.5,
                           label=sub["level_label"].iloc[0], zorder=3)
        mask = x.notna() & y.notna()
        if mask.sum() >= 2 and x[mask].nunique() >= 2:
            coeffs = np.polyfit(x[mask], y[mask], 1)
            xs = np.linspace(x[mask].min(), x[mask].max(), 50)
            ax.plot(xs, np.polyval(coeffs, xs), color="#c0392b", linewidth=1.5, zorder=2)

        title = f"{label}\nPearson r={pr:.2f} (p={pp:.2g})" if np.isfinite(pr) else \
            f"{label}\n(need >=3 varied points)"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("z_index (0=simple, 1=complex)")
        ax.set_ylabel(f"{label} F1")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(linestyle=":", alpha=0.4)

    axes[0].legend(title="level", fontsize=8, loc="best")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Per-case F1 vs description complexity (each point = one case at one level)",
                 y=1.0, fontsize=13)
    fig.tight_layout()
    _save(fig, cfg, "levels_f1_vs_complexity")
    return pd.DataFrame(records)


def plot_f1_per_case(
    merged: pd.DataFrame, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG
) -> None:
    """Heatmap of global F1 per case (rows) x level (cols), sorted by real F1.

    An "Average" row (the column-wise mean across all cases) is appended
    below the per-case rows and set off with a heavier border, so the
    corpus-wide trend is visible in the same figure as the individual cases.
    """
    labels_ranks = (
        merged[["level_rank", "level_label"]].drop_duplicates().sort_values("level_rank")
    )
    labels = list(labels_ranks["level_label"])
    ranks = list(labels_ranks["level_rank"])
    pivot = merged.pivot_table(
        index="sub_folder_name", columns="level_rank", values="f1_global"
    ).reindex(columns=ranks)
    pivot.columns = labels
    sort_col = labels[-1] if labels[-1] in pivot.columns else labels[0]
    pivot = pivot.reindex(pivot.sort_values(sort_col).index)

    average_row = pivot.mean(axis=0, skipna=True)
    pivot_with_avg = pd.concat([pivot, average_row.to_frame("Average").T])

    height = max(4.0, 0.28 * len(pivot_with_avg))
    fig, ax = plt.subplots(figsize=(1.7 * len(labels) + 3, height))
    sns.heatmap(
        pivot_with_avg, cmap="RdYlGn", vmin=0.0, vmax=1.0, linewidths=0.4, linecolor="white",
        annot=True, fmt=".2f", annot_kws={"fontsize": 7},
        cbar_kws={"label": "global F1"}, ax=ax,
    )
    # Heavier border between the per-case rows and the summary "Average" row.
    ax.axhline(len(pivot), color="black", linewidth=2.5)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_title("Per-case global F1 across the three complexity levels")
    ax.set_xlabel("description level")
    ax.set_ylabel("dataset")
    _save(fig, cfg, "levels_f1_per_case")


def case_level_score_correlation(
    case: str, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG
) -> pd.DataFrame:
    """Correlate ``level_rank`` directly against F1 (global + categories) for one case.

    Unlike ``plot_f1_vs_complexity`` (which correlates F1 against the
    continuous ``z_index`` across every case x level point), this looks at a
    single case's own trajectory across its configured levels — one point per
    level, ranked 0..N. With few levels the p-values are not meaningful; this
    is a descriptive summary, not a significance test.

    Writes ``{case}_levels_score_correlation.csv`` under ``cfg.output_dir``.
    """
    f1 = pd.read_csv(cfg.f1_csv)
    sub = f1[f1["sub_folder_name"] == case].sort_values("level_rank")
    if sub.empty:
        raise ValueError(f"No F1 rows for case {case!r} in {cfg.f1_csv.name}.")

    x = sub["level_rank"].astype(float)
    series = [("f1_global", "global")] + [(f"f1_{c}", c) for c in CATEGORIES]
    rows = []
    for col, label in series:
        y = sub[col].astype(float)
        pr, pp, sr, sp = _corr(x, y)
        rows.append({"series": label, "pearson_r": pr, "pearson_p": pp,
                     "spearman_r": sr, "spearman_p": sp, "n": int(len(x))})

    corr = pd.DataFrame(rows)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / f"{case}_levels_score_correlation.csv"
    corr.to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)
    return corr


def generate_all(cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> pd.DataFrame:
    """Merge, plot both figures, and write the correlation table."""
    merged = load_merged(cfg)
    if merged.empty:
        logger.warning("Nothing to correlate.")
        return merged
    corr = plot_f1_vs_complexity(merged, cfg)
    plot_f1_per_case(merged, cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    corr_path = cfg.output_dir / "levels_f1_complexity_correlation.csv"
    corr.to_csv(corr_path, index=False)
    logger.info("Wrote correlation table to %s", corr_path)
    if corr["pearson_r"].notna().any():
        logger.info("F1 vs z_index correlation:\n%s", corr.to_string(index=False))
    else:
        logger.info(
            "Not enough varied (case, level) points for a correlation yet — "
            "run the evaluate stage across many cases first."
        )
    return corr


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Correlate per-case F1 with description complexity."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    try:
        generate_all(DEFAULT_LEVELS_CONFIG)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    logger.info("Done. Outputs in %s", DEFAULT_LEVELS_CONFIG.output_dir)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
