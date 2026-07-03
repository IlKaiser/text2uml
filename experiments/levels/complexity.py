"""Complexity matrix for the three description levels.

Scores each project's three description variants (level one / two / real) on the
project's linguistic-complexity metric suite, normalized to the corpus
``z_index`` scale, and plots:

1. ``levels_complexity_matrix`` - the requested matrix: the three levels (rows)
   against every metric plus the overall z_index (columns), as oriented
   z-scores (higher = more complex), averaged over all projects.
2. ``levels_zindex_matrix``     - per-project z_index (rows = datasets) at each
   level (columns), which shows directly whether L1 < L2 < L3 holds.

Reuses ``text.metrics`` for scoring and ``text.rewrite.scorer`` for the corpus
reference and z_index mapping.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")  # headless / reproducible rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from text.config import DEFAULT_CONFIG, TextConfig
from text.metrics import compute_all, metric_names
from text.rewrite.scorer import build_reference

from .config import DEFAULT_LEVELS_CONFIG, LevelsConfig

logger = logging.getLogger(__name__)

_COMPLEXITY_CSV = "levels_complexity.csv"


def compute_level_metrics(
    levels_cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG,
    text_cfg: TextConfig = DEFAULT_CONFIG,
    only: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Score every (dataset, level) description on the metric suite + z_index.

    Args:
        only: Restrict to these dataset folder names (else every dataset).

    Returns:
        Tidy DataFrame: sub_folder_name, level, level_label, level_rank,
        every metric name, z_index, n_tokens.

    Raises:
        FileNotFoundError: When the corpus metrics CSV does not exist.
    """
    reference = build_reference(text_cfg)
    metrics = metric_names()
    datasets = sorted(p for p in levels_cfg.dataset_dir.iterdir() if p.is_dir())
    if only:
        wanted = set(only)
        datasets = [p for p in datasets if p.name in wanted]
    level_meta = {t: (fname, label, rank) for t, fname, label, rank in levels_cfg.levels}

    rows: List[dict] = []
    for dataset in datasets:
        for tag, (fname, label, rank) in level_meta.items():
            desc = dataset / fname
            if not desc.is_file():
                logger.debug("  %s: missing %s; skipping level %s", dataset.name, fname, tag)
                continue
            text = desc.read_text(encoding="utf-8")
            result = compute_all(text, sample_id=dataset.name, model_name=text_cfg.spacy_model)
            if result.error:
                logger.warning("  %s/%s: %s; skipping", dataset.name, tag, result.error)
                continue
            row = {
                "sub_folder_name": dataset.name,
                "level": tag,
                "level_label": label,
                "level_rank": rank,
                "z_index": reference.z_index(result.values),
                "n_tokens": result.n_tokens,
            }
            row.update({m: float(result.values.get(m, float("nan"))) for m in metrics})
            rows.append(row)
            logger.info("  %s/%s: z_index=%.2f", dataset.name, tag, row["z_index"])

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["sub_folder_name", "level_rank"]).reset_index(drop=True)
    return df


def write_complexity_csv(
    df: pd.DataFrame, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG
) -> "pd.Path | None":
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.output_dir / _COMPLEXITY_CSV
    df.to_csv(out, index=False)
    logger.info("Wrote %d rows to %s", len(df), out)
    return out


def _save(fig, cfg: LevelsConfig, stem: str) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in cfg.figure_formats:
        path = cfg.output_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=cfg.dpi)
        logger.info("Saved %s", path)
    plt.close(fig)


def _ordered_levels(df: pd.DataFrame):
    seen = df[["level_rank", "level_label"]].drop_duplicates().sort_values("level_rank")
    return list(seen["level_label"]), list(seen["level_rank"])


def plot_complexity_matrix(
    df: pd.DataFrame,
    text_cfg: TextConfig = DEFAULT_CONFIG,
    cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG,
) -> None:
    """Heatmap: levels (rows) x metrics + z_index (cols), oriented z-scores.

    Each metric is z-scored against the corpus (mean/std from the metrics CSV)
    and oriented so higher always means more complex (Flesch Reading Ease is
    flipped). Cells are the mean oriented z-score across projects at that level.
    A trailing ``z_index`` column shows the overall [0, 1] complexity.
    """
    reference = build_reference(text_cfg)
    metrics = [m for m in metric_names() if m in df.columns]
    labels, ranks = _ordered_levels(df)

    # Oriented mean z-score per (level, metric).
    means, stds = reference.means, reference.stds
    grid = np.full((len(ranks), len(metrics)), np.nan)
    for i, rank in enumerate(ranks):
        sub = df[df["level_rank"] == rank]
        for j, m in enumerate(metrics):
            sd = stds.get(m)
            mu = means.get(m)
            if not sd or sd <= 0 or mu is None:
                continue
            z = (sub[m].astype(float) - mu) / sd
            if m == "flesch_reading_ease":
                z = -z
            grid[i, j] = z.mean()

    z_index_col = df.groupby("level_rank")["z_index"].mean().reindex(ranks).to_numpy()

    matrix = pd.DataFrame(grid, index=labels, columns=metrics)
    matrix["z_index"] = z_index_col

    fig, (ax, ax_idx) = plt.subplots(
        1, 2,
        figsize=(1.1 * len(metrics) + 4, 0.9 * len(ranks) + 2.5),
        gridspec_kw={"width_ratios": [len(metrics), 1], "wspace": 0.06},
        sharey=True,
    )
    sns.heatmap(
        matrix[metrics], cmap="RdBu_r", center=0, linewidths=0.5, linecolor="white",
        annot=True, fmt=".2f", annot_kws={"fontsize": 8},
        cbar_kws={"label": "mean oriented z-score", "pad": 0.02}, ax=ax,
    )
    ax.set_title("Complexity profile per level (oriented z-scores; higher = more complex)")
    ax.set_xlabel("metric")
    ax.set_ylabel("description level")
    ax.tick_params(axis="x", rotation=45)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")

    sns.heatmap(
        matrix[["z_index"]], cmap="viridis", vmin=0.0, vmax=1.0,
        linewidths=0.5, linecolor="white", annot=True, fmt=".2f",
        annot_kws={"fontsize": 8},
        cbar_kws={"label": "z_index (0=simple, 1=complex)", "pad": 0.12}, ax=ax_idx,
    )
    ax_idx.set_xlabel("")
    ax_idx.set_ylabel("")
    ax_idx.tick_params(left=False)
    _save(fig, cfg, "levels_complexity_matrix")


def plot_zindex_matrix(
    df: pd.DataFrame, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG
) -> None:
    """Heatmap: per-project z_index (rows = datasets) at each level (columns)."""
    labels, ranks = _ordered_levels(df)
    pivot = df.pivot_table(index="sub_folder_name", columns="level_rank", values="z_index")
    pivot = pivot.reindex(columns=ranks)
    pivot.columns = labels
    # Sort projects by their real-spec (highest-rank) complexity for readability.
    order = pivot.sort_values(labels[-1]).index
    pivot = pivot.reindex(order)

    height = max(6.0, 0.28 * len(pivot))
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 3, height))
    sns.heatmap(
        pivot, cmap="viridis", vmin=0.0, vmax=1.0, linewidths=0.4, linecolor="white",
        annot=True, fmt=".2f", annot_kws={"fontsize": 7},
        cbar_kws={"label": "z_index (0=simple, 1=complex)"}, ax=ax,
    )
    ax.set_title("Per-project complexity (z_index) across the three levels")
    ax.set_xlabel("description level")
    ax.set_ylabel("dataset")
    _save(fig, cfg, "levels_zindex_matrix")


def generate_all(
    levels_cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG,
    text_cfg: TextConfig = DEFAULT_CONFIG,
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute (unless ``df`` given) and plot the complexity matrices."""
    if df is None:
        df = compute_level_metrics(levels_cfg, text_cfg)
    if df.empty:
        logger.warning("No level metrics computed; nothing to plot.")
        return df
    write_complexity_csv(df, levels_cfg)
    plot_complexity_matrix(df, text_cfg, levels_cfg)
    plot_zindex_matrix(df, levels_cfg)
    return df


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Plot the complexity matrix for the three levels.")
    parser.add_argument(
        "--plots-only", action="store_true",
        help="Skip scoring; plot from an existing levels_complexity.csv.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    cfg = DEFAULT_LEVELS_CONFIG
    df = None
    if args.plots_only:
        csv = cfg.output_dir / _COMPLEXITY_CSV
        if not csv.is_file():
            logger.error("No CSV at %s — run without --plots-only first.", csv)
            return 1
        df = pd.read_csv(csv)
    generate_all(cfg, DEFAULT_CONFIG, df=df)
    logger.info("Done. Outputs in %s", cfg.output_dir)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
