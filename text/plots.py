"""Comparative-results plots for the complexity metrics.

Produces two figure families:

1. Complexity profile  - a heatmap of z-scored metrics per dataset.
2. Complexity vs LLM F1 - per-metric scatter against the mean F1 across models
   (with Pearson/Spearman r and a trend line), plus a correlation-summary bar
   chart that tests the paper's thesis directly.

All figures are written under ``cfg.output_dir`` in every configured format.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless / reproducible rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .config import DEFAULT_CONFIG, TextConfig
from .metrics import metric_names

logger = logging.getLogger(__name__)

# F1 columns from evaluation_results_llm.csv to correlate against.
_F1_COLUMNS = ["f1_class", "f1_rel"]


def _save(fig, cfg: TextConfig, stem: str) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in cfg.figure_formats:
        path = cfg.output_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        logger.info("Saved %s", path)
    plt.close(fig)


def _present_metrics(metrics_df: pd.DataFrame) -> List[str]:
    return [m for m in metric_names() if m in metrics_df.columns]


def _z_scores(metrics_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (raw metric values, per-metric z-scores), indexed by dataset.

    Each metric column is z-scored across datasets so that heterogeneous
    scales become directly comparable.
    """
    metrics = _present_metrics(metrics_df)
    data = metrics_df.set_index("sub_folder_name")[metrics].astype(float)
    z = (data - data.mean()) / data.std(ddof=0)
    return data, z


def _raw_complexity(z: pd.DataFrame) -> pd.Series:
    """Mean oriented z-score per dataset (higher = more complex).

    Flesch Reading Ease is inverted (higher = simpler) so every oriented metric
    grows with linguistic complexity before averaging.
    """
    oriented = z.copy()
    if "flesch_reading_ease" in oriented.columns:
        oriented["flesch_reading_ease"] = -oriented["flesch_reading_ease"]
    return oriented.mean(axis=1)


def complexity_index(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset complexity index normalized to ``[0, 1]``.

    The mean oriented z-score (see :func:`_raw_complexity`) is the raw index;
    it is then min-max scaled across datasets so the result is interpretable on
    a fixed scale: ``0`` = simplest dataset in the set, ``1`` = most complex.
    When every dataset scores identically the index is undefined, so ``0.5`` is
    returned as a neutral fallback.

    Args:
        metrics_df: Loaded ``complexity_metrics.csv`` (one row per dataset).

    Returns:
        DataFrame with ``sub_folder_name``, the raw ``complexity_z`` index, and
        the normalized ``z_index`` in ``[0, 1]``, sorted simplest -> most complex.
    """
    _, z = _z_scores(metrics_df)
    raw = _raw_complexity(z)

    span = raw.max() - raw.min()
    if not np.isfinite(span) or span == 0:
        normalized = pd.Series(0.5, index=raw.index)
    else:
        normalized = (raw - raw.min()) / span

    out = pd.DataFrame(
        {
            "sub_folder_name": raw.index,
            "complexity_z": raw.to_numpy(),
            "z_index": normalized.to_numpy(),
        }
    )
    return out.sort_values("z_index").reset_index(drop=True)


def write_complexity_index(
    metrics_df: pd.DataFrame, cfg: TextConfig = DEFAULT_CONFIG
) -> pd.DataFrame:
    """Compute the normalized complexity index and persist it to ``cfg.index_csv``."""
    index_df = complexity_index(metrics_df)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(cfg.index_csv, index=False)
    logger.info("Wrote complexity index for %d datasets to %s", len(index_df), cfg.index_csv)
    return index_df


def load_metrics(cfg: TextConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Load the computed metrics CSV."""
    if not cfg.metrics_csv.is_file():
        raise FileNotFoundError(
            f"Metrics CSV not found: {cfg.metrics_csv}. Run the pipeline first."
        )
    return pd.read_csv(cfg.metrics_csv)


def aggregate_results(cfg: TextConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Mean F1 per dataset across all models, from the evaluation results CSV."""
    if not cfg.results_csv.is_file():
        logger.warning("Results CSV not found: %s; correlation plots skipped.", cfg.results_csv)
        return pd.DataFrame(columns=["sub_folder_name", *_F1_COLUMNS])
    results = pd.read_csv(cfg.results_csv)
    cols = [c for c in _F1_COLUMNS if c in results.columns]
    agg = (
        results.groupby("sub_folder_name")[cols]
        .mean()
        .reset_index()
        .rename(columns={c: f"mean_{c}" for c in cols})
    )
    return agg


def plot_complexity_profile(metrics_df: pd.DataFrame, cfg: TextConfig = DEFAULT_CONFIG) -> None:
    """Heatmap of z-scored metrics per dataset (the complexity profile).

    The per-metric z-scores are drawn on a diverging, centre-0 scale; a narrow
    companion column on the right shows the normalized ``z_index`` (0 = simplest,
    1 = most complex) on its own sequential 0..1 scale.
    """
    metrics = _present_metrics(metrics_df)
    data, z = _z_scores(metrics_df)
    # Overall complexity = mean oriented z-score across metrics.
    complexity = _raw_complexity(z)
    # Sort datasets by increasing overall complexity (top = simplest).
    order = complexity.sort_values().index
    z = z.reindex(order)
    # Raw metric values, in the same order, used as cell annotations.
    annot = data.reindex(order)

    # Normalized [0, 1] index per dataset, in the same order, for the side column.
    index_df = complexity_index(metrics_df).set_index("sub_folder_name")
    z_index = index_df["z_index"].reindex(order).to_frame("z_index")

    height = max(6.0, 0.28 * len(z))
    fig, (ax, ax_idx) = plt.subplots(
        1,
        2,
        figsize=(1.2 * len(metrics) + 5, height),
        gridspec_kw={"width_ratios": [len(metrics), 1], "wspace": 0.05},
        sharey=True,
    )
    sns.heatmap(
        z,
        cmap="RdBu_r",
        center=0,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "z-score", "pad": 0.02},
        annot=annot,
        fmt=".2f",
        annot_kws={"fontsize": 7},
        ax=ax,
    )
    ax.set_title("Linguistic-complexity profile per dataset (z-scored)")
    ax.set_xlabel("metric")
    ax.set_ylabel("dataset")

    # Companion z_index column on its own 0..1 sequential scale.
    sns.heatmap(
        z_index,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "z_index (0=simple, 1=complex)", "pad": 0.1},
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7},
        ax=ax_idx,
    )
    ax_idx.set_xlabel("")
    ax_idx.set_ylabel("")
    ax_idx.tick_params(left=False)
    _save(fig, cfg, "complexity_profile_heatmap")


def plot_z_index_distribution(metrics_df: pd.DataFrame, cfg: TextConfig = DEFAULT_CONFIG) -> None:
    """Scatter of the normalized ``z_index`` across datasets (cases).

    Each dataset is a point at its ``z_index`` (x), ordered simplest -> most
    complex (y), coloured by the same value. Mean and median reference lines
    summarise where the bulk of the cases sit on the 0..1 scale.
    """
    index_df = complexity_index(metrics_df)  # already sorted simplest -> complex
    if index_df.empty:
        logger.warning("No data for z_index distribution; plot skipped.")
        return

    values = index_df["z_index"].to_numpy()
    names = index_df["sub_folder_name"].tolist()
    y = np.arange(len(values))

    height = max(6.0, 0.28 * len(values))
    fig, ax = plt.subplots(figsize=(8, height))
    scatter = ax.scatter(
        values, y, c=values, cmap="viridis", vmin=0.0, vmax=1.0,
        s=70, edgecolor="white", linewidth=0.6, zorder=3,
    )
    # Thin connector from the simple end to each point, for readability.
    ax.hlines(y, 0, values, color="#cccccc", linewidth=0.8, zorder=1)

    mean_v, median_v = float(values.mean()), float(np.median(values))
    ax.axvline(mean_v, color="#c0392b", linestyle="--", linewidth=1.2,
               label=f"mean = {mean_v:.2f}", zorder=2)
    ax.axvline(median_v, color="#2c3e50", linestyle=":", linewidth=1.2,
               label=f"median = {median_v:.2f}", zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("z_index (0 = simplest, 1 = most complex)")
    ax.set_ylabel("dataset")
    ax.set_title("z_index distribution across datasets")
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()  # simplest at top, matching the heatmap order
    fig.colorbar(scatter, ax=ax, label="z_index", pad=0.02)
    _save(fig, cfg, "z_index_distribution")


def _corr(x: pd.Series, y: pd.Series) -> Tuple[float, float, float, float]:
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return pr, pp, sr, sp


def plot_metric_vs_f1(
    merged: pd.DataFrame, f1_col: str, cfg: TextConfig = DEFAULT_CONFIG
) -> pd.DataFrame:
    """Grid of scatter plots: each metric vs ``f1_col``. Returns a correlation table."""
    metrics = _present_metrics(merged)
    n = len(metrics)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    records = []
    for ax, metric in zip(axes, metrics):
        x = merged[metric].astype(float)
        y = merged[f1_col].astype(float)
        pr, pp, sr, sp = _corr(x, y)
        records.append(
            {"metric": metric, "target": f1_col, "pearson_r": pr, "pearson_p": pp,
             "spearman_r": sr, "spearman_p": sp}
        )

        ax.scatter(x, y, alpha=0.7, edgecolor="white", linewidth=0.5, color="#3b6fb0")
        # Trend line when a correlation could be computed.
        mask = x.notna() & y.notna()
        if mask.sum() >= 2 and x[mask].nunique() >= 2:
            coeffs = np.polyfit(x[mask], y[mask], 1)
            xs = np.linspace(x[mask].min(), x[mask].max(), 50)
            ax.plot(xs, np.polyval(coeffs, xs), color="#c0392b", linewidth=1.5)
        ax.set_title(f"{metric}\nPearson r={pr:.2f} (p={pp:.2g})", fontsize=9)
        ax.set_xlabel(metric)
        ax.set_ylabel(f1_col)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Complexity metrics vs {f1_col} (mean across models)", y=1.0, fontsize=13)
    fig.tight_layout()
    _save(fig, cfg, f"scatter_metrics_vs_{f1_col}")
    return pd.DataFrame(records)


def plot_correlation_summary(corr_df: pd.DataFrame, cfg: TextConfig = DEFAULT_CONFIG) -> None:
    """Grouped bar chart of Pearson r for every metric and F1 target."""
    if corr_df.empty:
        return
    pivot = corr_df.pivot(index="metric", columns="target", values="pearson_r")
    fig, ax = plt.subplots(figsize=(1.1 * len(pivot) + 4, 6))
    pivot.plot(kind="bar", ax=ax, color=["#3b6fb0", "#e08a3c"], edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Correlation of linguistic complexity with LLM F1")
    ax.set_ylabel("Pearson r")
    ax.set_xlabel("metric")
    ax.legend(title="target")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    _save(fig, cfg, "correlation_summary")


def generate_all_plots(cfg: TextConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Build the metrics-per-dataset figure only (no correlation plots).

    Returns an empty DataFrame for backward compatibility with callers that
    expect the (now unused) correlation table.
    """
    metrics_df = load_metrics(cfg)
    plot_complexity_profile(metrics_df, cfg)
    plot_z_index_distribution(metrics_df, cfg)
    write_complexity_index(metrics_df, cfg)
    return pd.DataFrame()
