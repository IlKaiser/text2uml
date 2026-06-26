"""Pipeline: compute complexity metrics for every dataset description.

Walks ``dataset/<Name>/description.md``, runs the registered metric suite on each,
and writes one row per dataset to ``text/output/complexity_metrics.csv``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

from .config import DEFAULT_CONFIG, TextConfig
from .metrics import compute_all, metric_names

logger = logging.getLogger(__name__)


def _iter_description_paths(cfg: TextConfig) -> List[Path]:
    """Return sorted ``description.md`` paths under the dataset directory."""
    if not cfg.dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {cfg.dataset_dir}")
    paths = sorted(cfg.dataset_dir.glob(f"*/{cfg.description_filename}"))
    if not paths:
        logger.warning("No '%s' files found under %s", cfg.description_filename, cfg.dataset_dir)
    return paths


def compute_dataset_metrics(cfg: TextConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Compute metrics for all datasets and return them as a DataFrame.

    Args:
        cfg: Pipeline configuration.

    Returns:
        DataFrame with columns ``sub_folder_name``, ``n_tokens``, ``n_sentences``,
        every metric name, and ``error``. One row per dataset folder.
    """
    rows = []
    for path in _iter_description_paths(cfg):
        dataset_name = path.parent.name
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("Could not read %s: %s", path, exc)
            continue

        result = compute_all(text, sample_id=dataset_name, model_name=cfg.spacy_model)
        row = {
            "sub_folder_name": result.sample_id,
            "n_tokens": result.n_tokens,
            "n_sentences": result.n_sentences,
            **result.values,
            "error": result.error or "",
        }
        rows.append(row)
        logger.info("Computed metrics for %s", dataset_name)

    if not rows:
        return pd.DataFrame(
            columns=["sub_folder_name", "n_tokens", "n_sentences", *metric_names(), "error"]
        )

    df = pd.DataFrame(rows).sort_values("sub_folder_name").reset_index(drop=True)
    return df


def run_pipeline(cfg: TextConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Compute metrics and persist them to ``cfg.metrics_csv``."""
    df = compute_dataset_metrics(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.metrics_csv, index=False)
    logger.info("Wrote %d rows to %s", len(df), cfg.metrics_csv)
    return df
