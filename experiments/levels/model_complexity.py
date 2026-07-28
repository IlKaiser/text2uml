"""Gold UML model size, used to weight F1 by how structurally demanding a
case's model actually is (number of classes, attributes, and associations),
as distinct from the *description text's* linguistic complexity
(``experiments.levels.complexity``).

A case with a small gold model (few classes/attributes/associations) scoring
F1=0.9 is a much easier win than a case with a large, richly-connected model
scoring the same F1 -- an unweighted corpus-wide mean treats them identically.
``weighted_f1`` computes a complexity-weighted mean per level alongside the
plain mean, so bigger/harder models count proportionally more.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

from .config import DEFAULT_LEVELS_CONFIG, LevelsConfig
from .evaluate import load_evaluator

logger = logging.getLogger(__name__)

_MODEL_COMPLEXITY_CSV = "gold_model_complexity.csv"

# Default complexity weight: classes + attributes + associations, matching
# exactly the three dimensions requested -- inheritance edges are reported
# separately (n_inheritance) but excluded from the default weight.
DEFAULT_WEIGHT_COL = "total_size"


def compute_gold_complexity(cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> pd.DataFrame:
    """Class/attribute/association counts for every case's gold plantuml.txt.

    Reuses ``src/eval.py``'s grammar-based parser (``init_parser`` /
    ``get_class_and_assoc_size``) so counts stay consistent with how F1
    itself is computed -- no separate ad-hoc parsing.
    """
    ev = load_evaluator(cfg)
    parser = ev.init_parser(str(cfg.grammar_path))

    rows: List[dict] = []
    for dataset in sorted(p for p in cfg.dataset_dir.iterdir() if p.is_dir()):
        gold = dataset / cfg.gold_filename
        if not gold.is_file():
            continue
        try:
            n_classes, n_rel, n_attrs, n_inh = ev.get_class_and_assoc_size(str(gold), parser)
        except Exception as exc:  # noqa: BLE001 - one bad gold file must not abort the batch
            logger.warning("Could not parse gold for %s: %s", dataset.name, exc)
            continue
        rows.append({
            "sub_folder_name": dataset.name,
            "n_classes": n_classes,
            "n_attributes": n_attrs,
            "n_associations": n_rel,
            "n_inheritance": n_inh,
            DEFAULT_WEIGHT_COL: n_classes + n_attrs + n_rel,
        })
    return pd.DataFrame(rows)


def write_gold_complexity_csv(df: pd.DataFrame, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.output_dir / _MODEL_COMPLEXITY_CSV
    df.to_csv(out, index=False)
    logger.info("Wrote %d rows to %s", len(df), out)
    return out


def weighted_f1(
    f1_df: pd.DataFrame,
    complexity_df: pd.DataFrame,
    weight_col: str = DEFAULT_WEIGHT_COL,
) -> pd.DataFrame:
    """Per-level plain mean F1 vs. complexity-weighted mean F1.

    weighted_mean = sum(f1_i * weight_i) / sum(weight_i), grouped by level.
    Raises ValueError if there are no overlapping cases (e.g. mismatched
    ``sub_folder_name`` values between the two frames).
    """
    merged = f1_df.merge(
        complexity_df[["sub_folder_name", weight_col]], on="sub_folder_name", how="inner"
    )
    if merged.empty:
        raise ValueError("No overlapping cases between f1_df and complexity_df.")

    rows: List[dict] = []
    for (level, rank), group in merged.groupby(["level", "level_rank"]):
        w = group[weight_col]
        f1 = group["f1_global"]
        total_w = w.sum()
        rows.append({
            "level": level,
            "level_rank": rank,
            "f1_mean": f1.mean(),
            "f1_weighted_mean": (f1 * w).sum() / total_w if total_w else float("nan"),
            "n_cases": len(group),
            "total_weight": total_w,
        })
    return pd.DataFrame(rows).sort_values("level_rank").reset_index(drop=True)
