"""End-to-end single-case pipeline across every configured complexity level.

Runs text-metric computation, generation, scoring, and plotting for one
dataset case at a time, merging into the shared ``levels_complexity.csv`` /
``levels_f1.csv`` (only replacing that case's rows) instead of recomputing the
whole corpus. Safe to re-run repeatedly while iterating on a single dataset.

Usage:
    # Full pipeline for one case:
    python -m experiments.levels.case_pipeline --case Menso \
        --provider anthropic --model claude-sonnet-4-6

    # Only refresh scores + plots after editing an existing result file:
    python -m experiments.levels.case_pipeline --case Menso \
        --model claude-sonnet-4-6 --stage evaluate plots correlate
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd

from .complexity import compute_level_metrics
from .config import DEFAULT_LEVELS_CONFIG, LevelsConfig
from .correlation import case_level_score_correlation
from .case_metrics import plot_case
from .evaluate import evaluate_all
from .plots import plot_case_bars

logger = logging.getLogger("experiments.levels.case_pipeline")


def _merge_case_rows(
    csv_path: Path, case: str, new_rows: pd.DataFrame, sort_cols: List[str]
) -> pd.DataFrame:
    """Replace ``case``'s rows in ``csv_path`` with ``new_rows``; leave all other rows untouched."""
    if csv_path.is_file():
        existing = pd.read_csv(csv_path)
        existing = existing[existing["sub_folder_name"] != case]
        merged = pd.concat([existing, new_rows], ignore_index=True)
    else:
        merged = new_rows.copy()
    merged = merged.sort_values(sort_cols).reset_index(drop=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(csv_path, index=False)
    return merged


def run_complexity(case: str, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> pd.DataFrame:
    """Recompute text-complexity metrics for every configured level of ``case``."""
    df = compute_level_metrics(cfg, only=[case])
    if df.empty:
        raise ValueError(f"No complexity rows computed for case {case!r} (missing description files?).")
    merged = _merge_case_rows(cfg.output_dir / "levels_complexity.csv", case, df,
                               ["sub_folder_name", "level_rank"])
    logger.info("complexity: %d row(s) for %s (%d total in file)", len(df), case, len(merged))
    return df


def run_generate(
    case: str, provider: str, model: str, provider_cfg: dict,
    force: bool = False, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG,
) -> int:
    """Generate the two-shot UML result for every level of ``case`` missing a result file."""
    from .generate import generate

    written = generate(provider, model, provider_cfg, only=[case], force=force, cfg=cfg)
    logger.info("generate: wrote %d result file(s) for %s", written, case)
    return written


def run_evaluate(case: str, model: str, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> pd.DataFrame:
    """Score every generated level result of ``case`` against its gold PlantUML."""
    df = evaluate_all(model, only=[case], cfg=cfg)
    if df.empty:
        raise ValueError(f"No results scored for case {case!r} — run the generate stage first.")
    merged = _merge_case_rows(cfg.f1_csv, case, df, ["sub_folder_name", "level_rank"])
    logger.info("evaluate: %d row(s) for %s (%d total in file)", len(df), case, len(merged))
    return df


def run_plots(case: str, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> None:
    """Redraw the per-case text-metric trajectory and F1-by-category bar chart."""
    plot_case(case, cfg)
    f1_df = pd.read_csv(cfg.f1_csv)
    plot_case_bars(case, f1_df, cfg)


def run_correlate(case: str, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> pd.DataFrame:
    """Correlate level_rank against F1 (global + categories) for ``case`` alone."""
    corr = case_level_score_correlation(case, cfg)
    logger.info("correlate:\n%s", corr.to_string(index=False))
    return corr


def _load_provider_cfg(provider: str, cfg: LevelsConfig) -> dict:
    """Pull the provider sub-config from src/config.yaml (for _make_llm kwargs)."""
    import yaml

    try:
        with open(cfg.run_config) as f:
            full = yaml.safe_load(f) or {}
        return (full.get("providers", {}) or {}).get(provider, {}) or {}
    except OSError as exc:
        logger.warning("Could not read provider config (%s); using empty dict.", exc)
        return {}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True, help="Dataset folder name (e.g. Menso).")
    parser.add_argument("--provider", default="anthropic", help="Provider key (required for generate).")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Model id (required for generate/evaluate).")
    parser.add_argument(
        "--stage", nargs="+", default=["complexity", "generate", "evaluate", "plots", "correlate"],
        choices=["complexity", "generate", "evaluate", "plots", "correlate"],
        help="Which stages to run (default: all five, in order).",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate existing result files.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    cfg = DEFAULT_LEVELS_CONFIG
    stages = set(args.stage)

    if "complexity" in stages:
        run_complexity(args.case, cfg)
    if "generate" in stages:
        provider_cfg = _load_provider_cfg(args.provider, cfg)
        run_generate(args.case, args.provider, args.model, provider_cfg, args.force, cfg)
    if "evaluate" in stages:
        run_evaluate(args.case, args.model, cfg)
    if "plots" in stages:
        run_plots(args.case, cfg)
    if "correlate" in stages:
        run_correlate(args.case, cfg)

    logger.info("Done. Outputs in %s", cfg.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
