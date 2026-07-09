"""CLI: compare two-shot UML generation across three complexity levels.

Pipeline (each stage can be run independently):
  generate  -- run the two-shot chain on description_level_one / _two / real spec
  evaluate  -- score every generated result against the gold PlantUML
  plot      -- draw global + per-category F1 figures

Examples:
    # End-to-end for one model:
    python -m experiments.levels.run --provider anthropic --model claude-sonnet-4-6

    # Evaluate + plot from already-generated results (no LLM calls):
    python -m experiments.levels.run --stage evaluate plot --model claude-sonnet-4-6

    # Plot only, from an existing levels_f1.csv:
    python -m experiments.levels.run --stage plot

    # Smoke test on a few projects, one level:
    python -m experiments.levels.run --provider anthropic --model claude-sonnet-4-6 \
        --datasets AirTravel Cruise --levels three
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from .config import DEFAULT_LEVELS_CONFIG, LevelsConfig, TECHNIQUE_RESULT_PREFIXES, safe_subdir_model

logger = logging.getLogger("experiments.levels")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )


def _resolve_plot_scope(df: pd.DataFrame, model: Optional[str], f1_csv: Path) -> Tuple[pd.DataFrame, str, str]:
    """Decide the plot stage's (dataframe, output subdir, title suffix).

    Once a second model's rows land in the same F1 CSV (see
    ``evaluate.write_f1_csv``), blending them into one "corpus" plot mixes
    two models' scores into a meaningless average. If ``model`` narrows to
    one of several present, scope the plots to it and save under a
    model-specific subdir instead of overwriting the shared default.

    The "one of several present" check looks at every model in the
    persisted CSV, not just ``df`` -- when generate+evaluate+plot run
    together for one model, ``df`` (from ``evaluate_all``) only ever holds
    that model's freshly-scored rows, so checking ``df`` alone would never
    detect other models already on disk and would silently overwrite the
    shared ``corpus/`` plots with single-model data.
    """
    all_models = set(df["model"].unique()) if "model" in df.columns else set()
    if f1_csv.is_file():
        all_models |= set(pd.read_csv(f1_csv, usecols=["model"])["model"].unique())

    if model and all_models and all_models != {model}:
        plot_df = df[df["model"] == model] if "model" in df.columns else df
        subdir = f"corpus_{safe_subdir_model(model)}"
        title_suffix = f" — {model}"
        return plot_df, subdir, title_suffix
    return df, "corpus", ""


def _load_provider_cfg(provider: str, cfg: LevelsConfig) -> dict:
    """Pull the provider sub-config from src/config.yaml (for _make_llm kwargs)."""
    try:
        import yaml

        with open(cfg.run_config) as f:
            full = yaml.safe_load(f) or {}
        return (full.get("providers", {}) or {}).get(provider, {}) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read provider config (%s); using empty dict.", exc)
        return {}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stage", nargs="+", default=["generate", "evaluate", "plot"],
        choices=["generate", "evaluate", "plot", "complexity", "correlate"],
        help="Which stages to run (default: generate, evaluate, plot). "
        "'complexity' scores the three level descriptions and plots the "
        "complexity matrix; 'correlate' joins F1 with complexity per case and "
        "plots score-vs-complexity (both independent of generation).",
    )
    parser.add_argument("--provider", help="Provider key (required for generate).")
    parser.add_argument("--model", help="Model id (required for generate/evaluate).")
    parser.add_argument(
        "--technique", choices=sorted(TECHNIQUE_RESULT_PREFIXES), default="few_shot",
        help="Generation technique from src.run._CHAIN_BUILDERS (default: few_shot, "
        "the original two-shot pipeline). Non-default techniques get their own "
        "output_dir subfolder (experiments/levels/output/<technique>/) so their "
        "CSVs/plots never mix with or overwrite the few_shot corpus results.",
    )
    parser.add_argument("--datasets", nargs="*", help="Only these dataset folder names.")
    parser.add_argument(
        "--levels", nargs="*", choices=["zero", "one", "two", "three", "four", "zeroalt", "zeroalt2"],
        help="Only these complexity levels (default: all).",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate existing result files.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    cfg = DEFAULT_LEVELS_CONFIG
    if args.technique != cfg.technique:
        cfg = _dc_replace(
            cfg,
            technique=args.technique,
            result_prefix=TECHNIQUE_RESULT_PREFIXES[args.technique],
            output_dir=DEFAULT_LEVELS_CONFIG.output_dir / args.technique,
        )
    stages = set(args.stage)

    if "complexity" in stages:
        from .complexity import generate_all as complexity_all

        complexity_all(cfg)

    if "generate" in stages:
        if not (args.provider and args.model):
            logger.error("--provider and --model are required for the generate stage.")
            return 2
        from .generate import generate

        pcfg = _load_provider_cfg(args.provider, cfg)
        generate(
            args.provider, args.model, pcfg,
            only=args.datasets, levels=args.levels, force=args.force, cfg=cfg,
        )

    df = pd.DataFrame()
    if "evaluate" in stages:
        if not args.model:
            logger.error("--model is required for the evaluate stage.")
            return 2
        from .evaluate import evaluate_all, write_f1_csv

        df = evaluate_all(args.model, only=args.datasets, levels=args.levels, cfg=cfg)
        if df.empty:
            logger.error("No results scored — generate first, or check --datasets/--levels.")
            return 1
        write_f1_csv(df, cfg)
        # Console summary: mean F1 per level.
        summary = df.groupby(["level_rank", "level_label"]).agg(
            n=("sub_folder_name", "count"),
            f1_global=("f1_global", "mean"),
            f1_class=("f1_class", "mean"),
            f1_association=("f1_association", "mean"),
            f1_attribute=("f1_attribute", "mean"),
            f1_cardinality=("f1_cardinality", "mean"),
        ).reset_index().drop(columns="level_rank")
        logger.info("Mean F1 per level:\n%s", summary.to_string(index=False))

    if "plot" in stages:
        from .plots import generate_all_plots

        if df.empty:
            if not cfg.f1_csv.is_file():
                logger.error("No F1 CSV at %s — run the evaluate stage first.", cfg.f1_csv)
                return 1
            df = pd.read_csv(cfg.f1_csv)

        plot_df, subdir, title_suffix = _resolve_plot_scope(df, args.model, cfg.f1_csv)
        if args.model and plot_df.empty:
            logger.error("No rows for model %s in %s", args.model, cfg.f1_csv)
            return 1
        generate_all_plots(plot_df, cfg, subdir=subdir, title_suffix=title_suffix)

    if "correlate" in stages:
        from .correlation import generate_all as correlate_all

        try:
            correlate_all(cfg, model=args.model)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("%s", exc)
            return 1

    logger.info("Done. Outputs in %s", cfg.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
