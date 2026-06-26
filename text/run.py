"""CLI entrypoint for the text-complexity metrics package.

Examples:
    # From the repository root:
    python -m text.run                # compute metrics + generate all plots
    python -m text.run --no-plots     # metrics CSV only
    python -m text.run --self-check   # run metrics on the paper's example sentences
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import DEFAULT_CONFIG
from .metrics import compute_all, metric_names

logger = logging.getLogger("text")

# Sentences taken from the paper, used as a deterministic self-check.
_SELF_CHECK_SAMPLES = {
    "elaboration": "Luca bought a new car. It is a red Panda.",
    "contrast": "I love the sea, but Sara prefers the mountains.",
    "concession": "I go out, even though it is raining.",
}


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )


def _run_self_check() -> int:
    """Compute all metrics on the paper's examples and assert finite values."""
    import math

    ok = True
    for sample_id, text in _SELF_CHECK_SAMPLES.items():
        result = compute_all(text, sample_id=sample_id)
        print(f"\n[{sample_id}] {text}")
        for name in metric_names():
            value = result.values[name]
            flag = "" if math.isfinite(value) else "  <-- non-finite!"
            if not math.isfinite(value):
                ok = False
            print(f"  {name:28s} = {value:.4f}{flag}")
    print("\nSelf-check:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compute linguistic-complexity metrics for the Text2UML dataset.")
    parser.add_argument("--no-plots", action="store_true", help="Compute the metrics CSV but skip plotting.")
    parser.add_argument("--plots-only", action="store_true", help="Skip computation; plot from an existing metrics CSV.")
    parser.add_argument("--self-check", action="store_true", help="Run metrics on the paper's example sentences and exit.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    if args.self_check:
        return _run_self_check()

    # Imported here so --self-check does not require pandas/matplotlib.
    from .pipeline import run_pipeline
    from .plots import generate_all_plots

    if not args.plots_only:
        df = run_pipeline(DEFAULT_CONFIG)
        logger.info("Computed metrics for %d datasets.", len(df))

    if not args.no_plots:
        corr = generate_all_plots(DEFAULT_CONFIG)
        if not corr.empty:
            top = corr.reindex(corr["pearson_r"].abs().sort_values(ascending=False).index).head(5)
            logger.info("Strongest complexity/F1 correlations:\n%s", top.to_string(index=False))

    logger.info("Done. Outputs in %s", DEFAULT_CONFIG.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
