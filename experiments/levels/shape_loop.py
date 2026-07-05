"""Corpus-level closed feedback loop for the four description complexity levels.

Runs the per-case shape-aware rewrite (``text.rewrite.run.process_dataset``)
for every dataset, recomputes the corpus metrics, and re-checks each case's
four-level shape independently from the freshly computed CSV (not the
per-text loop's self-report). Any case that still fails goes back through
``process_dataset`` for just its failing level(s), up to ``max_retries``
extra corpus-level passes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from text.rewrite.config import DEFAULT_REWRITE_CONFIG, RewriteConfig
from text.rewrite.run import process_dataset
from text.rewrite.scorer import build_reference
from text.rewrite.shape_targets import check_case_shape, shape_ok

from .case_metrics import plot_case
from .complexity import compute_level_metrics, generate_all, write_complexity_csv
from .config import DEFAULT_LEVELS_CONFIG, LevelsConfig

logger = logging.getLogger(__name__)

_METRIC_COLS = ("mdd", "subordination_index", "context_dependence_proxy")
_COMPLIANCE_CSV = "shape_compliance_report.csv"


_ALL_LEVELS = frozenset({"zero", "one", "two", "three"})
_REGENERATABLE_LEVELS = frozenset({"zero", "one", "two"})
# _global_max_check's own fingerprint (text/rewrite/shape_targets.py): every
# real per-level band in RATIO_BANDS is a distinct, non-(0.0, None) tuple, so
# this is a reliable way to tell a cross-level check apart from a local
# check_shape() check that happens to report the same (metric, level) pair
# (e.g. context_dependence_proxy's cross-level check is reported at
# level="zero", exactly where its own local check also lives).
_GLOBAL_CHECK_BAND = (0.0, None)


def find_noncompliant_cases(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Case name -> failing non-three level tags, per ``check_case_shape``.

    Per-level checks (``check_shape``) name the exact offending level
    ("zero"/"one"/"two"). The two cross-level max checks in
    ``check_case_shape`` name the level that's *supposed* to be the max
    ("three" for subordination_index, "zero" for context_dependence_proxy) —
    that tells us which constraint broke, not which of zero/one/two is the
    culprit that needs to change, and (for context_dependence_proxy) its
    reported level collides with a local check at the same level. Cross-level
    checks are identified by their ``_GLOBAL_CHECK_BAND`` fingerprint and
    excluded from ``bad_levels`` so they can't be mistaken for a local
    failure. When only a cross-level check fails (no specific zero/one/two
    level also failed its own local check), the safe fix is to regenerate all
    three non-real levels for that case, since we can't cheaply tell which
    one is currently over/under the real spec.

    A case missing an expected level's row entirely (its metrics failed to
    compute -- ``compute_level_metrics`` drops such rows rather than leaving
    them as failing) is treated as non-compliant too, since a case that was
    never actually checked must not be silently reported as passing.
    """
    failing: Dict[str, List[str]] = {}
    for case, sub in df.groupby("sub_folder_name"):
        levels = {
            row["level"]: {m: row[m] for m in _METRIC_COLS if m in row}
            for _, row in sub.iterrows()
        }
        missing = _ALL_LEVELS - set(levels)
        if missing:
            failing[case] = sorted(missing & _REGENERATABLE_LEVELS) or sorted(_REGENERATABLE_LEVELS)
            continue

        checks = check_case_shape(levels)
        if shape_ok(checks):
            continue
        bad_levels = sorted({
            c.level for c in checks
            if not (c.degenerate or (c.rank_ok and c.band_ok))
            and c.level in _REGENERATABLE_LEVELS
            and c.band != _GLOBAL_CHECK_BAND
        })
        failing[case] = bad_levels or sorted(_REGENERATABLE_LEVELS)
    return failing


def _iter_dataset_paths(levels_cfg: LevelsConfig, only: Optional[List[str]] = None) -> List[Path]:
    paths = sorted(p / "description.md" for p in levels_cfg.dataset_dir.iterdir() if p.is_dir() and (p / "description.md").is_file())
    if only:
        wanted = set(only)
        paths = [p for p in paths if p.parent.name in wanted]
    return paths


def run_shape_loop(
    levels_cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG,
    rewrite_cfg: RewriteConfig = DEFAULT_REWRITE_CONFIG,
    max_retries: int = 2,
    only: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Full closed loop: generate -> recheck -> retry outliers -> report.

    ``only`` restricts the run to these dataset folder names (else every
    case under ``levels_cfg.dataset_dir``) -- e.g. for a smoke-test run
    against a scratch ``levels_cfg`` (a different ``output_dir``) without
    touching the corpus-wide ``levels_complexity.csv``/compliance report.
    """
    from text.config import DEFAULT_CONFIG
    from text.rewrite.client import make_client

    reference = build_reference(DEFAULT_CONFIG)
    client = make_client()

    for desc_path in _iter_dataset_paths(levels_cfg, only):
        case = desc_path.parent.name
        try:
            process_dataset(case, desc_path, rewrite_cfg, reference, DEFAULT_CONFIG, client)
        except Exception as exc:  # noqa: BLE001 - one bad case must not abort the whole corpus run
            logger.error("process_dataset failed for %s (initial pass): %s", case, exc)

    df = compute_level_metrics(levels_cfg, only=only)
    write_complexity_csv(df, levels_cfg)

    for attempt in range(1, max_retries + 1):
        failing = find_noncompliant_cases(df)
        if not failing:
            logger.info("Shape loop converged after %d retry pass(es).", attempt - 1)
            break
        logger.info("Retry pass %d: %d non-compliant case(s): %s", attempt, len(failing), sorted(failing))
        for case, bad_levels in failing.items():
            desc_path = levels_cfg.dataset_dir / case / "description.md"
            try:
                process_dataset(case, desc_path, rewrite_cfg, reference, DEFAULT_CONFIG, client, levels=tuple(bad_levels), force=True)
            except Exception as exc:  # noqa: BLE001 - one bad case must not abort the whole retry pass
                logger.error("process_dataset failed for %s (retry pass %d, levels=%s): %s", case, attempt, bad_levels, exc)
        df = compute_level_metrics(levels_cfg, only=only)
        write_complexity_csv(df, levels_cfg)

    report_rows = []
    for case, sub in df.groupby("sub_folder_name"):
        levels = {row["level"]: {m: row[m] for m in _METRIC_COLS if m in row} for _, row in sub.iterrows()}
        for check in check_case_shape(levels):
            report_rows.append({
                "sub_folder_name": case, "metric": check.metric, "level": check.level,
                "value": check.value, "l3_value": check.l3_value, "ratio": check.ratio,
                "rank_ok": check.rank_ok, "band_ok": check.band_ok, "degenerate": check.degenerate,
            })
    report = pd.DataFrame(report_rows)
    levels_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(levels_cfg.output_dir / _COMPLIANCE_CSV, index=False)
    logger.info("Wrote shape compliance report to %s", levels_cfg.output_dir / _COMPLIANCE_CSV)

    generate_all(levels_cfg, DEFAULT_CONFIG, df=df)
    for case in df["sub_folder_name"].unique():
        plot_case(case, levels_cfg)

    return report


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the corpus-level shape closed loop.")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    report = run_shape_loop(max_retries=args.max_retries)
    # Mirror shape_ok's real semantics exactly: a check fails only when it is
    # NOT degenerate and it fails its rank or its band (a degenerate check —
    # L3 reference value of exactly 0 for that metric — is fully exempted).
    still_failing = ~report["degenerate"] & (~report["rank_ok"] | ~report["band_ok"])
    n_fail = int(still_failing.sum())
    logger.info("Done. %d shape-check row(s) still failing after all retries.", n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
