"""CLI: rewrite every dataset's description toward two complexity levels.

For each ``dataset/<Name>/description.md`` this:
  1. scores the original on the corpus ``z_index`` scale,
  2. drives it toward ``level_one`` (target z_index 0.0 — as clear as possible),
  3. drives it toward ``level_two`` (midpoint between actual and simplest),
  4. writes ``description_level_one.md`` / ``description_level_two.md`` beside the
     original, and appends a row to ``text/output/rewrite_summary.csv``.

Originals are never overwritten.

Examples:
    python -m text.rewrite.run                    # all datasets, both levels
    python -m text.rewrite.run --dry-run          # score originals + targets only
    python -m text.rewrite.run --datasets Sober AirTravel
    python -m text.rewrite.run --limit 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd

from ..config import DEFAULT_CONFIG, TextConfig
from .config import DEFAULT_REWRITE_CONFIG, RewriteConfig
from .scorer import build_reference, score_text

logger = logging.getLogger("text.rewrite")


def _iter_description_paths(cfg: TextConfig, only: List[str] | None) -> List[Path]:
    paths = sorted(cfg.dataset_dir.glob(f"*/{cfg.description_filename}"))
    if only:
        wanted = set(only)
        paths = [p for p in paths if p.parent.name in wanted]
    return paths


def _write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", help="Only these dataset folder names.")
    parser.add_argument("--limit", type=int, help="Process at most N datasets.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Score originals and print targets; make no API calls and write nothing.",
    )
    parser.add_argument("--model", help="Override the Claude model id.")
    parser.add_argument("--effort", choices=["low", "medium", "high", "max"])
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument(
        "--verify-meaning", action="store_true",
        help="Audit semantic equivalence with a second Claude call before "
        "accepting a complexity-matching rewrite (extra cost).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    tconf = DEFAULT_CONFIG
    cfg = RewriteConfig(
        model=args.model or DEFAULT_REWRITE_CONFIG.model,
        effort=args.effort or DEFAULT_REWRITE_CONFIG.effort,
        max_iterations=args.max_iterations or DEFAULT_REWRITE_CONFIG.max_iterations,
        tolerance=args.tolerance if args.tolerance is not None else DEFAULT_REWRITE_CONFIG.tolerance,
        verify_meaning=args.verify_meaning,
    )

    try:
        reference = build_reference(tconf)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    paths = _iter_description_paths(tconf, args.datasets)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        logger.error("No description files found under %s", tconf.dataset_dir)
        return 1

    client = None
    if not args.dry_run:
        from .client import make_client

        client = make_client()
        from .loop import rewrite_to_target

    rows = []
    for path in paths:
        name = path.parent.name
        original = path.read_text(encoding="utf-8")
        try:
            base = score_text(original, reference, tconf)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", name, exc)
            continue

        t1 = cfg.level_one_target
        t2 = base.z_index * cfg.level_two_factor
        logger.info(
            "%s: actual z_index=%.2f -> level_one target=%.2f, level_two target=%.2f",
            name, base.z_index, t1, t2,
        )

        row = {
            "sub_folder_name": name,
            "actual_z": round(base.z_index, 4),
            "level_one_target": round(t1, 4),
            "level_two_target": round(t2, 4),
        }

        if not args.dry_run:
            for level_name, target in (
                (cfg.level_one_name, t1),
                (cfg.level_two_name, t2),
            ):
                res = rewrite_to_target(
                    client, cfg, original, base, reference, level_name, target
                )
                _write(cfg.output_path(path.parent, level_name), res.text)
                row[f"{level_name}_z"] = round(res.final_z, 4)
                row[f"{level_name}_reached"] = res.reached
                row[f"{level_name}_iters"] = res.iterations
                logger.info(
                    "%s/%s: z=%.2f (target %.2f) reached=%s in %d iter(s)",
                    name, level_name, res.final_z, target, res.reached, res.iterations,
                )
        rows.append(row)

    if rows and not args.dry_run:
        out = tconf.output_dir / cfg.summary_csv_name
        tconf.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        logger.info("Wrote summary for %d datasets to %s", len(rows), out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
