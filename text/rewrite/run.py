"""CLI: rewrite every dataset's description toward all three complexity levels.

For each ``dataset/<Name>/description.md`` this delegates to
``process_dataset``, which:
  1. scores the original on the corpus ``z_index`` scale,
  2. drives it toward ``level_zero`` (compact structural UML notes genre),
     ``level_one`` (simplest narrative), and ``level_two`` (mid-complexity
     narrative) via the shape-matching feedback loop (``rewrite_to_shape``),
  3. writes ``description_level_zero.md`` / ``description_level_one.md`` /
     ``description_level_two.md`` beside the original, and appends a row to
     ``text/output/rewrite_shape_summary.csv``.

Originals are never overwritten.

Examples:
    python -m text.rewrite.run                    # all datasets, levels zero/one/two
    python -m text.rewrite.run --dry-run          # score originals only
    python -m text.rewrite.run --datasets Sober AirTravel
    python -m text.rewrite.run --limit 5
    python -m text.rewrite.run --levels four      # level four only, all datasets
"""

from __future__ import annotations

import argparse
import functools
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from ..config import DEFAULT_CONFIG, TextConfig
from .config import DEFAULT_REWRITE_CONFIG, RewriteConfig
from .flat_prompts import build_flat_feedback, build_flat_user_prompt, flat_system_prompt
from .loop import rewrite_to_minimize, rewrite_to_shape
from .prompts import METRIC_GUIDANCE, build_shape_user_prompt, system_prompt
from .scorer import ComplexityReference, build_reference, score_text
from .shape_targets import shape_ok
from .structural_prompts import (
    STRUCTURAL_METRIC_GUIDANCE,
    build_structural_user_prompt,
    structural_system_prompt,
)

logger = logging.getLogger("text.rewrite")

_LEVEL_TAGS: Tuple[str, ...] = ("zero", "one", "two")
_LEVEL_LABELS = {
    "one": "a simplified, easy-to-read narrative (level one: simplest)",
    "two": "a moderately simplified narrative (level two: mid-complexity)",
}


def _iter_description_paths(cfg: TextConfig, only: List[str] | None) -> List[Path]:
    paths = sorted(cfg.dataset_dir.glob(f"*/{cfg.description_filename}"))
    if only:
        wanted = set(only)
        paths = [p for p in paths if p.parent.name in wanted]
    return paths


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )


def _level_setup(
    cfg: RewriteConfig, tag: str
) -> Tuple[str, str, Callable[[str, str, Optional[str]], str], Dict[str, str]]:
    """(output level-name, system prompt, user-prompt fn, metric guidance) for one level tag."""
    if tag == "zero":
        return cfg.level_zero_name, structural_system_prompt(), build_structural_user_prompt, STRUCTURAL_METRIC_GUIDANCE
    out_name = cfg.level_one_name if tag == "one" else cfg.level_two_name
    user_fn = functools.partial(build_shape_user_prompt, level_label=_LEVEL_LABELS[tag])
    return out_name, system_prompt(), user_fn, METRIC_GUIDANCE


def process_dataset(
    name: str,
    description_path: Path,
    cfg: RewriteConfig,
    reference: ComplexityReference,
    tconf: TextConfig,
    client,
    levels: Tuple[str, ...] = _LEVEL_TAGS,
    force: bool = False,
) -> dict:
    """Regenerate the requested complexity levels for one dataset case.

    Returns a summary row: ``sub_folder_name``, ``actual_z``, and per
    processed level ``<tag>_reached`` / ``<tag>_iterations`` / ``<tag>_shape_ok``.
    """
    original = description_path.read_text(encoding="utf-8")
    base = score_text(original, reference, tconf)
    row: dict = {"sub_folder_name": name, "actual_z": round(base.z_index, 4)}

    for tag in levels:
        out_level_name, sprompt, user_fn, guidance = _level_setup(cfg, tag)
        out_path = cfg.output_path(description_path.parent, out_level_name)
        if out_path.is_file() and not force:
            logger.info("%s/%s: exists, skipping (pass force=True to regenerate)", name, tag)
            continue
        result = rewrite_to_shape(
            client=client, cfg=cfg, original=original, original_score=base,
            reference=reference, level_name=tag, l3_values=base.values,
            system_prompt=sprompt, user_prompt_fn=user_fn, metric_guidance=guidance,
        )
        if result.text == original:
            # No candidate ever improved on the untouched source (most often
            # a total API failure on the very first call, e.g. an outage or
            # exhausted credits) -- writing it out would silently replace a
            # level file with a byte-identical copy of description.md.
            # Mirrors generate.py's guard against clobbering a good result
            # with a failed one: leave whatever was already on disk alone.
            logger.error(
                "%s/%s: no candidate ever improved on the source description "
                "(likely a total generation failure); leaving %s untouched",
                name, tag, out_path.name,
            )
            row[f"{tag}_reached"] = False
            row[f"{tag}_iterations"] = result.iterations
            row[f"{tag}_shape_ok"] = False
            continue
        out_path.write_text(result.text.rstrip() + "\n", encoding="utf-8")
        row[f"{tag}_reached"] = result.reached
        row[f"{tag}_iterations"] = result.iterations
        row[f"{tag}_shape_ok"] = shape_ok(result.shape_checks)
        logger.info("%s/%s: reached=%s in %d iter(s)", name, tag, result.reached, result.iterations)

    return row


def process_dataset_level_four(
    name: str,
    description_path: Path,
    cfg: RewriteConfig,
    reference: ComplexityReference,
    tconf: TextConfig,
    client,
    force: bool = False,
) -> dict:
    """Generate ``description_level_four.md``: minimize parse_tree_depth vs.
    the real spec (description.md), preserving meaning. Unlike
    ``process_dataset``'s shape-matching levels, this targets one metric
    directly rather than a rank+band shape (see ``text.rewrite.flat_prompts``
    for why: parse_tree_depth is the strongest single F1 predictor at L3 in
    this corpus, so this tests whether directly minimizing it helps further).

    Returns a summary row: ``sub_folder_name``, ``source_value``,
    ``best_value``, ``improved``, ``iterations``.
    """
    out_path = cfg.output_path(description_path.parent, cfg.level_four_name)
    if out_path.is_file() and not force:
        logger.info("%s/four: exists, skipping (pass force=True to regenerate)", name)
        return {"sub_folder_name": name}

    original = description_path.read_text(encoding="utf-8")
    base = score_text(original, reference, tconf)
    result = rewrite_to_minimize(
        client=client, cfg=cfg, original=original, original_score=base,
        reference=reference, metric_name="parse_tree_depth",
        system_prompt=flat_system_prompt(), user_prompt_fn=build_flat_user_prompt,
        feedback_fn=build_flat_feedback,
    )
    if result.text == original:
        # Mirrors process_dataset's guard: never write a level file that
        # ends up byte-identical to the source (total generation failure).
        logger.error(
            "%s/four: no candidate ever improved on the source description "
            "(likely a total generation failure); leaving %s untouched",
            name, out_path.name,
        )
        return {"sub_folder_name": name, "four_improved": False, "four_iterations": result.iterations}

    out_path.write_text(result.text.rstrip() + "\n", encoding="utf-8")
    logger.info(
        "%s/four: parse_tree_depth %.2f -> %.2f (improved=%s) in %d iter(s)",
        name, result.source_value, result.best_value, result.improved, result.iterations,
    )
    return {
        "sub_folder_name": name,
        "source_value": round(result.source_value, 4),
        "best_value": round(result.best_value, 4),
        "improved": result.improved,
        "iterations": result.iterations,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", help="Only these dataset folder names.")
    parser.add_argument("--limit", type=int, help="Process at most N datasets.")
    parser.add_argument(
        "--levels", nargs="*", choices=["zero", "one", "two", "four"],
        help="Which levels to (re)generate (default: zero one two). 'four' "
        "(direct parse_tree_depth minimization) is dispatched to "
        "process_dataset_level_four instead of the shape-matching loop.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate level files that already exist.",
    )
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

    levels_arg = tuple(args.levels) if args.levels else _LEVEL_TAGS
    shape_levels = tuple(tag for tag in levels_arg if tag != "four")
    want_four = "four" in levels_arg

    rows = []
    for path in paths:
        name = path.parent.name
        try:
            base = score_text(path.read_text(encoding="utf-8"), reference, tconf)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", name, exc)
            continue

        if args.dry_run:
            logger.info("%s: actual z_index=%.2f (dry run, no rewrite)", name, base.z_index)
            continue

        row: dict = {}
        if shape_levels:
            row.update(process_dataset(name, path, cfg, reference, tconf, client, levels=shape_levels, force=args.force))
        if want_four:
            four_row = process_dataset_level_four(name, path, cfg, reference, tconf, client, force=args.force)
            row.update(four_row)
        row.setdefault("sub_folder_name", name)
        rows.append(row)

    if rows and not args.dry_run:
        out = tconf.output_dir / "rewrite_shape_summary.csv"
        tconf.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        logger.info("Wrote shape summary for %d datasets to %s", len(rows), out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
