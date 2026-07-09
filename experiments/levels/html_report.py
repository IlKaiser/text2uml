"""Generate the L0-vs-L3 global F1 comparison as a standalone interactive HTML report.

Reads whichever ``levels_f1.csv`` files ``experiments.levels.run``'s evaluate
stage has written (the few-shot corpus at ``cfg.f1_csv`` and the multi-model
zero-shot corpus at ``cfg.output_dir / "zero_shot" / "levels_f1.csv"``),
computes each run's mean F1 at L0 (minimal) and L3 (real spec), and renders
``report_assets/l0_vs_l3_template.html`` with the result substituted in.

Usage:
    python -m experiments.levels.html_report
    python -m experiments.levels.html_report --out path/to/report.html
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import pandas as pd

from .config import CATEGORIES, DEFAULT_LEVELS_CONFIG, TECHNIQUE_RESULT_PREFIXES, LevelsConfig
from .generate import load_runner

logger = logging.getLogger("experiments.levels.html_report")

_TEMPLATE_PATH = Path(__file__).parent / "report_assets" / "l0_vs_l3_template.html"


class RunSpec(NamedTuple):
    """One comparison row: a display label plus the (csv, model, technique) it comes from.

    ``l0_tag`` defaults to "zero" (the Claude-authored ``description_level_
    zero.md``) but can be pointed at any other L0-region level tag from
    ``config._LEVELS`` -- e.g. "zeroalt" for the gemma4:e4b-mlx rewrite --
    so alternate L0 variants can be compared against the same L3 baseline
    without a separate report.
    """

    label: str
    csv_path: Path
    model: str
    technique: str
    l0_tag: str = "zero"


def default_runs(cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> List[RunSpec]:
    """The runs compared so far this session.

    Edit this list to add, remove, or relabel runs -- each entry only needs a
    display label, the ``levels_f1.csv`` it lives in, the exact value in that
    CSV's ``model`` column, and the technique (a key in
    ``config.TECHNIQUE_RESULT_PREFIXES``) used to generate it.
    """
    zero_shot_csv = cfg.output_dir / "zero_shot" / "levels_f1.csv"
    return [
        RunSpec("few-shot · claude-sonnet-4-6", cfg.f1_csv, "claude-sonnet-4-6", "few_shot"),
        RunSpec("few-shot · gpt-4o-mini", cfg.f1_csv, "gpt-4o-mini", "few_shot"),
        RunSpec("few-shot · gemma4:e4b-mlx (local)", cfg.f1_csv, "gemma4:e4b-mlx", "few_shot"),
        RunSpec("zero-shot · claude-sonnet-4-6", zero_shot_csv, "claude-sonnet-4-6", "zero_shot"),
        RunSpec("zero-shot · gpt-4o-mini", zero_shot_csv, "gpt-4o-mini", "zero_shot"),
        RunSpec("zero-shot · gemma4:e4b (local)", zero_shot_csv, "gemma4:e4b", "zero_shot"),
        RunSpec("zero-shot · gemma4:e4b-mlx (local)", zero_shot_csv, "gemma4:e4b-mlx", "zero_shot"),
        RunSpec(
            "few-shot · claude-sonnet-4-6 (L0-alt: gemma4:e4b-mlx rewrite)",
            cfg.f1_csv, "claude-sonnet-4-6", "few_shot", l0_tag="zeroalt",
        ),
        RunSpec(
            "few-shot · claude-sonnet-4-6 (L0-alt: gpt-4o-mini rewrite)",
            cfg.f1_csv, "claude-sonnet-4-6", "few_shot", l0_tag="zeroalt2",
        ),
        RunSpec(
            "zero-shot · claude-sonnet-4-6 (L0-alt: gemma4:e4b-mlx rewrite)",
            zero_shot_csv, "claude-sonnet-4-6", "zero_shot", l0_tag="zeroalt",
        ),
        RunSpec(
            "zero-shot · claude-sonnet-4-6 (L0-alt: gpt-4o-mini rewrite)",
            zero_shot_csv, "claude-sonnet-4-6", "zero_shot", l0_tag="zeroalt2",
        ),
        RunSpec(
            "zero-shot · gemma4:e4b-mlx (L0-alt: gemma4:e4b-mlx rewrite, local)",
            zero_shot_csv, "gemma4:e4b-mlx", "zero_shot", l0_tag="zeroalt",
        ),
        RunSpec(
            "few-shot · gemma4:e4b-mlx (L0-alt: gemma4:e4b-mlx rewrite, local)",
            cfg.f1_csv, "gemma4:e4b-mlx", "few_shot", l0_tag="zeroalt",
        ),
    ]


def _avg_seconds_per_case(spec: RunSpec, cfg: LevelsConfig) -> Tuple[Optional[float], bool]:
    """Mean generation time (seconds) per (case, level) at L0/L3, plus whether
    it's a live measurement or a backfilled estimate.

    ``levels_generation_time.csv`` is a sibling of ``spec.csv_path`` (same
    output dir as the F1 CSV it came from). Its ``estimated`` column
    distinguishes rows timed live via ``time.time()`` in ``generate.generate``
    (``False``) from rows reconstructed after the fact from result-file mtimes
    for runs that predate that instrumentation (``True``). A run whose rows
    are a mix of both is reported as estimated overall -- if any measurement
    isn't a real per-call timing, the average isn't either. Some historical
    runs (e.g. few-shot claude-sonnet-4-6, whose result files were touched in
    an unrelated bulk operation spanning days) have no reliable mtime signal
    at all and were deliberately left unbackfilled -- that's fine, the report
    just omits the stat rather than fabricating a number.
    """
    time_csv = spec.csv_path.parent / cfg.time_csv_name
    if not time_csv.is_file():
        return None, False
    tdf = pd.read_csv(time_csv)
    sub = tdf[
        (tdf["model"] == spec.model)
        & (tdf["technique"] == spec.technique)
        & (tdf["level"].isin((spec.l0_tag, "three")))
    ]
    if sub.empty:
        return None, False
    if "estimated" in sub.columns:
        estimated = bool(sub["estimated"].apply(lambda v: True if pd.isna(v) else bool(v)).any())
    else:
        estimated = True
    return round(float(sub["seconds"].mean()), 1), estimated


def _run_summary(spec: RunSpec, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> dict:
    """Mean f1_global and per-category F1 at L0 (or ``spec.l0_tag``) and L3 for one (csv, model)."""
    if not spec.csv_path.is_file():
        raise FileNotFoundError(f"{spec.label}: no CSV at {spec.csv_path} -- run the evaluate stage first.")
    df = pd.read_csv(spec.csv_path)
    compare_levels = (spec.l0_tag, "three")
    sub = df[(df["model"] == spec.model) & (df["level"].isin(compare_levels))]
    if sub.empty:
        raise ValueError(f"{spec.label}: no rows for model={spec.model!r} in {spec.csv_path}")

    value_cols = ["f1_global"] + [f"f1_{c}" for c in CATEGORIES]
    means = sub.groupby("level")[value_cols].mean()
    missing = [lvl for lvl in compare_levels if lvl not in means.index]
    if missing:
        raise ValueError(f"{spec.label}: missing level(s) {missing} in {spec.csv_path}")

    l0, l3 = means.loc[spec.l0_tag], means.loc["three"]
    avg_seconds, avg_seconds_estimated = _avg_seconds_per_case(spec, cfg)
    counts = sub.groupby("level").size()
    return {
        "run": spec.label,
        "l0": round(float(l0["f1_global"]), 6),
        "l3": round(float(l3["f1_global"]), 6),
        "cat": {
            "l0": [round(float(l0[f"f1_{c}"]), 6) for c in CATEGORIES],
            "l3": [round(float(l3[f"f1_{c}"]), 6) for c in CATEGORIES],
        },
        "avg_seconds": avg_seconds,
        "avg_seconds_estimated": avg_seconds_estimated,
        # n_l0 < n_l3 means some cases have no L0 (or L0-alt) result at all --
        # for an alt-L0 run this is almost always the rewrite step failing to
        # produce a usable candidate for that case (see text.rewrite.run's
        # "no candidate ever improved" guard), not a generation-stage failure.
        "n_l0": int(counts.get(spec.l0_tag, 0)),
        "n_l3": int(counts.get("three", 0)),
    }


def build_data(runs: Optional[List[RunSpec]] = None, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> List[dict]:
    return [_run_summary(r, cfg) for r in (runs or default_runs(cfg))]


def _case_f1(spec: RunSpec, level_tag: str) -> dict:
    """{sub_folder_name: f1_global} for one run at one level tag.

    Reads straight from the CSV rather than reusing ``_run_summary``'s
    grouped means -- the matrix needs the individual per-case values the
    mean collapses away.
    """
    df = pd.read_csv(spec.csv_path)
    sub = df[(df["model"] == spec.model) & (df["level"] == level_tag)]
    return {row.sub_folder_name: round(float(row.f1_global), 4) for row in sub.itertuples()}


def build_case_matrix(runs: Optional[List[RunSpec]] = None, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> dict:
    """Per-case f1_global at L0 (``spec.l0_tag``) and at L3 for every run.

    This is the full matrix behind ``build_data``'s aggregate means -- a
    per-run mean can hide a case that's an outlier, or hide that a run only
    covers a subset of cases (e.g. an alt-L0 rewrite that failed on some
    datasets). ``cases`` is the sorted union of every case that appears in
    any run, so a run missing a case just renders a blank cell instead of
    shifting the row alignment.
    """
    runs = runs or default_runs(cfg)
    per_run = []
    all_cases: set = set()
    max_val = 0.0
    for spec in runs:
        if not spec.csv_path.is_file():
            continue
        l0_vals = _case_f1(spec, spec.l0_tag)
        l3_vals = _case_f1(spec, "three")
        all_cases |= set(l0_vals) | set(l3_vals)
        if l0_vals:
            max_val = max(max_val, max(l0_vals.values()))
        if l3_vals:
            max_val = max(max_val, max(l3_vals.values()))
        per_run.append({"run": spec.label, "l0": l0_vals, "l3": l3_vals})
    return {
        "cases": sorted(all_cases),
        "runs": per_run,
        "domain_max": round(max_val + 0.05, 2),
    }


def _build_findings(data: List[dict]) -> str:
    """A short, data-derived summary: how much L0 scores higher than L3, per run.

    Ranking still uses the raw ``l3 - l0`` delta (unaffected by choice of
    baseline), but the reported percentage is L0's increase over L3 -- i.e.
    L3 is the baseline/denominator -- instead of L3's decrease relative to L0.
    """
    deltas = [(d["run"], d["l3"] - d["l0"], (d["l0"] - d["l3"]) / d["l3"] * 100 if d["l3"] else 0.0) for d in data]
    best_run, best_delta, best_pct = max(deltas, key=lambda t: t[1])
    worst_run, worst_delta, worst_pct = min(deltas, key=lambda t: t[1])

    if best_delta > -0.003:
        lead = (
            f"<strong>Reading it:</strong> {best_run} is the only run where L0 (minimal) "
            f"doesn't score higher than L3 (real spec) ({best_pct:+.1f}% vs. L3)."
        )
    else:
        lead = (
            f"<strong>Reading it:</strong> every run scores higher on L0 (minimal) than "
            f"L3 (real spec); {best_run} shows the smallest L0 increase over L3 ({best_pct:+.1f}%)."
        )
    tail = f" {worst_run} shows the largest L0 increase over L3 ({worst_pct:+.1f}%)."
    return lead + tail


def _result_file_pattern(spec: RunSpec, cfg: LevelsConfig) -> str:
    """The text_output/<case>/result_<prefix>_<level>_<model>.txt pattern for one run.

    "<case>"/"<level>" are HTML-escaped -- this string is embedded directly
    into a template's <pre> block, and raw angle brackets would be parsed as
    (invalid) tags rather than displayed literally.
    """
    safe_model = load_runner(cfg)._safe_model_name(spec.model)
    prefix = TECHNIQUE_RESULT_PREFIXES[spec.technique]
    return f"text_output/&lt;case&gt;/result_{prefix}_&lt;level&gt;_{safe_model}.txt"


def _display_path(path: Path, cfg: LevelsConfig) -> str:
    """Path relative to the repo root when possible (e.g. in tests, csv_path
    may live under a tmp_path outside the repo), else the absolute path."""
    try:
        return str(path.relative_to(cfg.repo_root))
    except ValueError:
        return str(path)


def render_html(
    data: List[dict],
    runs: List[RunSpec],
    cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG,
    matrix: Optional[dict] = None,
) -> str:
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    markers = (
        "__SUBTITLE__", "__FINDINGS__", "__SOURCE_FILES__", "__SCORED_CSVS__",
        "__DATA_JSON__", "__MATRIX_JSON__",
    )
    missing = [m for m in markers if m not in template]
    if missing:
        raise ValueError(f"Template {_TEMPLATE_PATH} is missing marker(s): {missing}")

    source_files = "\n".join(
        f'<span class="path">{_result_file_pattern(r, cfg)}</span>  # {r.label}' for r in runs
    )
    scored_csvs = "\n".join(sorted({_display_path(r.csv_path, cfg) for r in runs}))
    if matrix is None:
        matrix = build_case_matrix(runs, cfg)

    html = template
    html = html.replace("__SUBTITLE__", f"Direct comparison across {len(data)} runs — generated from levels_f1.csv")
    html = html.replace("__FINDINGS__", _build_findings(data))
    html = html.replace("__SOURCE_FILES__", source_files)
    html = html.replace("__SCORED_CSVS__", scored_csvs)
    html = html.replace("__DATA_JSON__", json.dumps(data, indent=2))
    html = html.replace("__MATRIX_JSON__", json.dumps(matrix, indent=2))
    return html


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_LEVELS_CONFIG.output_dir / "reports" / "l0_vs_l3.html",
        help="Output HTML path (default: experiments/levels/output/reports/l0_vs_l3.html).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    runs = default_runs()
    try:
        data = build_data(runs)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    html = render_html(data, runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    logger.info("Wrote %s (%d runs)", args.out, len(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
