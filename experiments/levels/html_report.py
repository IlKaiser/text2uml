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
from typing import List, NamedTuple, Optional

import pandas as pd

from .config import CATEGORIES, DEFAULT_LEVELS_CONFIG, TECHNIQUE_RESULT_PREFIXES, LevelsConfig
from .generate import load_runner

logger = logging.getLogger("experiments.levels.html_report")

_TEMPLATE_PATH = Path(__file__).parent / "report_assets" / "l0_vs_l3_template.html"
_COMPARE_LEVELS = ("zero", "three")  # L0, L3


class RunSpec(NamedTuple):
    """One comparison row: a display label plus the (csv, model, technique) it comes from."""

    label: str
    csv_path: Path
    model: str
    technique: str


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
        RunSpec("zero-shot · claude-sonnet-4-6", zero_shot_csv, "claude-sonnet-4-6", "zero_shot"),
        RunSpec("zero-shot · gpt-4o-mini", zero_shot_csv, "gpt-4o-mini", "zero_shot"),
        RunSpec("zero-shot · gemma4:e4b (local)", zero_shot_csv, "gemma4:e4b", "zero_shot"),
    ]


def _run_summary(spec: RunSpec) -> dict:
    """Mean f1_global and per-category F1 at L0 and L3 for one (csv, model)."""
    if not spec.csv_path.is_file():
        raise FileNotFoundError(f"{spec.label}: no CSV at {spec.csv_path} -- run the evaluate stage first.")
    df = pd.read_csv(spec.csv_path)
    sub = df[(df["model"] == spec.model) & (df["level"].isin(_COMPARE_LEVELS))]
    if sub.empty:
        raise ValueError(f"{spec.label}: no rows for model={spec.model!r} in {spec.csv_path}")

    value_cols = ["f1_global"] + [f"f1_{c}" for c in CATEGORIES]
    means = sub.groupby("level")[value_cols].mean()
    missing = [lvl for lvl in _COMPARE_LEVELS if lvl not in means.index]
    if missing:
        raise ValueError(f"{spec.label}: missing level(s) {missing} in {spec.csv_path}")

    l0, l3 = means.loc["zero"], means.loc["three"]
    return {
        "run": spec.label,
        "l0": round(float(l0["f1_global"]), 6),
        "l3": round(float(l3["f1_global"]), 6),
        "cat": {
            "l0": [round(float(l0[f"f1_{c}"]), 6) for c in CATEGORIES],
            "l3": [round(float(l3[f"f1_{c}"]), 6) for c in CATEGORIES],
        },
    }


def build_data(runs: Optional[List[RunSpec]] = None, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> List[dict]:
    return [_run_summary(r) for r in (runs or default_runs(cfg))]


def _build_findings(data: List[dict]) -> str:
    """A short, data-derived summary: which run's L3 holds up best/worst vs. L0."""
    deltas = [(d["run"], d["l3"] - d["l0"], (d["l3"] - d["l0"]) / d["l0"] * 100 if d["l0"] else 0.0) for d in data]
    best_run, best_delta, best_pct = max(deltas, key=lambda t: t[1])
    worst_run, worst_delta, worst_pct = min(deltas, key=lambda t: t[1])

    if best_delta > -0.003:
        lead = (
            f"<strong>Reading it:</strong> {best_run} is the only run where L3 (real spec) "
            f"doesn't lose to L0 (minimal) ({best_delta:+.3f}, {best_pct:+.1f}%)."
        )
    else:
        lead = (
            f"<strong>Reading it:</strong> every run scores worse on L3 (real spec) than "
            f"L0 (minimal); {best_run} degrades least ({best_delta:+.3f}, {best_pct:+.1f}%)."
        )
    tail = f" {worst_run} shows the largest drop from L0 to L3 ({worst_delta:+.3f}, {worst_pct:+.1f}%)."
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


def render_html(data: List[dict], runs: List[RunSpec], cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> str:
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    markers = ("__SUBTITLE__", "__FINDINGS__", "__SOURCE_FILES__", "__SCORED_CSVS__", "__DATA_JSON__")
    missing = [m for m in markers if m not in template]
    if missing:
        raise ValueError(f"Template {_TEMPLATE_PATH} is missing marker(s): {missing}")

    source_files = "\n".join(
        f'<span class="path">{_result_file_pattern(r, cfg)}</span>  # {r.label}' for r in runs
    )
    scored_csvs = "\n".join(sorted({_display_path(r.csv_path, cfg) for r in runs}))

    html = template
    html = html.replace("__SUBTITLE__", f"Direct comparison across {len(data)} runs — generated from levels_f1.csv")
    html = html.replace("__FINDINGS__", _build_findings(data))
    html = html.replace("__SOURCE_FILES__", source_files)
    html = html.replace("__SCORED_CSVS__", scored_csvs)
    html = html.replace("__DATA_JSON__", json.dumps(data, indent=2))
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
