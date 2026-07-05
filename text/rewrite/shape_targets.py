"""Rank + ratio-band shape checks for the four description complexity levels.

Encodes the empirical GasStation_KUL / GasStation_TUW pattern for three
metrics (mdd, subordination_index, context_dependence_proxy) as a pure,
API-free checker: given a level's metric values and that same case's own
(untouched) L3 values, decide whether the level's shape matches the
reference pattern. Two tiers:

* Rank constraints (hard): e.g. mdd(L0) must exceed mdd(L3).
* Ratio bands (soft): the level's metric value, as a multiple of that case's
  own L3 value, must fall in an empirically-derived band.

When the L3 reference value is exactly zero, there is no meaningful ratio or
rank to compare against for that metric (e.g. a real spec with zero
subordinate clauses would force every other level to also have zero, an
unreasonable requirement) — the whole check is skipped (never fails) and
marked ``degenerate``, rather than only skipping the ratio band.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

SHAPE_METRICS: Tuple[str, ...] = ("mdd", "subordination_index", "context_dependence_proxy")

# level tag -> metric -> (min_ratio, max_ratio); max_ratio=None means one-sided (>= min_ratio).
RATIO_BANDS: Dict[str, Dict[str, Tuple[float, Optional[float]]]] = {
    "zero": {
        "mdd": (1.05, None),
        "subordination_index": (0.35, 0.65),
        "context_dependence_proxy": (1.5, None),
    },
    "one": {
        "mdd": (0.60, 0.90),
        "subordination_index": (0.20, 0.70),
        "context_dependence_proxy": (0.5, 1.3),
    },
    "two": {
        "mdd": (0.75, 0.95),
        "subordination_index": (0.45, 0.75),
        "context_dependence_proxy": (0.7, 1.2),
    },
}


@dataclass(frozen=True)
class ShapeCheck:
    """One metric's shape verdict for one level of one case."""

    metric: str
    level: str
    value: float
    l3_value: float
    ratio: float
    band: Tuple[float, Optional[float]]
    rank_ok: bool
    band_ok: bool
    degenerate: bool


def _rank_ok(metric: str, level: str, value: float, l3_value: float) -> bool:
    """Hard rank constraints from the design spec (Section: Target shape rules)."""
    if metric == "mdd" and level == "zero":
        return value > l3_value
    if metric == "subordination_index":
        return l3_value >= value  # L3 must dominate every other level.
    if metric == "context_dependence_proxy" and level == "zero":
        return value >= l3_value
    return True


def check_shape(
    level: str, level_values: Dict[str, float], l3_values: Dict[str, float]
) -> List[ShapeCheck]:
    """Check one level's metrics against that case's own L3 values.

    Only used as the *local* (vs L3 alone) check, e.g. inside the per-text
    rewrite loop. ``check_case_shape`` adds the cross-level max constraints
    that need every level's values at once.
    """
    checks: List[ShapeCheck] = []
    bands = RATIO_BANDS.get(level, {})
    for metric in SHAPE_METRICS:
        value = level_values.get(metric)
        l3_value = l3_values.get(metric)
        if value is None or l3_value is None:
            continue
        degenerate = l3_value == 0
        ratio = float("nan") if degenerate else value / l3_value
        lo, hi = bands.get(metric, (0.0, None))
        band_ok = degenerate or (ratio >= lo and (hi is None or ratio <= hi))
        checks.append(
            ShapeCheck(
                metric=metric,
                level=level,
                value=value,
                l3_value=l3_value,
                ratio=ratio,
                band=(lo, hi),
                rank_ok=_rank_ok(metric, level, value, l3_value),
                band_ok=band_ok,
                degenerate=degenerate,
            )
        )
    return checks


def _global_max_check(metric: str, expected_max_level: str, levels: Dict[str, Dict[str, float]]) -> Optional[ShapeCheck]:
    """Cross-level constraint: ``metric`` must peak at ``expected_max_level``."""
    values = {lvl: vals.get(metric) for lvl, vals in levels.items() if metric in vals}
    if expected_max_level not in values or len(values) < 2:
        return None
    max_level = max(values, key=lambda lvl: values[lvl])
    l3_value = levels.get("three", {}).get(metric, float("nan"))
    return ShapeCheck(
        metric=metric,
        level=expected_max_level,
        value=values[expected_max_level],
        l3_value=l3_value,
        ratio=float("nan"),
        band=(0.0, None),
        rank_ok=(max_level == expected_max_level),
        band_ok=True,
        degenerate=False,
    )


def check_case_shape(levels: Dict[str, Dict[str, float]]) -> List[ShapeCheck]:
    """Full corpus-level check: per-level bands vs L3, plus cross-level maxima.

    ``levels`` maps level tag ("zero", "one", "two", "three") to that level's
    metric-value dict (e.g. one row of ``levels_complexity.csv`` per level).
    """
    l3_values = levels.get("three", {})
    checks: List[ShapeCheck] = []
    for level in ("zero", "one", "two"):
        if level in levels:
            checks.extend(check_shape(level, levels[level], l3_values))
    for metric, expected_max_level in (
        ("subordination_index", "three"),
        ("context_dependence_proxy", "zero"),
    ):
        gmax = _global_max_check(metric, expected_max_level, levels)
        if gmax is not None:
            checks.append(gmax)
    return checks


def shape_ok(checks: List[ShapeCheck]) -> bool:
    """True only if every non-degenerate check holds both its rank and band.

    A degenerate check (L3 reference value is exactly 0 for that metric) has
    no meaningful rank or ratio to compare against, so it is exempted
    entirely rather than only skipping its ratio band.
    """
    return all(c.degenerate or (c.rank_ok and c.band_ok) for c in checks)


def format_feedback(checks: List[ShapeCheck], guidance: Dict[str, str]) -> str:
    """Turn failing checks into concrete editing guidance for the rewrite prompt."""
    lines: List[str] = []
    for c in checks:
        if c.degenerate or (c.rank_ok and c.band_ok):
            continue
        lo, hi = c.band
        band_str = f">= {lo:.2f}x" if hi is None else f"{lo:.2f}x-{hi:.2f}x"
        ratio_str = "n/a" if math.isnan(c.ratio) else f"{c.ratio:.2f}x"
        tip = guidance.get(c.metric, "adjust the phrasing")
        lines.append(
            f"- {c.metric}: measured {ratio_str} of the real spec's value "
            f"(target band {band_str} of the real spec). {tip}."
        )
    if not lines:
        return ""
    return "The following metrics don't yet match the target shape:\n" + "\n".join(lines)
