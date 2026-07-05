from __future__ import annotations

import pandas as pd

from experiments.levels.shape_loop import find_noncompliant_cases


def _row(case, level, rank, mdd, sub, ctx):
    return {
        "sub_folder_name": case, "level": level, "level_rank": rank,
        "mdd": mdd, "subordination_index": sub, "context_dependence_proxy": ctx,
    }


def test_find_noncompliant_cases_flags_only_failing_case():
    rows = [
        # Case A: shape holds.
        _row("A", "zero", 0, 3.4, 0.2, 0.9),
        _row("A", "one", 1, 2.0, 0.15, 0.2),
        _row("A", "two", 2, 2.4, 0.3, 0.35),
        _row("A", "three", 3, 3.0, 0.5, 0.4),
        # Case B: zero's mdd does not exceed three's -> fails.
        _row("B", "zero", 0, 2.5, 0.2, 0.9),
        _row("B", "one", 1, 2.0, 0.15, 0.2),
        _row("B", "two", 2, 2.4, 0.3, 0.35),
        _row("B", "three", 3, 3.0, 0.5, 0.4),
    ]
    df = pd.DataFrame(rows)
    failing = find_noncompliant_cases(df)
    assert "A" not in failing
    assert "B" in failing
    assert "zero" in failing["B"]


def test_find_noncompliant_cases_flags_case_with_missing_level_row():
    """A level whose metrics failed to compute is dropped by
    compute_level_metrics rather than left as a failing row -- a case with a
    missing level must still be treated as non-compliant, not silently
    passing because check_case_shape only sees the levels present."""
    rows = [
        _row("D", "zero", 0, 3.4, 0.2, 0.9),
        _row("D", "one", 1, 2.0, 0.15, 0.2),
        # "two" is missing entirely.
        _row("D", "three", 3, 3.0, 0.5, 0.4),
    ]
    df = pd.DataFrame(rows)
    failing = find_noncompliant_cases(df)
    assert failing.get("D") == ["two"]


def test_find_noncompliant_cases_excludes_global_max_check_from_bad_levels(monkeypatch):
    """The cross-level context_dependence_proxy check reports level="zero" --
    the same (metric, level) pair a local check_shape() failure would use --
    so it must be told apart via its (0.0, None) band fingerprint. Otherwise
    a failing global check could be mistaken for (or masked by) a local
    "zero" failure and never trigger the documented all-three-levels
    fallback."""
    from text.rewrite.shape_targets import ShapeCheck

    synthetic_checks = [
        # Local check_shape() check at level="zero": passes on its own.
        ShapeCheck(
            metric="context_dependence_proxy", level="zero", value=0.9, l3_value=0.4,
            ratio=2.25, band=(1.5, None), rank_ok=True, band_ok=True, degenerate=False,
        ),
        # Cross-level check (_global_max_check's (0.0, None) fingerprint): fails.
        ShapeCheck(
            metric="context_dependence_proxy", level="zero", value=0.9, l3_value=0.4,
            ratio=float("nan"), band=(0.0, None), rank_ok=False, band_ok=True, degenerate=False,
        ),
    ]
    monkeypatch.setattr("experiments.levels.shape_loop.check_case_shape", lambda levels: synthetic_checks)
    monkeypatch.setattr("experiments.levels.shape_loop.shape_ok", lambda checks: False)

    df = pd.DataFrame([
        _row("C", "zero", 0, 3.4, 0.2, 0.9),
        _row("C", "one", 1, 2.0, 0.15, 0.2),
        _row("C", "two", 2, 2.4, 0.3, 0.35),
        _row("C", "three", 3, 3.0, 0.5, 0.4),
    ])
    failing = find_noncompliant_cases(df)
    # No genuine local failure exists (only the excluded global check "fails"),
    # so this must fall back to retrying all three regenerable levels.
    assert failing["C"] == sorted(["zero", "one", "two"])
