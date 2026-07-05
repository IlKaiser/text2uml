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
