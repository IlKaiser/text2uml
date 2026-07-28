from __future__ import annotations

import math

from text.rewrite.shape_targets import (
    RATIO_BANDS,
    SHAPE_METRICS,
    check_case_shape,
    check_shape,
    format_feedback,
    shape_ok,
)

L3 = {"mdd": 3.0, "subordination_index": 0.5, "context_dependence_proxy": 0.4}


def test_check_shape_all_pass_for_level_zero():
    values = {"mdd": 3.4, "subordination_index": 0.2, "context_dependence_proxy": 0.7}
    checks = check_shape("zero", values, L3)
    assert len(checks) == len(SHAPE_METRICS)
    assert shape_ok(checks)


def test_check_shape_mdd_rank_failure_for_level_zero():
    values = {"mdd": 2.9, "subordination_index": 0.2, "context_dependence_proxy": 0.7}
    checks = check_shape("zero", values, L3)
    mdd_check = next(c for c in checks if c.metric == "mdd")
    assert mdd_check.rank_ok is False
    assert shape_ok(checks) is False


def test_check_shape_band_failure_for_level_one():
    # mdd band for "one" is 0.60-0.90x L3; 0.95x is out of band.
    values = {"mdd": 2.85, "subordination_index": 0.3, "context_dependence_proxy": 0.3}
    checks = check_shape("one", values, L3)
    mdd_check = next(c for c in checks if c.metric == "mdd")
    assert mdd_check.band_ok is False
    assert shape_ok(checks) is False


def test_check_shape_degenerate_l3_skips_band():
    l3_zero = dict(L3, subordination_index=0.0)
    values = {"mdd": 3.4, "subordination_index": 0.9, "context_dependence_proxy": 0.7}
    checks = check_shape("zero", values, l3_zero)
    sub_check = next(c for c in checks if c.metric == "subordination_index")
    assert sub_check.degenerate is True
    assert math.isnan(sub_check.ratio)
    assert shape_ok(checks) is True  # degenerate never fails on its own


def test_subordination_index_rank_not_enforced_at_zero_but_still_enforced_at_one_and_two():
    """Calibration against real generation (2026-07 shape-aware level-zero
    rollout) showed structured notes at level zero routinely exceed L3's
    subordination_index (their mdd-boosting participial clauses are counted
    as subordination too) -- L3 no longer needs to dominate at "zero", but
    must still dominate at "one"/"two" where simplification reliably lowers
    it in practice."""
    # subordination_index exceeds L3 (0.5) at every level here.
    values = {"mdd": 3.4, "subordination_index": 0.6, "context_dependence_proxy": 0.7}

    zero_checks = check_shape("zero", values, L3)
    zero_sub = next(c for c in zero_checks if c.metric == "subordination_index")
    assert zero_sub.rank_ok is True

    for level in ("one", "two"):
        checks = check_shape(level, values, L3)
        sub_check = next(c for c in checks if c.metric == "subordination_index")
        assert sub_check.rank_ok is False


def test_check_case_shape_global_max_constraints():
    levels = {
        "zero": {"mdd": 3.4, "subordination_index": 0.2, "context_dependence_proxy": 0.9},
        "one": {"mdd": 2.0, "subordination_index": 0.15, "context_dependence_proxy": 0.2},
        "two": {"mdd": 2.4, "subordination_index": 0.3, "context_dependence_proxy": 0.35},
        "three": L3,
    }
    checks = check_case_shape(levels)
    assert shape_ok(checks) is True


def test_check_case_shape_fails_when_zero_not_max_context_dependence():
    # zero's own local rank/band vs L3 already fails here (0.1 < L3's 0.4),
    # which alone would flip shape_ok to False without exercising the
    # cross-level max check at all. Assert on the _global_max_check's own
    # ShapeCheck entry (metric + level="zero" + the unbounded (0.0, None)
    # band it always reports) to prove that check independently caught
    # "one" (0.9) outranking "zero" (0.1), not just piggybacking on the
    # local failure.
    levels = {
        "zero": {"mdd": 3.4, "subordination_index": 0.2, "context_dependence_proxy": 0.1},
        "one": {"mdd": 2.0, "subordination_index": 0.15, "context_dependence_proxy": 0.9},
        "two": {"mdd": 2.4, "subordination_index": 0.3, "context_dependence_proxy": 0.35},
        "three": L3,
    }
    checks = check_case_shape(levels)
    global_max_check = next(
        c for c in checks
        if c.metric == "context_dependence_proxy" and c.level == "zero" and c.band == (0.0, None)
    )
    assert global_max_check.rank_ok is False
    assert shape_ok(checks) is False


def test_format_feedback_names_failing_metric():
    values = {"mdd": 2.9, "subordination_index": 0.2, "context_dependence_proxy": 0.7}
    checks = check_shape("zero", values, L3)
    text = format_feedback(checks, {"mdd": "pack more modifiers per entity"})
    assert "mdd" in text
    assert "pack more modifiers per entity" in text
