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
    levels = {
        "zero": {"mdd": 3.4, "subordination_index": 0.2, "context_dependence_proxy": 0.1},
        "one": {"mdd": 2.0, "subordination_index": 0.15, "context_dependence_proxy": 0.9},
        "two": {"mdd": 2.4, "subordination_index": 0.3, "context_dependence_proxy": 0.35},
        "three": L3,
    }
    checks = check_case_shape(levels)
    assert shape_ok(checks) is False


def test_format_feedback_names_failing_metric():
    values = {"mdd": 2.9, "subordination_index": 0.2, "context_dependence_proxy": 0.7}
    checks = check_shape("zero", values, L3)
    text = format_feedback(checks, {"mdd": "pack more modifiers per entity"})
    assert "mdd" in text
    assert "pack more modifiers per entity" in text
