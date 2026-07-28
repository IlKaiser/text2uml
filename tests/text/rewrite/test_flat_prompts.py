from __future__ import annotations

from text.rewrite.flat_prompts import (
    build_flat_feedback,
    build_flat_user_prompt,
    flat_system_prompt,
)


def test_system_prompt_targets_parse_tree_depth_minimization():
    prompt = flat_system_prompt()
    assert "dependency-tree depth" in prompt
    assert "relative clause" in prompt.lower()
    assert "Never add facts" in prompt


def test_user_prompt_embeds_source_and_feedback():
    prompt = build_flat_user_prompt("SOURCE TEXT", "CURRENT DRAFT", "FIX THE DEPTH")
    assert "SOURCE TEXT" in prompt
    assert "CURRENT DRAFT" in prompt
    assert "FIX THE DEPTH" in prompt


def test_user_prompt_omits_feedback_block_when_none():
    prompt = build_flat_user_prompt("SOURCE TEXT", "CURRENT DRAFT", None)
    assert "SOURCE TEXT" in prompt
    assert "FEEDBACK" not in prompt


def test_build_flat_feedback_names_improvement_vs_no_improvement():
    improved = build_flat_feedback(current_value=2.0, best_value=3.0, source_value=4.0)
    assert "improvement" in improved.lower()

    no_improvement = build_flat_feedback(current_value=3.5, best_value=3.0, source_value=4.0)
    assert "did not improve" in no_improvement.lower()
