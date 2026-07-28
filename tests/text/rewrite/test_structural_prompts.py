from __future__ import annotations

from text.rewrite.structural_prompts import (
    STRUCTURAL_METRIC_GUIDANCE,
    build_structural_user_prompt,
    structural_system_prompt,
)


def test_system_prompt_describes_classes_and_relationships_genre():
    prompt = structural_system_prompt()
    assert "Classes" in prompt
    assert "Relationships" in prompt
    assert "pronoun" in prompt.lower()
    assert "Never add facts" in prompt


def test_guidance_covers_the_three_shape_metrics():
    for metric in ("mdd", "subordination_index", "context_dependence_proxy"):
        assert metric in STRUCTURAL_METRIC_GUIDANCE


def test_user_prompt_embeds_source_and_feedback():
    prompt = build_structural_user_prompt("SOURCE TEXT", "CURRENT DRAFT", "FIX THE MDD")
    assert "SOURCE TEXT" in prompt
    assert "CURRENT DRAFT" in prompt
    assert "FIX THE MDD" in prompt


def test_user_prompt_omits_feedback_block_when_none():
    prompt = build_structural_user_prompt("SOURCE TEXT", "CURRENT DRAFT", None)
    assert "SOURCE TEXT" in prompt
    assert "FEEDBACK" not in prompt
