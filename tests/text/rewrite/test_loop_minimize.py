from __future__ import annotations

from text.rewrite.config import RewriteConfig
from text.rewrite.loop import rewrite_to_minimize
from text.rewrite.scorer import ComplexityReference, ScoreResult


def _reference() -> ComplexityReference:
    metrics = ["parse_tree_depth"]
    return ComplexityReference(metrics=metrics, means={"parse_tree_depth": 1.0}, stds={"parse_tree_depth": 1.0}, raw_min=-1.0, raw_max=1.0)


def _score(parse_tree_depth: float, n_tokens: int = 50) -> ScoreResult:
    values = {"parse_tree_depth": parse_tree_depth}
    ref = _reference()
    return ScoreResult(z_index=ref.z_index(values), oriented_z=ref.oriented_z(values), values=values, n_tokens=n_tokens)


def _feedback_fn(current_value, best_value, source_value) -> str:
    return f"current={current_value} best={best_value} source={source_value}"


def test_rewrite_to_minimize_keeps_the_lowest_value_seen(monkeypatch):
    """Feeds a non-monotonic sequence of candidates (down, up, down further)
    and confirms the function returns the overall lowest, not the last."""
    original_score = _score(parse_tree_depth=5.0)
    texts = iter(["BETTER", "WORSE", "BEST"])
    scores = iter([_score(3.0), _score(4.0), _score(2.0)])

    monkeypatch.setattr("text.rewrite.loop.rewrite_once", lambda client, cfg, system, user: next(texts))
    monkeypatch.setattr("text.rewrite.loop.score_text", lambda text, reference, cfg: next(scores))

    cfg = RewriteConfig(verify_meaning=False, max_iterations=3)
    result = rewrite_to_minimize(
        client=None, cfg=cfg, original="ORIGINAL", original_score=original_score,
        reference=_reference(), metric_name="parse_tree_depth",
        system_prompt="SYSTEM", user_prompt_fn=lambda o, c, f: "USER",
        feedback_fn=_feedback_fn,
    )
    assert result.text == "BEST"
    assert result.best_value == 2.0
    assert result.source_value == 5.0
    assert result.improved is True
    assert result.iterations == 3


def test_rewrite_to_minimize_never_beats_source_reports_not_improved(monkeypatch):
    original_score = _score(parse_tree_depth=3.0)
    monkeypatch.setattr("text.rewrite.loop.rewrite_once", lambda client, cfg, system, user: "WORSE")
    monkeypatch.setattr("text.rewrite.loop.score_text", lambda text, reference, cfg: _score(4.0))

    cfg = RewriteConfig(verify_meaning=False, max_iterations=2)
    result = rewrite_to_minimize(
        client=None, cfg=cfg, original="ORIGINAL", original_score=original_score,
        reference=_reference(), metric_name="parse_tree_depth",
        system_prompt="SYSTEM", user_prompt_fn=lambda o, c, f: "USER",
        feedback_fn=_feedback_fn,
    )
    assert result.improved is False
    assert result.text == "ORIGINAL"
    assert result.best_value == 3.0


def test_rewrite_to_minimize_rejects_over_shortened_candidate(monkeypatch):
    """A candidate with a lower metric value but too few tokens must not
    become the new best (content-floor guard, matching the shape loop)."""
    original_score = _score(parse_tree_depth=5.0, n_tokens=100)
    monkeypatch.setattr("text.rewrite.loop.rewrite_once", lambda client, cfg, system, user: "TOO SHORT")
    monkeypatch.setattr("text.rewrite.loop.score_text", lambda text, reference, cfg: _score(1.0, n_tokens=10))

    cfg = RewriteConfig(verify_meaning=False, max_iterations=1, min_token_ratio=0.5)
    result = rewrite_to_minimize(
        client=None, cfg=cfg, original="ORIGINAL", original_score=original_score,
        reference=_reference(), metric_name="parse_tree_depth",
        system_prompt="SYSTEM", user_prompt_fn=lambda o, c, f: "USER",
        feedback_fn=_feedback_fn,
    )
    assert result.improved is False
    assert result.text == "ORIGINAL"
