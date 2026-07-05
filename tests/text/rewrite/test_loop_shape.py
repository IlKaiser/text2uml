from __future__ import annotations

from text.rewrite.config import RewriteConfig
from text.rewrite.loop import rewrite_to_shape
from text.rewrite.scorer import ComplexityReference, ScoreResult


def _reference() -> ComplexityReference:
    metrics = ["mdd", "subordination_index", "context_dependence_proxy"]
    means = {m: 1.0 for m in metrics}
    stds = {m: 1.0 for m in metrics}
    return ComplexityReference(metrics=metrics, means=means, stds=stds, raw_min=-1.0, raw_max=1.0)


def _score(mdd, sub, ctx, n_tokens=50) -> ScoreResult:
    values = {"mdd": mdd, "subordination_index": sub, "context_dependence_proxy": ctx}
    ref = _reference()
    return ScoreResult(z_index=ref.z_index(values), oriented_z=ref.oriented_z(values), values=values, n_tokens=n_tokens)


def test_rewrite_to_shape_accepts_first_candidate_when_shape_already_ok(monkeypatch):
    l3_values = {"mdd": 3.0, "subordination_index": 0.5, "context_dependence_proxy": 0.4}
    good_candidate_score = _score(mdd=3.4, sub=0.2, ctx=0.7)

    monkeypatch.setattr("text.rewrite.loop.rewrite_once", lambda client, cfg, system, user: "GOOD CANDIDATE")
    monkeypatch.setattr("text.rewrite.loop.score_text", lambda text, reference, cfg: good_candidate_score)
    monkeypatch.setattr("text.rewrite.loop.verify_meaning", lambda client, cfg, original, candidate: type("C", (), {"equivalent": True, "feedback": lambda self: ""})())

    cfg = RewriteConfig(verify_meaning=True, max_iterations=3)
    original_score = _score(mdd=3.0, sub=0.5, ctx=0.4)
    result = rewrite_to_shape(
        client=None,
        cfg=cfg,
        original="ORIGINAL",
        original_score=original_score,
        reference=_reference(),
        level_name="zero",
        l3_values=l3_values,
        system_prompt="SYSTEM",
        user_prompt_fn=lambda original, current, feedback: "USER",
        metric_guidance={},
    )
    assert result.reached is True
    assert result.text == "GOOD CANDIDATE"
    assert result.iterations == 1


def test_rewrite_to_shape_stops_at_max_iterations_when_never_reached(monkeypatch):
    l3_values = {"mdd": 3.0, "subordination_index": 0.5, "context_dependence_proxy": 0.4}
    bad_candidate_score = _score(mdd=2.0, sub=0.5, ctx=0.1)  # mdd rank fails for "zero"

    monkeypatch.setattr("text.rewrite.loop.rewrite_once", lambda client, cfg, system, user: "BAD CANDIDATE")
    monkeypatch.setattr("text.rewrite.loop.score_text", lambda text, reference, cfg: bad_candidate_score)

    cfg = RewriteConfig(verify_meaning=False, max_iterations=2)
    original_score = _score(mdd=3.0, sub=0.5, ctx=0.4)
    result = rewrite_to_shape(
        client=None,
        cfg=cfg,
        original="ORIGINAL",
        original_score=original_score,
        reference=_reference(),
        level_name="zero",
        l3_values=l3_values,
        system_prompt="SYSTEM",
        user_prompt_fn=lambda original, current, feedback: "USER",
        metric_guidance={},
    )
    assert result.reached is False
    assert result.iterations == 2
