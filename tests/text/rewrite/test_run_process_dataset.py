from __future__ import annotations

from pathlib import Path

from text.config import DEFAULT_CONFIG
from text.rewrite.config import RewriteConfig
from text.rewrite.run import _level_setup, process_dataset
from text.rewrite.scorer import ComplexityReference


def _reference() -> ComplexityReference:
    metrics = ["mdd", "subordination_index", "context_dependence_proxy", "flesch_reading_ease"]
    means = {m: 1.0 for m in metrics}
    stds = {m: 1.0 for m in metrics}
    return ComplexityReference(metrics=metrics, means=means, stds=stds, raw_min=-1.0, raw_max=1.0)


def test_process_dataset_writes_all_three_levels(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "ToyCase"
    dataset_dir.mkdir()
    desc = dataset_dir / "description.md"
    desc.write_text(
        "A Customer places many Orders. Each Order has a Total. "
        "Because the Total may be zero, the Order is flagged.",
        encoding="utf-8",
    )

    def fake_rewrite_to_shape(client, cfg, original, original_score, reference, level_name, l3_values, system_prompt, user_prompt_fn, metric_guidance):
        from text.rewrite.loop import ShapeLevelResult
        return ShapeLevelResult(
            level_name=level_name, final_z=0.1, shape_checks=(), reached=True,
            iterations=1, text=f"[{level_name} rewrite of source]",
        )

    monkeypatch.setattr("text.rewrite.run.rewrite_to_shape", fake_rewrite_to_shape)

    cfg = RewriteConfig()
    row = process_dataset(
        name="ToyCase", description_path=desc, cfg=cfg, reference=_reference(),
        tconf=DEFAULT_CONFIG, client=None,
    )

    assert (dataset_dir / "description_level_zero.md").read_text(encoding="utf-8").strip() == "[zero rewrite of source]"
    assert (dataset_dir / "description_level_one.md").read_text(encoding="utf-8").strip() == "[one rewrite of source]"
    assert (dataset_dir / "description_level_two.md").read_text(encoding="utf-8").strip() == "[two rewrite of source]"
    assert row["sub_folder_name"] == "ToyCase"
    assert row["zero_reached"] is True
    assert row["one_reached"] is True
    assert row["two_reached"] is True


def test_process_dataset_restricts_to_requested_levels(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "ToyCase2"
    dataset_dir.mkdir()
    desc = dataset_dir / "description.md"
    desc.write_text("A Customer places an Order.", encoding="utf-8")

    def fake_rewrite_to_shape(client, cfg, original, original_score, reference, level_name, l3_values, system_prompt, user_prompt_fn, metric_guidance):
        from text.rewrite.loop import ShapeLevelResult
        return ShapeLevelResult(
            level_name=level_name, final_z=0.1, shape_checks=(), reached=True,
            iterations=1, text=f"[{level_name} rewrite of source]",
        )

    monkeypatch.setattr("text.rewrite.run.rewrite_to_shape", fake_rewrite_to_shape)

    cfg = RewriteConfig()
    row = process_dataset(
        name="ToyCase2", description_path=desc, cfg=cfg, reference=_reference(),
        tconf=DEFAULT_CONFIG, client=None, levels=("zero",),
    )

    assert (dataset_dir / "description_level_zero.md").is_file()
    assert not (dataset_dir / "description_level_one.md").exists()
    assert "one_reached" not in row


def test_process_dataset_skips_existing_output_without_force(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "ToyCase3"
    dataset_dir.mkdir()
    desc = dataset_dir / "description.md"
    desc.write_text("A Customer places an Order.", encoding="utf-8")
    existing = dataset_dir / "description_level_zero.md"
    existing.write_text("PRE-EXISTING CONTENT\n", encoding="utf-8")

    calls = []

    def fake_rewrite_to_shape(client, cfg, original, original_score, reference, level_name, l3_values, system_prompt, user_prompt_fn, metric_guidance):
        from text.rewrite.loop import ShapeLevelResult
        calls.append(level_name)
        return ShapeLevelResult(
            level_name=level_name, final_z=0.1, shape_checks=(), reached=True,
            iterations=1, text=f"[{level_name} rewrite of source]",
        )

    monkeypatch.setattr("text.rewrite.run.rewrite_to_shape", fake_rewrite_to_shape)

    cfg = RewriteConfig()
    row = process_dataset(
        name="ToyCase3", description_path=desc, cfg=cfg, reference=_reference(),
        tconf=DEFAULT_CONFIG, client=None, levels=("zero",),
    )

    assert calls == []  # rewrite_to_shape never called for an existing output
    assert existing.read_text(encoding="utf-8") == "PRE-EXISTING CONTENT\n"
    assert "zero_reached" not in row


def test_process_dataset_force_overwrites_existing_output(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "ToyCase4"
    dataset_dir.mkdir()
    desc = dataset_dir / "description.md"
    desc.write_text("A Customer places an Order.", encoding="utf-8")
    existing = dataset_dir / "description_level_zero.md"
    existing.write_text("PRE-EXISTING CONTENT\n", encoding="utf-8")

    def fake_rewrite_to_shape(client, cfg, original, original_score, reference, level_name, l3_values, system_prompt, user_prompt_fn, metric_guidance):
        from text.rewrite.loop import ShapeLevelResult
        return ShapeLevelResult(
            level_name=level_name, final_z=0.1, shape_checks=(), reached=True,
            iterations=1, text=f"[{level_name} rewrite of source]",
        )

    monkeypatch.setattr("text.rewrite.run.rewrite_to_shape", fake_rewrite_to_shape)

    cfg = RewriteConfig()
    row = process_dataset(
        name="ToyCase4", description_path=desc, cfg=cfg, reference=_reference(),
        tconf=DEFAULT_CONFIG, client=None, levels=("zero",), force=True,
    )

    assert existing.read_text(encoding="utf-8").strip() == "[zero rewrite of source]"
    assert row["zero_reached"] is True


def test_process_dataset_does_not_clobber_existing_file_on_total_generation_failure(tmp_path, monkeypatch):
    """A total generation failure (e.g. an API outage/exhausted credits on the
    very first call) leaves rewrite_to_shape's best_text equal to the
    untouched original -- writing that out would silently replace a level
    file with a byte-identical copy of description.md. This happened for
    real during a 2026-07 batch run when Anthropic API credits ran out
    mid-retry, corrupting two cases' level files with no way to recover
    them (they were never backed up, since only the two GasStation
    reference cases were anticipated as calibration-overwrite targets)."""
    dataset_dir = tmp_path / "ToyCase5"
    dataset_dir.mkdir()
    desc = dataset_dir / "description.md"
    original_text = "A Customer places an Order."
    desc.write_text(original_text, encoding="utf-8")
    existing = dataset_dir / "description_level_zero.md"
    existing.write_text("PREVIOUSLY GOOD CONTENT\n", encoding="utf-8")

    def fake_rewrite_to_shape(client, cfg, original, original_score, reference, level_name, l3_values, system_prompt, user_prompt_fn, metric_guidance):
        from text.rewrite.loop import ShapeLevelResult
        # Simulates rewrite_to_shape's own fallback when every iteration
        # raised before producing a scoreable candidate: best_text stays
        # equal to `original` (the untouched description.md content).
        return ShapeLevelResult(
            level_name=level_name, final_z=0.0, shape_checks=(), reached=False,
            iterations=1, text=original,
        )

    monkeypatch.setattr("text.rewrite.run.rewrite_to_shape", fake_rewrite_to_shape)

    cfg = RewriteConfig()
    row = process_dataset(
        name="ToyCase5", description_path=desc, cfg=cfg, reference=_reference(),
        tconf=DEFAULT_CONFIG, client=None, levels=("zero",), force=True,
    )

    assert existing.read_text(encoding="utf-8") == "PREVIOUSLY GOOD CONTENT\n"
    assert row["zero_reached"] is False
    assert row["zero_shape_ok"] is False


def test_process_dataset_skips_write_on_total_failure_when_no_prior_file_existed(tmp_path, monkeypatch):
    """Same total-failure scenario, but no output file existed yet -- must
    not create one filled with a verbatim copy of the source."""
    dataset_dir = tmp_path / "ToyCase6"
    dataset_dir.mkdir()
    desc = dataset_dir / "description.md"
    original_text = "A Customer places an Order."
    desc.write_text(original_text, encoding="utf-8")

    def fake_rewrite_to_shape(client, cfg, original, original_score, reference, level_name, l3_values, system_prompt, user_prompt_fn, metric_guidance):
        from text.rewrite.loop import ShapeLevelResult
        return ShapeLevelResult(
            level_name=level_name, final_z=0.0, shape_checks=(), reached=False,
            iterations=1, text=original,
        )

    monkeypatch.setattr("text.rewrite.run.rewrite_to_shape", fake_rewrite_to_shape)

    cfg = RewriteConfig()
    row = process_dataset(
        name="ToyCase6", description_path=desc, cfg=cfg, reference=_reference(),
        tconf=DEFAULT_CONFIG, client=None, levels=("zero",),
    )

    assert not (dataset_dir / "description_level_zero.md").exists()
    assert row["zero_reached"] is False


def test_level_setup_user_prompt_fn_is_callable_positionally_for_narrative_levels():
    """rewrite_to_shape (Task 3) always calls user_prompt_fn(original, current,
    feedback) with exactly 3 positional args -- reproduce that exact call
    against the real functools.partial object _level_setup builds for "one"/
    "two", instead of a hand-written lambda, so a level_label keyword-vs-
    positional collision (functools.partial(f, level_label=X) then calling
    f(a, b, c) positionally) is actually caught."""
    cfg = RewriteConfig()
    for tag in ("one", "two"):
        _out_level_name, _sprompt, user_fn, _guidance = _level_setup(cfg, tag)
        prompt = user_fn("ORIGINAL TEXT", "CURRENT DRAFT", "some feedback")
        assert "ORIGINAL TEXT" in prompt
        assert "CURRENT DRAFT" in prompt
        assert "some feedback" in prompt

        prompt_no_feedback = user_fn("ORIGINAL TEXT", "CURRENT DRAFT", None)
        assert "ORIGINAL TEXT" in prompt_no_feedback
