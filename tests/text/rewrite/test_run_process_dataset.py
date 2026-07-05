from __future__ import annotations

from pathlib import Path

from text.config import DEFAULT_CONFIG
from text.rewrite.config import RewriteConfig
from text.rewrite.run import process_dataset
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
