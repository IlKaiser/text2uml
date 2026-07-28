from __future__ import annotations

from text.config import DEFAULT_CONFIG
from text.rewrite.config import RewriteConfig
from text.rewrite.run import process_dataset_level_four
from text.rewrite.scorer import ComplexityReference


def _reference() -> ComplexityReference:
    metrics = ["parse_tree_depth", "parse_tree_depth_max"]
    means = {m: 1.0 for m in metrics}
    stds = {m: 1.0 for m in metrics}
    return ComplexityReference(metrics=metrics, means=means, stds=stds, raw_min=-1.0, raw_max=1.0)


def test_process_dataset_level_four_writes_the_flattened_file(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "ToyCase"
    dataset_dir.mkdir()
    desc = dataset_dir / "description.md"
    desc.write_text("A Customer, which places Orders, is billed monthly.", encoding="utf-8")

    def fake_rewrite_to_minimize(client, cfg, original, original_score, reference, metric_name, system_prompt, user_prompt_fn, feedback_fn):
        from text.rewrite.loop import MinimizeResult
        return MinimizeResult(
            metric_name=metric_name, source_value=5.0, best_value=2.0,
            improved=True, iterations=3, text="A Customer places Orders. The customer is billed monthly.",
        )

    monkeypatch.setattr("text.rewrite.run.rewrite_to_minimize", fake_rewrite_to_minimize)

    cfg = RewriteConfig()
    row = process_dataset_level_four(
        name="ToyCase", description_path=desc, cfg=cfg, reference=_reference(), tconf=DEFAULT_CONFIG, client=None,
    )

    out = dataset_dir / "description_level_four.md"
    assert out.is_file()
    assert out.read_text(encoding="utf-8").strip() == "A Customer places Orders. The customer is billed monthly."
    assert row["sub_folder_name"] == "ToyCase"
    assert row["source_value"] == 5.0
    assert row["best_value"] == 2.0
    assert row["improved"] is True
    assert row["iterations"] == 3


def test_process_dataset_level_four_skips_existing_without_force(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "ToyCase2"
    dataset_dir.mkdir()
    desc = dataset_dir / "description.md"
    desc.write_text("A Customer places an Order.", encoding="utf-8")
    existing = dataset_dir / "description_level_four.md"
    existing.write_text("PRE-EXISTING\n", encoding="utf-8")

    calls = []
    monkeypatch.setattr(
        "text.rewrite.run.rewrite_to_minimize",
        lambda *a, **k: calls.append(1),
    )

    cfg = RewriteConfig()
    row = process_dataset_level_four(
        name="ToyCase2", description_path=desc, cfg=cfg, reference=_reference(), tconf=DEFAULT_CONFIG, client=None,
    )

    assert calls == []
    assert existing.read_text(encoding="utf-8") == "PRE-EXISTING\n"
    assert "source_value" not in row


def test_process_dataset_level_four_does_not_clobber_on_total_failure(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "ToyCase3"
    dataset_dir.mkdir()
    desc = dataset_dir / "description.md"
    original_text = "A Customer places an Order."
    desc.write_text(original_text, encoding="utf-8")
    existing = dataset_dir / "description_level_four.md"
    existing.write_text("PREVIOUSLY GOOD CONTENT\n", encoding="utf-8")

    def fake_rewrite_to_minimize(client, cfg, original, original_score, reference, metric_name, system_prompt, user_prompt_fn, feedback_fn):
        from text.rewrite.loop import MinimizeResult
        return MinimizeResult(
            metric_name=metric_name, source_value=5.0, best_value=5.0,
            improved=False, iterations=1, text=original,
        )

    monkeypatch.setattr("text.rewrite.run.rewrite_to_minimize", fake_rewrite_to_minimize)

    cfg = RewriteConfig()
    row = process_dataset_level_four(
        name="ToyCase3", description_path=desc, cfg=cfg, reference=_reference(), tconf=DEFAULT_CONFIG, client=None, force=True,
    )

    assert existing.read_text(encoding="utf-8") == "PREVIOUSLY GOOD CONTENT\n"
    assert row["four_improved"] is False
