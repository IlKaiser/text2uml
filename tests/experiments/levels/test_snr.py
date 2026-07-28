from __future__ import annotations

import pandas as pd
import pytest

from experiments.levels.snr import GoldComponents, Sentence, gold_components, split_sentences


def test_gold_components_renders_classes_attributes_associations_inheritance(tmp_path, monkeypatch):
    gold_path = tmp_path / "plantuml.txt"
    gold_path.write_text("@startuml\n@enduml\n", encoding="utf-8")

    def fake_parse_path(path, parser):
        classes = ["CardHolder", "Invoice"]
        # One relationship has a plain class-name key, the other an
        # embedded role name after a comma -- both must render cleanly.
        relationships = [
            {"CardHolder": '"1"', "Invoice": '"0..*"'},
            {"Pump, refillingPump": '"1"', "RefuelTurn": '"0..*"'},
        ]
        attributes = [("CardHolder", "Name:String"), ("Invoice", "Number:Int")]
        inheritance = [("Manager", "Employee")]  # (parent, child) format as returned by real function
        return classes, relationships, attributes, inheritance

    import experiments.levels.snr as snr_module
    fake_eval = type("FakeEval", (), {"parse_path": staticmethod(fake_parse_path)})()
    monkeypatch.setattr(snr_module, "_eval_module", lambda: fake_eval)

    gold = gold_components(gold_path, parser=None)
    assert isinstance(gold, GoldComponents)

    assert gold.classes == ("CardHolder", "Invoice")
    assert gold.attributes == ("CardHolder.Name:String", "Invoice.Number:Int")
    assert gold.associations == ("CardHolder -- Invoice", "Pump -- RefuelTurn")
    assert gold.inheritance == ("Employee <|-- Manager",)
    assert "CardHolder" in gold.all_names()
    assert "CardHolder.Name:String" in gold.all_names()


def test_split_sentences_counts_tokens_per_sentence():
    sentences = split_sentences("A Customer places Orders. The customer is billed monthly.")
    assert [s.text for s in sentences] == [
        "A Customer places Orders.",
        "The customer is billed monthly.",
    ]
    assert all(isinstance(s, Sentence) for s in sentences)
    assert all(s.n_tokens > 0 for s in sentences)


def test_split_sentences_drops_empty_sentences():
    sentences = split_sentences("First sentence.   \n\n  Second sentence.")
    assert len(sentences) == 2


from experiments.levels.snr import classify_sentences


def _toy_gold():
    return GoldComponents(
        classes=("Customer", "Order"),
        attributes=("Customer.name:String",),
        associations=("Customer -- Order",),
        inheritance=(),
    )


def test_classify_sentences_parses_llm_response(monkeypatch):
    import experiments.levels.snr as snr_module

    sentences = [
        Sentence("A Customer places an Order.", 5),
        Sentence("This is a fun fact about the company history.", 9),
    ]

    def fake_invoke_chain(sentences_arg, gold_arg):
        return "1: SIGNAL\n2: NOISE\n"

    monkeypatch.setattr(snr_module, "_invoke_classification_chain", fake_invoke_chain)

    labels = classify_sentences(sentences, _toy_gold())
    assert labels == ["SIGNAL", "NOISE"]


def test_classify_sentences_defaults_missing_index_to_noise(monkeypatch):
    import experiments.levels.snr as snr_module

    sentences = [Sentence("A Customer places an Order.", 5), Sentence("Unrelated aside.", 3)]

    monkeypatch.setattr(snr_module, "_invoke_classification_chain", lambda s, g: "1: SIGNAL\n")

    labels = classify_sentences(sentences, _toy_gold())
    assert labels == ["SIGNAL", "NOISE"]


def test_classify_sentences_falls_back_to_heuristic_on_failure(monkeypatch):
    import experiments.levels.snr as snr_module

    sentences = [
        Sentence("A Customer places an Order.", 5),
        Sentence("Unrelated narrative aside.", 3),
    ]

    def raise_error(sentences_arg, gold_arg):
        raise RuntimeError("API down")

    monkeypatch.setattr(snr_module, "_invoke_classification_chain", raise_error)

    labels = classify_sentences(sentences, _toy_gold())
    assert labels == ["SIGNAL", "NOISE"]


from dataclasses import replace

from experiments.levels.config import DEFAULT_LEVELS_CONFIG
from experiments.levels.snr import compute_all, compute_case_snr, write_snr_csv


def test_compute_case_snr_aggregates_tokens_and_ratios(monkeypatch, tmp_path):
    import experiments.levels.snr as snr_module

    description_path = tmp_path / "description.md"
    description_path.write_text(
        "A Customer places an Order. This is unrelated narrative filler text here.",
        encoding="utf-8",
    )
    gold_path = tmp_path / "plantuml.txt"
    gold_path.write_text("@startuml\n@enduml\n", encoding="utf-8")

    fake_gold = GoldComponents(
        classes=("Customer", "Order"), attributes=(), associations=("Customer -- Order",), inheritance=(),
    )
    monkeypatch.setattr(snr_module, "gold_components", lambda path, parser: fake_gold)
    monkeypatch.setattr(snr_module, "classify_sentences", lambda sentences, gold: ["SIGNAL", "NOISE"])

    row = compute_case_snr("ToyCase", description_path, gold_path, parser=None)

    assert row["sub_folder_name"] == "ToyCase"
    assert row["n_sentences"] == 2
    assert row["n_signal"] == 1
    assert row["n_noise"] == 1
    assert row["signal_tokens"] > 0
    assert row["noise_tokens"] > 0
    assert row["snr"] == row["signal_tokens"] / row["noise_tokens"]
    assert 0.0 < row["signal_ratio"] < 1.0
    assert row["n_classes"] == 2
    assert row["n_associations"] == 1


def test_compute_all_skips_cases_missing_gold_or_description(tmp_path, monkeypatch):
    cfg = replace(DEFAULT_LEVELS_CONFIG, dataset_dir=tmp_path)

    complete = tmp_path / "Complete"
    complete.mkdir()
    (complete / "description.md").write_text("A Customer places an Order.", encoding="utf-8")
    (complete / "plantuml.txt").write_text("@startuml\n@enduml\n", encoding="utf-8")

    missing_gold = tmp_path / "MissingGold"
    missing_gold.mkdir()
    (missing_gold / "description.md").write_text("Some text.", encoding="utf-8")

    import experiments.levels.snr as snr_module
    monkeypatch.setattr(
        snr_module, "compute_case_snr",
        lambda case, desc, gold, parser: {"sub_folder_name": case, "n_sentences": 1},
    )
    monkeypatch.setattr(snr_module, "_eval_module", lambda cfg=None: type(
        "FakeEval", (), {"init_parser": staticmethod(lambda path: None)}
    )())

    df = compute_all(cfg)
    assert set(df["sub_folder_name"]) == {"Complete"}


def test_compute_all_skips_folders_in_skip_folders(tmp_path):
    cfg = replace(DEFAULT_LEVELS_CONFIG, dataset_dir=tmp_path, skip_folders=("AlphaInsurance",))

    skipped = tmp_path / "AlphaInsurance"
    skipped.mkdir()
    (skipped / "description.md").write_text("A Customer places an Order.", encoding="utf-8")
    (skipped / "plantuml.txt").write_text("@startuml\n@enduml\n", encoding="utf-8")

    kept = tmp_path / "KeptCase"
    kept.mkdir()
    (kept / "description.md").write_text("A Customer places an Order.", encoding="utf-8")
    (kept / "plantuml.txt").write_text("@startuml\n@enduml\n", encoding="utf-8")

    import experiments.levels.snr as snr_module
    import experiments.levels.snr as _snr_for_monkeypatch  # noqa: F401 (name kept for clarity)

    from unittest.mock import patch

    with patch.object(
        snr_module, "compute_case_snr",
        side_effect=lambda case, desc, gold, parser: {"sub_folder_name": case, "n_sentences": 1},
    ), patch.object(
        snr_module, "_eval_module",
        return_value=type("FakeEval", (), {"init_parser": staticmethod(lambda path: None)})(),
    ):
        df = compute_all(cfg)

    assert set(df["sub_folder_name"]) == {"KeptCase"}


def test_compute_all_isolates_one_bad_case_from_the_rest(tmp_path):
    cfg = replace(DEFAULT_LEVELS_CONFIG, dataset_dir=tmp_path)

    bad = tmp_path / "BadCase"
    bad.mkdir()
    (bad / "description.md").write_text("Some text.", encoding="utf-8")
    (bad / "plantuml.txt").write_text("@startuml\n@enduml\n", encoding="utf-8")

    good = tmp_path / "GoodCase"
    good.mkdir()
    (good / "description.md").write_text("Some text.", encoding="utf-8")
    (good / "plantuml.txt").write_text("@startuml\n@enduml\n", encoding="utf-8")

    import experiments.levels.snr as snr_module

    def fake_compute_case_snr(case, description_path, gold_path, parser):
        if case == "BadCase":
            raise RuntimeError("simulated failure")
        return {"sub_folder_name": case, "n_sentences": 1}

    from unittest.mock import patch

    with patch.object(snr_module, "compute_case_snr", side_effect=fake_compute_case_snr), patch.object(
        snr_module, "_eval_module",
        return_value=type("FakeEval", (), {"init_parser": staticmethod(lambda path: None)})(),
    ):
        df = compute_all(cfg)

    assert set(df["sub_folder_name"]) == {"GoodCase"}


def test_write_snr_csv_overwrites_unconditionally(tmp_path):
    cfg = replace(DEFAULT_LEVELS_CONFIG, output_dir=tmp_path)
    write_snr_csv(pd.DataFrame([{"sub_folder_name": "A", "snr": 1.0}]), cfg)
    write_snr_csv(pd.DataFrame([{"sub_folder_name": "B", "snr": 2.0}]), cfg)

    result = pd.read_csv(cfg.f1_csv.parent / "levels_snr.csv")
    assert set(result["sub_folder_name"]) == {"B"}


from experiments.levels.snr import plot_snr_vs_f1


def test_plot_snr_vs_f1_returns_pearson_r_and_saves_files(tmp_path):
    cfg = replace(DEFAULT_LEVELS_CONFIG, output_dir=tmp_path)

    snr_df = pd.DataFrame([
        {"sub_folder_name": "A", "signal_ratio": 0.1},
        {"sub_folder_name": "B", "signal_ratio": 0.5},
        {"sub_folder_name": "C", "signal_ratio": 0.9},
    ])
    f1_df = pd.DataFrame([
        {"sub_folder_name": "A", "level": "three", "model": "claude-sonnet-4-6", "f1_global": 0.1},
        {"sub_folder_name": "B", "level": "three", "model": "claude-sonnet-4-6", "f1_global": 0.5},
        {"sub_folder_name": "C", "level": "three", "model": "claude-sonnet-4-6", "f1_global": 0.9},
        {"sub_folder_name": "A", "level": "zero", "model": "claude-sonnet-4-6", "f1_global": 0.99},  # must be filtered out (wrong level)
        {"sub_folder_name": "A", "level": "three", "model": "gpt-4o-mini", "f1_global": 0.01},  # must be filtered out (wrong model)
    ])

    r = plot_snr_vs_f1(snr_df, f1_df, model="claude-sonnet-4-6", cfg=cfg)

    assert r == pytest.approx(1.0, abs=1e-6)
    assert (tmp_path / "corpus" / "levels_snr_vs_f1.png").is_file()
    assert (tmp_path / "corpus" / "levels_snr_vs_f1.svg").is_file()
