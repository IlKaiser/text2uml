from __future__ import annotations

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
