# L3 Signal-to-Noise Ratio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For each case's real (L3) `description.md`, compute how much of its content maps to something that actually appears in the gold `plantuml.txt` diagram ("signal") versus content that doesn't ("noise"), as one row per case in a new `levels_snr.csv`, plus a scatter plot correlating the resulting `signal_ratio` against L3 F1.

**Architecture:** A new standalone module `experiments/levels/snr.py`, mirroring the shape of `experiments/levels/model_complexity.py`: reuse `src/eval.py`'s Lark parser (via `load_evaluator`/`init_parser`) for the gold component list, reuse `text.metrics.base.parse` (spaCy) for sentence splitting, and one `langchain_openai.ChatOpenAI` call per case (same shape as `src/eval.py`'s `check_class_llm`) to label each sentence SIGNAL/NOISE.

**Tech Stack:** `langchain_openai.ChatOpenAI` (`gpt-4o-mini`), `langchain_core.prompts.PromptTemplate` + `StrOutputParser`, spaCy (`en_core_web_sm` via `text.metrics.base.parse`), pandas, matplotlib, `scipy.stats.pearsonr`.

## Global Constraints

- All work runs in the `kul` conda env: `/Users/marcocalamo/anaconda3/envs/kul/bin/python`. Every test command in this plan uses that interpreter.
- L3 only (`description.md` / `plantuml.txt`) — no zero/one/two/four handling in this module.
- `compute_all` always recomputes every case in one pass; there is no `only=[...]` partial-scope parameter and no merge-on-write CSV logic — `write_snr_csv` overwrites `levels_snr.csv` unconditionally.
- Same case exclusion as the rest of the levels pipeline: skip any folder in `LevelsConfig.skip_folders` (`AlphaInsurance`, `GasStation`), and skip (don't error on) any case missing `description.md` or `plantuml.txt`.
- Gold components include inheritance edges, in addition to classes, attributes, and associations.
- Token counts use spaCy's per-sentence token length (`len(span)`).
- No real OpenAI API calls in any test — every test touching `classify_sentences` (or anything that calls it) monkeypatches the LLM call.
- No CLI entry point / argparse `main()` is required for this module.

---

### Task 1: Gold component extraction and sentence splitting

**Files:**
- Create: `experiments/levels/snr.py`
- Test: `tests/experiments/levels/test_snr.py`

**Interfaces:**
- Consumes: `ev.parse_path(path: str, parser) -> (classes: list[str], relationships: list[dict], attributes: list[tuple[str,str]], inheritance: list)` from `src/eval.py` (loaded via `experiments.levels.evaluate.load_evaluator(cfg)`); `text.metrics.base.parse(text: str) -> spacy.tokens.Doc`.
- Produces (for Task 2 and Task 3):
  - `@dataclass(frozen=True) class GoldComponents: classes: Tuple[str, ...]; attributes: Tuple[str, ...]; associations: Tuple[str, ...]; inheritance: Tuple[str, ...]` with method `all_names(self) -> Tuple[str, ...]`.
  - `def gold_components(gold_path: Path, parser) -> GoldComponents`
  - `@dataclass(frozen=True) class Sentence: text: str; n_tokens: int`
  - `def split_sentences(text: str) -> List[Sentence]`

The inheritance shape from `ev.get_inheritance_from_parsed` (called inside `parse_path`) is a list of `(child, parent)` string pairs — confirm this by reading `src/eval.py:319-357` (`get_inheritance_from_parsed`) if unsure; the render step below assumes `(child, parent)` tuples.

- [ ] **Step 1: Write the failing tests**

Create `tests/experiments/levels/test_snr.py`:

```python
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
        inheritance = [("Manager", "Employee")]
        return classes, relationships, attributes, inheritance

    import experiments.levels.snr as snr_module
    fake_eval = type("FakeEval", (), {"parse_path": staticmethod(fake_parse_path)})()
    monkeypatch.setattr(snr_module, "_eval_module", lambda: fake_eval)

    gold = gold_components(gold_path, parser=None)

    assert gold.classes == ("CardHolder", "Invoice")
    assert gold.attributes == ("CardHolder.Name:String", "Invoice.Number:Int")
    assert gold.associations == ("CardHolder -- Invoice", "Pump -- RefuelTurn")
    assert gold.inheritance == ("Manager <|-- Employee",)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/marcocalamo/anaconda3/envs/kul/bin/python -m pytest tests/experiments/levels/test_snr.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.levels.snr'` (or `ImportError`).

- [ ] **Step 3: Implement `experiments/levels/snr.py`**

```python
"""L3 description signal-to-noise ratio.

For each case's real (L3) description, measures how much of the description's
content maps to something that actually appears in the gold ``plantuml.txt``
diagram ("signal") versus narrative/business-rule elaboration that never
surfaces in the diagram ("noise"). See
``docs/superpowers/specs/2026-07-06-l3-signal-noise-ratio-design.md``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from text.metrics.base import parse as spacy_parse

from .config import DEFAULT_LEVELS_CONFIG, LevelsConfig
from .evaluate import load_evaluator

logger = logging.getLogger(__name__)

_SNR_CSV = "levels_snr.csv"


def _eval_module(cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG):
    """Indirection point so tests can stub out src/eval.py's parser."""
    return load_evaluator(cfg)


@dataclass(frozen=True)
class GoldComponents:
    classes: Tuple[str, ...]
    attributes: Tuple[str, ...]
    associations: Tuple[str, ...]
    inheritance: Tuple[str, ...]

    def all_names(self) -> Tuple[str, ...]:
        return self.classes + self.attributes


def _class_name(raw: str) -> str:
    """Strip an embedded role name (e.g. "Pump, refillingPump" -> "Pump")."""
    return raw.split(",")[0].strip()


def gold_components(gold_path: Path, parser) -> GoldComponents:
    ev = _eval_module()
    classes, relationships, attributes, inheritance = ev.parse_path(str(gold_path), parser)

    attrs = tuple(f"{cls}.{attr}" for cls, attr in attributes)
    assocs = []
    for rel in relationships:
        names = [_class_name(k) for k in rel.keys()]
        assocs.append(" -- ".join(names))
    inh = tuple(f"{child} <|-- {parent}" for child, parent in inheritance)

    return GoldComponents(
        classes=tuple(classes),
        attributes=attrs,
        associations=tuple(assocs),
        inheritance=inh,
    )


@dataclass(frozen=True)
class Sentence:
    text: str
    n_tokens: int


def split_sentences(text: str) -> List[Sentence]:
    doc = spacy_parse(text)
    sentences: List[Sentence] = []
    for span in doc.sents:
        stripped = span.text.strip()
        if not stripped:
            continue
        sentences.append(Sentence(text=stripped, n_tokens=len(span)))
    return sentences
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/marcocalamo/anaconda3/envs/kul/bin/python -m pytest tests/experiments/levels/test_snr.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add experiments/levels/snr.py tests/experiments/levels/test_snr.py
git commit -m "feat(levels): add gold component extraction and sentence splitting for SNR"
```

---

### Task 2: Sentence classification (LLM + fallback heuristic)

**Files:**
- Modify: `experiments/levels/snr.py`
- Test: `tests/experiments/levels/test_snr.py`

**Interfaces:**
- Consumes: `GoldComponents`, `Sentence` from Task 1.
- Produces (for Task 3): `def classify_sentences(sentences: List[Sentence], gold: GoldComponents) -> List[str]` — returns one of `"SIGNAL"`/`"NOISE"` per sentence, same order/length as `sentences`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/experiments/levels/test_snr.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/marcocalamo/anaconda3/envs/kul/bin/python -m pytest tests/experiments/levels/test_snr.py -v`
Expected: FAIL with `ImportError: cannot import name 'classify_sentences'` (or `AttributeError` on `_invoke_classification_chain`).

- [ ] **Step 3: Implement classification in `experiments/levels/snr.py`**

`List` and `Tuple` are already imported from Task 1's `from typing import List, Tuple` — no new imports needed. Append to `experiments/levels/snr.py`:

```python
_LABEL_LINE_RE = re.compile(r"^\s*(\d+)\s*:\s*(SIGNAL|NOISE)\s*$", re.IGNORECASE)

_CLASSIFICATION_PROMPT = """You will be given a numbered list of sentences from a software specification, \
and four lists of UML diagram components (classes, attributes, associations, inheritance edges) that were \
manually modeled from that same specification.

For each sentence, decide:
- SIGNAL: the sentence introduces, describes, or gives a cardinality/relationship for at least one of the \
listed components.
- NOISE: the sentence is narrative elaboration, an example, a business rule, or a process description that \
does not correspond to any listed component.

Output exactly one line per sentence, in the form "N: SIGNAL" or "N: NOISE", where N is the sentence number. \
Output nothing else.

##############

Classes: {classes}

Attributes: {attributes}

Associations: {associations}

Inheritance: {inheritance}

Sentences:
{sentences}

##############

Labels:
"""


def _invoke_classification_chain(sentences: List[Sentence], gold: GoldComponents) -> str:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    numbered = "\n".join(f"{i}. {s.text}" for i, s in enumerate(sentences, start=1))
    prompt = PromptTemplate.from_template(_CLASSIFICATION_PROMPT)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | model | StrOutputParser()
    return chain.invoke({
        "classes": list(gold.classes),
        "attributes": list(gold.attributes),
        "associations": list(gold.associations),
        "inheritance": list(gold.inheritance),
        "sentences": numbered,
    })


def _heuristic_labels(sentences: List[Sentence], gold: GoldComponents) -> List[str]:
    names = [n.lower() for n in gold.all_names()]
    labels = []
    for s in sentences:
        low = s.text.lower()
        labels.append("SIGNAL" if any(n in low for n in names) else "NOISE")
    return labels


def classify_sentences(sentences: List[Sentence], gold: GoldComponents) -> List[str]:
    if not sentences:
        return []
    try:
        response = _invoke_classification_chain(sentences, gold)
        parsed: dict[int, str] = {}
        for line in response.splitlines():
            m = _LABEL_LINE_RE.match(line)
            if m:
                parsed[int(m.group(1))] = m.group(2).upper()
        if not parsed:
            raise ValueError("no parseable SIGNAL/NOISE lines in LLM response")
        return [parsed.get(i, "NOISE") for i in range(1, len(sentences) + 1)]
    except Exception as exc:  # noqa: BLE001 - any LLM/parsing failure falls back
        logger.warning("Sentence classification failed (%s); using substring heuristic.", exc)
        return _heuristic_labels(sentences, gold)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/marcocalamo/anaconda3/envs/kul/bin/python -m pytest tests/experiments/levels/test_snr.py -v`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add experiments/levels/snr.py tests/experiments/levels/test_snr.py
git commit -m "feat(levels): add LLM sentence classification with substring-heuristic fallback"
```

---

### Task 3: Per-case orchestration, corpus driver, and CSV write

**Files:**
- Modify: `experiments/levels/snr.py`
- Test: `tests/experiments/levels/test_snr.py`

**Interfaces:**
- Consumes: `gold_components`, `split_sentences`, `classify_sentences` from Tasks 1-2; `LevelsConfig` fields `dataset_dir`, `gold_filename`, `skip_folders`, `output_dir`; `load_evaluator(cfg)` and `ev.init_parser(str(cfg.grammar_path))` (same pattern as `experiments/levels/model_complexity.py:41-42`).
- Produces (for Task 4 and for manual corpus runs):
  - `def compute_case_snr(case: str, description_path: Path, gold_path: Path, parser) -> dict`
  - `def compute_all(cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> pd.DataFrame`
  - `def write_snr_csv(df: pd.DataFrame, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> Path`

- [ ] **Step 1: Write the failing tests**

Append to `tests/experiments/levels/test_snr.py`:

```python
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


def test_write_snr_csv_overwrites_unconditionally(tmp_path):
    cfg = replace(DEFAULT_LEVELS_CONFIG, output_dir=tmp_path)
    write_snr_csv(pd.DataFrame([{"sub_folder_name": "A", "snr": 1.0}]), cfg)
    write_snr_csv(pd.DataFrame([{"sub_folder_name": "B", "snr": 2.0}]), cfg)

    result = pd.read_csv(cfg.f1_csv.parent / "levels_snr.csv")
    assert set(result["sub_folder_name"]) == {"B"}
```

Add `import pandas as pd` to the test file's imports if not already present (it is, transitively, but the test file should import it explicitly): add `import pandas as pd` near the top of `tests/experiments/levels/test_snr.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/marcocalamo/anaconda3/envs/kul/bin/python -m pytest tests/experiments/levels/test_snr.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_case_snr'` (or similar for `compute_all`/`write_snr_csv`).

- [ ] **Step 3: Implement orchestration in `experiments/levels/snr.py`**

Add `from pathlib import Path` (already present from Task 1) and append:

```python
def compute_case_snr(case: str, description_path: Path, gold_path: Path, parser) -> dict:
    gold = gold_components(gold_path, parser)
    sentences = split_sentences(description_path.read_text(encoding="utf-8"))
    labels = classify_sentences(sentences, gold)

    n_sentences = len(sentences)
    n_signal = sum(1 for l in labels if l == "SIGNAL")
    n_noise = n_sentences - n_signal
    signal_tokens = sum(s.n_tokens for s, l in zip(sentences, labels) if l == "SIGNAL")
    noise_tokens = sum(s.n_tokens for s, l in zip(sentences, labels) if l == "NOISE")
    snr = signal_tokens / noise_tokens if noise_tokens else float("inf")
    total_tokens = signal_tokens + noise_tokens
    signal_ratio = signal_tokens / total_tokens if total_tokens else float("nan")

    return {
        "sub_folder_name": case,
        "n_sentences": n_sentences,
        "n_signal": n_signal,
        "n_noise": n_noise,
        "signal_tokens": signal_tokens,
        "noise_tokens": noise_tokens,
        "snr": snr,
        "signal_ratio": signal_ratio,
        "n_classes": len(gold.classes),
        "n_attributes": len(gold.attributes),
        "n_associations": len(gold.associations),
        "n_inheritance": len(gold.inheritance),
    }


def compute_all(cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> pd.DataFrame:
    ev = _eval_module(cfg)
    parser = ev.init_parser(str(cfg.grammar_path))

    rows: List[dict] = []
    for dataset in sorted(p for p in cfg.dataset_dir.iterdir() if p.is_dir()):
        if dataset.name in cfg.skip_folders:
            continue
        description_path = dataset / "description.md"
        gold_path = dataset / cfg.gold_filename
        if not description_path.is_file() or not gold_path.is_file():
            logger.debug("%s: missing description.md or %s; skipping", dataset.name, cfg.gold_filename)
            continue
        try:
            rows.append(compute_case_snr(dataset.name, description_path, gold_path, parser))
        except Exception as exc:  # noqa: BLE001 - one bad case must not abort the batch
            logger.warning("%s: SNR computation failed (%s); skipping", dataset.name, exc)
    return pd.DataFrame(rows)


def write_snr_csv(df: pd.DataFrame, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.output_dir / _SNR_CSV
    df.to_csv(out, index=False)
    logger.info("Wrote %d rows to %s", len(df), out)
    return out
```

Update `_eval_module` from Task 1 to accept an optional `cfg` argument (it already does: `def _eval_module(cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG)`), and change its one existing call site in `gold_components` to keep passing no argument (uses the default) — no change needed there since `gold_components` doesn't have a `cfg` parameter; it already calls `_eval_module()` with the default.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/marcocalamo/anaconda3/envs/kul/bin/python -m pytest tests/experiments/levels/test_snr.py -v`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add experiments/levels/snr.py tests/experiments/levels/test_snr.py
git commit -m "feat(levels): add per-case SNR orchestration, corpus driver, and CSV write"
```

---

### Task 4: Correlation plot against L3 F1

**Files:**
- Modify: `experiments/levels/snr.py`
- Test: `tests/experiments/levels/test_snr.py`

**Interfaces:**
- Consumes: `levels_snr.csv` columns (`sub_folder_name`, `signal_ratio`); `levels_f1.csv` columns (`sub_folder_name`, `level`, `f1_global`) as produced by `experiments/levels/evaluate.py`.
- Produces: `def plot_snr_vs_f1(snr_df: pd.DataFrame, f1_df: pd.DataFrame, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> float` — returns the Pearson r, and saves `levels_snr_vs_f1.{png,svg}` to `cfg.figure_dir("corpus")`.

**Note on a concurrent repo change:** since this plan was written, another session added `LevelsConfig.figure_dir(subdir: str) -> Path` (`experiments/levels/config.py`) and routed every other plotting module's save helper through it (`experiments/levels/plots.py`'s `_save`, `case_metrics.py`, `complexity.py`, `correlation.py`) so corpus-wide figures land in `output/corpus/` instead of `output/` directly (CSVs are unaffected and still live at `output_dir` root). This task's plot must follow the same convention: save through `cfg.figure_dir("corpus")`, not `cfg.output_dir` directly.

- [ ] **Step 1: Write the failing test**

Append to `tests/experiments/levels/test_snr.py`:

```python
from experiments.levels.snr import plot_snr_vs_f1


def test_plot_snr_vs_f1_returns_pearson_r_and_saves_files(tmp_path):
    cfg = replace(DEFAULT_LEVELS_CONFIG, output_dir=tmp_path)

    snr_df = pd.DataFrame([
        {"sub_folder_name": "A", "signal_ratio": 0.1},
        {"sub_folder_name": "B", "signal_ratio": 0.5},
        {"sub_folder_name": "C", "signal_ratio": 0.9},
    ])
    f1_df = pd.DataFrame([
        {"sub_folder_name": "A", "level": "three", "f1_global": 0.1},
        {"sub_folder_name": "B", "level": "three", "f1_global": 0.5},
        {"sub_folder_name": "C", "level": "three", "f1_global": 0.9},
        {"sub_folder_name": "A", "level": "zero", "f1_global": 0.99},  # must be filtered out
    ])

    r = plot_snr_vs_f1(snr_df, f1_df, cfg)

    assert r == pytest.approx(1.0, abs=1e-6)
    assert (tmp_path / "corpus" / "levels_snr_vs_f1.png").is_file()
    assert (tmp_path / "corpus" / "levels_snr_vs_f1.svg").is_file()
```

Add `import pytest` to the top of `tests/experiments/levels/test_snr.py` if not already present.

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/marcocalamo/anaconda3/envs/kul/bin/python -m pytest tests/experiments/levels/test_snr.py -v`
Expected: FAIL with `ImportError: cannot import name 'plot_snr_vs_f1'`.

- [ ] **Step 3: Implement the plot in `experiments/levels/snr.py`**

Add these imports at the top of `experiments/levels/snr.py` alongside the existing ones:

```python
import matplotlib

matplotlib.use("Agg")  # headless / reproducible rendering
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
```

Append:

```python
def plot_snr_vs_f1(
    snr_df: pd.DataFrame, f1_df: pd.DataFrame, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG
) -> float:
    l3 = f1_df[f1_df["level"] == "three"][["sub_folder_name", "f1_global"]]
    merged = snr_df.merge(l3, on="sub_folder_name", how="inner").dropna(
        subset=["signal_ratio", "f1_global"]
    )

    r, p = pearsonr(merged["signal_ratio"], merged["f1_global"])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(merged["signal_ratio"], merged["f1_global"], color="#3b6fb0", edgecolor="white")
    if len(merged) >= 2:
        slope, intercept = np.polyfit(merged["signal_ratio"], merged["f1_global"], 1)
        xs = np.linspace(merged["signal_ratio"].min(), merged["signal_ratio"].max(), 100)
        ax.plot(xs, slope * xs + intercept, color="#c0392b", linewidth=2)
    ax.set_xlabel("signal_ratio (L3 description)")
    ax.set_ylabel("f1_global (L3)")
    ax.set_title(f"L3 signal ratio vs F1 (r={r:+.3f}, p={p:.3g}, n={len(merged)})")
    ax.grid(linestyle=":", alpha=0.4)

    out_dir = cfg.figure_dir("corpus")
    for fmt in cfg.figure_formats:
        path = out_dir / f"levels_snr_vs_f1.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=cfg.dpi)
        logger.info("Saved %s", path)
    plt.close(fig)
    return float(r)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/marcocalamo/anaconda3/envs/kul/bin/python -m pytest tests/experiments/levels/test_snr.py -v`
Expected: `11 passed`

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `/Users/marcocalamo/anaconda3/envs/kul/bin/python -m pytest tests/ -q`
Expected: all tests pass (no failures introduced by the new module's imports).

- [ ] **Step 6: Commit**

```bash
git add experiments/levels/snr.py tests/experiments/levels/test_snr.py
git commit -m "feat(levels): add SNR-vs-F1 correlation plot"
```

---

## After This Plan (not part of the plan's tasks)

Once all four tasks are merged, running the corpus-wide computation is a manual step (matching how `model_complexity.py` is invoked), not part of this plan:

```bash
/Users/marcocalamo/anaconda3/envs/kul/bin/python -c "
from experiments.levels.snr import compute_all, write_snr_csv, plot_snr_vs_f1
from experiments.levels.config import DEFAULT_LEVELS_CONFIG
import pandas as pd

cfg = DEFAULT_LEVELS_CONFIG
df = compute_all(cfg)
write_snr_csv(df, cfg)
f1_df = pd.read_csv(cfg.f1_csv)
r = plot_snr_vs_f1(df, f1_df, cfg)
print('Pearson r:', r)
"
```

This costs one `gpt-4o-mini` call per case (~170 cases) — flag the cost to the user before running it, the same way earlier corpus-wide batch runs in this project were confirmed first.
