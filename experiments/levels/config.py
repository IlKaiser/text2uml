"""Configuration for the complexity-levels vs two-shot F1 experiment.

Compares the ``few_shot`` (two-shot) UML-generation technique across three
complexity variants of each project's description:

* ``one``   -> ``description_level_one.md``  (simplest, from text.rewrite)
* ``two``   -> ``description_level_two.md``  (midpoint)
* ``three`` -> ``description.md``            (the real, original spec)

Everything is config-driven and immutable so the pipeline stays reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# experiments/levels/config.py -> repo root is two parents up.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# (tag, description filename, display label, rank). Rank orders the x-axis.
LevelSpec = Tuple[str, str, str, int]
_LEVELS: List[LevelSpec] = [
    ("zero", "description_level_zero.md", "L0 (minimal)", 0),
    ("one", "description_level_one.md", "L1 (simple)", 1),
    ("two", "description_level_two.md", "L2 (mid)", 2),
    ("three", "description.md", "L3 (real)", 3),
]

# Per-category F1 series drawn in the plots. "cardinality" reuses the relation
# score (endpoint + cardinality correctness) as score_rel / max_score.
CATEGORIES: List[str] = ["class", "association", "attribute", "cardinality"]


@dataclass(frozen=True)
class LevelsConfig:
    """Immutable configuration for the levels experiment."""

    repo_root: Path = _REPO_ROOT
    dataset_dir: Path = _REPO_ROOT / "dataset"
    src_dir: Path = _REPO_ROOT / "src"
    grammar_path: Path = _REPO_ROOT / "grammar.ebnf"
    run_config: Path = _REPO_ROOT / "src" / "config.yaml"
    eval_config: Path = _REPO_ROOT / "src" / "eval_config.yaml"
    output_dir: Path = Path(__file__).resolve().parent / "output"

    # The two-shot technique in the existing pipeline is ``few_shot`` (two
    # worked examples), written with the ``few_`` result prefix.
    technique: str = "few_shot"
    result_prefix: str = "few"

    # Folders used verbatim as the two few-shot examples -> excluded.
    skip_folders: Tuple[str, ...] = ("AlphaInsurance", "GasStation")

    gold_filename: str = "plantuml.txt"
    timeout: int = 360

    levels: Tuple[LevelSpec, ...] = tuple(_LEVELS)
    f1_csv_name: str = "levels_f1.csv"
    figure_formats: Tuple[str, ...] = ("svg", "png")
    dpi: int = 200

    def result_path(self, dataset_dir: Path, level_tag: str, safe_model: str) -> Path:
        """Level-tagged result file: ``result_few_<tag>_<model>.txt``."""
        return dataset_dir / f"result_{self.result_prefix}_{level_tag}_{safe_model}.txt"

    def legacy_result_path(self, dataset_dir: Path, safe_model: str) -> Path:
        """The original two-shot result on the real spec: ``result_few_<model>.txt``."""
        return dataset_dir / f"result_{self.result_prefix}_{safe_model}.txt"

    @property
    def f1_csv(self) -> Path:
        return self.output_dir / self.f1_csv_name


DEFAULT_LEVELS_CONFIG = LevelsConfig()
