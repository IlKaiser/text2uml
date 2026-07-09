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
from typing import Dict, List, Tuple

# experiments/levels/config.py -> repo root is two parents up.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# (tag, description filename, display label, rank). Rank orders the x-axis;
# float so a level can be inserted between two integer-ranked ones.
LevelSpec = Tuple[str, str, str, float]
_LEVELS: List[LevelSpec] = [
    ("zero", "description_level_zero.md", "L0 (minimal)", 0),
    ("one", "description_level_one.md", "L1 (simple)", 1),
    ("two", "description_level_two.md", "L2 (mid)", 2),
    ("three", "description.md", "L3 (real)", 3),
    # Minimizes parse_tree_depth vs. L3 directly (text.rewrite.flat_prompts),
    # rather than the zero/one/two shape-matching genres -- motivated by
    # parse_tree_depth being the strongest single F1 predictor at L3 in this
    # corpus. Rank 4 places it after L3 on plot x-axes as an additional
    # derived variant, not a "more complex than L3" level.
    ("four", "description_level_four.md", "L4 (flat)", 4),
    # An alternate L0 description rewritten by a different model
    # (gemma4:e4b-mlx via Ollama, vs. the Claude-authored description_level_
    # zero.md) -- same structural-notes genre and shape target, different
    # rewriter. Rank 0.5 places it right after L0 on plot x-axes without
    # disturbing the existing 0/1/2/3/4 ordering. Own filename/tag so its
    # generated diagrams (result_<prefix>_zeroalt_<model>.txt) never collide
    # with or overwrite the original L0 results.
    ("zeroalt", "description_level_zero_gemma4e4bmlx.md", "L0-alt (gemma rewrite)", 0.5),
    # A second alternate L0 description, rewritten by gpt-4o-mini instead of
    # gemma4:e4b-mlx or Claude -- same structural-notes genre and shape
    # target, third rewriter. Rank 0.6 keeps it right after "zeroalt" (0.5)
    # and before L1 (1) without disturbing any existing ordering. Own
    # filename/tag so its generated diagrams (result_<prefix>_zeroalt2_
    # <model>.txt) never collide with or overwrite any existing results.
    ("zeroalt2", "description_level_zero_gpt4omini.md", "L0-alt (gpt-4o-mini rewrite)", 0.6),
]

# Per-category F1 series drawn in the plots. "cardinality" reuses the relation
# score (endpoint + cardinality correctness) as score_rel / max_score.
CATEGORIES: List[str] = ["class", "association", "attribute", "cardinality"]

# Result-file prefix per src.run._CHAIN_BUILDERS technique key. "few_shot" ->
# "few" matches the filenames already on disk from before multi-technique
# support existed; the others are spelled out to stay unambiguous next to the
# level tags ("zero"/"one"/"two"/... appear in both technique names and level
# tags, e.g. result_zeroshot_zero_<model>.txt is technique=zero_shot,
# level=zero).
TECHNIQUE_RESULT_PREFIXES: Dict[str, str] = {
    "zero_shot": "zeroshot",
    "one_shot": "oneshot",
    "few_shot": "few",
    "cot": "cot",
    "cot_domain": "cotdomain",
}


@dataclass(frozen=True)
class LevelsConfig:
    """Immutable configuration for the levels experiment."""

    repo_root: Path = _REPO_ROOT
    dataset_dir: Path = _REPO_ROOT / "dataset"
    # Level-tagged UML generation results (result_few_<level>_<model>.txt) live
    # here, mirrored per case, instead of alongside the dataset's own
    # description/gold files -- keeps generated model output separate from
    # the case study's source material.
    text_output_dir: Path = _REPO_ROOT / "text_output"
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
    time_csv_name: str = "levels_generation_time.csv"
    figure_formats: Tuple[str, ...] = ("svg", "png")
    dpi: int = 200

    def result_path(self, dataset_dir: Path, level_tag: str, safe_model: str) -> Path:
        """Level-tagged result file: ``text_output/<case>/result_few_<tag>_<model>.txt``."""
        return self.text_output_dir / dataset_dir.name / f"result_{self.result_prefix}_{level_tag}_{safe_model}.txt"

    def legacy_result_path(self, dataset_dir: Path, safe_model: str) -> Path:
        """The original two-shot result on the real spec: ``text_output/<case>/result_few_<model>.txt``."""
        return self.text_output_dir / dataset_dir.name / f"result_{self.result_prefix}_{safe_model}.txt"

    @property
    def f1_csv(self) -> Path:
        return self.output_dir / self.f1_csv_name

    @property
    def time_csv(self) -> Path:
        return self.output_dir / self.time_csv_name

    def figure_dir(self, subdir: str) -> Path:
        """Subdirectory of ``output_dir`` for one category of figures.

        Keeps the ~180-file flat ``output/`` directory navigable: per-case
        plots and corpus-wide plots each get their own folder instead of
        sharing one directory by filename prefix alone. CSVs are unaffected
        and stay at ``output_dir`` root.
        """
        d = self.output_dir / subdir
        d.mkdir(parents=True, exist_ok=True)
        return d


def safe_subdir_model(model: str) -> str:
    """Sanitize a model id for use in an output subdir name.

    "/" for HuggingFace-style "org/model" ids; ":" for Ollama tags (e.g.
    "gemma4:e4b") -- mirrors ``src.run._safe_model_name``'s filename
    sanitization so a model's subdir and its result filenames agree.
    """
    return model.replace("/", "_").replace(":", "_")


DEFAULT_LEVELS_CONFIG = LevelsConfig()
