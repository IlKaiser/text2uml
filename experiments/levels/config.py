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
    # A third alternate L0 description, rewritten by qwen/qwen3.6-27b via
    # OpenRouter -- same structural-notes genre and shape target, fourth
    # rewriter (Claude, gemma4:e4b-mlx, gpt-4o-mini, now qwen). Rank 0.7
    # keeps it right after "zeroalt2" (0.6) and before L1 (1). Own
    # filename/tag so its generated diagrams (result_<prefix>_zeroalt3_
    # <model>.txt) never collide with or overwrite any existing results.
    ("zeroalt3", "description_level_zero_qwen36_27b.md", "L0-alt (qwen3.6-27b rewrite)", 0.7),
    # A fourth alternate L0 description, rewritten by z-ai/glm-5.2 via
    # OpenRouter -- same structural-notes genre and shape target, fifth
    # rewriter (Claude, gemma4:e4b-mlx, gpt-4o-mini, qwen3.6-27b, now glm-5.2).
    # Rank 0.8 keeps it right after "zeroalt3" (0.7) and before L1 (1). Own
    # filename/tag so its generated diagrams (result_<prefix>_zeroalt4_
    # <model>.txt) never collide with or overwrite any existing results.
    ("zeroalt4", "description_level_zero_glm52.md", "L0-alt (glm-5.2 rewrite)", 0.8),
    # A fifth alternate L0 description, rewritten by gpt-5.5 via OpenRouter.
    # Rank 0.85 keeps it right after "zeroalt4" (0.8) and before L1 (1). Own
    # filename/tag so its generated diagrams never collide with or overwrite
    # any existing results.
    ("zeroalt5", "description_level_zero_gpt55.md", "L0-alt (gpt-5.5 rewrite, OpenRouter)", 0.85),
    # A sixth alternate L0 description, rewritten by gpt-5.5 via the direct
    # OpenAI API (rather than OpenRouter) -- same model, different backend,
    # for comparing routing/backend effects independent of the model itself.
    # Rank 0.9 keeps it right after "zeroalt5" (0.85) and before L1 (1).
    ("zeroalt6", "description_level_zero_gpt55_direct.md", "L0-alt (gpt-5.5 rewrite, direct OpenAI)", 0.9),
    # Not a shape-matching rewrite at all: directly maximizes recall_density
    # (gold class/attribute-name recall per token) via Claude, the metric
    # this project's own per-case correlation analysis found actually
    # predicts F1 (r=0.56-0.60, p<0.0001, n=213) -- unlike mdd/subordination/
    # context_dependence, which the zero/zeroalt* levels above target and
    # which did NOT hold up at the per-case level (mdd_ratio: r=-0.12, not
    # significant). See text.rewrite.recall_prompts. Rank -1 places it
    # before L0 on plot x-axes, reflecting that it's generated by a
    # different objective, not a same-target alternate rewriter. Own
    # filename/tag so its generated diagrams never collide with or
    # overwrite any existing results.
    ("levelminus1", "description_level_minus_one.md", "L-1 (recall-density optimized)", -1),
    # Same recall_density objective as "levelminus1", but the target entity
    # list is extracted from the source description itself (text.rewrite.
    # source_entities) instead of the gold PlantUML -- levelminus1 is leaky
    # (it optimizes against the same file evaluate.py scores against, an
    # upper-bound/ceiling experiment rather than a deployable method). This
    # level tests how much of that F1 gain survives without answer-key
    # access. Rank -2 places it before levelminus1 on plot x-axes.
    ("levelminus2", "description_level_minus_two.md", "L-2 (recall-density, source-only)", -2),
    # Same recall_density objective as "levelminus2" (no gold access), but
    # the target entity list comes from one LLM extraction call over the
    # source description instead of noun-chunk frequency counting --
    # levelminus2's frequency heuristic scored *below* the L0 baseline
    # (F1 0.405 vs 0.425), so this tests whether leaning on the model's own
    # UML-modeling judgment closes that gap while staying gold-free. See
    # text.rewrite.llm_source_entities. Rank -3 places it before
    # levelminus2 on plot x-axes.
    ("levelminus3", "description_level_minus_three.md", "L-3 (recall-density, LLM-extracted)", -3),
    # Extends "levelminus3"'s recall_density objective to also credit
    # relationship pairs (both endpoint classes stated on the same line),
    # not just entity names -- levelminus2/levelminus3 both plateaued ~5%
    # below the L0 baseline despite ~100% entity-name recall, suggesting
    # compression traded away relationship facts entity recall can't see.
    # Still no gold access (one LLM extraction call for entities AND
    # relationship pairs). See text.rewrite.relation_loop. Rank -4 places
    # it before levelminus3 on plot x-axes.
    ("levelminus4", "description_level_minus_four.md", "L-4 (recall-density, entities+relations)", -4),
    # Extends "levelminus4"'s objective to also credit multiplicity/
    # cardinality notation on each relationship line -- levelminus4's
    # relationship check only verified both endpoint names co-occur, with
    # zero signal for cardinality (its own full F1 scoring category), and
    # levelminus4 landed near L0 parity (0.4249 vs 0.4252) but still well
    # below levelminus1's gold-access ceiling (0.5207). Still no gold
    # access. See text.rewrite.cardinality_loop. Rank -5 places it before
    # levelminus4 on plot x-axes.
    ("levelminus5", "description_level_minus_five.md", "L-5 (recall-density, +cardinality)", -5),
    # Same three signals as "levelminus5" (entities + relations +
    # cardinality), but combined via min(...) instead of average(...) --
    # levelminus5's naive average let compression pressure cut multiplicity
    # words first (cheapest 1/3-point loss vs. a full relationship's 2/3),
    # landing below levelminus4 on both global F1 (0.4233 vs 0.4249) and
    # cardinality F1 (0.344 vs 0.363). Scoring on the weakest signal removes
    # that exploit. Still no gold access. See text.rewrite.complete_loop.
    # Rank -6 places it before levelminus5 on plot x-axes.
    ("levelminus6", "description_level_minus_six.md", "L-6 (recall-density, min-gated)", -6),
    # Reuses "levelminus6"'s min-gated recall_density architecture (entity
    # naming + relationship co-location), but retargets the third signal at
    # parse_tree_depth reduction instead of cardinality-presence -- unlike
    # "four" (which minimizes parse_tree_depth alone via a single-metric
    # loop, guarded only by a token-count ratio), this scores flattening and
    # entity/relationship completeness together in one min-gated score, so
    # the rewrite can't trade one for the other. Output is ordinary
    # human-readable prose (short paragraphs of flat sentences), not the
    # recall-density family's compact "Classes:"/"Relationships:" notes
    # format. See text.rewrite.ptd_recall_loop. Still no gold access. Rank 6
    # places it after L4 (4) on plot x-axes, as a second parse_tree_depth-
    # targeted variant rather than "more complex than L4."
    ("six", "description_level_six.md", "L6 (ptd+recall, min-gated)", 6),
    # A generic, objective-free LLM rewrite: "make this clearer and more
    # concise," no recall-density target, no shape target, no PTD target --
    # a single-shot call (text.rewrite.naive_rewrite), not the iterative
    # feedback loop every other rewrite level uses. Isolates how much of
    # RaaS's gain comes from the recall-density objective specifically vs.
    # from any competent LLM rewrite of the specification. Rank -0.5 places
    # it between L0 (0) and L-1 (-1) on plot x-axes.
    ("levelnaive", "description_level_naive.md", "L-naive (generic rewrite)", -0.5),
    # A purely mechanical, non-LLM L0-style rewrite: spaCy noun-chunk
    # frequency for candidate classes, dependency-pattern matching (has/
    # contain + dobj, or appositives) for candidate attributes, same-sentence
    # co-occurrence for candidate relationships -- same Classes/Relationships
    # list structure as L0, zero LLM calls. Ablation for RQ2: isolates
    # whether L0's F1 gain requires an LLM's semantic disambiguation of the
    # specification, or whether the segmented-list structure alone accounts
    # for it. See scratchpad/mechanical_l0.py (repo-external, single-use).
    # Rank -0.25 places it between L-naive (-0.5) and L0 (0) on plot x-axes.
    ("mech", "description_level_mech.md", "L0-mech (non-LLM baseline)", -0.25),
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
