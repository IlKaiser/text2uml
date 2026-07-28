"""CLI: rewrite every dataset's description toward all three complexity levels.

For each ``dataset/<Name>/description.md`` this delegates to
``process_dataset``, which:
  1. scores the original on the corpus ``z_index`` scale,
  2. drives it toward ``level_zero`` (compact structural UML notes genre),
     ``level_one`` (simplest narrative), and ``level_two`` (mid-complexity
     narrative) via the shape-matching feedback loop (``rewrite_to_shape``),
  3. writes ``description_level_zero.md`` / ``description_level_one.md`` /
     ``description_level_two.md`` beside the original, and appends a row to
     ``text/output/rewrite_shape_summary.csv``.

Originals are never overwritten.

Examples:
    python -m text.rewrite.run                    # all datasets, levels zero/one/two
    python -m text.rewrite.run --dry-run          # score originals only
    python -m text.rewrite.run --datasets Sober AirTravel
    python -m text.rewrite.run --limit 5
    python -m text.rewrite.run --levels four      # level four only, all datasets
"""

from __future__ import annotations

import argparse
import functools
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from ..config import DEFAULT_CONFIG, TextConfig
from ..metrics import compute_all
from .config import DEFAULT_REWRITE_CONFIG, RewriteConfig
from .flat_prompts import build_flat_feedback, build_flat_user_prompt, flat_system_prompt
from .gold_entities import build_uml_parser, gold_entity_names
from .loop import rewrite_to_minimize, rewrite_to_shape
from .prompts import METRIC_GUIDANCE, build_shape_user_prompt, system_prompt
from .recall_loop import rewrite_to_maximize_recall_density
from .recall_prompts import (
    build_recall_density_feedback,
    build_recall_density_user_prompt,
    recall_density_system_prompt,
)
from .cardinality_loop import rewrite_to_maximize_cardinality_recall_density
from .complete_loop import rewrite_to_maximize_complete_recall_density
from .ptd_recall_loop import rewrite_to_maximize_ptd_recall_density
from .ptd_recall_prompts import build_ptd_recall_feedback, build_ptd_recall_user_prompt, ptd_recall_system_prompt
from .llm_source_entities import extract_llm_source_entities
from .recall_prompts import (
    build_cardinality_recall_feedback,
    build_combined_recall_feedback,
    build_complete_recall_feedback,
    cardinality_recall_system_prompt,
    combined_recall_system_prompt,
    complete_recall_system_prompt,
)
from .relation_entities import extract_entities_and_relations
from .relation_loop import rewrite_to_maximize_relation_recall_density
from .scorer import ComplexityReference, build_reference, score_text
from .source_entities import extract_source_entities
from .shape_targets import shape_ok
from .structural_prompts import (
    STRUCTURAL_METRIC_GUIDANCE,
    build_structural_user_prompt,
    structural_system_prompt,
)

logger = logging.getLogger("text.rewrite")

_LEVEL_TAGS: Tuple[str, ...] = ("zero", "one", "two")
_LEVEL_LABELS = {
    "one": "a simplified, easy-to-read narrative (level one: simplest)",
    "two": "a moderately simplified narrative (level two: mid-complexity)",
}


def _iter_description_paths(cfg: TextConfig, only: List[str] | None) -> List[Path]:
    paths = sorted(cfg.dataset_dir.glob(f"*/{cfg.description_filename}"))
    if only:
        wanted = set(only)
        paths = [p for p in paths if p.parent.name in wanted]
    return paths


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )


def _level_setup(
    cfg: RewriteConfig, tag: str
) -> Tuple[str, str, Callable[[str, str, Optional[str]], str], Dict[str, str]]:
    """(output level-name, system prompt, user-prompt fn, metric guidance) for one level tag."""
    if tag == "zero":
        return cfg.level_zero_name, structural_system_prompt(), build_structural_user_prompt, STRUCTURAL_METRIC_GUIDANCE
    out_name = cfg.level_one_name if tag == "one" else cfg.level_two_name
    user_fn = functools.partial(build_shape_user_prompt, level_label=_LEVEL_LABELS[tag])
    return out_name, system_prompt(), user_fn, METRIC_GUIDANCE


def process_dataset(
    name: str,
    description_path: Path,
    cfg: RewriteConfig,
    reference: ComplexityReference,
    tconf: TextConfig,
    client,
    levels: Tuple[str, ...] = _LEVEL_TAGS,
    force: bool = False,
) -> dict:
    """Regenerate the requested complexity levels for one dataset case.

    Returns a summary row: ``sub_folder_name``, ``actual_z``, and per
    processed level ``<tag>_reached`` / ``<tag>_iterations`` / ``<tag>_shape_ok``.
    """
    original = description_path.read_text(encoding="utf-8")
    base = score_text(original, reference, tconf)
    row: dict = {"sub_folder_name": name, "actual_z": round(base.z_index, 4)}

    for tag in levels:
        out_level_name, sprompt, user_fn, guidance = _level_setup(cfg, tag)
        out_path = cfg.output_path(description_path.parent, out_level_name)
        if out_path.is_file() and not force:
            logger.info("%s/%s: exists, skipping (pass force=True to regenerate)", name, tag)
            continue
        result = rewrite_to_shape(
            client=client, cfg=cfg, original=original, original_score=base,
            reference=reference, level_name=tag, l3_values=base.values,
            system_prompt=sprompt, user_prompt_fn=user_fn, metric_guidance=guidance,
        )
        if result.text == original:
            # No candidate ever improved on the untouched source (most often
            # a total API failure on the very first call, e.g. an outage or
            # exhausted credits) -- writing it out would silently replace a
            # level file with a byte-identical copy of description.md.
            # Mirrors generate.py's guard against clobbering a good result
            # with a failed one: leave whatever was already on disk alone.
            logger.error(
                "%s/%s: no candidate ever improved on the source description "
                "(likely a total generation failure); leaving %s untouched",
                name, tag, out_path.name,
            )
            row[f"{tag}_reached"] = False
            row[f"{tag}_iterations"] = result.iterations
            row[f"{tag}_shape_ok"] = False
            continue
        out_path.write_text(result.text.rstrip() + "\n", encoding="utf-8")
        row[f"{tag}_reached"] = result.reached
        row[f"{tag}_iterations"] = result.iterations
        row[f"{tag}_shape_ok"] = shape_ok(result.shape_checks)
        logger.info("%s/%s: reached=%s in %d iter(s)", name, tag, result.reached, result.iterations)

    return row


def process_dataset_level_four(
    name: str,
    description_path: Path,
    cfg: RewriteConfig,
    reference: ComplexityReference,
    tconf: TextConfig,
    client,
    force: bool = False,
) -> dict:
    """Generate ``description_level_four.md``: minimize parse_tree_depth vs.
    the real spec (description.md), preserving meaning. Unlike
    ``process_dataset``'s shape-matching levels, this targets one metric
    directly rather than a rank+band shape (see ``text.rewrite.flat_prompts``
    for why: parse_tree_depth is the strongest single F1 predictor at L3 in
    this corpus, so this tests whether directly minimizing it helps further).

    Returns a summary row: ``sub_folder_name``, ``source_value``,
    ``best_value``, ``improved``, ``iterations``.
    """
    out_path = cfg.output_path(description_path.parent, cfg.level_four_name)
    if out_path.is_file() and not force:
        logger.info("%s/four: exists, skipping (pass force=True to regenerate)", name)
        return {"sub_folder_name": name}

    original = description_path.read_text(encoding="utf-8")
    base = score_text(original, reference, tconf)
    result = rewrite_to_minimize(
        client=client, cfg=cfg, original=original, original_score=base,
        reference=reference, metric_name="parse_tree_depth",
        system_prompt=flat_system_prompt(), user_prompt_fn=build_flat_user_prompt,
        feedback_fn=build_flat_feedback,
    )
    if result.text == original:
        # Mirrors process_dataset's guard: never write a level file that
        # ends up byte-identical to the source (total generation failure).
        logger.error(
            "%s/four: no candidate ever improved on the source description "
            "(likely a total generation failure); leaving %s untouched",
            name, out_path.name,
        )
        return {"sub_folder_name": name, "four_improved": False, "four_iterations": result.iterations}

    out_path.write_text(result.text.rstrip() + "\n", encoding="utf-8")
    logger.info(
        "%s/four: parse_tree_depth %.2f -> %.2f (improved=%s) in %d iter(s)",
        name, result.source_value, result.best_value, result.improved, result.iterations,
    )
    return {
        "sub_folder_name": name,
        "source_value": round(result.source_value, 4),
        "best_value": round(result.best_value, 4),
        "improved": result.improved,
        "iterations": result.iterations,
    }


def process_dataset_level_minus_one(
    name: str,
    description_path: Path,
    cfg: RewriteConfig,
    tconf: TextConfig,
    client,
    eval_mod,
    uml_parser,
    force: bool = False,
) -> dict:
    """Generate ``description_level_minus_one.md``: maximize recall_density
    (gold entity-name recall per token) instead of matching a linguistic
    shape. See ``text.rewrite.recall_prompts`` for why: this project's own
    per-case correlation analysis found recall_density predicts downstream
    UML-generation F1 far more robustly (Pearson r=0.56-0.60, p<0.0001,
    n=213 case/rewriter pairs) than any of level zero's three shape metrics
    (mdd_ratio: r=-0.12, not significant at the per-case level) -- this
    level tests the ceiling of optimizing for the metric that actually
    predicts F1, rather than one merely correlated with the target genre.

    Skips (with a warning) any case missing a gold PlantUML file -- unlike
    every other level here, this one needs it as the rewrite target.

    Returns a summary row: ``sub_folder_name``, ``source_recall``,
    ``source_density``, ``best_recall``, ``best_density``, ``improved``,
    ``iterations``.
    """
    out_path = cfg.output_path(description_path.parent, cfg.level_minus_one_name)
    if out_path.is_file() and not force:
        logger.info("%s/minus_one: exists, skipping (pass force=True to regenerate)", name)
        return {"sub_folder_name": name}

    gold_path = description_path.parent / "plantuml.txt"
    if not gold_path.is_file():
        logger.warning("%s/minus_one: no gold PlantUML at %s; skipping", name, gold_path)
        return {"sub_folder_name": name}

    gold_names = gold_entity_names(gold_path, eval_mod, uml_parser)
    original = description_path.read_text(encoding="utf-8")

    def n_tokens(text: str) -> int:
        return compute_all(text, sample_id=name, model_name=tconf.spacy_model).n_tokens

    result = rewrite_to_maximize_recall_density(
        client=client, cfg=cfg, original=original, gold_names=gold_names,
        n_tokens_fn=n_tokens, system_prompt=recall_density_system_prompt(),
        user_prompt_fn=build_recall_density_user_prompt,
        feedback_fn=build_recall_density_feedback,
    )
    if result.text == original:
        logger.error(
            "%s/minus_one: no candidate ever improved on the source description "
            "(likely a total generation failure); leaving %s untouched",
            name, out_path.name,
        )
        return {"sub_folder_name": name, "minus_one_improved": False, "minus_one_iterations": result.iterations}

    out_path.write_text(result.text.rstrip() + "\n", encoding="utf-8")
    logger.info(
        "%s/minus_one: recall_density %.3f -> %.3f (recall %.0f%%->%.0f%%, improved=%s) in %d iter(s)",
        name, result.source_density, result.best_density,
        result.source_recall * 100, result.best_recall * 100, result.improved, result.iterations,
    )
    return {
        "sub_folder_name": name,
        "source_recall": round(result.source_recall, 4),
        "source_density": round(result.source_density, 4),
        "best_recall": round(result.best_recall, 4),
        "best_density": round(result.best_density, 4),
        "improved": result.improved,
        "iterations": result.iterations,
    }


def process_dataset_level_minus_two(
    name: str,
    description_path: Path,
    cfg: RewriteConfig,
    tconf: TextConfig,
    client,
    force: bool = False,
) -> dict:
    """Generate ``description_level_minus_two.md``: maximize recall_density
    like level minus-one, but against entity names extracted from the
    *source description itself* (``text.rewrite.source_entities``) instead
    of the gold PlantUML. Level minus-one is leaky -- it optimizes against
    the same file ``evaluate.py`` scores against, so it's an upper-bound/
    ceiling experiment, not a deployable method (a real pipeline never has
    the gold UML in advance; that's what generation is trying to produce).
    This level tests how much of level minus-one's F1 gain survives without
    that answer-key access. Reuses the same loop/prompts as level minus-one
    (``text.rewrite.recall_loop``/``recall_prompts``) -- only the entity
    source differs.

    Summary-row columns are prefixed ``minus_two_`` (level minus-one's are
    unprefixed) so the two levels' rows never collide if ever regenerated
    in the same invocation.

    Returns a summary row: ``sub_folder_name``, ``minus_two_source_recall``,
    ``minus_two_source_density``, ``minus_two_best_recall``,
    ``minus_two_best_density``, ``minus_two_improved``,
    ``minus_two_iterations``.
    """
    out_path = cfg.output_path(description_path.parent, cfg.level_minus_two_name)
    if out_path.is_file() and not force:
        logger.info("%s/minus_two: exists, skipping (pass force=True to regenerate)", name)
        return {"sub_folder_name": name}

    original = description_path.read_text(encoding="utf-8")
    target_names = extract_source_entities(original, tconf.spacy_model)

    def n_tokens(text: str) -> int:
        return compute_all(text, sample_id=name, model_name=tconf.spacy_model).n_tokens

    result = rewrite_to_maximize_recall_density(
        client=client, cfg=cfg, original=original, gold_names=target_names,
        n_tokens_fn=n_tokens, system_prompt=recall_density_system_prompt(),
        user_prompt_fn=build_recall_density_user_prompt,
        feedback_fn=build_recall_density_feedback,
    )
    if result.text == original:
        logger.error(
            "%s/minus_two: no candidate ever improved on the source description "
            "(likely a total generation failure); leaving %s untouched",
            name, out_path.name,
        )
        return {"sub_folder_name": name, "minus_two_improved": False, "minus_two_iterations": result.iterations}

    out_path.write_text(result.text.rstrip() + "\n", encoding="utf-8")
    logger.info(
        "%s/minus_two: recall_density %.3f -> %.3f (recall %.0f%%->%.0f%%, improved=%s) in %d iter(s)",
        name, result.source_density, result.best_density,
        result.source_recall * 100, result.best_recall * 100, result.improved, result.iterations,
    )
    return {
        "sub_folder_name": name,
        "minus_two_source_recall": round(result.source_recall, 4),
        "minus_two_source_density": round(result.source_density, 4),
        "minus_two_best_recall": round(result.best_recall, 4),
        "minus_two_best_density": round(result.best_density, 4),
        "minus_two_improved": result.improved,
        "minus_two_iterations": result.iterations,
    }


def process_dataset_level_minus_three(
    name: str,
    description_path: Path,
    cfg: RewriteConfig,
    tconf: TextConfig,
    client,
    force: bool = False,
) -> dict:
    """Generate ``description_level_minus_three.md``: same recall_density
    objective as level minus-two, but the target entity list comes from one
    LLM extraction call over the source description (``text.rewrite.
    llm_source_entities``) instead of noun-chunk frequency counting.

    Level minus-two's frequency heuristic can't distinguish a real class/
    attribute name from any other repeated word and scored *below* the L0
    baseline (F1 0.405 vs 0.425) -- this tests whether leaning on the
    model's own UML-modeling judgment (still with zero gold access) closes
    that gap. Costs one extra API call per case (the extraction pass)
    before the usual recall_density loop.

    Summary-row columns are prefixed ``minus_three_`` so this level's rows
    never collide with level minus-one's or minus-two's if ever regenerated
    together.

    Returns a summary row: ``sub_folder_name``, ``minus_three_source_recall``,
    ``minus_three_source_density``, ``minus_three_best_recall``,
    ``minus_three_best_density``, ``minus_three_improved``,
    ``minus_three_iterations``.
    """
    out_path = cfg.output_path(description_path.parent, cfg.level_minus_three_name)
    if out_path.is_file() and not force:
        logger.info("%s/minus_three: exists, skipping (pass force=True to regenerate)", name)
        return {"sub_folder_name": name}

    original = description_path.read_text(encoding="utf-8")
    try:
        target_names = extract_llm_source_entities(client, cfg, original)
    except Exception as exc:  # noqa: BLE001 - one bad extraction call must not kill the whole batch
        logger.error("%s/minus_three: entity extraction call failed (%s); skipping this case", name, exc)
        return {"sub_folder_name": name}

    def n_tokens(text: str) -> int:
        return compute_all(text, sample_id=name, model_name=tconf.spacy_model).n_tokens

    result = rewrite_to_maximize_recall_density(
        client=client, cfg=cfg, original=original, gold_names=target_names,
        n_tokens_fn=n_tokens, system_prompt=recall_density_system_prompt(),
        user_prompt_fn=build_recall_density_user_prompt,
        feedback_fn=build_recall_density_feedback,
    )
    if result.text == original:
        logger.error(
            "%s/minus_three: no candidate ever improved on the source description "
            "(likely a total generation failure); leaving %s untouched",
            name, out_path.name,
        )
        return {"sub_folder_name": name, "minus_three_improved": False, "minus_three_iterations": result.iterations}

    out_path.write_text(result.text.rstrip() + "\n", encoding="utf-8")
    logger.info(
        "%s/minus_three: recall_density %.3f -> %.3f (recall %.0f%%->%.0f%%, improved=%s) in %d iter(s)",
        name, result.source_density, result.best_density,
        result.source_recall * 100, result.best_recall * 100, result.improved, result.iterations,
    )
    return {
        "sub_folder_name": name,
        "minus_three_source_recall": round(result.source_recall, 4),
        "minus_three_source_density": round(result.source_density, 4),
        "minus_three_best_recall": round(result.best_recall, 4),
        "minus_three_best_density": round(result.best_density, 4),
        "minus_three_improved": result.improved,
        "minus_three_iterations": result.iterations,
    }


def process_dataset_level_minus_four(
    name: str,
    description_path: Path,
    cfg: RewriteConfig,
    tconf: TextConfig,
    client,
    force: bool = False,
) -> dict:
    """Generate ``description_level_minus_four.md``: recall_density extended
    to credit relationship pairs, not just entity names (``text.rewrite.
    relation_loop``). Level minus-two and minus-three (both gold-free)
    plateaued ~5% *below* the L0 baseline despite ~100% entity-name recall
    of their own target list -- entity recall alone gives the rewrite no
    signal to protect relationship/cardinality facts, and compression
    pressure seems to trade those away. This tests whether also crediting
    "both endpoints of a relationship stated on the same line" closes that
    gap while staying gold-free (one LLM extraction call for entities AND
    relationship pairs, ``text.rewrite.relation_entities``).

    Summary-row columns are prefixed ``minus_four_``.

    Returns a summary row: ``sub_folder_name``,
    ``minus_four_source_entity_recall``, ``minus_four_source_relation_recall``,
    ``minus_four_source_density``, ``minus_four_best_entity_recall``,
    ``minus_four_best_relation_recall``, ``minus_four_best_density``,
    ``minus_four_improved``, ``minus_four_iterations``.
    """
    out_path = cfg.output_path(description_path.parent, cfg.level_minus_four_name)
    if out_path.is_file() and not force:
        logger.info("%s/minus_four: exists, skipping (pass force=True to regenerate)", name)
        return {"sub_folder_name": name}

    original = description_path.read_text(encoding="utf-8")
    try:
        entity_names, relation_pairs = extract_entities_and_relations(client, cfg, original)
    except Exception as exc:  # noqa: BLE001 - one bad extraction call must not kill the whole batch
        logger.error("%s/minus_four: entity/relation extraction call failed (%s); skipping this case", name, exc)
        return {"sub_folder_name": name}

    def n_tokens(text: str) -> int:
        return compute_all(text, sample_id=name, model_name=tconf.spacy_model).n_tokens

    result = rewrite_to_maximize_relation_recall_density(
        client=client, cfg=cfg, original=original,
        entity_names=entity_names, relation_pairs=relation_pairs,
        n_tokens_fn=n_tokens, system_prompt=combined_recall_system_prompt(),
        user_prompt_fn=build_recall_density_user_prompt,
        feedback_fn=build_combined_recall_feedback,
    )
    if result.text == original:
        logger.error(
            "%s/minus_four: no candidate ever improved on the source description "
            "(likely a total generation failure); leaving %s untouched",
            name, out_path.name,
        )
        return {"sub_folder_name": name, "minus_four_improved": False, "minus_four_iterations": result.iterations}

    out_path.write_text(result.text.rstrip() + "\n", encoding="utf-8")
    logger.info(
        "%s/minus_four: density %.3f -> %.3f (entities %.0f%%->%.0f%%, relations %.0f%%->%.0f%%, "
        "improved=%s) in %d iter(s)",
        name, result.source_density, result.best_density,
        result.source_entity_recall * 100, result.best_entity_recall * 100,
        result.source_relation_recall * 100, result.best_relation_recall * 100,
        result.improved, result.iterations,
    )
    return {
        "sub_folder_name": name,
        "minus_four_source_entity_recall": round(result.source_entity_recall, 4),
        "minus_four_source_relation_recall": round(result.source_relation_recall, 4),
        "minus_four_source_density": round(result.source_density, 4),
        "minus_four_best_entity_recall": round(result.best_entity_recall, 4),
        "minus_four_best_relation_recall": round(result.best_relation_recall, 4),
        "minus_four_best_density": round(result.best_density, 4),
        "minus_four_improved": result.improved,
        "minus_four_iterations": result.iterations,
    }


def process_dataset_level_minus_five(
    name: str,
    description_path: Path,
    cfg: RewriteConfig,
    tconf: TextConfig,
    client,
    force: bool = False,
) -> dict:
    """Generate ``description_level_minus_five.md``: level minus-four's
    entity+relation recall_density extended to also credit multiplicity/
    cardinality notation on each relationship line (``text.rewrite.
    cardinality_loop``). Level minus-four's relationship check only
    verified both endpoint names co-occur, with zero signal for
    cardinality -- its own full F1 scoring category, and likely most of
    the remaining gap to level minus-one's gold-access ceiling (0.52 vs.
    L-4's 0.42). Reuses the exact same extraction as level minus-four
    (``text.rewrite.relation_entities.extract_entities_and_relations`` --
    no new extraction needed, the cardinality check is presence-based, not
    a comparison against an extracted target string). Still no gold access.

    Summary-row columns are prefixed ``minus_five_``.

    Returns a summary row: ``sub_folder_name``,
    ``minus_five_source_entity_recall``, ``minus_five_source_relation_recall``,
    ``minus_five_source_cardinality_recall``, ``minus_five_source_density``,
    ``minus_five_best_entity_recall``, ``minus_five_best_relation_recall``,
    ``minus_five_best_cardinality_recall``, ``minus_five_best_density``,
    ``minus_five_improved``, ``minus_five_iterations``.
    """
    out_path = cfg.output_path(description_path.parent, cfg.level_minus_five_name)
    if out_path.is_file() and not force:
        logger.info("%s/minus_five: exists, skipping (pass force=True to regenerate)", name)
        return {"sub_folder_name": name}

    original = description_path.read_text(encoding="utf-8")
    try:
        entity_names, relation_pairs = extract_entities_and_relations(client, cfg, original)
    except Exception as exc:  # noqa: BLE001 - one bad extraction call must not kill the whole batch
        logger.error("%s/minus_five: entity/relation extraction call failed (%s); skipping this case", name, exc)
        return {"sub_folder_name": name}

    def n_tokens(text: str) -> int:
        return compute_all(text, sample_id=name, model_name=tconf.spacy_model).n_tokens

    result = rewrite_to_maximize_cardinality_recall_density(
        client=client, cfg=cfg, original=original,
        entity_names=entity_names, relation_pairs=relation_pairs,
        n_tokens_fn=n_tokens, system_prompt=cardinality_recall_system_prompt(),
        user_prompt_fn=build_recall_density_user_prompt,
        feedback_fn=build_cardinality_recall_feedback,
    )
    if result.text == original:
        logger.error(
            "%s/minus_five: no candidate ever improved on the source description "
            "(likely a total generation failure); leaving %s untouched",
            name, out_path.name,
        )
        return {"sub_folder_name": name, "minus_five_improved": False, "minus_five_iterations": result.iterations}

    out_path.write_text(result.text.rstrip() + "\n", encoding="utf-8")
    logger.info(
        "%s/minus_five: density %.3f -> %.3f (entities %.0f%%->%.0f%%, relations %.0f%%->%.0f%%, "
        "cardinality %.0f%%->%.0f%%, improved=%s) in %d iter(s)",
        name, result.source_density, result.best_density,
        result.source_entity_recall * 100, result.best_entity_recall * 100,
        result.source_relation_recall * 100, result.best_relation_recall * 100,
        result.source_cardinality_recall * 100, result.best_cardinality_recall * 100,
        result.improved, result.iterations,
    )
    return {
        "sub_folder_name": name,
        "minus_five_source_entity_recall": round(result.source_entity_recall, 4),
        "minus_five_source_relation_recall": round(result.source_relation_recall, 4),
        "minus_five_source_cardinality_recall": round(result.source_cardinality_recall, 4),
        "minus_five_source_density": round(result.source_density, 4),
        "minus_five_best_entity_recall": round(result.best_entity_recall, 4),
        "minus_five_best_relation_recall": round(result.best_relation_recall, 4),
        "minus_five_best_cardinality_recall": round(result.best_cardinality_recall, 4),
        "minus_five_best_density": round(result.best_density, 4),
        "minus_five_improved": result.improved,
        "minus_five_iterations": result.iterations,
    }


def process_dataset_level_minus_six(
    name: str,
    description_path: Path,
    cfg: RewriteConfig,
    tconf: TextConfig,
    client,
    force: bool = False,
) -> dict:
    """Generate ``description_level_minus_six.md``: level minus-five's
    entity + relation + cardinality recall_density, but combined via
    min(...) instead of average(...) (``text.rewrite.complete_loop``).
    Level minus-five's naive average let compression pressure cut
    multiplicity words first (cheapest 1/3-point loss vs. a full
    relationship's 2/3), landing *below* level minus-four on both global F1
    (0.4233 vs 0.4249) and cardinality F1 (0.344 vs 0.363). Scoring on the
    weakest of the three signals removes that exploit: there is no way to
    raise the score except by improving whichever signal is currently
    lowest. Reuses the exact same extraction as level minus-four/minus-five
    (``text.rewrite.relation_entities.extract_entities_and_relations``).
    Still no gold access.

    Summary-row columns are prefixed ``minus_six_``.

    Returns a summary row: ``sub_folder_name``,
    ``minus_six_source_entity_recall``, ``minus_six_source_relation_recall``,
    ``minus_six_source_cardinality_recall``, ``minus_six_source_density``,
    ``minus_six_best_entity_recall``, ``minus_six_best_relation_recall``,
    ``minus_six_best_cardinality_recall``, ``minus_six_best_density``,
    ``minus_six_improved``, ``minus_six_iterations``.
    """
    out_path = cfg.output_path(description_path.parent, cfg.level_minus_six_name)
    if out_path.is_file() and not force:
        logger.info("%s/minus_six: exists, skipping (pass force=True to regenerate)", name)
        return {"sub_folder_name": name}

    original = description_path.read_text(encoding="utf-8")
    try:
        entity_names, relation_pairs = extract_entities_and_relations(client, cfg, original)
    except Exception as exc:  # noqa: BLE001 - one bad extraction call must not kill the whole batch
        logger.error("%s/minus_six: entity/relation extraction call failed (%s); skipping this case", name, exc)
        return {"sub_folder_name": name}

    def n_tokens(text: str) -> int:
        return compute_all(text, sample_id=name, model_name=tconf.spacy_model).n_tokens

    result = rewrite_to_maximize_complete_recall_density(
        client=client, cfg=cfg, original=original,
        entity_names=entity_names, relation_pairs=relation_pairs,
        n_tokens_fn=n_tokens, system_prompt=complete_recall_system_prompt(),
        user_prompt_fn=build_recall_density_user_prompt,
        feedback_fn=build_complete_recall_feedback,
    )
    if result.text == original:
        logger.error(
            "%s/minus_six: no candidate ever improved on the source description "
            "(likely a total generation failure); leaving %s untouched",
            name, out_path.name,
        )
        return {"sub_folder_name": name, "minus_six_improved": False, "minus_six_iterations": result.iterations}

    out_path.write_text(result.text.rstrip() + "\n", encoding="utf-8")
    logger.info(
        "%s/minus_six: density %.3f -> %.3f (entities %.0f%%->%.0f%%, relations %.0f%%->%.0f%%, "
        "cardinality %.0f%%->%.0f%%, improved=%s) in %d iter(s)",
        name, result.source_density, result.best_density,
        result.source_entity_recall * 100, result.best_entity_recall * 100,
        result.source_relation_recall * 100, result.best_relation_recall * 100,
        result.source_cardinality_recall * 100, result.best_cardinality_recall * 100,
        result.improved, result.iterations,
    )
    return {
        "sub_folder_name": name,
        "minus_six_source_entity_recall": round(result.source_entity_recall, 4),
        "minus_six_source_relation_recall": round(result.source_relation_recall, 4),
        "minus_six_source_cardinality_recall": round(result.source_cardinality_recall, 4),
        "minus_six_source_density": round(result.source_density, 4),
        "minus_six_best_entity_recall": round(result.best_entity_recall, 4),
        "minus_six_best_relation_recall": round(result.best_relation_recall, 4),
        "minus_six_best_cardinality_recall": round(result.best_cardinality_recall, 4),
        "minus_six_best_density": round(result.best_density, 4),
        "minus_six_improved": result.improved,
        "minus_six_iterations": result.iterations,
    }


def process_dataset_level_six(
    name: str,
    description_path: Path,
    cfg: RewriteConfig,
    tconf: TextConfig,
    client,
    force: bool = False,
) -> dict:
    """Generate ``description_level_six.md``: level minus-four/five/six's
    entity + relation recall_density architecture, retargeted at
    parse_tree_depth instead of cardinality (``text.rewrite.
    ptd_recall_loop``). Level four already minimizes parse_tree_depth
    directly (``text.rewrite.flat_prompts`` + ``loop.rewrite_to_minimize``),
    but its only content guard is a token-count ratio -- nothing scores
    whether specific entities/relationships survived the flattening. This
    level instead scores parse-tree-depth reduction alongside entity-name
    and relationship-pair recall in a single min-gated density (level
    minus-six's min-gating, applied here instead of to cardinality), so the
    rewrite cannot trade completeness for flatness or vice versa. Output
    reads as ordinary prose (short paragraphs of flat sentences), not the
    recall-density family's compact 'Classes:'/'Relationships:' notes
    format -- see ``text.rewrite.ptd_recall_prompts``.

    Reuses the same extraction as level minus-four/five/six
    (``text.rewrite.relation_entities.extract_entities_and_relations``). No
    gold access.

    Returns a summary row: ``sub_folder_name``, ``six_source_entity_recall``,
    ``six_source_relation_recall``, ``six_source_ptd``,
    ``six_source_density``, ``six_best_entity_recall``,
    ``six_best_relation_recall``, ``six_best_ptd``, ``six_best_density``,
    ``six_improved``, ``six_iterations``.
    """
    out_path = cfg.output_path(description_path.parent, cfg.level_six_name)
    if out_path.is_file() and not force:
        logger.info("%s/six: exists, skipping (pass force=True to regenerate)", name)
        return {"sub_folder_name": name}

    original = description_path.read_text(encoding="utf-8")
    try:
        entity_names, relation_pairs = extract_entities_and_relations(client, cfg, original)
    except Exception as exc:  # noqa: BLE001 - one bad extraction call must not kill the whole batch
        logger.error("%s/six: entity/relation extraction call failed (%s); skipping this case", name, exc)
        return {"sub_folder_name": name}

    def metrics_fn(text: str) -> Tuple[int, float]:
        result = compute_all(text, sample_id=name, model_name=tconf.spacy_model)
        return result.n_tokens, result.values.get("parse_tree_depth", float("nan"))

    result = rewrite_to_maximize_ptd_recall_density(
        client=client, cfg=cfg, original=original,
        entity_names=entity_names, relation_pairs=relation_pairs,
        metrics_fn=metrics_fn, system_prompt=ptd_recall_system_prompt(),
        user_prompt_fn=build_ptd_recall_user_prompt,
        feedback_fn=build_ptd_recall_feedback,
    )
    if result.text == original:
        logger.error(
            "%s/six: no candidate ever improved on the source description "
            "(likely a total generation failure); leaving %s untouched",
            name, out_path.name,
        )
        return {"sub_folder_name": name, "six_improved": False, "six_iterations": result.iterations}

    out_path.write_text(result.text.rstrip() + "\n", encoding="utf-8")
    logger.info(
        "%s/six: density %.3f -> %.3f (entities %.0f%%->%.0f%%, relations %.0f%%->%.0f%%, "
        "parse_tree_depth %.2f->%.2f, improved=%s) in %d iter(s)",
        name, result.source_density, result.best_density,
        result.source_entity_recall * 100, result.best_entity_recall * 100,
        result.source_relation_recall * 100, result.best_relation_recall * 100,
        result.source_ptd, result.best_ptd,
        result.improved, result.iterations,
    )
    return {
        "sub_folder_name": name,
        "six_source_entity_recall": round(result.source_entity_recall, 4),
        "six_source_relation_recall": round(result.source_relation_recall, 4),
        "six_source_ptd": round(result.source_ptd, 4),
        "six_source_density": round(result.source_density, 4),
        "six_best_entity_recall": round(result.best_entity_recall, 4),
        "six_best_relation_recall": round(result.best_relation_recall, 4),
        "six_best_ptd": round(result.best_ptd, 4),
        "six_best_density": round(result.best_density, 4),
        "six_improved": result.improved,
        "six_iterations": result.iterations,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", help="Only these dataset folder names.")
    parser.add_argument("--limit", type=int, help="Process at most N datasets.")
    parser.add_argument(
        "--levels", nargs="*",
        choices=[
            "zero", "one", "two", "four", "six",
            "minus_one", "minus_two", "minus_three", "minus_four", "minus_five", "minus_six",
        ],
        help="Which levels to (re)generate (default: zero one two). 'four' "
        "(direct parse_tree_depth minimization) is dispatched to "
        "process_dataset_level_four instead of the shape-matching loop. "
        "'minus_one' (direct recall_density maximization against the gold "
        "PlantUML) is dispatched to process_dataset_level_minus_one -- "
        "skips any case missing a gold file. 'minus_two' is the same "
        "recall_density objective but against entity names extracted from "
        "the source description via noun-chunk frequency -- no gold file "
        "needed. 'minus_three' is the same objective again, but the entity "
        "list comes from one LLM extraction call over the source "
        "description instead of frequency counting -- also no gold access. "
        "'minus_four' extends minus_three's objective to also credit "
        "relationship pairs (both endpoint classes stated on the same "
        "line), not just entity names -- still no gold access. 'minus_five' "
        "extends minus_four further to also credit multiplicity/cardinality "
        "notation on each relationship line -- still no gold access. "
        "'minus_six' scores the same three signals as minus_five but "
        "combines them via min(...) instead of average(...), so cardinality "
        "can no longer be cheaply traded away under compression pressure -- "
        "still no gold access. 'six' reuses minus_six's min-gated "
        "entity+relation recall_density architecture but retargets the "
        "third signal at parse_tree_depth reduction instead of cardinality "
        "-- unlike 'four' (which minimizes parse_tree_depth alone with only "
        "a token-ratio content guard), this scores flattening and "
        "entity/relationship completeness together so neither can be "
        "traded for the other; output is ordinary prose, not compact "
        "notes -- still no gold access.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate level files that already exist.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Score originals and print targets; make no API calls and write nothing.",
    )
    parser.add_argument(
        "--provider", choices=["anthropic", "openai", "openrouter", "ollama"], default="anthropic",
        help="LLM backend for the rewrite calls (default: anthropic).",
    )
    parser.add_argument("--model", help="Override the model id (Claude id for anthropic, OpenAI id for openai, OpenRouter slug for openrouter, tag for ollama).")
    parser.add_argument("--effort", choices=["low", "medium", "high", "max"])
    parser.add_argument(
        "--temperature", type=float,
        help="Sampling temperature for the openai/openrouter/ollama providers "
        "(default: 0.0). Some newer OpenAI models (e.g. gpt-5.5) only support "
        "the default temperature=1 and reject 0 with a 400 error -- pass "
        "--temperature 1 for those.",
    )
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument(
        "--verify-meaning", action="store_true",
        help="Audit semantic equivalence with a second Claude call before "
        "accepting a complexity-matching rewrite (extra cost, anthropic only).",
    )
    parser.add_argument(
        "--output-suffix", default="",
        help="Appended to the level output name before writing (e.g. "
        "'_gemma4e4bmlx' -> description_level_zero_gemma4e4bmlx.md instead "
        "of description_level_zero.md). Use this to compare an alternate "
        "provider/model's rewrite without ever overwriting the existing "
        "level file.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    tconf = DEFAULT_CONFIG
    default_model = (
        DEFAULT_REWRITE_CONFIG.model if args.provider == "anthropic" else None
    )
    if args.provider != "anthropic" and not args.model:
        logger.error("--model is required when --provider is not anthropic.")
        return 2
    cfg = RewriteConfig(
        provider=args.provider,
        model=args.model or default_model,
        effort=args.effort or DEFAULT_REWRITE_CONFIG.effort,
        max_iterations=args.max_iterations or DEFAULT_REWRITE_CONFIG.max_iterations,
        tolerance=args.tolerance if args.tolerance is not None else DEFAULT_REWRITE_CONFIG.tolerance,
        temperature=args.temperature if args.temperature is not None else DEFAULT_REWRITE_CONFIG.temperature,
        verify_meaning=args.verify_meaning,
        level_zero_name=DEFAULT_REWRITE_CONFIG.level_zero_name + args.output_suffix,
        level_one_name=DEFAULT_REWRITE_CONFIG.level_one_name + args.output_suffix,
        level_two_name=DEFAULT_REWRITE_CONFIG.level_two_name + args.output_suffix,
        level_four_name=DEFAULT_REWRITE_CONFIG.level_four_name + args.output_suffix,
        level_six_name=DEFAULT_REWRITE_CONFIG.level_six_name + args.output_suffix,
        level_minus_one_name=DEFAULT_REWRITE_CONFIG.level_minus_one_name + args.output_suffix,
        level_minus_two_name=DEFAULT_REWRITE_CONFIG.level_minus_two_name + args.output_suffix,
        level_minus_three_name=DEFAULT_REWRITE_CONFIG.level_minus_three_name + args.output_suffix,
        level_minus_four_name=DEFAULT_REWRITE_CONFIG.level_minus_four_name + args.output_suffix,
        level_minus_five_name=DEFAULT_REWRITE_CONFIG.level_minus_five_name + args.output_suffix,
        level_minus_six_name=DEFAULT_REWRITE_CONFIG.level_minus_six_name + args.output_suffix,
    )

    try:
        reference = build_reference(tconf)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    paths = _iter_description_paths(tconf, args.datasets)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        logger.error("No description files found under %s", tconf.dataset_dir)
        return 1

    client = None
    if not args.dry_run:
        from .client import make_client

        client = make_client(cfg.provider)

    levels_arg = tuple(args.levels) if args.levels else _LEVEL_TAGS
    shape_levels = tuple(
        tag for tag in levels_arg
        if tag not in (
            "four", "six", "minus_one", "minus_two", "minus_three", "minus_four", "minus_five", "minus_six",
        )
    )
    want_four = "four" in levels_arg
    want_six = "six" in levels_arg
    want_minus_one = "minus_one" in levels_arg
    want_minus_two = "minus_two" in levels_arg
    want_minus_three = "minus_three" in levels_arg
    want_minus_four = "minus_four" in levels_arg
    want_minus_five = "minus_five" in levels_arg
    want_minus_six = "minus_six" in levels_arg

    eval_mod = uml_parser = None
    if want_minus_one and not args.dry_run:
        eval_mod, uml_parser = build_uml_parser(tconf)

    rows = []
    for path in paths:
        name = path.parent.name
        try:
            base = score_text(path.read_text(encoding="utf-8"), reference, tconf)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", name, exc)
            continue

        if args.dry_run:
            logger.info("%s: actual z_index=%.2f (dry run, no rewrite)", name, base.z_index)
            continue

        row: dict = {}
        if shape_levels:
            row.update(process_dataset(name, path, cfg, reference, tconf, client, levels=shape_levels, force=args.force))
        if want_four:
            four_row = process_dataset_level_four(name, path, cfg, reference, tconf, client, force=args.force)
            row.update(four_row)
        if want_six:
            six_row = process_dataset_level_six(name, path, cfg, tconf, client, force=args.force)
            row.update(six_row)
        if want_minus_one:
            minus_one_row = process_dataset_level_minus_one(
                name, path, cfg, tconf, client, eval_mod, uml_parser, force=args.force
            )
            row.update(minus_one_row)
        if want_minus_two:
            minus_two_row = process_dataset_level_minus_two(name, path, cfg, tconf, client, force=args.force)
            row.update(minus_two_row)
        if want_minus_three:
            minus_three_row = process_dataset_level_minus_three(name, path, cfg, tconf, client, force=args.force)
            row.update(minus_three_row)
        if want_minus_four:
            minus_four_row = process_dataset_level_minus_four(name, path, cfg, tconf, client, force=args.force)
            row.update(minus_four_row)
        if want_minus_five:
            minus_five_row = process_dataset_level_minus_five(name, path, cfg, tconf, client, force=args.force)
            row.update(minus_five_row)
        if want_minus_six:
            minus_six_row = process_dataset_level_minus_six(name, path, cfg, tconf, client, force=args.force)
            row.update(minus_six_row)
        row.setdefault("sub_folder_name", name)
        rows.append(row)

    if rows and not args.dry_run:
        out = tconf.output_dir / "rewrite_shape_summary.csv"
        tconf.output_dir.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame(rows)
        # Merge on sub_folder_name instead of overwriting: this is routinely
        # invoked with --datasets scoped to one or a few cases, and an
        # unconditional overwrite would truncate every other case's row out
        # of the corpus-wide summary. A row-level replace isn't enough,
        # though: each level (zero/four/minus_one/minus_two/...) contributes
        # its own prefixed columns to the SAME row per case, so re-running
        # one level for a case that another level already touched must
        # merge column-wise -- dropping and replacing the whole row (the
        # previous approach) silently wiped every other level's columns for
        # that case (discovered when a minus_three run zeroed out an
        # already-computed minus_two column for all 44 shared cases).
        # combine_first prefers this run's fresh values where present and
        # keeps every existing column/row it doesn't touch.
        if out.is_file():
            existing = pd.read_csv(out).set_index("sub_folder_name")
            new_df = new_df.set_index("sub_folder_name").combine_first(existing).reset_index()
        new_df.to_csv(out, index=False)
        logger.info("Wrote shape summary for %d datasets to %s", len(rows), out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
