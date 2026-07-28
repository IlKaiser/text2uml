"""Evaluate level-tagged two-shot results against the gold PlantUML.

Reuses ``src/eval.py`` scoring (grammar parse + per-category checks) and emits a
tidy per-(dataset, level) row with a global F1 and four category F1s:

    class        -> f1_class      (LLM-assisted class match)
    association  -> f1_rel        (relation endpoints)
    attribute    -> f1_attr_llm   (LLM-assisted attribute match)
    cardinality  -> score_rel / max_score (endpoint + cardinality correctness)

Global F1 is the mean of the four category scores (per project, per level).
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List

import pandas as pd

from .config import CATEGORIES, DEFAULT_LEVELS_CONFIG, LevelsConfig
from .generate import _iter_datasets, _load_src_module, load_runner

logger = logging.getLogger(__name__)


def load_evaluator(cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> ModuleType:
    return _load_src_module("t2u_eval", cfg.src_dir / "eval.py")


def _cardinality_f1(rel_result: dict) -> float:
    """Relation cardinality accuracy as score_rel / max_score in [0, 1]."""
    max_score = rel_result.get("len", 0) or rel_result.get("max_score", 0)
    if not max_score:
        return 0.0
    return min(1.0, rel_result.get("total_score", 0.0) / max_score)


def _category_f1s(results) -> Dict[str, float]:
    """Map ``src.eval.evaluate`` output to the four experiment categories.

    ``results`` is (class, rel, attr_naive, attr_syn, attr_llm, inh).
    """
    res_cl, res_re, _res_at_naive, _res_at_syn, res_at_llm, _res_in = results
    return {
        "class": float(res_cl["f1"]),
        "association": float(res_re["f1"]),
        "attribute": float(res_at_llm["f1"]),
        "cardinality": _cardinality_f1(res_re),
    }


def evaluate_all(
    model: str,
    only: List[str] | None = None,
    levels: List[str] | None = None,
    cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG,
) -> pd.DataFrame:
    """Score every generated (dataset, level) result for ``model``.

    Returns a tidy DataFrame with columns: sub_folder_name, level, level_label,
    level_rank, model, f1_class, f1_association, f1_attribute, f1_cardinality,
    f1_global.
    """
    ev = load_evaluator(cfg)
    # Reuse src.run's own sanitization instead of duplicating it inline --
    # generate.py names result files via runner._safe_model_name(model), and
    # a second, drifted copy here (e.g. missing ":" handling for Ollama tags
    # like "gemma4:e4b") would silently fail to find any result file.
    runner_safe = load_runner(cfg)._safe_model_name(model)
    parser = ev.init_parser(str(cfg.grammar_path))
    # Read the strip config from the eval yaml so scaffolding removal matches
    # the main pipeline; fall back to no stripping if unavailable.
    strip_cfg = None
    try:
        import yaml

        with open(cfg.eval_config) as f:
            strip_cfg = (yaml.safe_load(f) or {}).get("strip_prompt")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load strip config (%s); evaluating unstripped.", exc)

    level_meta = {t: (label, rank) for t, _f, label, rank in cfg.levels}
    want_levels = set(levels) if levels else set(level_meta)

    rows: List[dict] = []
    for dataset in _iter_datasets(cfg, only):
        gold = dataset / cfg.gold_filename
        if not gold.is_file():
            logger.debug("  %s: no gold %s; skipping", dataset.name, cfg.gold_filename)
            continue
        for tag in want_levels:
            result_file = cfg.result_path(dataset, tag, runner_safe)
            if not result_file.is_file():
                logger.debug("  %s/%s: no result file; skipping", dataset.name, tag)
                continue
            if not ev._contains_uml(str(result_file), strip_cfg or {}):
                logger.info("  %s/%s: non-UML output; scored as 0.", dataset.name, tag)
                cats = {c: 0.0 for c in CATEGORIES}
            else:
                try:
                    results = ev.evaluate(
                        str(gold), str(result_file), parser, strip_cfg=strip_cfg or None
                    )
                    cats = _category_f1s(results)
                except Exception as exc:  # noqa: BLE001 - isolate a bad file
                    logger.warning("  %s/%s eval failed (%s); scored as 0.", dataset.name, tag, exc)
                    cats = {c: 0.0 for c in CATEGORIES}

            label, rank = level_meta[tag]
            row = {
                "sub_folder_name": dataset.name,
                "level": tag,
                "level_label": label,
                "level_rank": rank,
                "model": model,
                "f1_global": sum(cats.values()) / len(cats),
            }
            row.update({f"f1_{c}": cats[c] for c in CATEGORIES})
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["sub_folder_name", "level_rank"]).reset_index(drop=True)
    return df


def write_f1_csv(df: pd.DataFrame, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> Path:
    """Merge ``df`` into the existing F1 CSV: rows matching ``df``'s
    (sub_folder_name, level, model) are replaced, every other row is left
    untouched.

    This matters because ``evaluate_all`` is routinely called with
    ``only=[...]`` scoped to one or two cases (e.g. after regenerating a
    single case's description) -- an unconditional overwrite would silently
    truncate the corpus-wide CSV down to just those cases. Mirrors
    ``case_pipeline.py``'s ``_merge_case_rows`` pattern, generalized to
    multiple cases in one call.

    Keying on ``sub_folder_name`` alone (the original implementation) also
    silently deleted every *other* model's or level's rows for a touched
    case once a second model was evaluated into the same CSV -- it went
    unnoticed as long as only one model (``claude-sonnet-4-6``) had ever
    been scored here. Keying on every identifying column present in both
    frames fixes that without changing behavior for single-model CSVs.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.f1_csv.is_file() and not df.empty:
        existing = pd.read_csv(cfg.f1_csv)
        key_cols = [c for c in ("sub_folder_name", "level", "model") if c in df.columns and c in existing.columns]
        touched = set(map(tuple, df[key_cols].drop_duplicates().to_numpy()))
        existing = existing[~existing[key_cols].apply(tuple, axis=1).isin(touched)]
        merged = pd.concat([existing, df], ignore_index=True)
        merged = merged.sort_values(["sub_folder_name", "level_rank"]).reset_index(drop=True)
    else:
        merged = df
    merged.to_csv(cfg.f1_csv, index=False)
    logger.info("Wrote %d row(s) for %d case(s) to %s (%d rows total in file)",
                len(df), df["sub_folder_name"].nunique() if not df.empty else 0, cfg.f1_csv, len(merged))
    return cfg.f1_csv
