"""Generate UML for each project at every complexity level, for one technique.

Reuses the existing generation stack in ``src/run.py`` (LLM construction,
``_CHAIN_BUILDERS[cfg.technique]``, timeout-guarded invocation) but drives it
from a level-specific description file and writes level-tagged result files:
``result_<prefix>_<level>_<model>.txt`` (prefix from
``config.TECHNIQUE_RESULT_PREFIXES``). Originals are never overwritten.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import List

import pandas as pd

from .config import DEFAULT_LEVELS_CONFIG, LevelsConfig

logger = logging.getLogger(__name__)


def _load_src_module(name: str, path: Path) -> ModuleType:
    """Import a module from ``src/`` by file path (src is not a package)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_runner(cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> ModuleType:
    """Import ``src/run.py`` as a module so we can reuse its helpers."""
    return _load_src_module("t2u_run", cfg.src_dir / "run.py")


def _iter_datasets(cfg: LevelsConfig, only: List[str] | None) -> List[Path]:
    paths = sorted(p for p in cfg.dataset_dir.iterdir() if p.is_dir())
    if only:
        wanted = set(only)
        paths = [p for p in paths if p.name in wanted]
    return [p for p in paths if p.name not in set(cfg.skip_folders)]


def _merge_time_csv(rows: List[dict], cfg: LevelsConfig) -> None:
    """Merge newly measured generation times into ``levels_generation_time.csv``.

    Mirrors ``evaluate.write_f1_csv``'s replace-matching-rows-only merge, keyed
    on (sub_folder_name, level, model, technique), so a partial/rerun never
    wipes out another model's or technique's timing rows.
    """
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    key_cols = ["sub_folder_name", "level", "model", "technique"]
    if cfg.time_csv.is_file():
        existing = pd.read_csv(cfg.time_csv)
        touched = set(map(tuple, new_df[key_cols].to_numpy()))
        existing = existing[~existing[key_cols].apply(tuple, axis=1).isin(touched)]
        merged = pd.concat([existing, new_df], ignore_index=True)
    else:
        merged = new_df
    merged.to_csv(cfg.time_csv, index=False)


def generate(
    provider: str,
    model: str,
    provider_cfg: dict,
    only: List[str] | None = None,
    levels: List[str] | None = None,
    force: bool = False,
    cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG,
) -> int:
    """Run the two-shot chain for one model over the requested levels.

    Args:
        provider: Provider key understood by ``src/run.py`` (e.g. "anthropic").
        model: Model id for that provider.
        provider_cfg: Provider sub-config (passed to ``_make_llm``).
        only: Restrict to these dataset folder names (else all, minus skips).
        levels: Restrict to these level tags (else all three).
        force: Regenerate even if a result file already exists.

    Returns:
        Number of (dataset, level) result files written.
    """
    runner = load_runner(cfg)
    safe_model = runner._safe_model_name(model)
    llm = runner._make_llm(provider, model, provider_cfg)
    build_chain = runner._CHAIN_BUILDERS[cfg.technique]
    chain = build_chain(llm)

    want_levels = set(levels) if levels else {t for t, *_ in cfg.levels}
    datasets = _iter_datasets(cfg, only)
    written = 0
    time_rows: List[dict] = []

    for dataset in datasets:
        for tag, desc_name, _label, _rank in cfg.levels:
            if tag not in want_levels:
                continue
            desc = dataset / desc_name
            if not desc.is_file():
                logger.debug("  %s: missing %s; skipping level %s", dataset.name, desc_name, tag)
                continue
            out_file = cfg.result_path(dataset, tag, safe_model)
            if out_file.exists() and out_file.stat().st_size > 0 and not force:
                logger.info("  skip existing: %s", out_file.name)
                continue

            text = desc.read_text(encoding="utf-8")
            cb = runner._UsageCallback()
            t0 = time.time()
            result = runner._run_chain(chain, text, cfg.timeout, cb)
            elapsed = time.time() - t0
            if provider == "huggingface_local":
                result = runner._parse_hf_local_output(result)
            if not result:
                # _run_chain swallows exceptions and returns "" on failure (timeout,
                # API error, etc.). Never let that blank write clobber an existing
                # non-empty result file.
                logger.error(
                    "  %s/%s: generation returned empty output; leaving %s untouched",
                    dataset.name, tag, out_file.name,
                )
                continue
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(result, encoding="utf-8")
            written += 1
            time_rows.append({
                "sub_folder_name": dataset.name, "level": tag,
                "model": model, "technique": cfg.technique, "seconds": elapsed,
                "estimated": False,  # timed live via time.time(), not backfilled from file mtimes
            })
            logger.info(
                "  %s/%s -> %s (%d chars, %.1fs)", dataset.name, tag, out_file.name, len(result), elapsed
            )

    _merge_time_csv(time_rows, cfg)
    logger.info("Generated %d result file(s) for %s", written, model)
    return written
