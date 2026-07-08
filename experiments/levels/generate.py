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
from pathlib import Path
from types import ModuleType
from typing import List

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
            result = runner._run_chain(chain, text, cfg.timeout, cb)
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
            logger.info(
                "  %s/%s -> %s (%d chars)", dataset.name, tag, out_file.name, len(result)
            )

    logger.info("Generated %d result file(s) for %s", written, model)
    return written
