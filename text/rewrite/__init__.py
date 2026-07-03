"""Claude-powered rewrite pipeline: simplify descriptions to target complexity.

Public API:
    ComplexityReference / build_reference / score_text  -- scoring on the z_index scale
    RewriteConfig / DEFAULT_REWRITE_CONFIG              -- configuration
    rewrite_to_target / LevelResult                     -- the feedback loop
"""

from __future__ import annotations

from .config import DEFAULT_REWRITE_CONFIG, RewriteConfig
from .loop import LevelResult, rewrite_to_target
from .scorer import ComplexityReference, ScoreResult, build_reference, score_text
from .verifier import MeaningCheck, verify_meaning

__all__ = [
    "RewriteConfig",
    "DEFAULT_REWRITE_CONFIG",
    "ComplexityReference",
    "ScoreResult",
    "build_reference",
    "score_text",
    "LevelResult",
    "rewrite_to_target",
    "MeaningCheck",
    "verify_meaning",
]
