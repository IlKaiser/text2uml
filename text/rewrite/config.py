"""Configuration for the description-simplification (rewrite) pipeline.

Immutable, config-driven settings for the Claude-powered rewrite loop that
drives each dataset's ``description.md`` toward a target linguistic-complexity
level while preserving its meaning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import DEFAULT_CONFIG, TextConfig


@dataclass(frozen=True)
class RewriteConfig:
    """Immutable configuration for the rewrite loop."""

    # --- Claude API ---
    model: str = "claude-opus-4-8"
    effort: str = "medium"  # low | medium | high | max
    max_tokens: int = 16000

    # --- Feedback loop ---
    max_iterations: int = 5
    tolerance: float = 0.05  # |measured z_index - target| accepted as "reached"
    # Reject a candidate as "best" if it drops below this fraction of the source
    # token count (a crude guard against over-shortening that loses content).
    min_token_ratio: float = 0.5
    # When True, a second Claude call audits semantic equivalence before a
    # complexity-matching candidate is accepted (costs one extra call per
    # tolerance-hit iteration). See rewrite/verifier.py.
    verify_meaning: bool = False

    # --- Targets ---
    # level_one = "as clear as possible" -> the simplest end of the scale.
    level_one_target: float = 0.0
    # level_two target is computed per dataset as actual_z_index * this factor
    # (the midpoint between the actual complexity and the simplest, z_index 0).
    level_two_factor: float = 0.5

    # --- Output ---
    level_zero_name: str = "level_zero"
    level_one_name: str = "level_one"
    level_two_name: str = "level_two"
    summary_csv_name: str = "rewrite_summary.csv"

    text_config: TextConfig = field(default_factory=lambda: DEFAULT_CONFIG)

    def output_path(self, dataset_dir, level_name: str):
        """Sibling ``description_<level>.md`` path next to the source description."""
        return dataset_dir / f"description_{level_name}.md"


DEFAULT_REWRITE_CONFIG = RewriteConfig()
