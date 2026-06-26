"""Configuration for the text-complexity metrics package.

All paths and runtime knobs live here as a frozen dataclass so the rest of the
package stays config-driven and the configuration itself is immutable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Repository root is the parent of the ``text`` package directory.
_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent


@dataclass(frozen=True)
class TextConfig:
    """Immutable configuration for the complexity-metrics pipeline."""

    repo_root: Path = _REPO_ROOT
    dataset_dir: Path = _REPO_ROOT / "dataset"
    description_filename: str = "description.md"
    results_csv: Path = _REPO_ROOT / "dataset" / "evaluation_results_llm.csv"
    output_dir: Path = _PACKAGE_DIR / "output"
    metrics_csv_name: str = "complexity_metrics.csv"
    spacy_model: str = "en_core_web_sm"
    # Figure formats written for every plot (matches the repo's SVG convention).
    figure_formats: List[str] = field(default_factory=lambda: ["svg", "png"])

    @property
    def metrics_csv(self) -> Path:
        return self.output_dir / self.metrics_csv_name


DEFAULT_CONFIG = TextConfig()
