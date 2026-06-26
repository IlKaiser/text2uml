"""Base types and shared NLP resources for complexity metrics.

A metric takes a parsed spaCy ``Doc`` and returns a single ``float`` value
(``nan`` when it cannot be computed). The shared spaCy pipeline is loaded once
and reused, since parsing is the dominant cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Protocol

logger = logging.getLogger(__name__)

# Lazily-initialized singleton spaCy pipeline.
_NLP = None


def get_nlp(model_name: str = "en_core_web_sm"):
    """Load (once) and return the shared spaCy pipeline.

    Args:
        model_name: Name of the installed spaCy model.

    Returns:
        A loaded spaCy ``Language`` object.

    Raises:
        OSError: When the model is not installed.
    """
    global _NLP
    if _NLP is None:
        import spacy

        try:
            _NLP = spacy.load(model_name)
        except OSError:
            logger.error(
                "spaCy model '%s' not found. Install it with: "
                "python -m spacy download %s",
                model_name,
                model_name,
            )
            raise
    return _NLP


class Metric(Protocol):
    """A complexity metric: maps a parsed document to a scalar score."""

    name: str

    def compute(self, doc) -> float:  # pragma: no cover - structural typing
        ...


@dataclass(frozen=True)
class MetricResult:
    """The value of every metric for a single text sample."""

    sample_id: str
    values: Dict[str, float]
    n_tokens: int
    n_sentences: int
    error: Optional[str] = None
