"""Base types and shared NLP resources for complexity metrics.

A metric takes a parsed spaCy ``Doc`` and returns a single ``float`` value
(``nan`` when it cannot be computed). The shared spaCy pipeline is loaded once
and reused, since parsing is the dominant cost.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Protocol

logger = logging.getLogger(__name__)

# Lazily-initialized singleton spaCy pipeline, guarded by _NLP_LOCK: both the
# one-time load and every parse call go through it, since a shared Language
# object is not guaranteed safe for concurrent nlp(text) calls (tokenization
# mutates the shared Vocab/StringStore) and concurrent callers (e.g.
# experiments.levels.shape_loop's ThreadPoolExecutor-based dispatch) can
# otherwise race on both the cold-cache load and inference.
_NLP = None
_NLP_LOCK = threading.Lock()


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
        with _NLP_LOCK:
            if _NLP is None:  # re-check: another thread may have won the race
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


def parse(text: str, model_name: str = "en_core_web_sm"):
    """Load the shared pipeline (if needed) and parse ``text`` under the same
    lock, so concurrent callers never invoke ``nlp(text)`` on the shared
    ``Language`` object at the same time."""
    nlp = get_nlp(model_name)
    with _NLP_LOCK:
        return nlp(text)


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
