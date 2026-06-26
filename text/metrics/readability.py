"""Readability metric: Flesch Reading Ease.

Flesch (1948); Farr et al. (1951); Klare et al. (1969).

    Flesch Reading Ease = 206.835
                        - 1.015 * (words / sentences)
                        - 84.6  * (syllables / words)

Syllables are counted with a vowel-group heuristic, refined by the CMU
Pronouncing Dictionary (via nltk) when the word is known. The CMU dict is loaded
lazily and the package degrades gracefully to the heuristic if it is absent.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional

from .base import Metric

logger = logging.getLogger(__name__)

_VOWELS = "aeiouy"
_WORD_RE = re.compile(r"[a-zA-Z]+")

# Lazily-loaded CMU dict: maps lowercase word -> syllable count.
_CMU: Optional[Dict[str, int]] = None
_CMU_TRIED = False


def _load_cmu() -> Optional[Dict[str, int]]:
    """Load the CMU pronouncing dictionary if available, else ``None``."""
    global _CMU, _CMU_TRIED
    if _CMU_TRIED:
        return _CMU
    _CMU_TRIED = True
    try:
        from nltk.corpus import cmudict

        raw = cmudict.dict()
        _CMU = {
            word: min(
                sum(1 for ph in pron if ph[-1].isdigit()) for pron in prons
            )
            for word, prons in raw.items()
        }
    except (ImportError, LookupError):
        logger.warning("nltk CMU dict unavailable; using syllable heuristic only.")
        _CMU = None
    return _CMU


def count_syllables(word: str) -> int:
    """Estimate syllables in a single word.

    Prefers the CMU dictionary; falls back to a vowel-group heuristic with the
    common silent-``e`` correction. Always returns at least 1 for a word that
    contains a letter.
    """
    word = word.lower()
    cmu = _load_cmu()
    if cmu is not None and word in cmu:
        return cmu[word]

    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and not word.endswith("le") and count > 1:
        count -= 1
    return max(count, 1)


class FleschReadingEase(Metric):
    """Flesch Reading Ease. Higher = easier to read (lower complexity)."""

    name = "flesch_reading_ease"

    def compute(self, doc) -> float:
        sentences = sum(1 for _ in doc.sents)
        words = [t.text for t in doc if _WORD_RE.fullmatch(t.text)]
        n_words = len(words)
        if sentences == 0 or n_words == 0:
            return float("nan")
        n_syllables = sum(count_syllables(w) for w in words)
        return (
            206.835
            - 1.015 * (n_words / sentences)
            - 84.6 * (n_syllables / n_words)
        )
