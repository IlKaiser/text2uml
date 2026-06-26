"""Syntactic complexity metrics computed from spaCy dependency parses.

Implements three metrics from the paper:

* Mean Dependency Distance (MDD)   - Liu (2008); Jing & Liu (2015).
* Subordination Index               - Csub / Ctotal.
* Parse Tree Depth (PTD)            - Kauchak et al. (2017); Massung et al. (2013).
"""

from __future__ import annotations

import logging
from statistics import mean
from typing import List

from .base import Metric

logger = logging.getLogger(__name__)

# Dependency labels that head a subordinate clause.
_SUBORDINATE_DEPS = {"advcl", "ccomp", "acl", "relcl", "xcomp", "csubj"}


class MeanDependencyDistance(Metric):
    """MDD = mean over arcs of |pos(head) - pos(dependent)|.

    Implements the paper's definition:
        MDD = (1/n) * sum_i |pos(h_i) - pos(d_i)|
    over the ``n`` dependency relations of the document (the ROOT self-arc,
    where the token is its own head, contributes no relation).
    """

    name = "mdd"

    def compute(self, doc) -> float:
        distances: List[int] = [
            abs(token.i - token.head.i)
            for token in doc
            if token.head.i != token.i and not token.is_space
        ]
        if not distances:
            return float("nan")
        return float(mean(distances))


class SubordinationIndex(Metric):
    """Subordination Index = Csub / Ctotal.

    ``Csub`` counts subordinate-clause heads (clausal dependency labels or any
    token bearing a ``mark`` child, e.g. "even though"). ``Ctotal`` counts all
    clause heads: one ROOT per sentence plus the subordinate heads.
    """

    name = "subordination_index"

    def compute(self, doc) -> float:
        n_sub = 0
        n_root = 0
        for token in doc:
            if token.dep_ == "ROOT":
                n_root += 1
            is_sub = token.dep_ in _SUBORDINATE_DEPS or any(
                child.dep_ == "mark" for child in token.children
            )
            if is_sub:
                n_sub += 1
        total = n_root + n_sub
        if total == 0:
            return float("nan")
        return n_sub / total


class ParseTreeDepth(Metric):
    """Mean per-sentence depth of the dependency tree (root -> deepest leaf).

        PTD = max_{l in leaves} d(root, l)
    computed per sentence and averaged across the document. The companion
    ``ParseTreeDepthMax`` reports the document-level maximum.
    """

    name = "parse_tree_depth"

    @staticmethod
    def _sentence_depth(sent) -> int:
        max_depth = 0
        for token in sent:
            depth = 0
            current = token
            # Walk up to the sentence root; cap to avoid cycles on odd parses.
            # spaCy returns a fresh Token on each .head access, so compare by
            # index (the root is its own head: current.head.i == current.i).
            while current.head.i != current.i and depth < len(sent) + 1:
                depth += 1
                current = current.head
            max_depth = max(max_depth, depth)
        return max_depth

    def compute(self, doc) -> float:
        depths = [self._sentence_depth(sent) for sent in doc.sents]
        if not depths:
            return float("nan")
        return float(mean(depths))


class ParseTreeDepthMax(ParseTreeDepth):
    """Document-level maximum parse-tree depth across sentences."""

    name = "parse_tree_depth_max"

    def compute(self, doc) -> float:
        depths = [self._sentence_depth(sent) for sent in doc.sents]
        if not depths:
            return float("nan")
        return float(max(depths))
