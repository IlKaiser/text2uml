# Shape-Aware Complexity-Level Generation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate `description_level_zero/one/two.md` for every dataset case so `mdd`, `subordination_index`, and `context_dependence_proxy` evolve across L0→L1→L2→L3 in the same shape observed in `GasStation_KUL`/`GasStation_TUW`, via a per-text closed rewrite loop plus a corpus-level closed retry loop.

**Architecture:** A pure rank+band checker (`shape_targets.py`) is the single source of truth for "does this shape hold," used both inside the per-text iterative Claude rewrite loop (`loop.py`, extended) and by a corpus-level driver (`shape_loop.py`, new) that re-verifies from freshly computed metrics and retries non-compliant cases. Level zero gets a new "structured UML notes" genre prompt (`structural_prompts.py`); levels one/two keep the existing narrative-simplification prompt but are now regenerated under the same shape gate.

**Tech Stack:** Python 3.10 (conda env `kul`, `/Users/marcocalamo/anaconda3/envs/kul/bin/python`), pandas, spaCy (`en_core_web_sm`), Anthropic SDK, pytest.

## Global Constraints

- Conda env `kul` (`/Users/marcocalamo/anaconda3/envs/kul/bin/python`) is the project's runtime; it is missing `pytest` — install it once (Task 0) before any test task.
- `description.md` (L3) is never modified by any code in this plan.
- Every rewrite acceptance still requires the existing meaning-preservation verifier (`text/rewrite/verifier.py`) to pass — no change to that contract.
- Rank constraints (see Task 1) are a hard gate; ratio bands are soft (degenerate L3 values skip the band check, never fail on that account alone).
- All new/modified dataclasses are frozen; no mutable default arguments; specific exception types only (no bare `except:`); every new function has type hints.
- Run all commands from the repo root `/Users/marcocalamo/text2uml`.

---

### Task 0: Install pytest in the `kul` conda env

**Files:** none (environment setup only)

- [ ] **Step 1: Install pytest**

```bash
/Users/marcocalamo/anaconda3/envs/kul/bin/python -m pip install pytest
```

- [ ] **Step 2: Verify**

```bash
/Users/marcocalamo/anaconda3/envs/kul/bin/python -m pytest --version
```
Expected: prints a pytest version (e.g. `pytest 8.x.x`).

No commit needed (environment change, not a repo file).

---

### Task 1: Shape-target checker (`text/rewrite/shape_targets.py`)

Pure logic, no API calls — the rank+band rules from the design spec, plus a
corpus-level cross-level max check and a feedback-text formatter shared by
both the narrative and structural prompt paths.

**Files:**
- Create: `text/rewrite/shape_targets.py`
- Test: `tests/text/rewrite/test_shape_targets.py`

**Interfaces:**
- Produces: `SHAPE_METRICS: Tuple[str, ...]`, `RATIO_BANDS: Dict[str, Dict[str, Tuple[float, Optional[float]]]]`, `ShapeCheck` (frozen dataclass: `metric: str`, `level: str`, `value: float`, `l3_value: float`, `ratio: float`, `band: Tuple[float, Optional[float]]`, `rank_ok: bool`, `band_ok: bool`, `degenerate: bool`), `check_shape(level: str, level_values: Dict[str, float], l3_values: Dict[str, float]) -> List[ShapeCheck]`, `check_case_shape(levels: Dict[str, Dict[str, float]]) -> List[ShapeCheck]`, `shape_ok(checks: List[ShapeCheck]) -> bool`, `format_feedback(checks: List[ShapeCheck], guidance: Dict[str, str]) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/text/rewrite/test_shape_targets.py
from __future__ import annotations

import math

from text.rewrite.shape_targets import (
    RATIO_BANDS,
    SHAPE_METRICS,
    check_case_shape,
    check_shape,
    format_feedback,
    shape_ok,
)

L3 = {"mdd": 3.0, "subordination_index": 0.5, "context_dependence_proxy": 0.4}


def test_check_shape_all_pass_for_level_zero():
    values = {"mdd": 3.4, "subordination_index": 0.2, "context_dependence_proxy": 0.7}
    checks = check_shape("zero", values, L3)
    assert len(checks) == len(SHAPE_METRICS)
    assert shape_ok(checks)


def test_check_shape_mdd_rank_failure_for_level_zero():
    values = {"mdd": 2.9, "subordination_index": 0.2, "context_dependence_proxy": 0.7}
    checks = check_shape("zero", values, L3)
    mdd_check = next(c for c in checks if c.metric == "mdd")
    assert mdd_check.rank_ok is False
    assert shape_ok(checks) is False


def test_check_shape_band_failure_for_level_one():
    # mdd band for "one" is 0.60-0.90x L3; 0.95x is out of band.
    values = {"mdd": 2.85, "subordination_index": 0.3, "context_dependence_proxy": 0.3}
    checks = check_shape("one", values, L3)
    mdd_check = next(c for c in checks if c.metric == "mdd")
    assert mdd_check.band_ok is False
    assert shape_ok(checks) is False


def test_check_shape_degenerate_l3_skips_band():
    l3_zero = dict(L3, subordination_index=0.0)
    values = {"mdd": 3.4, "subordination_index": 0.9, "context_dependence_proxy": 0.7}
    checks = check_shape("zero", values, l3_zero)
    sub_check = next(c for c in checks if c.metric == "subordination_index")
    assert sub_check.degenerate is True
    assert math.isnan(sub_check.ratio)
    assert shape_ok(checks) is True  # degenerate never fails on its own


def test_check_case_shape_global_max_constraints():
    levels = {
        "zero": {"mdd": 3.4, "subordination_index": 0.2, "context_dependence_proxy": 0.9},
        "one": {"mdd": 2.0, "subordination_index": 0.15, "context_dependence_proxy": 0.2},
        "two": {"mdd": 2.4, "subordination_index": 0.3, "context_dependence_proxy": 0.35},
        "three": L3,
    }
    checks = check_case_shape(levels)
    assert shape_ok(checks) is True


def test_check_case_shape_fails_when_zero_not_max_context_dependence():
    levels = {
        "zero": {"mdd": 3.4, "subordination_index": 0.2, "context_dependence_proxy": 0.1},
        "one": {"mdd": 2.0, "subordination_index": 0.15, "context_dependence_proxy": 0.9},
        "two": {"mdd": 2.4, "subordination_index": 0.3, "context_dependence_proxy": 0.35},
        "three": L3,
    }
    checks = check_case_shape(levels)
    assert shape_ok(checks) is False


def test_format_feedback_names_failing_metric():
    values = {"mdd": 2.9, "subordination_index": 0.2, "context_dependence_proxy": 0.7}
    checks = check_shape("zero", values, L3)
    text = format_feedback(checks, {"mdd": "pack more modifiers per entity"})
    assert "mdd" in text
    assert "pack more modifiers per entity" in text
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/marcocalamo/text2uml && /Users/marcocalamo/anaconda3/envs/kul/bin/python -m pytest tests/text/rewrite/test_shape_targets.py -v
```
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'text.rewrite.shape_targets'`.

- [ ] **Step 3: Implement `text/rewrite/shape_targets.py`**

```python
"""Rank + ratio-band shape checks for the four description complexity levels.

Encodes the empirical GasStation_KUL / GasStation_TUW pattern for three
metrics (mdd, subordination_index, context_dependence_proxy) as a pure,
API-free checker: given a level's metric values and that same case's own
(untouched) L3 values, decide whether the level's shape matches the
reference pattern. Two tiers:

* Rank constraints (hard): e.g. mdd(L0) must exceed mdd(L3).
* Ratio bands (soft): the level's metric value, as a multiple of that case's
  own L3 value, must fall in an empirically-derived band. A band is skipped
  (never fails) when the L3 value is exactly zero (degenerate ratio).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

SHAPE_METRICS: Tuple[str, ...] = ("mdd", "subordination_index", "context_dependence_proxy")

# level tag -> metric -> (min_ratio, max_ratio); max_ratio=None means one-sided (>= min_ratio).
RATIO_BANDS: Dict[str, Dict[str, Tuple[float, Optional[float]]]] = {
    "zero": {
        "mdd": (1.05, None),
        "subordination_index": (0.35, 0.65),
        "context_dependence_proxy": (1.5, None),
    },
    "one": {
        "mdd": (0.60, 0.90),
        "subordination_index": (0.20, 0.70),
        "context_dependence_proxy": (0.5, 1.3),
    },
    "two": {
        "mdd": (0.75, 0.95),
        "subordination_index": (0.45, 0.75),
        "context_dependence_proxy": (0.7, 1.2),
    },
}


@dataclass(frozen=True)
class ShapeCheck:
    """One metric's shape verdict for one level of one case."""

    metric: str
    level: str
    value: float
    l3_value: float
    ratio: float
    band: Tuple[float, Optional[float]]
    rank_ok: bool
    band_ok: bool
    degenerate: bool


def _rank_ok(metric: str, level: str, value: float, l3_value: float) -> bool:
    """Hard rank constraints from the design spec (Section: Target shape rules)."""
    if metric == "mdd" and level == "zero":
        return value > l3_value
    if metric == "subordination_index":
        return l3_value >= value  # L3 must dominate every other level.
    if metric == "context_dependence_proxy" and level == "zero":
        return value >= l3_value
    return True


def check_shape(
    level: str, level_values: Dict[str, float], l3_values: Dict[str, float]
) -> List[ShapeCheck]:
    """Check one level's metrics against that case's own L3 values.

    Only used as the *local* (vs L3 alone) check, e.g. inside the per-text
    rewrite loop. ``check_case_shape`` adds the cross-level max constraints
    that need every level's values at once.
    """
    checks: List[ShapeCheck] = []
    bands = RATIO_BANDS.get(level, {})
    for metric in SHAPE_METRICS:
        value = level_values.get(metric)
        l3_value = l3_values.get(metric)
        if value is None or l3_value is None:
            continue
        degenerate = l3_value == 0
        ratio = float("nan") if degenerate else value / l3_value
        lo, hi = bands.get(metric, (0.0, None))
        band_ok = degenerate or (ratio >= lo and (hi is None or ratio <= hi))
        checks.append(
            ShapeCheck(
                metric=metric,
                level=level,
                value=value,
                l3_value=l3_value,
                ratio=ratio,
                band=(lo, hi),
                rank_ok=_rank_ok(metric, level, value, l3_value),
                band_ok=band_ok,
                degenerate=degenerate,
            )
        )
    return checks


def _global_max_check(metric: str, expected_max_level: str, levels: Dict[str, Dict[str, float]]) -> Optional[ShapeCheck]:
    """Cross-level constraint: ``metric`` must peak at ``expected_max_level``."""
    values = {lvl: vals.get(metric) for lvl, vals in levels.items() if metric in vals}
    if expected_max_level not in values or len(values) < 2:
        return None
    max_level = max(values, key=lambda lvl: values[lvl])
    l3_value = levels.get("three", {}).get(metric, float("nan"))
    return ShapeCheck(
        metric=metric,
        level=expected_max_level,
        value=values[expected_max_level],
        l3_value=l3_value,
        ratio=float("nan"),
        band=(0.0, None),
        rank_ok=(max_level == expected_max_level),
        band_ok=True,
        degenerate=False,
    )


def check_case_shape(levels: Dict[str, Dict[str, float]]) -> List[ShapeCheck]:
    """Full corpus-level check: per-level bands vs L3, plus cross-level maxima.

    ``levels`` maps level tag ("zero", "one", "two", "three") to that level's
    metric-value dict (e.g. one row of ``levels_complexity.csv`` per level).
    """
    l3_values = levels.get("three", {})
    checks: List[ShapeCheck] = []
    for level in ("zero", "one", "two"):
        if level in levels:
            checks.extend(check_shape(level, levels[level], l3_values))
    for metric, expected_max_level in (
        ("subordination_index", "three"),
        ("context_dependence_proxy", "zero"),
    ):
        gmax = _global_max_check(metric, expected_max_level, levels)
        if gmax is not None:
            checks.append(gmax)
    return checks


def shape_ok(checks: List[ShapeCheck]) -> bool:
    """True only if every non-degenerate check holds both its rank and band.

    A degenerate check (L3 reference value is exactly 0 for that metric) has
    no meaningful rank or ratio to compare against, so it is exempted
    entirely rather than only skipping its ratio band.
    """
    return all(c.degenerate or (c.rank_ok and c.band_ok) for c in checks)


def format_feedback(checks: List[ShapeCheck], guidance: Dict[str, str]) -> str:
    """Turn failing checks into concrete editing guidance for the rewrite prompt."""
    lines: List[str] = []
    for c in checks:
        if c.degenerate or (c.rank_ok and c.band_ok):
            continue
        lo, hi = c.band
        band_str = f">= {lo:.2f}x" if hi is None else f"{lo:.2f}x-{hi:.2f}x"
        ratio_str = "n/a" if math.isnan(c.ratio) else f"{c.ratio:.2f}x"
        tip = guidance.get(c.metric, "adjust the phrasing")
        lines.append(
            f"- {c.metric}: measured {ratio_str} of the real spec's value "
            f"(target band {band_str} of the real spec). {tip}."
        )
    if not lines:
        return ""
    return "The following metrics don't yet match the target shape:\n" + "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/marcocalamo/text2uml && /Users/marcocalamo/anaconda3/envs/kul/bin/python -m pytest tests/text/rewrite/test_shape_targets.py -v
```
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add text/rewrite/shape_targets.py tests/text/rewrite/test_shape_targets.py
git commit -m "feat(rewrite): add rank+band shape checker for complexity levels"
```

---

### Task 2: Structural (level-zero) genre prompt (`text/rewrite/structural_prompts.py`)

**Files:**
- Create: `text/rewrite/structural_prompts.py`
- Test: `tests/text/rewrite/test_structural_prompts.py`

**Interfaces:**
- Consumes: nothing new (pure string builders).
- Produces: `structural_system_prompt() -> str`, `build_structural_user_prompt(original: str, current_text: str, feedback: Optional[str]) -> str`, `STRUCTURAL_METRIC_GUIDANCE: Dict[str, str]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/text/rewrite/test_structural_prompts.py
from __future__ import annotations

from text.rewrite.structural_prompts import (
    STRUCTURAL_METRIC_GUIDANCE,
    build_structural_user_prompt,
    structural_system_prompt,
)


def test_system_prompt_describes_classes_and_relationships_genre():
    prompt = structural_system_prompt()
    assert "Classes" in prompt
    assert "Relationships" in prompt
    assert "pronoun" in prompt.lower()
    assert "Never add facts" in prompt


def test_guidance_covers_the_three_shape_metrics():
    for metric in ("mdd", "subordination_index", "context_dependence_proxy"):
        assert metric in STRUCTURAL_METRIC_GUIDANCE


def test_user_prompt_embeds_source_and_feedback():
    prompt = build_structural_user_prompt("SOURCE TEXT", "CURRENT DRAFT", "FIX THE MDD")
    assert "SOURCE TEXT" in prompt
    assert "CURRENT DRAFT" in prompt
    assert "FIX THE MDD" in prompt


def test_user_prompt_omits_feedback_block_when_none():
    prompt = build_structural_user_prompt("SOURCE TEXT", "CURRENT DRAFT", None)
    assert "SOURCE TEXT" in prompt
    assert "FEEDBACK" not in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/marcocalamo/text2uml && /Users/marcocalamo/anaconda3/envs/kul/bin/python -m pytest tests/text/rewrite/test_structural_prompts.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'text.rewrite.structural_prompts'`.

- [ ] **Step 3: Implement `text/rewrite/structural_prompts.py`**

```python
"""Prompt construction for the level-zero "structured UML notes" genre.

Unlike level one/two (linguistic simplification of the narrative), level zero
is a different genre entirely: a compact "Classes / Relationships" analyst
fact-sheet, modelled on the existing GasStation_KUL / GasStation_TUW
references. That genre naturally produces the target shape for three
metrics — high mdd (parenthetical/appositive density), low
subordination_index (no subordinating conjunctions), high
context_dependence_proxy (pronoun back-reference instead of repeated names)
— so the prompt encodes those stylistic rules directly instead of chasing an
aggregate complexity number.
"""

from __future__ import annotations

from typing import Dict, Optional

STRUCTURAL_METRIC_GUIDANCE: Dict[str, str] = {
    "mdd": "pack qualifiers onto each entity via commas, parentheses, or a "
    "trailing participial clause (e.g. 'many Pumps (1 -> 0..*), each Pump "
    "belonging to one station') instead of a separate sentence",
    "subordination_index": "never use subordinating conjunctions (when, "
    "because, although, since, if); state each fact as a flat declarative "
    "bullet",
    "context_dependence_proxy": "after an entity's first mention, refer back "
    "to it with a pronoun (it, its, they) instead of repeating its full name",
}


def structural_system_prompt() -> str:
    """The level-zero persona: compact UML analyst notes, not simplified prose."""
    return (
        "You are a UML analyst producing a compact fact-sheet from a software "
        "system description. Convert the description into two sections:\n\n"
        "Classes\n"
        "One line per class: Name — short attribute/role list.\n\n"
        "Relationships\n"
        "One line per relationship: state the two classes, the multiplicity "
        "(e.g. '1 -> 0..*'), and any qualifying condition, packed into a "
        "single sentence via commas/parentheses/participial clauses.\n\n"
        "Style rules (these produce the target notes genre):\n"
        "- Pack qualifiers onto each entity via commas, parentheses, or a "
        "trailing participial clause instead of a separate sentence.\n"
        "- Never use subordinating conjunctions (when, because, although, "
        "since, if); every line is a flat declarative statement.\n"
        "- After an entity's first mention, refer back to it with a pronoun "
        "(it, its, they) instead of repeating its full name.\n\n"
        "Hard rules:\n"
        "- Never add facts, entities, relationships, or constraints not in "
        "the source.\n"
        "- Never remove or merge any entity, attribute, relationship, "
        "multiplicity, action, or constraint from the source.\n"
        "- Output ONLY the rewritten notes in Markdown. No preamble, no "
        "explanation, no commentary."
    )


def build_structural_user_prompt(
    original: str, current_text: str, feedback: Optional[str]
) -> str:
    """Assemble the per-iteration structural-rewrite request."""
    feedback_block = f"\nFEEDBACK ON YOUR LAST ATTEMPT:\n{feedback}\n" if feedback else ""
    return (
        "TASK: rewrite the description as compact 'Classes / Relationships' "
        "UML analyst notes, per the style rules in the system prompt.\n"
        f"{feedback_block}\n"
        "SOURCE OF TRUTH — preserve every fact, entity, attribute, "
        "relationship, action, and constraint from here:\n"
        f"<source>\n{original}\n</source>\n\n"
        "VERSION TO IMPROVE — rewrite this, keeping all meaning from the "
        "source:\n"
        f"<current>\n{current_text}\n</current>\n\n"
        "Return only the rewritten Markdown notes."
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/marcocalamo/text2uml && /Users/marcocalamo/anaconda3/envs/kul/bin/python -m pytest tests/text/rewrite/test_structural_prompts.py -v
```
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add text/rewrite/structural_prompts.py tests/text/rewrite/test_structural_prompts.py
git commit -m "feat(rewrite): add structured-notes prompt for level zero"
```

---

### Task 3: Shape-aware rewrite loop (`text/rewrite/loop.py`, `text/rewrite/config.py`)

Generalizes `rewrite_to_target` into `rewrite_to_shape`, which accepts
pluggable prompt builders and a metric-guidance dict, so it drives level
zero (structural genre) and levels one/two (narrative genre) through the
same shape-checked loop.

**Files:**
- Modify: `text/rewrite/loop.py`
- Modify: `text/rewrite/config.py`
- Test: `tests/text/rewrite/test_loop_shape.py`

**Interfaces:**
- Consumes: `text.rewrite.shape_targets.check_shape`, `shape_ok`, `format_feedback` (Task 1); `text.rewrite.scorer.ScoreResult`, `score_text`, `ComplexityReference` (existing); `text.rewrite.verifier.verify_meaning` (existing).
- Produces: `ShapeLevelResult` (frozen dataclass: `level_name: str`, `final_z: float`, `shape_checks: tuple`, `reached: bool`, `iterations: int`, `text: str`), `rewrite_to_shape(client, cfg: RewriteConfig, original: str, original_score: ScoreResult, reference: ComplexityReference, level_name: str, l3_values: Dict[str, float], system_prompt: str, user_prompt_fn: Callable[[str, str, Optional[str]], str], metric_guidance: Dict[str, str]) -> ShapeLevelResult`. `RewriteConfig.level_zero_name: str = "level_zero"` (new field).

- [ ] **Step 1: Write the failing test**

```python
# tests/text/rewrite/test_loop_shape.py
from __future__ import annotations

from text.rewrite.config import RewriteConfig
from text.rewrite.loop import rewrite_to_shape
from text.rewrite.scorer import ComplexityReference, ScoreResult


def _reference() -> ComplexityReference:
    metrics = ["mdd", "subordination_index", "context_dependence_proxy"]
    means = {m: 1.0 for m in metrics}
    stds = {m: 1.0 for m in metrics}
    return ComplexityReference(metrics=metrics, means=means, stds=stds, raw_min=-1.0, raw_max=1.0)


def _score(mdd, sub, ctx, n_tokens=50) -> ScoreResult:
    values = {"mdd": mdd, "subordination_index": sub, "context_dependence_proxy": ctx}
    ref = _reference()
    return ScoreResult(z_index=ref.z_index(values), oriented_z=ref.oriented_z(values), values=values, n_tokens=n_tokens)


def test_rewrite_to_shape_accepts_first_candidate_when_shape_already_ok(monkeypatch):
    l3_values = {"mdd": 3.0, "subordination_index": 0.5, "context_dependence_proxy": 0.4}
    good_candidate_score = _score(mdd=3.4, sub=0.2, ctx=0.7)

    monkeypatch.setattr("text.rewrite.loop.rewrite_once", lambda client, cfg, system, user: "GOOD CANDIDATE")
    monkeypatch.setattr("text.rewrite.loop.score_text", lambda text, reference, cfg: good_candidate_score)
    monkeypatch.setattr("text.rewrite.loop.verify_meaning", lambda client, cfg, original, candidate: type("C", (), {"equivalent": True, "feedback": lambda self: ""})())

    cfg = RewriteConfig(verify_meaning=True, max_iterations=3)
    original_score = _score(mdd=3.0, sub=0.5, ctx=0.4)
    result = rewrite_to_shape(
        client=None,
        cfg=cfg,
        original="ORIGINAL",
        original_score=original_score,
        reference=_reference(),
        level_name="zero",
        l3_values=l3_values,
        system_prompt="SYSTEM",
        user_prompt_fn=lambda original, current, feedback: "USER",
        metric_guidance={},
    )
    assert result.reached is True
    assert result.text == "GOOD CANDIDATE"
    assert result.iterations == 1


def test_rewrite_to_shape_stops_at_max_iterations_when_never_reached(monkeypatch):
    l3_values = {"mdd": 3.0, "subordination_index": 0.5, "context_dependence_proxy": 0.4}
    bad_candidate_score = _score(mdd=2.0, sub=0.5, ctx=0.1)  # mdd rank fails for "zero"

    monkeypatch.setattr("text.rewrite.loop.rewrite_once", lambda client, cfg, system, user: "BAD CANDIDATE")
    monkeypatch.setattr("text.rewrite.loop.score_text", lambda text, reference, cfg: bad_candidate_score)

    cfg = RewriteConfig(verify_meaning=False, max_iterations=2)
    original_score = _score(mdd=3.0, sub=0.5, ctx=0.4)
    result = rewrite_to_shape(
        client=None,
        cfg=cfg,
        original="ORIGINAL",
        original_score=original_score,
        reference=_reference(),
        level_name="zero",
        l3_values=l3_values,
        system_prompt="SYSTEM",
        user_prompt_fn=lambda original, current, feedback: "USER",
        metric_guidance={},
    )
    assert result.reached is False
    assert result.iterations == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/marcocalamo/text2uml && /Users/marcocalamo/anaconda3/envs/kul/bin/python -m pytest tests/text/rewrite/test_loop_shape.py -v
```
Expected: FAIL with `ImportError: cannot import name 'rewrite_to_shape'`.

- [ ] **Step 3: Add `level_zero_name` to `RewriteConfig`**

In `text/rewrite/config.py`, add one field next to the existing level names:

```python
    # --- Output ---
    level_zero_name: str = "level_zero"
    level_one_name: str = "level_one"
    level_two_name: str = "level_two"
```

- [ ] **Step 4: Implement `rewrite_to_shape` in `text/rewrite/loop.py`**

Add alongside the existing `rewrite_to_target` (keep that function — it stays
available, just unused by the new driver):

```python
from typing import Callable, Dict, Optional, Tuple

from .shape_targets import check_shape, format_feedback, shape_ok


@dataclass(frozen=True)
class ShapeLevelResult:
    """Outcome of driving one description toward one level's target shape."""

    level_name: str
    final_z: float
    shape_checks: Tuple  # Tuple[ShapeCheck, ...]
    reached: bool
    iterations: int
    text: str


def rewrite_to_shape(
    client,
    cfg: RewriteConfig,
    original: str,
    original_score: ScoreResult,
    reference: ComplexityReference,
    level_name: str,
    l3_values: Dict[str, float],
    system_prompt: str,
    user_prompt_fn: Callable[[str, str, Optional[str]], str],
    metric_guidance: Dict[str, str],
) -> ShapeLevelResult:
    """Iteratively rewrite ``original`` until its shape (rank + band) matches.

    Generalizes ``rewrite_to_target``: instead of chasing a single aggregate
    z_index, checks ``mdd`` / ``subordination_index`` / ``context_dependence_proxy``
    against ``l3_values`` (that case's own, untouched real-spec metrics) via
    ``text.rewrite.shape_targets``. ``system_prompt`` / ``user_prompt_fn`` are
    supplied by the caller so this one loop drives both the structural
    (level zero) and narrative (level one/two) genres.
    """
    tconf: TextConfig = cfg.text_config
    min_tokens = cfg.min_token_ratio * max(1, original_score.n_tokens)

    best_text = original
    best_checks = check_shape(level_name, original_score.values, l3_values)
    best_ok = shape_ok(best_checks)
    candidate_text = original
    feedback: Optional[str] = None
    reached = False
    iterations = 0
    final_z = original_score.z_index

    for i in range(cfg.max_iterations):
        iterations = i + 1
        user = user_prompt_fn(original, candidate_text, feedback)
        try:
            candidate_text = rewrite_once(client, cfg, system_prompt, user)
        except Exception as exc:  # noqa: BLE001 - report and stop iterating
            logger.error("Shape rewrite call failed on '%s' iter %d: %s", level_name, i, exc)
            break

        try:
            latest_score = score_text(candidate_text, reference, tconf)
        except ValueError as exc:
            logger.warning("Could not score shape candidate (%s); retrying.", exc)
            feedback = "The previous output could not be parsed. Return a normal, complete Markdown document."
            continue

        checks = check_shape(level_name, latest_score.values, l3_values)
        ok = shape_ok(checks)
        keeps_content = latest_score.n_tokens >= min_tokens

        if ok and keeps_content:
            if not cfg.verify_meaning:
                best_text, best_checks, best_ok, final_z = candidate_text, checks, True, latest_score.z_index
                reached = True
                break
            check = verify_meaning(client, cfg, original, candidate_text)
            if check.equivalent:
                best_text, best_checks, best_ok, final_z = candidate_text, checks, True, latest_score.z_index
                reached = True
                break
            feedback = "Your shape is on target — keep it there. " + check.feedback()
            continue

        if ok and not keeps_content:
            feedback = format_feedback(checks, metric_guidance) or "Restore every entity, attribute, relationship, action, and constraint from the source; your version is too short."
            continue

        if not best_ok:
            best_text, best_checks, final_z = candidate_text, checks, latest_score.z_index
        feedback = format_feedback(checks, metric_guidance)

    return ShapeLevelResult(
        level_name=level_name,
        final_z=final_z,
        shape_checks=tuple(best_checks),
        reached=reached,
        iterations=iterations,
        text=best_text,
    )
```

Add the two new imports at the top of `text/rewrite/loop.py` (`Callable`,
`Optional`, `Tuple` from `typing`; `check_shape`, `format_feedback`,
`shape_ok` from `.shape_targets`) alongside the existing ones.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /Users/marcocalamo/text2uml && /Users/marcocalamo/anaconda3/envs/kul/bin/python -m pytest tests/text/rewrite/test_loop_shape.py -v
```
Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add text/rewrite/loop.py text/rewrite/config.py tests/text/rewrite/test_loop_shape.py
git commit -m "feat(rewrite): generalize the rewrite loop to shape-aware acceptance"
```

---

### Task 4: Per-case driver refactor (`text/rewrite/run.py`)

Extract the per-dataset logic into a standalone, reusable function so both
the CLI and the corpus-level closed loop (Task 5) can call it; drive all
three levels (zero/one/two) through `rewrite_to_shape`.

**Files:**
- Modify: `text/rewrite/run.py`
- Test: `tests/text/rewrite/test_run_process_dataset.py`

**Interfaces:**
- Consumes: `rewrite_to_shape`, `ShapeLevelResult` (Task 3); `structural_system_prompt`, `build_structural_user_prompt`, `STRUCTURAL_METRIC_GUIDANCE` (Task 2); `text.rewrite.prompts.system_prompt`, `METRIC_GUIDANCE` (existing `_METRIC_GUIDANCE`, renamed — see Step 1; the existing numeric-target `build_user_prompt` is NOT reused here, since it bakes in a z_index target/level_name that no longer drives acceptance — Step 1 adds a shape-appropriate replacement instead).
- Produces: `build_shape_user_prompt(original: str, current_text: str, level_label: str, feedback: Optional[str]) -> str` (in `text/rewrite/prompts.py`); `process_dataset(name: str, description_path: Path, cfg: RewriteConfig, reference: ComplexityReference, tconf: TextConfig, client, levels: Tuple[str, ...] = ("zero", "one", "two"), force: bool = False) -> dict` (a summary row with keys `sub_folder_name`, `actual_z`, and per level `<level>_reached`, `<level>_iterations`, `<level>_shape_ok`).

- [ ] **Step 1: Rename `_METRIC_GUIDANCE` and add a shape-appropriate narrative user-prompt to `text/rewrite/prompts.py`**

Rename the constant (drop the leading underscore so Task 4 can import it) and
update its one existing use site in `build_feedback`:

```python
METRIC_GUIDANCE: Dict[str, str] = {
    ...  # same contents as _METRIC_GUIDANCE, just the public name
}
```

```python
def build_feedback(score: ScoreResult, target: float, tolerance: float) -> str:
    """Turn the measured metric breakdown into editing guidance."""
    top = sorted(score.oriented_z.items(), key=lambda kv: kv[1], reverse=True)[:3]
    lines = [
        f"Your last version measured z_index = {score.z_index:.2f} "
        f"(target {target:.2f}). You need to {_direction(score.z_index, target, tolerance)}.",
    ]
    if score.z_index > target:
        lines.append("The metrics contributing MOST to the excess complexity are:")
        for name, val in top:
            guidance = METRIC_GUIDANCE.get(name, "simplify the phrasing")
            lines.append(f"  - {name} (+{val:.2f} above corpus average): {guidance}.")
    else:
        lines.append(
            "It is now simpler than the target. Restore some detail and "
            "connective structure (without adding or removing facts) to raise "
            "the complexity back toward the target."
        )
    return "\n".join(lines)
```

Add a new function alongside `build_user_prompt` — the shape loop no longer
targets a numeric z_index, so this drops the `target`/`measured_z` framing
entirely in favor of a plain-language level label:

```python
def build_shape_user_prompt(
    original: str, current_text: str, level_label: str, feedback: Optional[str]
) -> str:
    """Per-iteration narrative-rewrite request, framed by level label instead
    of a numeric z_index target (acceptance is shape-based, see
    ``text.rewrite.shape_targets``)."""
    feedback_block = f"\nFEEDBACK ON YOUR LAST ATTEMPT:\n{feedback}\n" if feedback else ""
    return (
        f"TASK: rewrite the description as {level_label}.\n"
        f"{feedback_block}\n"
        f"SOURCE OF TRUTH — preserve every fact, entity, attribute, "
        f"relationship, action, and constraint from here:\n"
        f"<source>\n{original}\n</source>\n\n"
        f"VERSION TO IMPROVE — rewrite this, keeping all meaning from the "
        f"source:\n"
        f"<current>\n{current_text}\n</current>\n\n"
        f"Return only the rewritten Markdown description."
    )
```

`build_user_prompt` (the old numeric-target version) stays untouched for
`rewrite_to_target`, which also stays untouched.

- [ ] **Step 2: Write the failing test**

```python
# tests/text/rewrite/test_run_process_dataset.py
from __future__ import annotations

from pathlib import Path

from text.config import DEFAULT_CONFIG
from text.rewrite.config import RewriteConfig
from text.rewrite.run import process_dataset
from text.rewrite.scorer import ComplexityReference


def _reference() -> ComplexityReference:
    metrics = ["mdd", "subordination_index", "context_dependence_proxy", "flesch_reading_ease"]
    means = {m: 1.0 for m in metrics}
    stds = {m: 1.0 for m in metrics}
    return ComplexityReference(metrics=metrics, means=means, stds=stds, raw_min=-1.0, raw_max=1.0)


def test_process_dataset_writes_all_three_levels(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "ToyCase"
    dataset_dir.mkdir()
    desc = dataset_dir / "description.md"
    desc.write_text(
        "A Customer places many Orders. Each Order has a Total. "
        "Because the Total may be zero, the Order is flagged.",
        encoding="utf-8",
    )

    def fake_rewrite_to_shape(client, cfg, original, original_score, reference, level_name, l3_values, system_prompt, user_prompt_fn, metric_guidance):
        from text.rewrite.loop import ShapeLevelResult
        return ShapeLevelResult(
            level_name=level_name, final_z=0.1, shape_checks=(), reached=True,
            iterations=1, text=f"[{level_name} rewrite of source]",
        )

    monkeypatch.setattr("text.rewrite.run.rewrite_to_shape", fake_rewrite_to_shape)

    cfg = RewriteConfig()
    row = process_dataset(
        name="ToyCase", description_path=desc, cfg=cfg, reference=_reference(),
        tconf=DEFAULT_CONFIG, client=None,
    )

    assert (dataset_dir / "description_level_zero.md").read_text(encoding="utf-8").strip() == "[zero rewrite of source]"
    assert (dataset_dir / "description_level_one.md").read_text(encoding="utf-8").strip() == "[one rewrite of source]"
    assert (dataset_dir / "description_level_two.md").read_text(encoding="utf-8").strip() == "[two rewrite of source]"
    assert row["sub_folder_name"] == "ToyCase"
    assert row["zero_reached"] is True
    assert row["one_reached"] is True
    assert row["two_reached"] is True


def test_process_dataset_restricts_to_requested_levels(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "ToyCase2"
    dataset_dir.mkdir()
    desc = dataset_dir / "description.md"
    desc.write_text("A Customer places an Order.", encoding="utf-8")

    def fake_rewrite_to_shape(client, cfg, original, original_score, reference, level_name, l3_values, system_prompt, user_prompt_fn, metric_guidance):
        from text.rewrite.loop import ShapeLevelResult
        return ShapeLevelResult(
            level_name=level_name, final_z=0.1, shape_checks=(), reached=True,
            iterations=1, text=f"[{level_name} rewrite of source]",
        )

    monkeypatch.setattr("text.rewrite.run.rewrite_to_shape", fake_rewrite_to_shape)

    cfg = RewriteConfig()
    row = process_dataset(
        name="ToyCase2", description_path=desc, cfg=cfg, reference=_reference(),
        tconf=DEFAULT_CONFIG, client=None, levels=("zero",),
    )

    assert (dataset_dir / "description_level_zero.md").is_file()
    assert not (dataset_dir / "description_level_one.md").exists()
    assert "one_reached" not in row
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /Users/marcocalamo/text2uml && /Users/marcocalamo/anaconda3/envs/kul/bin/python -m pytest tests/text/rewrite/test_run_process_dataset.py -v
```
Expected: FAIL with `ImportError: cannot import name 'process_dataset'`.

- [ ] **Step 4: Implement `process_dataset` and rewire `main()` in `text/rewrite/run.py`**

Replace the body of the existing per-dataset loop in `main()` with a call to
a new top-level function, and drive all three levels:

```python
import functools
from pathlib import Path
from typing import Tuple

from .loop import rewrite_to_shape
from .prompts import METRIC_GUIDANCE, build_shape_user_prompt, system_prompt
from .shape_targets import shape_ok
from .structural_prompts import STRUCTURAL_METRIC_GUIDANCE, build_structural_user_prompt, structural_system_prompt

_LEVEL_TAGS: Tuple[str, ...] = ("zero", "one", "two")
_LEVEL_LABELS = {
    "one": "a simplified, easy-to-read narrative (level one: simplest)",
    "two": "a moderately simplified narrative (level two: mid-complexity)",
}


def _level_setup(cfg: RewriteConfig, tag: str):
    """(output level-name, system prompt, user-prompt fn, metric guidance) for one level tag."""
    if tag == "zero":
        return cfg.level_zero_name, structural_system_prompt(), build_structural_user_prompt, STRUCTURAL_METRIC_GUIDANCE
    out_name = cfg.level_one_name if tag == "one" else cfg.level_two_name
    user_fn = functools.partial(build_shape_user_prompt, level_label=_LEVEL_LABELS[tag])
    return out_name, system_prompt(), user_fn, METRIC_GUIDANCE


def process_dataset(
    name: str,
    description_path: Path,
    cfg: RewriteConfig,
    reference,
    tconf,
    client,
    levels: Tuple[str, ...] = _LEVEL_TAGS,
    force: bool = False,
) -> dict:
    """Regenerate the requested complexity levels for one dataset case.

    Returns a summary row: ``sub_folder_name``, ``actual_z``, and per
    processed level ``<tag>_reached`` / ``<tag>_iterations`` / ``<tag>_shape_ok``.
    """
    original = description_path.read_text(encoding="utf-8")
    base = score_text(original, reference, tconf)
    row: dict = {"sub_folder_name": name, "actual_z": round(base.z_index, 4)}

    for tag in levels:
        out_level_name, sprompt, user_fn, guidance = _level_setup(cfg, tag)
        out_path = cfg.output_path(description_path.parent, out_level_name)
        if out_path.is_file() and not force:
            logger.info("%s/%s: exists, skipping (pass force=True to regenerate)", name, tag)
            continue
        result = rewrite_to_shape(
            client=client, cfg=cfg, original=original, original_score=base,
            reference=reference, level_name=tag, l3_values=base.values,
            system_prompt=sprompt, user_prompt_fn=user_fn, metric_guidance=guidance,
        )
        out_path.write_text(result.text.rstrip() + "\n", encoding="utf-8")
        row[f"{tag}_reached"] = result.reached
        row[f"{tag}_iterations"] = result.iterations
        row[f"{tag}_shape_ok"] = shape_ok(result.shape_checks)
        logger.info("%s/%s: reached=%s in %d iter(s)", name, tag, result.reached, result.iterations)

    return row
```

Note `user_prompt_fn` in `rewrite_to_shape` (Task 3) is called as
`user_prompt_fn(original, candidate_text, feedback)` — three positional
args. `functools.partial(build_shape_user_prompt, level_label=...)` binds
the keyword-only `level_label`, leaving exactly `(original, current_text,
feedback)` as the remaining positional signature, so the partial is a drop-in
match; `build_structural_user_prompt(original, current_text, feedback)`
already matches directly.

Now replace `main()`'s existing per-dataset loop (the `for path in paths:`
block that calls `rewrite_to_target` twice per dataset) with a call to
`process_dataset`. The rest of `main()` — argument parsing, `_configure_logging`,
building `reference` via `build_reference`, `_iter_description_paths` — is
unchanged; only the body of the loop and the summary-write at the end change:

```python
    rows = []
    for path in paths:
        name = path.parent.name
        try:
            base = score_text(path.read_text(encoding="utf-8"), reference, tconf)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", name, exc)
            continue

        if args.dry_run:
            logger.info("%s: actual z_index=%.2f (dry run, no rewrite)", name, base.z_index)
            continue

        row = process_dataset(name, path, cfg, reference, tconf, client)
        rows.append(row)

    if rows and not args.dry_run:
        out = tconf.output_dir / "rewrite_shape_summary.csv"
        tconf.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        logger.info("Wrote shape summary for %d datasets to %s", len(rows), out)
```

This writes to `text/output/rewrite_shape_summary.csv` — a new filename, so
the existing `rewrite_summary.csv` (the historical record of the old
aggregate-z_index-only run) is left untouched.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /Users/marcocalamo/text2uml && /Users/marcocalamo/anaconda3/envs/kul/bin/python -m pytest tests/text/rewrite/test_run_process_dataset.py -v
```
Expected: both tests PASS.

- [ ] **Step 6: Run the full `text/rewrite` test suite to check nothing broke**

```bash
cd /Users/marcocalamo/text2uml && /Users/marcocalamo/anaconda3/envs/kul/bin/python -m pytest tests/text/rewrite/ -v
```
Expected: all tests across Tasks 1-4 PASS.

- [ ] **Step 7: Commit**

```bash
git add text/rewrite/run.py text/rewrite/prompts.py tests/text/rewrite/test_run_process_dataset.py
git commit -m "refactor(rewrite): extract process_dataset and drive all 3 levels through the shape loop"
```

---

### Task 5: Corpus-level closed loop (`experiments/levels/shape_loop.py`)

**Files:**
- Create: `experiments/levels/shape_loop.py`
- Test: `tests/experiments/levels/test_shape_loop.py`

**Interfaces:**
- Consumes: `text.rewrite.shape_targets.check_case_shape`, `shape_ok` (Task 1); `text.rewrite.run.process_dataset` (Task 4); `experiments.levels.complexity.compute_level_metrics`, `write_complexity_csv`, `generate_all` (existing); `experiments.levels.case_metrics.plot_case` (existing).
- Produces: `find_noncompliant_cases(df: pd.DataFrame) -> Dict[str, List[str]]` (case name -> list of failing level tags, pure function over an already-loaded `levels_complexity.csv`-shaped DataFrame), `run_shape_loop(cfg=DEFAULT_LEVELS_CONFIG, rewrite_cfg=DEFAULT_REWRITE_CONFIG, max_retries: int = 2) -> pd.DataFrame` (the compliance report).

- [ ] **Step 1: Write the failing test for the pure re-check function**

```python
# tests/experiments/levels/test_shape_loop.py
from __future__ import annotations

import pandas as pd

from experiments.levels.shape_loop import find_noncompliant_cases


def _row(case, level, rank, mdd, sub, ctx):
    return {
        "sub_folder_name": case, "level": level, "level_rank": rank,
        "mdd": mdd, "subordination_index": sub, "context_dependence_proxy": ctx,
    }


def test_find_noncompliant_cases_flags_only_failing_case():
    rows = [
        # Case A: shape holds.
        _row("A", "zero", 0, 3.4, 0.2, 0.9),
        _row("A", "one", 1, 2.0, 0.15, 0.2),
        _row("A", "two", 2, 2.4, 0.3, 0.35),
        _row("A", "three", 3, 3.0, 0.5, 0.4),
        # Case B: zero's mdd does not exceed three's -> fails.
        _row("B", "zero", 0, 2.5, 0.2, 0.9),
        _row("B", "one", 1, 2.0, 0.15, 0.2),
        _row("B", "two", 2, 2.4, 0.3, 0.35),
        _row("B", "three", 3, 3.0, 0.5, 0.4),
    ]
    df = pd.DataFrame(rows)
    failing = find_noncompliant_cases(df)
    assert "A" not in failing
    assert "B" in failing
    assert "zero" in failing["B"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/marcocalamo/text2uml && /Users/marcocalamo/anaconda3/envs/kul/bin/python -m pytest tests/experiments/levels/test_shape_loop.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.levels.shape_loop'`.

- [ ] **Step 3: Implement `experiments/levels/shape_loop.py`**

```python
"""Corpus-level closed feedback loop for the four description complexity levels.

Runs the per-case shape-aware rewrite (``text.rewrite.run.process_dataset``)
for every dataset, recomputes the corpus metrics, and re-checks each case's
four-level shape independently from the freshly computed CSV (not the
per-text loop's self-report). Any case that still fails goes back through
``process_dataset`` for just its failing level(s), up to ``max_retries``
extra corpus-level passes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from text.rewrite.config import DEFAULT_REWRITE_CONFIG, RewriteConfig
from text.rewrite.run import process_dataset
from text.rewrite.scorer import build_reference
from text.rewrite.shape_targets import check_case_shape, shape_ok

from .case_metrics import plot_case
from .complexity import compute_level_metrics, generate_all, write_complexity_csv
from .config import DEFAULT_LEVELS_CONFIG, LevelsConfig

logger = logging.getLogger(__name__)

_METRIC_COLS = ("mdd", "subordination_index", "context_dependence_proxy")
_COMPLIANCE_CSV = "shape_compliance_report.csv"


def find_noncompliant_cases(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Case name -> failing non-three level tags, per ``check_case_shape``.

    Per-level checks (``check_shape``) name the exact offending level
    ("zero"/"one"/"two"). The two cross-level max checks in
    ``check_case_shape`` name the level that's *supposed* to be the max
    ("three" for subordination_index, "zero" for context_dependence_proxy) —
    that tells us which constraint broke, not which of zero/one/two is the
    culprit that needs to change. When only a cross-level check fails (no
    specific zero/one/two level also failed its own local check), the safe
    fix is to regenerate all three non-real levels for that case, since we
    can't cheaply tell which one is currently over/under the real spec.
    """
    failing: Dict[str, List[str]] = {}
    for case, sub in df.groupby("sub_folder_name"):
        levels = {
            row["level"]: {m: row[m] for m in _METRIC_COLS if m in row}
            for _, row in sub.iterrows()
        }
        checks = check_case_shape(levels)
        if shape_ok(checks):
            continue
        bad_levels = sorted({
            c.level for c in checks
            if not (c.degenerate or (c.rank_ok and c.band_ok)) and c.level in ("zero", "one", "two")
        })
        failing[case] = bad_levels or ["zero", "one", "two"]
    return failing


def _iter_dataset_paths(levels_cfg: LevelsConfig) -> List[Path]:
    return sorted(p / "description.md" for p in levels_cfg.dataset_dir.iterdir() if p.is_dir() and (p / "description.md").is_file())


def run_shape_loop(
    levels_cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG,
    rewrite_cfg: RewriteConfig = DEFAULT_REWRITE_CONFIG,
    max_retries: int = 2,
) -> pd.DataFrame:
    """Full closed loop: generate -> recheck -> retry outliers -> report."""
    from text.config import DEFAULT_CONFIG
    from text.rewrite.client import make_client

    reference = build_reference(DEFAULT_CONFIG)
    client = make_client()

    for desc_path in _iter_dataset_paths(levels_cfg):
        process_dataset(desc_path.parent.name, desc_path, rewrite_cfg, reference, DEFAULT_CONFIG, client)

    df = compute_level_metrics(levels_cfg)
    write_complexity_csv(df, levels_cfg)

    for attempt in range(1, max_retries + 1):
        failing = find_noncompliant_cases(df)
        if not failing:
            logger.info("Shape loop converged after %d retry pass(es).", attempt - 1)
            break
        logger.info("Retry pass %d: %d non-compliant case(s): %s", attempt, len(failing), sorted(failing))
        for case, bad_levels in failing.items():
            desc_path = levels_cfg.dataset_dir / case / "description.md"
            process_dataset(case, desc_path, rewrite_cfg, reference, DEFAULT_CONFIG, client, levels=tuple(bad_levels), force=True)
        df = compute_level_metrics(levels_cfg, only=list(failing))
        write_complexity_csv(df, levels_cfg)
        df = compute_level_metrics(levels_cfg)

    report_rows = []
    for case, sub in df.groupby("sub_folder_name"):
        levels = {row["level"]: {m: row[m] for m in _METRIC_COLS if m in row} for _, row in sub.iterrows()}
        for check in check_case_shape(levels):
            report_rows.append({
                "sub_folder_name": case, "metric": check.metric, "level": check.level,
                "value": check.value, "l3_value": check.l3_value, "ratio": check.ratio,
                "rank_ok": check.rank_ok, "band_ok": check.band_ok, "degenerate": check.degenerate,
            })
    report = pd.DataFrame(report_rows)
    levels_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(levels_cfg.output_dir / _COMPLIANCE_CSV, index=False)
    logger.info("Wrote shape compliance report to %s", levels_cfg.output_dir / _COMPLIANCE_CSV)

    generate_all(levels_cfg, DEFAULT_CONFIG, df=df)
    for case in df["sub_folder_name"].unique():
        plot_case(case, levels_cfg)

    return report


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the corpus-level shape closed loop.")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    report = run_shape_loop(max_retries=args.max_retries)
    n_fail = int((~report["rank_ok"] | (~report["band_ok"] & ~report["degenerate"])).sum())
    logger.info("Done. %d shape-check row(s) still failing after all retries.", n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/marcocalamo/text2uml && /Users/marcocalamo/anaconda3/envs/kul/bin/python -m pytest tests/experiments/levels/test_shape_loop.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/levels/shape_loop.py tests/experiments/levels/test_shape_loop.py
git commit -m "feat(levels): add corpus-level shape closed loop with auto-retry"
```

---

### Task 6 (operational): Calibration dry run on the GasStation reference cases

Not a code task — validates Tasks 1-5 against the two cases whose shape we
are replicating, before spending API budget on the other 43.

- [ ] **Step 1: Back up the hand-authored GasStation level files**

```bash
cd /Users/marcocalamo/text2uml
mkdir -p /tmp/gasstation_reference_backup
cp dataset/GasStation_KUL/description_level_*.md /tmp/gasstation_reference_backup/
cp dataset/GasStation_TUW/description_level_*.md /tmp/gasstation_reference_backup/
```

- [ ] **Step 2: Regenerate just those two cases with `force=True` via a throwaway script**

```bash
/Users/marcocalamo/anaconda3/envs/kul/bin/python - <<'EOF'
from pathlib import Path
from text.config import DEFAULT_CONFIG
from text.rewrite.config import DEFAULT_REWRITE_CONFIG
from text.rewrite.client import make_client
from text.rewrite.scorer import build_reference
from text.rewrite.run import process_dataset

reference = build_reference(DEFAULT_CONFIG)
client = make_client()
for case in ("GasStation_KUL", "GasStation_TUW"):
    desc = Path("dataset") / case / "description.md"
    row = process_dataset(case, desc, DEFAULT_REWRITE_CONFIG, reference, DEFAULT_CONFIG, client, force=True)
    print(case, row)
EOF
```

- [ ] **Step 3: Recompute metrics and check the compliance report for just these two cases**

```bash
/Users/marcocalamo/anaconda3/envs/kul/bin/python -m experiments.levels.complexity -v
/Users/marcocalamo/anaconda3/envs/kul/bin/python - <<'EOF'
import pandas as pd
from experiments.levels.shape_loop import find_noncompliant_cases
df = pd.read_csv("experiments/levels/output/levels_complexity.csv")
df = df[df["sub_folder_name"].isin(["GasStation_KUL", "GasStation_TUW"])]
print(find_noncompliant_cases(df))
EOF
```
Expected: `{}` (empty dict — both cases pass). If not empty, inspect the
regenerated `description_level_*.md` files against the backups, adjust the
ratio bands in `text/rewrite/shape_targets.py` (Task 1) or the prompts
(Task 2/existing narrative prompt) if the bands prove systematically
unreachable, and repeat this task before proceeding.

- [ ] **Step 4: Restore the reference files (this task is calibration only, not the real run)**

```bash
cp /tmp/gasstation_reference_backup/*.md dataset/GasStation_KUL/  # then move TUW's files back to their own folder
```
Run per-file, matching filenames back to `dataset/GasStation_KUL/` and
`dataset/GasStation_TUW/` respectively; `git status` afterward should show
no diff in either case's `description_level_*.md`.

---

### Task 7 (operational): Full batch run + corpus closed loop

- [ ] **Step 1: Run the corpus-level shape loop across every case**

```bash
cd /Users/marcocalamo/text2uml
/Users/marcocalamo/anaconda3/envs/kul/bin/python -m experiments.levels.shape_loop -v --max-retries 2
```
This calls `process_dataset` for all cases (including GasStation_KUL/TUW —
expect those two to already be shape-compliant and need no further
retries), recomputes `levels_complexity.csv`, retries non-compliant cases up
to twice, and writes `experiments/levels/output/shape_compliance_report.csv`
plus refreshed plots.

- [ ] **Step 2: Review the compliance report**

```bash
/Users/marcocalamo/anaconda3/envs/kul/bin/python - <<'EOF'
import pandas as pd
report = pd.read_csv("experiments/levels/output/shape_compliance_report.csv")
failing = report[~report["rank_ok"] | (~report["band_ok"] & ~report["degenerate"])]
print(f"{len(failing)} failing shape-check row(s) out of {len(report)}")
print(failing.to_string(index=False))
EOF
```
Share this output with the user before proceeding — some residual failures
after the retry cap are expected to be discussed, not silently accepted.

- [ ] **Step 3: Commit the regenerated dataset descriptions and outputs**

```bash
git add dataset/*/description_level_zero.md dataset/*/description_level_one.md dataset/*/description_level_two.md
git add experiments/levels/output/ text/output/rewrite_shape_summary.csv
git commit -m "data: regenerate all description levels through the shape-aware loop"
```

---

### Task 8: Self-review

- [ ] Re-read `docs/superpowers/specs/2026-07-04-levels-shape-aware-generation-design.md` section by section and confirm each is covered: Target shape rules (Task 1), level-zero genre prompt (Task 2), shape-aware loop (Task 3), batch driver regenerating all three levels (Task 4), corpus-level closed loop with auto-retry (Task 5), rollout calibration-then-full-run sequencing (Tasks 6-7).
- [ ] Confirm no task references a function/type not defined by an earlier task (`ShapeLevelResult`, `check_shape`/`check_case_shape`/`shape_ok`/`format_feedback`, `structural_system_prompt`/`build_structural_user_prompt`/`STRUCTURAL_METRIC_GUIDANCE`, `process_dataset`, `find_noncompliant_cases`/`run_shape_loop` all defined before first use).
- [ ] Confirm `text/rewrite/prompts.py`'s rename (`_METRIC_GUIDANCE` → `METRIC_GUIDANCE`) is the only breaking change to an existing public name, and that Task 4 Step 1 updates its one call site.
