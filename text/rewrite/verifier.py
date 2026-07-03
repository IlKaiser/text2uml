"""Optional semantic-equivalence check for a rewritten description.

A second Claude call audits whether a candidate preserves every piece of
modelling-relevant information from the source (entities, attributes,
relationships, actions, constraints). Used by the feedback loop to *gate*
acceptance: a candidate that hits the complexity target but drops or alters
meaning is rejected, and the specific issues are fed back into the next rewrite.

Structured outputs constrain the response to a small JSON object, so parsing is
reliable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import List

from .config import RewriteConfig

logger = logging.getLogger(__name__)

_SCHEMA = {
    "type": "object",
    "properties": {
        "equivalent": {
            "type": "boolean",
            "description": "True only if the candidate preserves ALL modelling-"
            "relevant information from the source and adds none.",
        },
        "issues": {
            "type": "array",
            "items": {"type": "string"},
            "description": "One short line per problem: information dropped, "
            "altered, or added relative to the source.",
        },
    },
    "required": ["equivalent", "issues"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are a meticulous requirements auditor for UML model extraction. You "
    "compare a rewritten software description against its source and decide "
    "whether they are semantically EQUIVALENT for modelling purposes.\n\n"
    "They are equivalent only if the candidate preserves, with no additions:\n"
    "- every actor, entity, and class (and its name)\n"
    "- every attribute / data item each entity holds\n"
    "- every relationship or association (and its multiplicity / direction)\n"
    "- every action, operation, and use case, and which actor performs it\n"
    "- every business rule, constraint, and condition\n\n"
    "Wording, sentence structure, and reading difficulty are irrelevant — only "
    "the modelling content matters. Report each dropped, altered, or added item "
    "as one short issue."
)


@dataclass(frozen=True)
class MeaningCheck:
    """Result of the semantic-equivalence audit."""

    equivalent: bool
    issues: List[str] = field(default_factory=list)

    def feedback(self) -> str:
        if self.equivalent:
            return ""
        bullets = "\n".join(f"  - {i}" for i in self.issues) or "  - (unspecified)"
        return (
            "MEANING CHECK FAILED — your version changed the information relative "
            "to the source. Fix these while keeping the target complexity:\n"
            + bullets
        )


def verify_meaning(
    client, cfg: RewriteConfig, original: str, candidate: str
) -> MeaningCheck:
    """Audit whether ``candidate`` preserves the meaning of ``original``.

    On a call or parse failure, returns ``equivalent=True`` (does not block the
    loop) and logs a warning — the rewrite prompt already enforces meaning.
    """
    user = (
        "SOURCE:\n<source>\n" + original + "\n</source>\n\n"
        "CANDIDATE:\n<candidate>\n" + candidate + "\n</candidate>\n\n"
        "Return the JSON object per the schema."
    )
    try:
        with client.messages.stream(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            thinking={"type": "adaptive"},
            output_config={
                "effort": cfg.effort,
                "format": {"type": "json_schema", "schema": _SCHEMA},
            },
            system=_SYSTEM,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            message = stream.get_final_message()
        text = "".join(b.text for b in message.content if b.type == "text").strip()
        data = json.loads(text)
        return MeaningCheck(
            equivalent=bool(data.get("equivalent", True)),
            issues=[str(x) for x in data.get("issues", [])],
        )
    except Exception as exc:  # noqa: BLE001 - never block the loop on a checker error
        logger.warning("Meaning check failed to run/parse (%s); assuming equivalent.", exc)
        return MeaningCheck(equivalent=True, issues=[])
