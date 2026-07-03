"""Thin wrapper around the Anthropic SDK for single-shot rewrites.

Uses streaming with adaptive thinking (the recommended defaults for Claude
Opus 4.8) so large / thoughtful rewrites don't hit request timeouts.
"""

from __future__ import annotations

import logging

from .config import RewriteConfig

logger = logging.getLogger(__name__)


def make_client():
    """Construct an Anthropic client, loading a local .env if python-dotenv exists.

    Credentials resolve from the environment (``ANTHROPIC_API_KEY`` or
    ``ANTHROPIC_AUTH_TOKEN``).
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # dotenv is optional; env vars may already be set.
        logger.debug("python-dotenv not installed; relying on existing environment.")

    import anthropic

    return anthropic.Anthropic()


def rewrite_once(client, cfg: RewriteConfig, system: str, user: str) -> str:
    """Send one rewrite request and return the assistant's text response."""
    with client.messages.stream(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        thinking={"type": "adaptive"},
        output_config={"effort": cfg.effort},
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        message = stream.get_final_message()

    text = "".join(block.text for block in message.content if block.type == "text")
    return text.strip()
