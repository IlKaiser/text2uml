"""Thin wrapper around the Anthropic SDK, OpenAI SDK, and a local Ollama
client for single-shot rewrites.

The Anthropic path streams with adaptive thinking (the recommended defaults
for Claude Opus 4.8) so large / thoughtful rewrites don't hit request
timeouts. The OpenAI and Ollama paths are a plain single-shot chat call --
no thinking/effort knobs to set.
"""

from __future__ import annotations

import logging

from .config import RewriteConfig

logger = logging.getLogger(__name__)


def _load_dotenv_once() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # dotenv is optional; env vars may already be set.
        logger.debug("python-dotenv not installed; relying on existing environment.")


def make_client(provider: str = "anthropic"):
    """Construct the LLM client for ``provider``.

    anthropic: credentials resolve from the environment (``ANTHROPIC_API_KEY``
    or ``ANTHROPIC_AUTH_TOKEN``), loading a local .env if python-dotenv exists.
    openai: credentials resolve from ``OPENAI_API_KEY``, same .env loading.
    ollama: talks to the local Ollama server (default http://localhost:11434);
    no credentials needed.
    """
    if provider == "ollama":
        import ollama

        return ollama.Client()

    if provider == "openai":
        _load_dotenv_once()
        import openai

        return openai.OpenAI()

    _load_dotenv_once()
    import anthropic

    return anthropic.Anthropic()


def rewrite_once(client, cfg: RewriteConfig, system: str, user: str) -> str:
    """Send one rewrite request and return the assistant's text response."""
    if cfg.provider == "ollama":
        response = client.chat(
            model=cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"temperature": cfg.temperature},
        )
        return response["message"]["content"].strip()

    if cfg.provider == "openai":
        response = client.chat.completions.create(
            model=cfg.model,
            temperature=cfg.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content.strip()

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
