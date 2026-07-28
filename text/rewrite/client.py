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
        _load_dotenv_once()
        import ollama

        return ollama.Client(timeout=600.0)

    if provider == "openai":
        _load_dotenv_once()
        import openai

        return openai.OpenAI()

    if provider == "openrouter":
        _load_dotenv_once()
        import os

        import openai

        return openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

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
            stream=True,
        )
        parts = []
        import sys
        for chunk in response:
            content = chunk.get("message", {}).get("content", "")
            parts.append(content)
            sys.stdout.write(content)
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
        return "".join(parts).strip()

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

    if cfg.provider == "openrouter":
        # Separate branch from "openai" (rather than folding into it) so
        # neither provider's call path is touched by adding the other.
        response = client.chat.completions.create(
            model=cfg.model,
            temperature=cfg.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content.strip()

    # The installed anthropic SDK's typed `thinking=`/`output_config=` params
    # don't recognize "adaptive"/effort-based thinking (a newer server-side
    # feature this SDK version predates) -- extra_body bypasses the client's
    # type validation and merges these fields into the raw request body,
    # which the API itself does support for this model.
    with client.messages.stream(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        extra_body={"thinking": {"type": "adaptive"}, "output_config": {"effort": cfg.effort}},
    ) as stream:
        message = stream.get_final_message()

    text = "".join(block.text for block in message.content if block.type == "text")
    return text.strip()
