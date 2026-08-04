"""Model resolution — one place that knows how to build a chat model.

Providers are named by a `prefix:id` spec so nothing downstream has to care
which one is in use:

    openrouter:deepseek/deepseek-v4-flash
    anthropic:claude-sonnet-5
    openai:gpt-5.5

The eval matrix, the CLI and the server all pass strings around; only this
module constructs clients.
"""

from __future__ import annotations

import os
from typing import Any

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: Sent by OpenRouter's convention so usage is attributable to this app.
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/Freddy503/wayfinder",
    "X-Title": "Wayfinder",
}


def provider_of(spec: Any) -> str:
    """The provider prefix of a model spec, or "" for a model instance."""
    if not isinstance(spec, str) or ":" not in spec:
        return ""
    return spec.split(":", 1)[0]


def required_key_for(spec: Any) -> str | None:
    """Which environment variable this model needs, for the preflight check."""
    return {
        "openrouter": "OPENROUTER_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }.get(provider_of(spec))


def resolve_model(spec: Any) -> Any:
    """Turn a model spec into something `create_deep_agent` accepts.

    Model instances pass through untouched (the tests inject fakes), and
    anything LangChain already understands is handed on as a string. Only
    OpenRouter needs building here, because it is the OpenAI wire format
    pointed at a different host.
    """
    if not isinstance(spec, str):
        return spec

    provider, _, model_id = spec.partition(":")
    if provider != "openrouter":
        return spec

    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        msg = (
            "OPENROUTER_API_KEY is unset — set it in .env (see .env.example) "
            f"to use {spec!r}."
        )
        raise RuntimeError(msg)

    return ChatOpenAI(
        model=model_id,
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_BASE_URL,
        default_headers=OPENROUTER_HEADERS,
        # The harness lives or dies on well-formed tool calls; sampling
        # variation buys nothing here and costs schema validity.
        temperature=0,
        max_retries=3,
    )


def label_of(spec: Any) -> str:
    """A short human label for experiment names and the UI."""
    if not isinstance(spec, str):
        return type(spec).__name__
    return spec.split("/")[-1].split(":")[-1]
