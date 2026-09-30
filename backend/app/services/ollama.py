from typing import Any

from app.services.providers.base import ProviderUnavailable
from app.services.providers.ollama import close_client, ollama_status
from app.services.providers.registry import get_embedding_provider, get_llm_provider

# Group 7 provider abstraction: this module used to *be* the Ollama client.
# It is now a thin facade over the injected LLMProvider / EmbeddingProvider
# (providers/), kept for two reasons: every service already imports
# `generate` / `embed` / `OllamaUnavailable` from here (and existing tests
# patch those names on the consuming modules), and it means a service gets
# provider injection without changing a single call site. The Ollama-specific
# code lives in providers/ollama.py.

# Alias, not a subclass: an `except OllamaUnavailable` must also catch the
# failure of any other provider (or a test fake) -- see providers/base.py.
OllamaUnavailable = ProviderUnavailable

__all__ = ["OllamaUnavailable", "close_client", "embed", "generate", "status"]


async def embed(texts: list[str]) -> list[list[float]]:
    return await get_embedding_provider().embed(texts)


async def generate(messages: list[dict[str, str]], *, think: bool | None = None) -> str:
    return await get_llm_provider().generate(messages, think=think)


async def status() -> dict[str, Any]:
    return await ollama_status()
