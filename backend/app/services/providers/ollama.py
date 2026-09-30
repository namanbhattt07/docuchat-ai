import base64
import logging
from typing import Any

import httpx

from app.core.config import get_settings
from app.services.providers.base import ProviderUnavailable

# The only module that talks to the local Ollama runtime. Everything else
# reaches it through the LLMProvider / EmbeddingProvider / VisionProvider
# interfaces (providers/base.py) and their registry, so swapping the runtime
# -- or substituting a fake in a test -- never touches a service.

logger = logging.getLogger("docuchat.ollama")

_client: httpx.AsyncClient | None = None

# Room left for the model's own reply inside the context window, and a
# deliberately pessimistic characters-per-token figure (measured: ~3.4 for
# numeric-heavy prose, ~4 for ordinary prose) used only to *warn* that a prompt
# is likely to be truncated -- never to alter or drop anything.
_REPLY_TOKEN_RESERVE = 1024
_CHARS_PER_TOKEN = 3.0


def _warn_if_prompt_may_be_truncated(messages: list[dict[str, str]], num_ctx: int) -> None:
    estimated_tokens = sum(len(message.get("content", "")) for message in messages) / _CHARS_PER_TOKEN
    if estimated_tokens > num_ctx - _REPLY_TOKEN_RESERVE:
        logger.warning(
            "Prompt is ~%d tokens against a %d-token context window; Ollama will silently truncate it. "
            "Raise OLLAMA_NUM_CTX or lower the retrieval source limits.",
            estimated_tokens, num_ctx,
        )


def _http_client() -> httpx.AsyncClient:
    """Reuse one connection-pooled client for the process lifetime instead of
    opening a fresh connection for every embed/generate/status call -- with
    ~90 embedding requests per large-PDF upload this avoided overhead adds up.
    """
    global _client
    if _client is None:
        _client = httpx.AsyncClient(base_url=get_settings().ollama_base_url)
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


class OllamaLLMProvider:
    name = "ollama"

    async def generate(self, messages: list[dict[str, str]], *, think: bool | None = None) -> str:
        settings = get_settings()
        _warn_if_prompt_may_be_truncated(messages, settings.ollama_num_ctx)
        try:
            response = await _http_client().post(
                "/api/chat",
                json={
                    "model": settings.ollama_generation_model,
                    "messages": messages,
                    "stream": False,
                    "options": {"num_ctx": settings.ollama_num_ctx},
                    # Qwen3 emits a long internal <think>...</think> reasoning
                    # trace by default; we strip it anyway once parsed, and
                    # skipping it cuts generation time roughly 10x with no
                    # measurable answer-quality loss for this grounded-QA prompt.
                    "think": settings.ollama_generation_think if think is None else think,
                    # Keeps the model resident between requests so a user's next
                    # question doesn't pay the multi-second reload cost again.
                    "keep_alive": settings.ollama_keep_alive,
                },
                timeout=settings.ollama_generate_timeout_seconds,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except httpx.HTTPStatusError as exc:
            raise ProviderUnavailable(f"Ollama generation request failed ({exc.response.status_code}): {exc.response.text[:400]}") from exc
        except (httpx.RequestError, KeyError) as exc:
            raise ProviderUnavailable(f"Could not reach Ollama's generation service: {exc}") from exc

    async def status(self) -> dict[str, Any]:
        return await ollama_status()


class OllamaEmbeddingProvider:
    name = "ollama"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        settings = get_settings()
        try:
            response = await _http_client().post(
                "/api/embed",
                json={"model": settings.ollama_embedding_model, "input": texts},
                timeout=settings.ollama_embed_timeout_seconds,
            )
            response.raise_for_status()
            return response.json()["embeddings"]
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:400]
            raise ProviderUnavailable(f"Ollama embedding request failed ({exc.response.status_code}): {detail}") from exc
        except (httpx.RequestError, KeyError) as exc:
            raise ProviderUnavailable(f"Could not reach Ollama's embedding service: {exc}") from exc


class OllamaVisionProvider:
    """Sends one rendered image plus a question to an Ollama vision model via
    the standard `images` field of /api/chat. Only ever constructed when the
    configured vision model is verified installed (see
    providers/registry.py::vision_capability) -- it is never a fallback for
    the text model, which cannot read images.
    """

    name = "ollama"

    def __init__(self, model: str) -> None:
        self.model = model

    async def answer(self, image_png: bytes, question: str, context: str = "") -> str:
        settings = get_settings()
        prompt = question if not context else f"{question}\n\nText extracted from the same page (may be incomplete):\n{context}"
        try:
            response = await _http_client().post(
                "/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(image_png).decode("ascii")]}],
                    "stream": False,
                    "keep_alive": settings.ollama_keep_alive,
                },
                timeout=settings.ollama_generate_timeout_seconds,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except httpx.HTTPStatusError as exc:
            raise ProviderUnavailable(f"Ollama vision request failed ({exc.response.status_code}): {exc.response.text[:400]}") from exc
        except (httpx.RequestError, KeyError) as exc:
            raise ProviderUnavailable(f"Could not reach Ollama's vision service: {exc}") from exc


async def list_installed_models() -> list[dict[str, Any]]:
    """Raw /api/tags model entries; [] when Ollama is unreachable."""
    try:
        response = await _http_client().get("/api/tags", timeout=3)
        response.raise_for_status()
        return list(response.json().get("models", []))
    except (httpx.HTTPError, ValueError):
        return []


async def ollama_status() -> dict[str, Any]:
    settings = get_settings()
    try:
        response = await _http_client().get("/api/tags", timeout=3)
        response.raise_for_status()
        names = {model["name"] for model in response.json().get("models", [])}
        return {
            "available": True,
            "generation_model_ready": settings.ollama_generation_model in names,
            "embedding_model_ready": settings.ollama_embedding_model in names,
            "generation_model": settings.ollama_generation_model,
            "embedding_model": settings.ollama_embedding_model,
        }
    except httpx.HTTPError:
        return {
            "available": False, "generation_model_ready": False, "embedding_model_ready": False,
            "generation_model": settings.ollama_generation_model, "embedding_model": settings.ollama_embedding_model,
        }
