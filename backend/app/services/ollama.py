from typing import Any

import httpx

from app.core.config import get_settings

settings = get_settings()

_client: httpx.AsyncClient | None = None


class OllamaUnavailable(RuntimeError):
    pass


def _http_client() -> httpx.AsyncClient:
    """Reuse one connection-pooled client for the process lifetime instead of
    opening a fresh connection for every embed/generate/status call -- with
    ~90 embedding requests per large-PDF upload this avoided overhead adds up.
    """
    global _client
    if _client is None:
        _client = httpx.AsyncClient(base_url=settings.ollama_base_url)
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def embed(texts: list[str]) -> list[list[float]]:
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
        raise OllamaUnavailable(f"Ollama embedding request failed ({exc.response.status_code}): {detail}") from exc
    except (httpx.RequestError, KeyError) as exc:
        raise OllamaUnavailable(f"Could not reach Ollama's embedding service: {exc}") from exc


async def generate(messages: list[dict[str, str]], *, think: bool | None = None) -> str:
    try:
        response = await _http_client().post(
            "/api/chat",
            json={
                "model": settings.ollama_generation_model,
                "messages": messages,
                "stream": False,
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
        raise OllamaUnavailable(f"Ollama generation request failed ({exc.response.status_code}): {exc.response.text[:400]}") from exc
    except (httpx.RequestError, KeyError) as exc:
        raise OllamaUnavailable(f"Could not reach Ollama's generation service: {exc}") from exc


async def status() -> dict[str, Any]:
    try:
        response = await _http_client().get("/api/tags", timeout=3)
        response.raise_for_status()
        names = {model["name"] for model in response.json().get("models", [])}
        return {
            "available": True,
            "generation_model_ready": settings.ollama_generation_model in names,
            "embedding_model_ready": settings.ollama_embedding_model in names,
        }
    except httpx.HTTPError:
        return {"available": False, "generation_model_ready": False, "embedding_model_ready": False}
