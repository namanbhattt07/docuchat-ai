from typing import Any, Protocol, runtime_checkable


class ProviderUnavailable(RuntimeError):
    """A model provider could not serve a request (not running, timed out,
    returned an error). Every service that used to catch `OllamaUnavailable`
    now catches this -- `OllamaUnavailable` is kept as an alias of it (see
    services/ollama.py), so a fake or future provider raising it is handled
    exactly like Ollama being down.
    """


@runtime_checkable
class LLMProvider(Protocol):
    """Text generation, as the application actually uses it: one
    non-streaming chat completion. The app never streams and never asks the
    model for provider-enforced structured output (every JSON contract is
    prompted for and parsed defensively by its caller), so neither is part of
    this interface -- adding them here would be speculative surface nobody
    calls.
    """

    name: str

    async def generate(self, messages: list[dict[str, str]], *, think: bool | None = None) -> str: ...

    async def status(self) -> dict[str, Any]: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Batch text embedding. Vector dimensionality is the provider's
    property (it must match what is already stored in the vector index) --
    callers never assume one.
    """

    name: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class VisionProvider(Protocol):
    """Answer a question about one rendered page/figure image. Optional:
    nothing in normal document chat depends on it, and no implementation is
    active unless a local vision-capable model is configured *and* actually
    installed (see providers/registry.py::vision_capability).
    """

    name: str
    model: str

    async def answer(self, image_png: bytes, question: str, context: str = "") -> str: ...
