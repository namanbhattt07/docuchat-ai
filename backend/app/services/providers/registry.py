from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Iterator

from app.core.config import get_settings
from app.services.providers.base import EmbeddingProvider, LLMProvider, VisionProvider
from app.services.providers.ollama import OllamaEmbeddingProvider, OllamaLLMProvider, OllamaVisionProvider, list_installed_models

# Deliberately not a plugin framework: one active instance per role, created
# lazily (Ollama by default), replaceable with set_*() -- which is all a test
# or a future alternative backend needs. Services never import a concrete
# provider; they call services/ollama.py's `generate` / `embed` (or accept a
# provider argument, for the newer services), and those resolve the active
# instance here at call time.

_llm: LLMProvider | None = None
_embedding: EmbeddingProvider | None = None
_vision: VisionProvider | None = None


def get_llm_provider() -> LLMProvider:
    global _llm
    if _llm is None:
        _llm = OllamaLLMProvider()
    return _llm


def get_embedding_provider() -> EmbeddingProvider:
    global _embedding
    if _embedding is None:
        _embedding = OllamaEmbeddingProvider()
    return _embedding


def set_llm_provider(provider: LLMProvider | None) -> None:
    global _llm
    _llm = provider


def set_embedding_provider(provider: EmbeddingProvider | None) -> None:
    global _embedding
    _embedding = provider


def set_vision_provider(provider: VisionProvider | None) -> None:
    """An explicitly-set vision provider always wins over auto-detection --
    the hook tests (and any future non-Ollama backend) use."""
    global _vision
    _vision = provider


@contextmanager
def use_providers(*, llm: LLMProvider | None = None, embedding: EmbeddingProvider | None = None, vision: VisionProvider | None = None) -> Iterator[None]:
    """Temporarily install providers, restoring the previous ones on exit."""
    global _llm, _embedding, _vision
    previous = (_llm, _embedding, _vision)
    if llm is not None:
        _llm = llm
    if embedding is not None:
        _embedding = embedding
    if vision is not None:
        _vision = vision
    try:
        yield
    finally:
        _llm, _embedding, _vision = previous


def reset_providers() -> None:
    global _llm, _embedding, _vision
    _llm = _embedding = _vision = None


@dataclass
class VisionCapability:
    """Whether the app can *actually* interpret images right now. `supported`
    is only ever true when a vision provider is injected, or a vision model
    is both configured (OLLAMA_VISION_MODEL) and confirmed installed in
    Ollama -- never inferred from the text model, which cannot read images.
    """

    supported: bool
    model: str | None
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


NO_VISION_REASON = "No local vision model is configured (set OLLAMA_VISION_MODEL to an installed vision-capable model)."


async def vision_capability() -> VisionCapability:
    if _vision is not None:
        return VisionCapability(True, _vision.model, "Vision provider injected.")
    configured = get_settings().ollama_vision_model.strip()
    if not configured:
        return VisionCapability(False, None, NO_VISION_REASON)
    installed = {model.get("name"): model for model in await list_installed_models()}
    entry = installed.get(configured)
    if entry is None:
        return VisionCapability(False, configured, f'The configured vision model "{configured}" is not installed in Ollama.')
    capabilities = entry.get("capabilities")
    # Older Ollama builds don't report capabilities; only reject when they
    # are reported and "vision" is absent (e.g. someone pointed the setting
    # at a text-only model).
    if capabilities is not None and "vision" not in capabilities:
        return VisionCapability(False, configured, f'The configured model "{configured}" does not report vision capability.')
    return VisionCapability(True, configured, "Configured local vision model is installed.")


async def get_vision_provider() -> VisionProvider | None:
    """The active vision provider, or None when there is no verified local
    vision model. Callers must handle None -- vision is never required."""
    if _vision is not None:
        return _vision
    capability = await vision_capability()
    if not capability.supported or not capability.model:
        return None
    return OllamaVisionProvider(capability.model)
