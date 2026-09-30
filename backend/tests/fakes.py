from typing import Any, Callable

from app.services.ocr import OcrLine


class FakeLLMProvider:
    """LLMProvider stand-in. `responder` is a string, an exception to raise,
    or a callable(messages) returning either -- so a test decides exactly what
    the "model" says without Ollama."""

    name = "fake"

    def __init__(self, responder: str | Exception | Callable[[list[dict[str, str]]], str | Exception] = "") -> None:
        self.responder = responder
        self.calls: list[dict[str, Any]] = []

    async def generate(self, messages: list[dict[str, str]], *, think: bool | None = None) -> str:
        self.calls.append({"messages": messages, "think": think})
        outcome = self.responder(messages) if callable(self.responder) else self.responder
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def status(self) -> dict[str, Any]:
        return {"available": True, "generation_model_ready": True, "embedding_model_ready": True}


class FakeEmbeddingProvider:
    """Deterministic 4-dimensional vectors; records every batch it was asked for."""

    name = "fake"
    dimensions = 4

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text) % 7), 1.0, 0.0, 0.5] for text in texts]


class FakeVisionProvider:
    name = "fake"
    model = "fake-vision:1b"

    def __init__(self, reply: str = "A bar chart with three bars.") -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    async def answer(self, image_png: bytes, question: str, context: str = "") -> str:
        self.calls.append({"image_bytes": len(image_png), "question": question, "context": context})
        return self.reply


class FakeOcrEngine:
    """OcrEngine stand-in. `result` is a list of OcrLine, an exception to
    raise, or a callable(png_bytes) returning either."""

    name = "fake-ocr"

    def __init__(self, result: list[OcrLine] | Exception | Callable[[bytes], list[OcrLine] | Exception] | None = None, *, available: bool = True, reason: str | None = None) -> None:
        self.result = result if result is not None else []
        self.available = available
        self.reason = reason
        self.calls = 0
        self.last_png: bytes | None = None

    def availability(self) -> tuple[bool, str | None]:
        return (True, None) if self.available else (False, self.reason or "fake OCR engine is not installed")

    def recognize(self, image_png: bytes) -> list[OcrLine]:
        self.calls += 1
        self.last_png = image_png
        outcome = self.result(image_png) if callable(self.result) else self.result
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
