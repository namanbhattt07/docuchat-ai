import importlib.util
from dataclasses import dataclass
from threading import Lock
from typing import Protocol, runtime_checkable

# Group 7 OCR: a small engine interface so the ingestion pipeline (see
# services/pdf_provider.py) never depends on a specific OCR library, plus one
# real implementation that runs entirely on this machine (RapidOCR: ONNX
# models bundled in the pip wheel, no network access, no cloud API).

OCR_SETUP_MESSAGE = (
    "OCR is not available: the local OCR engine is not installed. Install it with "
    "`pip install rapidocr` (see backend/requirements.txt), restart the backend, and re-upload the PDF."
)


class OcrUnavailable(RuntimeError):
    pass


@dataclass
class OcrLine:
    """One recognized line of text. `bbox` is (x0, y0, x1, y1) in the pixel
    space of the image that was recognized; the caller maps it back to page
    coordinates."""

    text: str
    bbox: tuple[float, float, float, float]
    confidence: float


@runtime_checkable
class OcrEngine(Protocol):
    name: str

    def availability(self) -> tuple[bool, str | None]:
        """(usable, reason-if-not). Cheap and side-effect free."""
        ...

    def recognize(self, image_png: bytes) -> list[OcrLine]:
        """Lines in reading order. Raises on engine failure; returns [] when
        the image simply contains no text."""
        ...


class RapidOcrEngine:
    name = "rapidocr"

    def __init__(self) -> None:
        self._engine = None
        self._lock = Lock()

    def availability(self) -> tuple[bool, str | None]:
        if importlib.util.find_spec("rapidocr") is None:
            return False, OCR_SETUP_MESSAGE
        return True, None

    def _load(self):
        if self._engine is None:
            from rapidocr import RapidOCR  # imported lazily: heavy, and optional

            # RapidOCR logs several INFO lines per model load and resets its
            # own logger level from this config key, so this is the only knob
            # that quiets it -- that output is noise in the API log.
            self._engine = RapidOCR(params={"Global.log_level": "warning"})
        return self._engine

    def recognize(self, image_png: bytes) -> list[OcrLine]:
        available, reason = self.availability()
        if not available:
            raise OcrUnavailable(reason)
        # One inference at a time: the engine object is shared and OCR is
        # CPU-bound, so serializing costs nothing and avoids relying on the
        # library's thread-safety.
        with self._lock:
            result = self._load()(image_png)
        texts = list(getattr(result, "txts", None) or [])
        scores = list(getattr(result, "scores", None) or [])
        boxes = getattr(result, "boxes", None)
        lines: list[OcrLine] = []
        for index, text in enumerate(texts):
            if boxes is None or index >= len(boxes):
                continue
            quad = boxes[index]
            xs = [float(point[0]) for point in quad]
            ys = [float(point[1]) for point in quad]
            confidence = float(scores[index]) if index < len(scores) else 0.0
            lines.append(OcrLine(text=str(text), bbox=(min(xs), min(ys), max(xs), max(ys)), confidence=confidence))
        return lines


_engine: OcrEngine | None = None


def get_ocr_engine() -> OcrEngine:
    global _engine
    if _engine is None:
        _engine = RapidOcrEngine()
    return _engine


def set_ocr_engine(engine: OcrEngine | None) -> None:
    global _engine
    _engine = engine


def ocr_status() -> dict:
    """What /system/status reports: whether OCR can run right now, honestly."""
    from app.core.config import get_settings

    engine = get_ocr_engine()
    available, reason = engine.availability()
    return {"enabled": get_settings().ocr_enabled, "available": available, "engine": engine.name, "detail": reason}
