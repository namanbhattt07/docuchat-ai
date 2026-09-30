from collections import Counter
from pathlib import Path

from app.core.config import Settings

# Group 8 -- configuration. `.env.example` is what a new install copies to
# `.env`; a mistake there is invisible until it bites (it once defined
# OLLAMA_VISION_MODEL twice, and the empty second copy silently overrode the
# first).

EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"
FRONTEND_ONLY = {"NEXT_PUBLIC_API_BASE_URL"}  # read by Next.js, not by the backend Settings


def _example_keys() -> list[str]:
    lines = (line.strip() for line in EXAMPLE.read_text().splitlines())
    return [line.split("=", 1)[0] for line in lines if line and not line.startswith("#") and "=" in line]


def test_every_key_in_the_example_is_defined_once() -> None:
    duplicates = [key for key, count in Counter(_example_keys()).items() if count > 1]

    assert duplicates == []


def test_every_backend_key_in_the_example_is_a_real_setting() -> None:
    known = {name.upper() for name in Settings.model_fields}
    unknown = [key for key in _example_keys() if key not in known and key not in FRONTEND_ONLY]

    assert unknown == []  # a typo'd or removed setting would be silently ignored (extra="ignore")


def test_the_example_loads_and_enables_the_documented_defaults() -> None:
    settings = Settings(_env_file=EXAMPLE)

    assert settings.ollama_num_ctx == 8192
    assert settings.ollama_generation_think is False
    assert settings.ollama_vision_model == "qwen2.5vl:7b"  # the one definition, not overridden by a later blank
    assert settings.ocr_enabled is True
