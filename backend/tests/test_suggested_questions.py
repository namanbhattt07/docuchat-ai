import asyncio

from app.services import suggestions as suggestions_module
from app.services.ollama import OllamaUnavailable
from app.services.suggestions import _parse_questions, generate_suggested_questions


def test_parse_questions_reads_clean_json_array() -> None:
    raw = '["What is MQTT?", "What are the four pillars of IoT?", "How does M2M differ from IoT?"]'
    assert _parse_questions(raw, count=5) == [
        "What is MQTT?",
        "What are the four pillars of IoT?",
        "How does M2M differ from IoT?",
    ]


def test_parse_questions_strips_think_block() -> None:
    raw = '<think>reasoning here</think>["What is MQTT?"]'
    assert _parse_questions(raw, count=5) == ["What is MQTT?"]


def test_parse_questions_caps_to_requested_count() -> None:
    raw = '["a?", "b?", "c?", "d?"]'
    assert _parse_questions(raw, count=2) == ["a?", "b?"]


def test_parse_questions_deduplicates() -> None:
    raw = '["What is MQTT?", "What is MQTT?", "What is IoT?"]'
    assert _parse_questions(raw, count=5) == ["What is MQTT?", "What is IoT?"]


def test_parse_questions_falls_back_to_line_by_line_when_json_is_malformed() -> None:
    raw = "1. What is MQTT?\n2. What is IoT?\n"
    result = _parse_questions(raw, count=5)
    assert "What is MQTT?" in result
    assert "What is IoT?" in result


def test_generate_suggested_questions_uses_generate(monkeypatch) -> None:
    captured = {}

    async def fake_generate(messages, **kwargs):
        captured["messages"] = messages
        return '["What is MQTT?", "What is IoT?", "What are sensors?", "What is M2M?", "What protocols are covered?"]'

    monkeypatch.setattr(suggestions_module, "generate", fake_generate)

    questions = asyncio.run(generate_suggested_questions(["MQTT is a lightweight IoT protocol used for M2M communication with sensors."], count=5))

    assert len(questions) == 5
    assert "MQTT" in captured["messages"][0]["content"]


def test_generate_suggested_questions_returns_empty_when_ollama_is_down(monkeypatch) -> None:
    async def failing_generate(messages, **kwargs):
        raise OllamaUnavailable("down")

    monkeypatch.setattr(suggestions_module, "generate", failing_generate)

    questions = asyncio.run(generate_suggested_questions(["Some document text about IoT."], count=5))

    assert questions == []


def test_generate_suggested_questions_returns_empty_for_blank_document(monkeypatch) -> None:
    async def fake_generate(messages, **kwargs):
        raise AssertionError("should not call generate() for a document with no extractable text")

    monkeypatch.setattr(suggestions_module, "generate", fake_generate)

    questions = asyncio.run(generate_suggested_questions(["", "   "], count=5))

    assert questions == []


def test_generate_suggested_questions_never_raises_on_unparsable_output(monkeypatch) -> None:
    async def fake_generate(messages, **kwargs):
        return ""

    monkeypatch.setattr(suggestions_module, "generate", fake_generate)

    questions = asyncio.run(generate_suggested_questions(["Some document text about IoT."], count=5))

    assert questions == []
