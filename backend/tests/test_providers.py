import asyncio
import base64
import json

import httpx
import pytest
from fakes import FakeEmbeddingProvider, FakeLLMProvider, FakeVisionProvider

import app.api.v1.chat as chat_module
import app.services.providers.ollama as ollama_provider_module
import app.services.providers.registry as registry_module
from app.api.v1.chat import ChatRequest, ask
from app.core.config import get_settings
from app.models import Chunk, Document
from app.services import ollama as ollama_facade
from app.services.documents import embed_in_batches
from app.services.providers import (
    EmbeddingProvider,
    LLMProvider,
    ProviderUnavailable,
    VisionProvider,
    get_embedding_provider,
    get_llm_provider,
    get_vision_provider,
    set_embedding_provider,
    set_llm_provider,
    set_vision_provider,
    use_providers,
    vision_capability,
)
from app.services.providers.ollama import OllamaEmbeddingProvider, OllamaLLMProvider, OllamaVisionProvider
from app.services.retrieval import rewrite_standalone_question
from app.services.suggestions import generate_suggested_questions

# ---------------------------------------------------------------------------
# Group 7 provider abstraction: interface conformance, the Ollama
# implementation's HTTP contract (mock transport -- no Ollama needed), the
# injection seam, and proof that existing services run on fakes alone.
# ---------------------------------------------------------------------------


def _install_mock_ollama(monkeypatch, handler) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(recording), base_url="http://ollama.test")
    monkeypatch.setattr(ollama_provider_module, "_client", client)
    return seen


# ---------- interface conformance ----------


def test_ollama_and_fake_providers_satisfy_the_interfaces() -> None:
    assert isinstance(OllamaLLMProvider(), LLMProvider)
    assert isinstance(OllamaEmbeddingProvider(), EmbeddingProvider)
    assert isinstance(OllamaVisionProvider("some-vision-model"), VisionProvider)
    assert isinstance(FakeLLMProvider(), LLMProvider)
    assert isinstance(FakeEmbeddingProvider(), EmbeddingProvider)
    assert isinstance(FakeVisionProvider(), VisionProvider)


def test_default_providers_are_the_ollama_implementations() -> None:
    assert isinstance(get_llm_provider(), OllamaLLMProvider)
    assert isinstance(get_embedding_provider(), OllamaEmbeddingProvider)


# ---------- Ollama provider contract (mock transport) ----------


def test_ollama_llm_provider_sends_the_same_request_the_app_always_did(monkeypatch) -> None:
    seen = _install_mock_ollama(monkeypatch, lambda request: httpx.Response(200, json={"message": {"content": "hello"}}))
    settings = get_settings()

    result = asyncio.run(OllamaLLMProvider().generate([{"role": "user", "content": "hi"}]))

    assert result == "hello"
    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/api/chat"
    assert body["model"] == settings.ollama_generation_model
    assert body["stream"] is False  # the app never streams
    assert body["think"] is settings.ollama_generation_think
    assert body["keep_alive"] == settings.ollama_keep_alive
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_ollama_llm_provider_honours_an_explicit_think_override(monkeypatch) -> None:
    seen = _install_mock_ollama(monkeypatch, lambda request: httpx.Response(200, json={"message": {"content": "x"}}))

    asyncio.run(OllamaLLMProvider().generate([{"role": "user", "content": "hi"}], think=True))

    assert json.loads(seen[0].content)["think"] is True


@pytest.mark.parametrize("handler", [
    lambda request: httpx.Response(500, text="model exploded"),
    lambda request: httpx.Response(200, json={"unexpected": "shape"}),
])
def test_ollama_llm_provider_maps_failures_to_provider_unavailable(monkeypatch, handler) -> None:
    _install_mock_ollama(monkeypatch, handler)
    with pytest.raises(ProviderUnavailable):
        asyncio.run(OllamaLLMProvider().generate([{"role": "user", "content": "hi"}]))


def test_ollama_llm_provider_maps_connection_errors(monkeypatch) -> None:
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    _install_mock_ollama(monkeypatch, refuse)
    with pytest.raises(ProviderUnavailable, match="Could not reach Ollama"):
        asyncio.run(OllamaLLMProvider().generate([{"role": "user", "content": "hi"}]))


def test_ollama_embedding_provider_sends_batch_and_returns_vectors_unchanged(monkeypatch) -> None:
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    seen = _install_mock_ollama(monkeypatch, lambda request: httpx.Response(200, json={"embeddings": vectors}))

    result = asyncio.run(OllamaEmbeddingProvider().embed(["a", "b"]))

    assert result == vectors  # passed through untouched: dimensions are the model's, not ours
    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/api/embed"
    assert body == {"model": get_settings().ollama_embedding_model, "input": ["a", "b"]}


def test_ollama_embedding_provider_maps_failures(monkeypatch) -> None:
    _install_mock_ollama(monkeypatch, lambda request: httpx.Response(503, text="loading"))
    with pytest.raises(ProviderUnavailable, match="503"):
        asyncio.run(OllamaEmbeddingProvider().embed(["a"]))


def test_ollama_vision_provider_attaches_the_image(monkeypatch) -> None:
    seen = _install_mock_ollama(monkeypatch, lambda request: httpx.Response(200, json={"message": {"content": "a chart"}}))

    result = asyncio.run(OllamaVisionProvider("vision-model:7b").answer(b"\x89PNGdata", "What is shown?", context="Figure 1: Throughput"))

    assert result == "a chart"
    body = json.loads(seen[0].content)
    assert body["model"] == "vision-model:7b"
    message = body["messages"][0]
    assert base64.b64decode(message["images"][0]) == b"\x89PNGdata"
    assert "What is shown?" in message["content"] and "Figure 1: Throughput" in message["content"]


# ---------- injection seam ----------


def test_facade_delegates_to_the_injected_providers() -> None:
    llm = FakeLLMProvider("from the fake model")
    embedder = FakeEmbeddingProvider()
    with use_providers(llm=llm, embedding=embedder):
        assert asyncio.run(ollama_facade.generate([{"role": "user", "content": "q"}], think=False)) == "from the fake model"
        assert asyncio.run(ollama_facade.embed(["x", "yy"])) == [[1.0, 1.0, 0.0, 0.5], [2.0, 1.0, 0.0, 0.5]]

    assert llm.calls == [{"messages": [{"role": "user", "content": "q"}], "think": False}]
    assert embedder.calls == [["x", "yy"]]


def test_use_providers_restores_the_previous_providers() -> None:
    original_llm, original_embedding = get_llm_provider(), get_embedding_provider()
    with use_providers(llm=FakeLLMProvider(), embedding=FakeEmbeddingProvider()):
        assert isinstance(get_llm_provider(), FakeLLMProvider)
    assert get_llm_provider() is original_llm
    assert get_embedding_provider() is original_embedding


def test_setters_replace_and_none_restores_the_default() -> None:
    fake = FakeLLMProvider()
    set_llm_provider(fake)
    assert get_llm_provider() is fake
    set_llm_provider(None)
    assert isinstance(get_llm_provider(), OllamaLLMProvider)
    set_embedding_provider(FakeEmbeddingProvider())
    set_embedding_provider(None)
    assert isinstance(get_embedding_provider(), OllamaEmbeddingProvider)


def test_ollama_unavailable_is_the_provider_error_so_fakes_are_handled_like_ollama_being_down() -> None:
    assert ollama_facade.OllamaUnavailable is ProviderUnavailable
    with use_providers(llm=FakeLLMProvider(ProviderUnavailable("down"))):
        with pytest.raises(ollama_facade.OllamaUnavailable):
            asyncio.run(ollama_facade.generate([{"role": "user", "content": "q"}]))


# ---------- existing services run on fakes alone (no monkeypatching Ollama) ----------


def test_rewrite_standalone_question_uses_the_injected_llm() -> None:
    llm = FakeLLMProvider("What are the types of sensors?")
    with use_providers(llm=llm):
        rewritten = asyncio.run(rewrite_standalone_question("USER: sensors?\n", "what are its types"))
    assert rewritten == "What are the types of sensors?"
    assert llm.calls[0]["think"] is False


def test_rewrite_standalone_question_falls_back_when_the_provider_is_down() -> None:
    with use_providers(llm=FakeLLMProvider(ProviderUnavailable("down"))):
        assert asyncio.run(rewrite_standalone_question("history", "what about it")) == "what about it"


def test_suggested_questions_use_the_injected_llm() -> None:
    llm = FakeLLMProvider(json.dumps(["What is ZigBee?", "How does mesh routing work?"]))
    with use_providers(llm=llm):
        questions = asyncio.run(generate_suggested_questions(["ZigBee is a low-power mesh protocol."], 2))
    assert questions == ["What is ZigBee?", "How does mesh routing work?"]


def test_embed_in_batches_uses_the_injected_embedder_and_keeps_order() -> None:
    embedder = FakeEmbeddingProvider()
    texts = [f"text-{i}" * (i + 1) for i in range(40)]
    with use_providers(embedding=embedder):
        vectors = asyncio.run(embed_in_batches(texts, batch_size=16, concurrency=3))
    assert len(vectors) == 40
    assert [len(batch) for batch in embedder.calls] == [16, 16, 8]
    assert vectors[5][0] == float(len(texts[5]) % 7)  # order preserved across concurrent batches


class _NoVectorHits:
    def query(self, query_embeddings, n_results, where, include):
        return {"ids": [[]]}


def test_chat_pipeline_answers_end_to_end_using_only_fake_providers(db_session, monkeypatch) -> None:
    """The regression that matters: the whole /chat path (routing, hybrid
    retrieval with real FTS5, generation, grounding, citations) works with no
    Ollama and no module-level patching -- purely via the provider seam."""
    monkeypatch.setattr(chat_module, "get_collection", lambda: _NoVectorHits())
    db_session.add(Document(id="doc-1", filename="protocols.pdf", status="ready"))
    db_session.add(Chunk(
        id="doc-1-4-0", document_id="doc-1", page_number=4,
        content="ZigBee operates in the 2.4 GHz frequency band and uses a mesh topology.", start_offset=0, end_offset=72,
    ))
    db_session.commit()

    llm = FakeLLMProvider(json.dumps({"answer": "ZigBee uses the 2.4 GHz band. [1]", "source_ids": [1]}))
    embedder = FakeEmbeddingProvider()
    with use_providers(llm=llm, embedding=embedder):
        result = asyncio.run(ask(ChatRequest(question="Which frequency band does ZigBee use?", document_ids=["doc-1"]), db=db_session))

    assert "2.4 GHz" in result["answer"]
    assert result["citations"][0]["page_number"] == 4
    assert llm.calls and embedder.calls  # both providers were really exercised


def test_chat_pipeline_reports_provider_outage_as_503(db_session, monkeypatch) -> None:
    from fastapi import HTTPException

    monkeypatch.setattr(chat_module, "get_collection", lambda: _NoVectorHits())
    db_session.add(Document(id="doc-1", filename="p.pdf", status="ready"))
    db_session.add(Chunk(id="doc-1-1-0", document_id="doc-1", page_number=1, content="ZigBee uses a mesh topology.", start_offset=0, end_offset=28))
    db_session.commit()

    with use_providers(llm=FakeLLMProvider(ProviderUnavailable("down")), embedding=FakeEmbeddingProvider()):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(ask(ChatRequest(question="How does ZigBee topology work?", document_ids=["doc-1"]), db=db_session))
    assert exc_info.value.status_code == 503


# ---------- vision capability (never assumed) ----------


def _installed(monkeypatch, models: list[dict]) -> None:
    async def fake_list():
        return models

    monkeypatch.setattr(registry_module, "list_installed_models", fake_list)


def test_vision_is_unsupported_when_no_model_is_configured(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "ollama_vision_model", "")
    capability = asyncio.run(vision_capability())
    assert capability.supported is False
    assert capability.model is None
    assert asyncio.run(get_vision_provider()) is None


def test_the_installed_text_model_is_never_mistaken_for_a_vision_model(monkeypatch) -> None:
    # The environment this project ships with: qwen3 + embedding models only.
    monkeypatch.setattr(get_settings(), "ollama_vision_model", "")
    _installed(monkeypatch, [{"name": "qwen3:8b", "capabilities": ["completion", "tools", "thinking"]}])
    assert asyncio.run(vision_capability()).supported is False


def test_configured_but_not_installed_vision_model_is_unsupported(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "ollama_vision_model", "llava:7b")
    _installed(monkeypatch, [{"name": "qwen3:8b"}])
    capability = asyncio.run(vision_capability())
    assert capability.supported is False
    assert "not installed" in capability.reason
    assert asyncio.run(get_vision_provider()) is None


def test_configured_model_that_reports_no_vision_capability_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "ollama_vision_model", "qwen3:8b")
    _installed(monkeypatch, [{"name": "qwen3:8b", "capabilities": ["completion", "tools"]}])
    assert asyncio.run(vision_capability()).supported is False


def test_installed_vision_model_is_detected_and_yields_a_provider(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "ollama_vision_model", "llava:7b")
    _installed(monkeypatch, [{"name": "llava:7b", "capabilities": ["completion", "vision"]}])
    capability = asyncio.run(vision_capability())
    assert capability.supported is True and capability.model == "llava:7b"
    provider = asyncio.run(get_vision_provider())
    assert isinstance(provider, OllamaVisionProvider) and provider.model == "llava:7b"


def test_older_ollama_without_capability_reporting_trusts_the_explicit_configuration(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "ollama_vision_model", "llava:7b")
    _installed(monkeypatch, [{"name": "llava:7b"}])  # no "capabilities" key at all
    assert asyncio.run(vision_capability()).supported is True


def test_an_injected_vision_provider_wins() -> None:
    fake = FakeVisionProvider()
    set_vision_provider(fake)
    capability = asyncio.run(vision_capability())
    assert capability.supported is True and capability.model == fake.model
    assert asyncio.run(get_vision_provider()) is fake
