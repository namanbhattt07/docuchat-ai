import asyncio
import os

import fitz
import pytest
from fakes import FakeEmbeddingProvider, FakeLLMProvider
from ingest import RecordingCollection, ingest
from pdfs import add_bar_chart_page, build_pdf

import app.api.v1.chat as chat_module
import app.services.documents as documents_module
import app.services.providers.ollama as ollama_provider
from app.api.v1.chat import ChatRequest, ask
from app.core.config import get_settings
from app.services.providers import use_providers, vision_capability

# LIVE check of visual Q&A against a real local vision model. Opt-in, because it
# needs Ollama with a vision model installed and takes ~30s:
#
#   DOCUCHAT_LIVE_VISION=1 python -m pytest tests/test_vision_live.py -s
#   (model defaults to qwen2.5vl:7b; override with DOCUCHAT_LIVE_VISION_MODEL)
#
# Everything else here is scratch: an in-memory database, fake embeddings and a
# fake text LLM -- the only real component is the vision model. The chart's
# values exist only as pixels, so a correct answer proves the image was sent
# and read. A 7B model can occasionally misread; a failure here is worth a look
# but is not necessarily a code bug.

pytestmark = pytest.mark.skipif(
    os.environ.get("DOCUCHAT_LIVE_VISION") != "1",
    reason="opt-in live test: set DOCUCHAT_LIVE_VISION=1 (needs Ollama + an installed vision model)",
)


@pytest.fixture(autouse=True)
def fast_embedding(monkeypatch):
    async def fake_embed_in_batches(texts, **kwargs):
        return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr(documents_module, "embed_in_batches", fake_embed_in_batches)


def test_real_vision_model_reads_the_chart_and_the_answer_keeps_its_citation(db_session, tmp_path, monkeypatch) -> None:
    model = os.environ.get("DOCUCHAT_LIVE_VISION_MODEL", "qwen2.5vl:7b")
    settings = get_settings()
    monkeypatch.setattr(settings, "ollama_vision_model", model)
    monkeypatch.setattr(settings, "upload_directory", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "processed_directory", str(tmp_path / "processed"))
    monkeypatch.setattr(chat_module, "get_collection", lambda: RecordingCollection())
    monkeypatch.setattr(ollama_provider, "_client", None)  # a fresh HTTP client bound to this test's event loop
    (tmp_path / "uploads").mkdir()

    pdf = build_pdf(tmp_path / "uploads" / "doc-1.pdf", [add_bar_chart_page])
    text_layer = fitz.open(pdf)[0].get_text()
    assert "BLE" not in text_layer and "1000" not in text_layer  # the answer is not in the text
    ingest(db_session, pdf, filename="chart.pdf")

    text_llm = FakeLLMProvider("the text model must not be used for a visual question")

    async def run():
        try:
            capability = await vision_capability()
            with use_providers(llm=text_llm, embedding=FakeEmbeddingProvider()):
                result = await ask(
                    ChatRequest(question="What does Figure 1 show? Which protocol has the highest throughput and what is its value?", document_ids=["doc-1"]),
                    db=db_session,
                )
            return capability, result
        finally:
            await ollama_provider.close_client()

    capability, result = asyncio.run(run())

    assert capability.supported and capability.model == model, capability
    assert result["visual"]["supported"] is True and result["visual"]["model"] == model
    assert "not supported by the current model configuration" not in result["answer"]   # the fallback was NOT used
    assert f"Vision model interpretation ({model})" in result["answer"]
    lowered = result["answer"].lower()
    assert "ble" in lowered and "1000" in lowered, result["answer"]                     # read from the image
    assert text_llm.calls == []                                                          # qwen3 was not asked to interpret it
    assert result["citations"] and all(c["page_number"] == 1 and c["document_id"] == "doc-1" for c in result["citations"])
    assert result["citations"][0]["source_type"] == "figure" and result["citations"][0]["bbox"]
