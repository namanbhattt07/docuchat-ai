import asyncio
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.api.v1.chat as chat_module
import app.services.retrieval as retrieval_module
from app.api.v1.chat import (
    ChatRequest,
    ask,
    build_free_instructions,
    compose_grounded_answer,
    grounded_response_with_context,
)
from app.db import Base
from app.models import Chunk, Document
from app.services.query_router import FormatMode
from app.services.retrieval_types import RetrievalMode

# Group 5: GROUNDING CONTROL (Stick to Document / Go Freely). See
# GROUNDING RULES / GROUNDING ANSWER PRESENTATION / CITATION CONTRACT in the
# brief for the exact contract these tests pin down.


def test_chat_request_defaults_to_stick_to_document() -> None:
    request = ChatRequest(question="What is ZigBee?", document_ids=["doc-1"])
    assert request.grounding_mode == "document"


def test_build_free_instructions_schema_forbids_citing_additional_context() -> None:
    instructions = build_free_instructions(RetrievalMode.FACT, FormatMode.DEFAULT, max_sources=3)
    assert '"additional_context"' in instructions
    assert "never" in instructions.lower() and "citation" in instructions.lower()


def test_grounded_response_with_context_parses_answer_and_supplemental_knowledge() -> None:
    raw = json.dumps({
        "answer": "ZigBee uses the 2.4 GHz band. [1]",
        "source_ids": [1],
        "additional_context": "ZigBee is also commonly used in low-power mesh networks.",
    })

    answer, source_ids, additional_context = grounded_response_with_context(raw, source_count=1)

    assert source_ids == [1]
    assert "2.4 GHz" in answer
    assert additional_context == "ZigBee is also commonly used in low-power mesh networks."


def test_grounded_response_with_context_strips_citation_markers_from_supplemental_text() -> None:
    """A [n] marker inside additional_context would falsely imply that
    sentence came from the document -- it must always be stripped."""
    raw = json.dumps({
        "answer": "ZigBee uses the 2.4 GHz band. [1]",
        "source_ids": [1],
        "additional_context": "It is also widely deployed in smart-home hubs [1].",
    })

    _, _, additional_context = grounded_response_with_context(raw, source_count=1)

    assert "[1]" not in additional_context


def test_grounded_response_with_context_keeps_supplemental_knowledge_when_no_evidence_supports_answer() -> None:
    """GO FREELY: the assistant may still answer from general knowledge when
    the documents don't cover the question -- this must not be treated as a
    failure the way Stick to Document's grounded_response treats it."""
    raw = json.dumps({
        "answer": "The internet's precursor, ARPANET, went live in 1969.",
        "source_ids": [],
        "additional_context": "This is general knowledge, not stated in the supplied passages.",
    })

    answer, source_ids, additional_context = grounded_response_with_context(raw, source_count=2)

    assert source_ids == []
    assert additional_context  # supplemental knowledge survives
    assert "couldn't find enough evidence" in answer.lower()


def test_grounded_response_with_context_falls_back_when_nothing_usable_at_all() -> None:
    raw = json.dumps({"answer": "", "source_ids": [], "additional_context": ""})

    answer, source_ids, additional_context = grounded_response_with_context(raw, source_count=2)

    assert source_ids == []
    assert additional_context == ""
    assert "couldn't verify" in answer.lower()


def test_supplemental_field_holding_only_markup_debris_is_treated_as_empty() -> None:
    """Found in the browser: the model returned "```" as additional_context, so
    Go Freely rendered a bare 'Additional context' heading with nothing under it."""
    for debris in ("```", "-", "...", "  \n ", "[1]"):
        raw = json.dumps({"answer": "", "source_ids": [], "additional_context": debris})

        answer, source_ids, additional_context = grounded_response_with_context(raw, source_count=2)

        assert additional_context == "", debris
        assert "couldn't verify" in answer.lower(), debris  # fail closed, exactly as for an empty field
        assert "Additional context" not in compose_grounded_answer(answer, additional_context), debris


def test_real_supplemental_knowledge_still_survives_the_debris_filter() -> None:
    raw = json.dumps({"answer": "", "source_ids": [], "additional_context": "Paris is the capital of France."})

    _, _, additional_context = grounded_response_with_context(raw, source_count=2)

    assert additional_context == "Paris is the capital of France."


def test_compose_grounded_answer_adds_no_heading_when_no_supplemental_knowledge() -> None:
    answer = compose_grounded_answer("ZigBee uses the 2.4 GHz band. [1]", "")
    assert "###" not in answer


def test_compose_grounded_answer_separates_document_and_supplemental_sections() -> None:
    answer = compose_grounded_answer("ZigBee uses the 2.4 GHz band. [1]", "It is popular in smart-home mesh networks.")
    assert "### From your documents" in answer
    assert "### Additional context" in answer
    assert answer.index("### From your documents") < answer.index("### Additional context")


# ---------------------------------------------------------------------------
# Full /chat wiring
# ---------------------------------------------------------------------------


class _FakeCollection:
    def query(self, query_embeddings, n_results, where, include):
        return {"ids": [[]]}


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def seeded_db(db_session, monkeypatch):
    monkeypatch.setattr(chat_module, "get_collection", lambda: _FakeCollection())

    async def _fake_embed(texts):
        return [[0.0] * 4 for _ in texts]
    monkeypatch.setattr(retrieval_module, "embed", _fake_embed)

    db_session.add(Document(id="doc-1", filename="protocols.pdf", status="ready"))
    db_session.add(Chunk(id="doc-1-59-0", document_id="doc-1", page_number=59, content="ZigBee operates in the 2.4 GHz frequency band.", start_offset=0, end_offset=48))
    db_session.commit()
    return db_session


def test_stick_to_document_never_renders_a_supplemental_section(seeded_db, monkeypatch) -> None:
    async def _fake_generate(messages, **kwargs):
        return json.dumps({"answer": "ZigBee uses the 2.4 GHz band. [1]", "source_ids": [1], "additional_context": "smuggled-in text"})
    monkeypatch.setattr(chat_module, "generate", _fake_generate)
    request = ChatRequest(question="What frequency does ZigBee use?", document_ids=["doc-1"], grounding_mode="document")

    result = asyncio.run(ask(request, db=seeded_db))

    assert "smuggled-in text" not in result["answer"]
    assert "Additional context" not in result["answer"]


def test_stick_to_document_reports_insufficient_evidence_without_outside_knowledge(seeded_db, monkeypatch) -> None:
    async def _fake_generate(messages, **kwargs):
        return json.dumps({"answer": "", "source_ids": []})
    monkeypatch.setattr(chat_module, "generate", _fake_generate)
    # Retrieval still needs to find *some* passage to hand the model (that's
    # realistic -- top-k retrieval always returns its best matches); the
    # "insufficient evidence" scenario being tested here is the model's
    # response saying it can't answer from them, not retrieval finding zero.
    request = ChatRequest(question="What frequency does ZigBee use, and what year was the original Internet invented?", document_ids=["doc-1"], grounding_mode="document")

    result = asyncio.run(ask(request, db=seeded_db))

    assert result["citations"] == []
    assert "couldn't verify" in result["answer"].lower()


def test_go_freely_labels_supplemental_knowledge_and_keeps_document_claims_cited(seeded_db, monkeypatch) -> None:
    async def _fake_generate(messages, **kwargs):
        return json.dumps({
            "answer": "ZigBee operates at 2.4 GHz. [1]",
            "source_ids": [1],
            "additional_context": "ZigBee is commonly used for low-power wireless communication.",
        })
    monkeypatch.setattr(chat_module, "generate", _fake_generate)
    request = ChatRequest(question="What frequency does ZigBee use and why is it popular?", document_ids=["doc-1"], grounding_mode="free")

    result = asyncio.run(ask(request, db=seeded_db))

    assert "### From your documents" in result["answer"]
    assert "### Additional context" in result["answer"]
    supplemental_section = result["answer"].split("### Additional context")[1]
    assert "[1]" not in supplemental_section
    assert len(result["citations"]) == 1
    assert result["citations"][0]["document_id"] == "doc-1"


def test_go_freely_with_no_supplemental_knowledge_needed_renders_like_stick_to_document(seeded_db, monkeypatch) -> None:
    async def _fake_generate(messages, **kwargs):
        return json.dumps({"answer": "ZigBee operates at 2.4 GHz. [1]", "source_ids": [1], "additional_context": ""})
    monkeypatch.setattr(chat_module, "generate", _fake_generate)
    request = ChatRequest(question="What frequency does ZigBee use?", document_ids=["doc-1"], grounding_mode="free")

    result = asyncio.run(ask(request, db=seeded_db))

    assert "###" not in result["answer"]


def test_go_freely_permits_supplemental_answer_when_document_lacks_the_fact(seeded_db, monkeypatch) -> None:
    async def _fake_generate(messages, **kwargs):
        return json.dumps({
            "answer": "",
            "source_ids": [],
            "additional_context": "The internet's precursor, ARPANET, went live in 1969 -- this is not stated in your documents.",
        })
    monkeypatch.setattr(chat_module, "generate", _fake_generate)
    request = ChatRequest(question="What frequency does ZigBee use, and what year was the original Internet invented?", document_ids=["doc-1"], grounding_mode="free")

    result = asyncio.run(ask(request, db=seeded_db))

    assert "1969" in result["answer"]
    assert result["citations"] == []  # supplemental knowledge is never cited
