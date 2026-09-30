import asyncio
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.api.v1.chat as chat_module
import app.services.retrieval as retrieval_module
from app.api.v1.chat import ChatRequest, ask
from app.db import Base
from app.models import Chunk, Document
from app.services import brainstorm

# Group 6 BRAINSTORMING MODE: document evidence (cited) and generated ideas
# (never cited as document fact) are always kept apart, under both grounding
# modes, and insufficient evidence fails safely.


def _payload(**overrides) -> str:
    payload = {
        "evidence": [
            {"point": "The paper uses MQTT for sensor telemetry.", "source_ids": [1]},
            {"point": "Battery life is identified as a limitation [2].", "source_ids": []},
        ],
        "ideas": [
            {"title": "Solar-powered sensor node", "description": "Address the battery limitation with energy harvesting [2].", "kind": "project", "builds_on": [2]},
            {"title": "Compare MQTT with CoAP", "description": "Benchmark both on constrained devices.", "kind": "research question", "builds_on": [5]},
        ],
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_evidence_and_generated_ideas_are_parsed_separately() -> None:
    result = brainstorm.parse_brainstorm_response(_payload(), source_count=3, grounding_mode="document")

    assert [point.point for point in result.evidence] == ["The paper uses MQTT for sensor telemetry.", "Battery life is identified as a limitation."]
    assert result.evidence[1].source_ids == [2]  # inline marker recovered as a real citation
    assert [idea.title for idea in result.ideas] == ["Solar-powered sensor node", "Compare MQTT with CoAP"]
    assert "[2]" not in result.ideas[0].description  # an idea never carries a document citation in its own text
    assert result.ideas[0].builds_on == [2]
    assert result.ideas[1].builds_on == []  # [5] isn't a passage the evidence cites
    assert result.ideas[1].kind == "research_question"
    assert result.source_ids == [1, 2]


def test_evidence_without_a_verified_passage_is_dropped() -> None:
    raw = _payload(evidence=[{"point": "Unverified claim.", "source_ids": [9]}, {"point": "Real point.", "source_ids": [1]}])
    result = brainstorm.parse_brainstorm_response(raw, source_count=2, grounding_mode="document")
    assert [point.point for point in result.evidence] == ["Real point."]


def test_stick_to_document_with_no_evidence_generates_no_ideas() -> None:
    result = brainstorm.parse_brainstorm_response(_payload(evidence=[]), source_count=2, grounding_mode="document")
    assert result.ideas == [] and result.insufficient_evidence
    assert "couldn't find enough document evidence" in result.to_markdown()


def test_go_freely_with_no_evidence_keeps_ideas_but_says_they_are_general() -> None:
    result = brainstorm.parse_brainstorm_response(_payload(evidence=[]), source_count=2, grounding_mode="free")
    assert result.ideas and result.insufficient_evidence
    assert result.notice == brainstorm.FREE_NO_EVIDENCE_NOTICE


def test_malformed_output_fails_safely() -> None:
    result = brainstorm.parse_brainstorm_response("Here are some ideas: build a robot!", source_count=2, grounding_mode="document")
    assert result.evidence == [] and result.ideas == [] and result.insufficient_evidence


def test_markdown_labels_the_two_sections_distinctly() -> None:
    markdown = brainstorm.parse_brainstorm_response(_payload(), source_count=3, grounding_mode="document").to_markdown()
    evidence_part, ideas_part = markdown.split("### Generated ideas")
    assert "### Document evidence" in evidence_part and "[1]" in evidence_part
    assert "not claims made by the document" in ideas_part


def test_system_message_follows_the_grounding_mode() -> None:
    assert "Stick to Document" in brainstorm.build_brainstorm_system_message("document")
    assert "Go Freely" in brainstorm.build_brainstorm_system_message("free")


# ---------------------------------------------------------------------------
# Full /chat wiring (mode="brainstorm")
# ---------------------------------------------------------------------------


class _FakeCollection:
    def query(self, query_embeddings, n_results, where, include):
        return {"ids": [[]]}


@pytest.fixture()
def seeded_db(monkeypatch):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    monkeypatch.setattr(chat_module, "get_collection", lambda: _FakeCollection())

    async def _fake_embed(texts):
        return [[0.0] * 4 for _ in texts]
    monkeypatch.setattr(retrieval_module, "embed", _fake_embed)

    async def _no_rewrite(history, question):
        return question
    monkeypatch.setattr(chat_module, "rewrite_standalone_question", _no_rewrite)
    long = "The proposed system streams sensor telemetry over MQTT to a cloud broker; the authors note battery life as the main limitation of the field deployment."
    session.add_all([
        Document(id="doc-1", filename="paper.pdf", status="ready"),
        *[Chunk(id=f"doc-1-{page}-0", document_id="doc-1", page_number=page, content=f"{long} (page {page})", start_offset=0, end_offset=len(long)) for page in range(1, 5)],
    ])
    session.commit()
    try:
        yield session
    finally:
        session.close()


def _capture(monkeypatch, raw: str) -> list:
    calls: list = []

    async def _fake_generate(messages, **kwargs):
        calls.append(messages)
        return raw
    monkeypatch.setattr(chat_module, "generate", _fake_generate)
    return calls


def test_brainstorm_mode_returns_evidence_and_ideas_with_citations(seeded_db, monkeypatch) -> None:
    calls = _capture(monkeypatch, _payload())
    result = asyncio.run(ask(ChatRequest(question="Brainstorm project ideas based on this paper", document_ids=["doc-1"], mode="brainstorm"), db=seeded_db))

    assert result["mode"] == "brainstorm"
    assert result["brainstorm"]["evidence"] and result["brainstorm"]["ideas"]
    assert [citation["index"] for citation in result["citations"]] == [1, 2]
    assert all(citation["document_id"] == "doc-1" for citation in result["citations"])
    assert "Stick to Document" in calls[0][0]["content"]
    assert "USER REQUEST: Brainstorm project ideas" in calls[0][1]["content"]


def test_brainstorm_draws_on_document_wide_coverage(seeded_db, monkeypatch) -> None:
    calls = _capture(monkeypatch, _payload())
    asyncio.run(ask(ChatRequest(question="Suggest questions I could investigate further", document_ids=["doc-1"], mode="brainstorm"), db=seeded_db))
    prompt = calls[0][1]["content"]
    assert all(f"page {page}" in prompt for page in range(1, 5))


def test_brainstorm_on_a_selection_stays_on_the_selection(seeded_db, monkeypatch) -> None:
    calls = _capture(monkeypatch, _payload())
    asyncio.run(ask(ChatRequest(
        question="Give me possible applications of this concept", document_ids=["doc-1"], mode="brainstorm",
        selected_text="Energy harvesting could extend node lifetime.", selection_page=2,
    ), db=seeded_db))
    assert "user-selected passage\nEnergy harvesting could extend node lifetime." in calls[0][1]["content"]


def test_brainstorm_insufficient_evidence_end_to_end(seeded_db, monkeypatch) -> None:
    _capture(monkeypatch, _payload(evidence=[]))
    result = asyncio.run(ask(ChatRequest(question="Brainstorm ideas", document_ids=["doc-1"], mode="brainstorm"), db=seeded_db))
    assert result["citations"] == [] and result["brainstorm"]["ideas"] == []
    assert result["brainstorm"]["insufficient_evidence"]


def test_brainstorm_go_freely_uses_the_free_rule(seeded_db, monkeypatch) -> None:
    calls = _capture(monkeypatch, _payload())
    asyncio.run(ask(ChatRequest(question="Brainstorm ideas", document_ids=["doc-1"], mode="brainstorm", grounding_mode="free"), db=seeded_db))
    assert "Go Freely" in calls[0][0]["content"]
