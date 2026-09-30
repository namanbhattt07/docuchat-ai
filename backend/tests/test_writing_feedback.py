import asyncio
import json

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.api.v1.learning as learning_module
import app.services.retrieval as retrieval_module
from app.api.v1.learning import WritingEvaluationRequest, evaluate_writing
from app.db import Base
from app.models import Chunk, Document
from app.services import writing_feedback

# Group 6 WRITING EVALUATION: the original answer is preserved exactly, the
# suggested version is a separate field, feedback is grounded (citations only
# for document facts) and the rubric never invents a grade without evidence.


def _feedback(**overrides) -> str:
    payload = {
        "summary": "A solid start that misses the broker's routing role.",
        "strengths": ["Correctly names MQTT as publish/subscribe."],
        "improvements": ["Explain what the broker does."],
        "missing_points": [{"point": "The broker routes messages to subscribers.", "source_ids": [1]}],
        "factual_issues": [{"issue": "MQTT does not require HTTP.", "source_ids": [1]}],
        "rubric": {
            "factual_grounding": {"rating": "adequate", "comment": "Mostly accurate."},
            "completeness": {"rating": "needs work", "comment": "Missing the broker."},
            "clarity": {"rating": "strong", "comment": "Easy to follow."},
            "structure": "good",
            "terminology": {"rating": "adequate", "comment": ""},
        },
        "suggested_answer": "MQTT is a publish/subscribe protocol in which a broker routes messages to subscribers. [1]",
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_feedback_is_parsed_into_separate_structured_parts() -> None:
    feedback = writing_feedback.parse_writing_feedback(_feedback(), source_count=2)
    assert feedback.structured
    assert feedback.suggested_answer.endswith("[1]")
    assert feedback.missing_points[0].text == "The broker routes messages to subscribers." and feedback.missing_points[0].source_ids == [1]
    assert [entry.criterion for entry in feedback.rubric] == ["factual_grounding", "completeness", "clarity", "structure", "terminology"]
    assert [entry.rating for entry in feedback.rubric] == ["adequate", "needs_work", "strong", "strong", "adequate"]
    assert feedback.source_ids == [1]


def test_stick_to_document_drops_uncited_document_claims_but_go_freely_keeps_them() -> None:
    raw = _feedback(missing_points=[{"point": "Uncited claim about MQTT history.", "source_ids": []}], factual_issues=[])
    assert writing_feedback.parse_writing_feedback(raw, 2, "document").missing_points == []
    assert writing_feedback.parse_writing_feedback(raw, 2, "free").missing_points[0].text == "Uncited claim about MQTT history."


def test_invalid_citations_are_removed_from_the_suggested_version() -> None:
    feedback = writing_feedback.parse_writing_feedback(_feedback(suggested_answer="Fact [1]. Invented [7]."), source_count=2)
    assert "[7]" not in feedback.suggested_answer and "[1]" in feedback.suggested_answer


def test_grouped_citation_markers_are_split_and_validated() -> None:
    feedback = writing_feedback.parse_writing_feedback(_feedback(suggested_answer="Used in home automation [1, 2]. Also [1, 9]."), source_count=2)
    assert "automation [1] [2]." in feedback.suggested_answer and "[9]" not in feedback.suggested_answer
    assert feedback.source_ids == [1, 2]


def test_evidence_criteria_are_not_assessed_without_document_evidence() -> None:
    feedback = writing_feedback.parse_writing_feedback(_feedback(), source_count=0, grounding_mode="free")
    ratings = {entry.criterion: entry.rating for entry in feedback.rubric}
    assert ratings["factual_grounding"] == "not_assessed" and ratings["completeness"] == "not_assessed"
    assert ratings["clarity"] == "strong"
    assert "[1]" not in feedback.suggested_answer and feedback.missing_points[0].source_ids == []


def test_malformed_feedback_degrades_to_an_unstructured_summary() -> None:
    feedback = writing_feedback.parse_writing_feedback("Your answer is decent but vague.", source_count=2)
    assert not feedback.structured and feedback.summary == "Your answer is decent but vague."
    assert feedback.suggested_answer == ""
    assert all(entry.rating == "not_assessed" for entry in feedback.rubric)


def test_prompt_asks_for_every_feedback_dimension_and_respects_grounding() -> None:
    stick = writing_feedback.build_writing_prompt("my answer", "SOURCE 1 | a.pdf | page 1\ntext", "document", "What is MQTT?", 1)
    for dimension in ("factual grounding", "missing important points", "clarity", "structure", "terminology", "completeness"):
        assert dimension in stick
    assert "QUESTION THE STUDENT WAS ANSWERING: What is MQTT?" in stick
    assert "Do not add facts from outside them" in stick
    assert "general knowledge" in writing_feedback.build_writing_prompt("my answer", "", "free", None, 0)


# ---------------------------------------------------------------------------
# API wiring
# ---------------------------------------------------------------------------


class _FakeCollection:
    def query(self, query_embeddings, n_results, where, include):
        return {"ids": [[]]}


@pytest.fixture()
def seeded_db(monkeypatch):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    monkeypatch.setattr(learning_module, "get_collection", lambda: _FakeCollection())

    async def _fake_embed(texts):
        return [[0.0] * 4 for _ in texts]
    monkeypatch.setattr(retrieval_module, "embed", _fake_embed)
    session.add_all([
        Document(id="doc-1", filename="mqtt.pdf", status="ready"),
        Chunk(id="doc-1-1-0", document_id="doc-1", page_number=4, content="MQTT is a publish/subscribe protocol. A broker routes messages from publishers to subscribers.", start_offset=0, end_offset=90),
    ])
    session.commit()
    try:
        yield session
    finally:
        session.close()


def _fake_model(monkeypatch, raw: str) -> list:
    prompts: list = []

    async def _fake_generate(messages, **kwargs):
        prompts.append(messages[1]["content"])
        return raw
    monkeypatch.setattr(learning_module, "generate", _fake_generate)
    return prompts


def test_original_answer_is_preserved_exactly_and_suggestion_is_separate(seeded_db, monkeypatch) -> None:
    prompts = _fake_model(monkeypatch, _feedback())
    original = "  MQTT is a protocol.\n\nIt uses HTTP   underneath.  "
    result = asyncio.run(evaluate_writing(WritingEvaluationRequest(answer=original, question="What is MQTT?", document_ids=["doc-1"]), db=seeded_db))

    assert result["original_answer"] == original  # byte-for-byte, whitespace included
    assert result["suggested_answer"] != original and result["suggested_answer"].startswith("MQTT is a publish/subscribe")
    assert result["feedback"]["missing_points"][0]["point"] == "The broker routes messages to subscribers."
    assert result["citations"][0]["document_id"] == "doc-1" and result["citations"][0]["page_number"] == 4
    assert "STUDENT ANSWER:\n" + original in prompts[0]


def test_malformed_model_output_still_preserves_the_original(seeded_db, monkeypatch) -> None:
    _fake_model(monkeypatch, "not json")
    result = asyncio.run(evaluate_writing(WritingEvaluationRequest(answer="My answer.", document_ids=["doc-1"]), db=seeded_db))
    assert result["original_answer"] == "My answer." and result["feedback"]["structured"] is False


@pytest.mark.parametrize("answer", ["", "   \n\t "])
def test_empty_answer_is_rejected(seeded_db, monkeypatch, answer) -> None:
    _fake_model(monkeypatch, _feedback())
    with pytest.raises(HTTPException) as error:
        asyncio.run(evaluate_writing(WritingEvaluationRequest(answer=answer, document_ids=["doc-1"]), db=seeded_db))
    assert error.value.status_code == 400 and "Write an answer" in error.value.detail


def test_stick_to_document_needs_document_text(seeded_db, monkeypatch) -> None:
    _fake_model(monkeypatch, _feedback())
    seeded_db.add(Document(id="doc-empty", filename="scan.pdf", status="empty"))
    seeded_db.commit()
    with pytest.raises(HTTPException) as error:
        asyncio.run(evaluate_writing(WritingEvaluationRequest(answer="An answer.", document_ids=["doc-empty"]), db=seeded_db))
    assert "no document text" in error.value.detail

    result = asyncio.run(evaluate_writing(WritingEvaluationRequest(answer="An answer.", document_ids=["doc-empty"], grounding_mode="free"), db=seeded_db))
    assert result["grounded"] is False
    assert {entry["criterion"]: entry["rating"] for entry in result["feedback"]["rubric"]}["factual_grounding"] == "not_assessed"
