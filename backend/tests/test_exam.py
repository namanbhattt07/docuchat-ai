import asyncio
import json
import random

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.api.v1.learning as learning_module
import app.services.exam as exam_module
import app.services.retrieval as retrieval_module
from app.api.v1.learning import ExamEvaluationRequest, QuestionGenerationRequest, evaluate_exam_answer, generate_question_set
from app.db import Base
from app.models import Chunk, Collection, CollectionDocument, Document
from app.services import exam
from app.services.exam import ExamProgress, MCQQuestion, ShortAnswerQuestion, TrueFalseQuestion
from app.services.ollama import OllamaUnavailable

# Group 6 EXAM / STUDY MODE + QUESTION GENERATION: schema validation of
# generated questions (MCQ / true-false / short answer), malformed output,
# grading, session-local progress and score arithmetic, and the API wiring
# (grounding, collection scope, citations).


def _mcq(**overrides) -> dict:
    item = {"type": "mcq", "question": "Which layer does MQTT run on?", "options": ["Application", "Physical", "Link", "Network"], "answer": "Application", "explanation": "MQTT is an application-layer protocol.", "source_ids": [1]}
    item.update(overrides)
    return item


def _tf(**overrides) -> dict:
    item = {"type": "true_false", "statement": "MQTT uses a broker.", "answer": True, "explanation": "The passage says so.", "source_ids": [1]}
    item.update(overrides)
    return item


def _short(**overrides) -> dict:
    item = {"type": "short_answer", "question": "Explain what an MQTT broker does.", "expected_answer": "It receives published messages and routes them to subscribers.", "key_points": ["receives published messages", "routes them to subscribers"], "explanation": "Two key roles.", "source_ids": [2]}
    item.update(overrides)
    return item


def _parse(items, allowed=("mcq", "true_false", "short_answer"), grounding="document", source_count=3):
    return exam.parse_generated_questions(
        json.dumps({"questions": items}), allowed_types=set(allowed), source_count=source_count,
        grounding_mode=grounding, rng=random.Random(0),
    )


# ---------------------------------------------------------------------------
# Generation schema
# ---------------------------------------------------------------------------


def test_mcq_schema_has_question_options_answer_explanation_and_sources() -> None:
    outcome = _parse([_mcq()])
    question = outcome.questions[0]
    assert isinstance(question, MCQQuestion)
    assert sorted(question.options) == sorted(["Application", "Physical", "Link", "Network"])
    assert question.answer == "Application" and question.answer in question.options
    assert question.explanation and question.source_ids == [1] and question.grounded


def test_questions_never_refer_to_passages_the_student_cannot_see() -> None:
    outcome = _parse([_mcq(question="Which layer does MQTT run on according to the passage?"), _tf(statement="In the given text, MQTT uses a broker.")])
    assert outcome.questions[0].question == "Which layer does MQTT run on according to the document?"
    assert outcome.questions[1].statement == "In the document, MQTT uses a broker."


def test_incorrect_feedback_does_not_double_the_full_stop() -> None:
    question = MCQQuestion(question="Which is right?", options=["It is right.", "It is wrong.", "Neither."], answer="It is right.")
    assert exam.evaluate_objective(question, "Neither.").feedback.endswith("It is right.")


def test_mcq_letter_answers_and_prefixed_options_are_normalized() -> None:
    outcome = _parse([_mcq(options=["A) Application", "B) Physical", "C) Link", "D) Network"], answer="B")])
    assert outcome.questions[0].answer == "Physical"
    assert "B) Physical" not in outcome.questions[0].options


@pytest.mark.parametrize("bad", [
    {"answer": "Transport"},                          # answer not among the options
    {"options": ["Application", "Application", "Link", "Network"]},  # duplicate options
    {"options": ["Application", "Physical"]},         # too few options
    {"options": "Application, Physical"},             # not a list
    {"answer": 2},                                    # ambiguous numeric index
    {"question": ""},                                 # missing question
])
def test_invalid_mcqs_are_dropped_not_repaired(bad) -> None:
    outcome = _parse([_mcq(**bad)])
    assert outcome.questions == [] and outcome.dropped


def test_mcq_options_are_shuffled_so_the_answer_is_not_always_first() -> None:
    positions = set()
    for seed in range(12):
        outcome = exam.parse_generated_questions(json.dumps({"questions": [_mcq()]}), allowed_types={"mcq"}, source_count=1, rng=random.Random(seed))
        positions.add(outcome.questions[0].options.index("Application"))
    assert len(positions) > 1


def test_true_false_schema_and_string_booleans() -> None:
    outcome = _parse([_tf(answer="False"), _tf(statement=None, question="MQTT needs no broker.", answer="true")])
    first, second = outcome.questions
    assert isinstance(first, TrueFalseQuestion) and first.answer is False and first.statement == "MQTT uses a broker."
    assert second.statement == "MQTT needs no broker." and second.answer is True


def test_true_false_with_a_non_boolean_answer_is_dropped() -> None:
    assert _parse([_tf(answer="maybe")]).questions == []


def test_short_answer_schema_and_derived_key_points() -> None:
    explicit, derived = _parse([
        _short(),
        _short(question="What does QoS 2 guarantee?", key_points=[], expected_answer="Exactly-once delivery. It uses a four-step handshake."),
    ]).questions
    assert isinstance(explicit, ShortAnswerQuestion) and explicit.key_points == ["receives published messages", "routes them to subscribers"]
    assert derived.key_points == ["Exactly-once delivery.", "It uses a four-step handshake."]


def test_short_answer_without_an_expected_answer_is_dropped() -> None:
    assert _parse([_short(expected_answer="", key_points=[])]).questions == []


def test_unsupported_and_unrequested_types_are_dropped() -> None:
    outcome = _parse([_mcq(type="essay"), _tf()], allowed=("mcq",))
    assert outcome.questions == []
    assert any("unsupported" in reason for reason in outcome.dropped)
    assert any("unrequested" in reason for reason in outcome.dropped)


def test_stick_to_document_drops_questions_that_cite_no_real_passage() -> None:
    outcome = _parse([_mcq(source_ids=[]), _tf(source_ids=[9])], source_count=3)
    assert outcome.questions == []


def test_go_freely_keeps_uncited_questions_but_marks_them_ungrounded() -> None:
    outcome = _parse([_mcq(source_ids=[])], grounding="free")
    assert outcome.questions[0].grounded is False


@pytest.mark.parametrize("raw", ["", "not json at all", '{"questions": "nope"}', "[1, 2, 3]", '{"questions": [42, null]}'])
def test_malformed_model_output_never_raises(raw) -> None:
    outcome = exam.parse_generated_questions(raw, allowed_types={"mcq"}, source_count=2)
    assert outcome.questions == [] and outcome.dropped


def test_fenced_json_with_trailing_commas_is_tolerated() -> None:
    raw = "```json\n{\"questions\": [" + json.dumps(_tf()) + ",],}\n```"
    assert len(exam.parse_generated_questions(raw, allowed_types={"true_false"}, source_count=1).questions) == 1


def test_duplicate_questions_are_dropped() -> None:
    outcome = _parse([_tf(), _tf(statement="MQTT uses a broker!")])
    assert len(outcome.questions) == 1 and "duplicate question" in outcome.dropped


def test_requested_type_normalization_and_unsupported_type_message() -> None:
    assert exam.normalize_requested_type("True/False") == "true_false"
    assert exam.normalize_requested_type("Mixed") == "mixed"
    with pytest.raises(exam.UnsupportedQuestionType, match="Unsupported question type"):
        exam.normalize_requested_type("essay")


def test_mixed_type_counts_and_batches() -> None:
    assert exam.plan_type_counts("mixed", 10) == {"mcq": 4, "true_false": 3, "short_answer": 3}
    assert exam.plan_type_counts("mcq", 7) == {"mcq": 7}
    batches = exam.plan_batches({"mcq": 4, "true_false": 3, "short_answer": 3}, [{"i": i} for i in range(12)])
    assert [batch.size for batch in batches] == [5, 5]
    assert all(len(batch.type_counts) == 3 for batch in batches)  # every batch of a mixed exam is mixed
    assert {source["i"] for source in batches[0].sources}.isdisjoint({source["i"] for source in batches[1].sources})


# ---------------------------------------------------------------------------
# generate_questions orchestration
# ---------------------------------------------------------------------------


def _sources(count: int) -> list[dict]:
    return [{"document_id": "doc-1", "filename": "a.pdf", "page_number": i + 1, "content": f"Passage {i + 1} about MQTT brokers and topics."} for i in range(count)]


def _run_generation(monkeypatch, responses, question_type="mcq", count=5, grounding="document", sources=None):
    queue = list(responses)
    prompts: list[str] = []

    async def _fake_generate(messages, **kwargs):
        prompts.append(messages[1]["content"])
        item = queue.pop(0) if queue else json.dumps({"questions": []})
        if isinstance(item, Exception):
            raise item
        return item
    monkeypatch.setattr(exam_module, "generate", _fake_generate)
    result = asyncio.run(exam.generate_questions(
        sources or _sources(8), question_type=question_type, count=count, grounding_mode=grounding,
        build_citations=lambda srcs, ids: [{"index": i, "page_number": srcs[i - 1]["page_number"]} for i in ids],
        format_source=lambda source, index: f"SOURCE {index} | page {source['page_number']}\n{source['content']}", seed=1,
    ))
    return result, prompts


def test_generation_attaches_citations_from_each_batch_own_passages(monkeypatch) -> None:
    items = [_mcq(question=f"Question number {n} about brokers?", source_ids=[2]) for n in range(5)]
    result, prompts = _run_generation(monkeypatch, [json.dumps({"questions": items})])
    assert len(result.questions) == 5
    assert all(question.citations == [{"index": 2, "page_number": 2}] for question in result.questions)
    assert "Write exactly 5 exam questions: 5 multiple-choice" in prompts[0]
    assert "answerable from the passages alone" in prompts[0]


def test_generation_tops_up_a_shortfall_once_and_asks_for_no_repeats(monkeypatch) -> None:
    first = json.dumps({"questions": [_tf(statement=f"Statement {n} is true.") for n in range(3)]})
    top_up = json.dumps({"questions": [_tf(statement=f"Extra statement {n}.") for n in range(2)]})
    result, prompts = _run_generation(monkeypatch, [first, top_up], question_type="true_false", count=5)
    assert len(result.questions) == 5 and result.shortfall == 0
    assert "Do not repeat" in prompts[1] and "Statement 0 is true." in prompts[1]


def test_one_failed_batch_does_not_fail_the_whole_set(monkeypatch) -> None:
    good = json.dumps({"questions": [_mcq(question=f"Question {n} about topics?") for n in range(5)]})
    result, _ = _run_generation(monkeypatch, [good, OllamaUnavailable("timeout")], count=10, sources=_sources(12))
    assert len(result.questions) >= 5 and result.failed_batches == 1


def test_every_batch_failing_surfaces_the_model_error(monkeypatch) -> None:
    with pytest.raises(OllamaUnavailable):
        _run_generation(monkeypatch, [OllamaUnavailable("down"), OllamaUnavailable("down")], count=10, sources=_sources(12))


def test_mixed_generation_is_interleaved_and_capped(monkeypatch) -> None:
    items = [_mcq(question=f"MCQ {n} about brokers?") for n in range(4)] + [_tf(statement=f"TF {n} is right.") for n in range(4)] + [_short(question=f"Short {n} about topics?") for n in range(4)]
    result, _ = _run_generation(monkeypatch, [json.dumps({"questions": items})], question_type="mixed", count=5, sources=_sources(4))
    assert [question.type for question in result.questions][:3] == ["mcq", "true_false", "short_answer"]
    assert len(result.questions) == 5


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def _mcq_question() -> MCQQuestion:
    return MCQQuestion(question="Which layer does MQTT run on?", options=["Application", "Physical", "Link", "Network"], answer="Application")


def test_mcq_grading_is_deterministic() -> None:
    assert exam.evaluate_objective(_mcq_question(), "Application").verdict == "correct"
    wrong = exam.evaluate_objective(_mcq_question(), "Physical")
    assert wrong.verdict == "incorrect" and wrong.points == 0 and "Application" in wrong.feedback
    assert exam.evaluate_objective(_mcq_question(), "a").verdict == "correct"  # letter form


def test_true_false_grading() -> None:
    question = TrueFalseQuestion(statement="MQTT uses a broker.", answer=True)
    assert exam.evaluate_objective(question, "True").points == 1.0
    assert exam.evaluate_objective(question, "false").verdict == "incorrect"


@pytest.mark.parametrize("answer", ["", "   ", None])
def test_empty_answers_are_rejected(answer) -> None:
    with pytest.raises(exam.InvalidAnswer):
        exam.evaluate_objective(_mcq_question(), answer)


def test_answers_outside_the_options_are_rejected() -> None:
    with pytest.raises(exam.InvalidAnswer):
        exam.evaluate_objective(_mcq_question(), "Transport")
    with pytest.raises(exam.InvalidAnswer):
        exam.evaluate_objective(TrueFalseQuestion(statement="MQTT uses a broker.", answer=True), "maybe")


def _short_question() -> ShortAnswerQuestion:
    return ShortAnswerQuestion(
        question="Explain what an MQTT broker does.", expected_answer="It receives published messages and routes them to subscribers.",
        key_points=["receives published messages", "routes messages to subscribers"],
        citations=[{"excerpt": "The broker receives messages and routes them to subscribers."}],
    )


def _grade_short(monkeypatch, response, answer="The broker receives messages and sends them to subscribers."):
    prompts = []

    async def _fake_generate(messages, **kwargs):
        prompts.append(messages[1]["content"])
        if isinstance(response, Exception):
            raise response
        return response
    monkeypatch.setattr(exam_module, "generate", _fake_generate)
    return asyncio.run(exam.evaluate_short_answer(_short_question(), answer)), prompts


def test_short_answer_full_coverage_is_correct(monkeypatch) -> None:
    evaluation, prompts = _grade_short(monkeypatch, json.dumps({"covered": [1, 2], "incorrect_claims": [], "feedback": "Great answer."}))
    assert evaluation.verdict == "correct" and evaluation.points == 1.0 and evaluation.method == "model"
    assert "KEY POINTS:\n1. receives published messages" in prompts[0]
    assert "The broker receives messages and routes them" in prompts[0]  # grounded in the question's citation


def test_short_answer_partial_coverage_scores_proportionally(monkeypatch) -> None:
    evaluation, _ = _grade_short(monkeypatch, json.dumps({"covered": [1], "incorrect_claims": [], "feedback": "Half there."}))
    assert evaluation.verdict == "partial" and evaluation.points == 0.5
    assert evaluation.key_points_missing == ["routes messages to subscribers"]


def test_short_answer_with_a_contradicting_claim_is_not_fully_correct(monkeypatch) -> None:
    evaluation, _ = _grade_short(monkeypatch, json.dumps({"covered": [1, 2], "incorrect_claims": ["Brokers store messages forever"], "feedback": "Mostly right."}))
    assert evaluation.verdict == "partial" and "Brokers store messages forever" in evaluation.feedback


def test_short_answer_grading_falls_back_to_keywords_on_malformed_output(monkeypatch) -> None:
    evaluation, _ = _grade_short(monkeypatch, "I think it's good!")
    assert evaluation.method == "keyword_fallback" and "approximately" in evaluation.feedback


def test_short_answer_grading_falls_back_when_the_model_is_down(monkeypatch) -> None:
    evaluation, _ = _grade_short(monkeypatch, OllamaUnavailable("timeout"), answer="Something unrelated about cats.")
    assert evaluation.method == "keyword_fallback" and evaluation.verdict == "incorrect"


# ---------------------------------------------------------------------------
# Session-local progress + score calculation
# ---------------------------------------------------------------------------


def test_progress_tracks_counts_score_percentage_and_completion() -> None:
    progress = ExamProgress(total_questions=3)
    mcq = _mcq_question()
    progress = exam.apply_evaluation(progress, exam.evaluate_objective(mcq, "Application"))
    assert (progress.answered_questions, progress.correct_answers, progress.score, progress.percentage) == (1, 1, 1.0, 100.0)

    tf = TrueFalseQuestion(statement="MQTT uses a broker.", answer=True)
    progress = exam.apply_evaluation(progress, exam.evaluate_objective(tf, "False"))
    assert (progress.answered_questions, progress.correct_answers, progress.score, progress.percentage) == (2, 1, 1.0, 50.0)
    assert not progress.completed

    partial = exam.AnswerEvaluation(question_id="q3", question_type="short_answer", verdict="partial", points=0.5, selected_answer="x", correct_answer="y", feedback="", method="model")
    progress = exam.apply_evaluation(progress, partial)
    assert progress.partial_answers == 1 and progress.score == 1.5 and progress.percentage == 50.0
    assert progress.completed


def test_answering_the_same_question_twice_cannot_inflate_the_score() -> None:
    mcq = _mcq_question()
    evaluation = exam.evaluate_objective(mcq, "Application")
    once = exam.apply_evaluation(ExamProgress(total_questions=2), evaluation)
    assert exam.apply_evaluation(once, evaluation) == once


def test_inconsistent_progress_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ExamProgress(total_questions=2, answered_questions=3)
    with pytest.raises(ValidationError):
        ExamProgress(total_questions=2, answered_questions=1, correct_answers=2)


def test_client_sent_questions_are_revalidated_before_grading() -> None:
    tampered = _mcq_question().model_dump() | {"answer": "Not an option"}
    with pytest.raises(ValidationError):
        exam.parse_question(tampered)
    with pytest.raises(ValidationError):
        exam.parse_question({"type": "essay", "question": "Write an essay."})


# ---------------------------------------------------------------------------
# API wiring
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


LONG_TEXT = "MQTT is a lightweight publish/subscribe messaging protocol. Clients publish messages to topics on a broker, which routes them to every subscriber of that topic."


@pytest.fixture()
def seeded_db(db_session, monkeypatch):
    monkeypatch.setattr(learning_module, "get_collection", lambda: _FakeCollection())

    async def _fake_embed(texts):
        return [[0.0] * 4 for _ in texts]
    monkeypatch.setattr(retrieval_module, "embed", _fake_embed)

    db_session.add_all([
        Document(id="doc-1", filename="mqtt.pdf", status="ready"),
        Document(id="doc-2", filename="biology.pdf", status="ready"),
        Document(id="doc-empty", filename="scan.pdf", status="empty"),
        *[Chunk(id=f"doc-1-{page}-0", document_id="doc-1", page_number=page, content=f"{LONG_TEXT} (page {page})", start_offset=0, end_offset=len(LONG_TEXT)) for page in range(1, 7)],
        Chunk(id="doc-2-1-0", document_id="doc-2", page_number=1, content="Photosynthesis converts light energy into chemical energy stored in glucose, inside the chloroplasts of plant cells.", start_offset=0, end_offset=100),
    ])
    db_session.commit()
    return db_session


def _fake_model(monkeypatch, items):
    prompts = []

    async def _fake_generate(messages, **kwargs):
        prompts.append(messages[1]["content"])
        return json.dumps({"questions": items})
    monkeypatch.setattr(exam_module, "generate", _fake_generate)
    return prompts


def test_question_api_returns_structured_cited_questions(seeded_db, monkeypatch) -> None:
    _fake_model(monkeypatch, [_mcq(question=f"Question {n} about MQTT?", source_ids=[1]) for n in range(3)])
    result = asyncio.run(generate_question_set(QuestionGenerationRequest(document_ids=["doc-1"], question_type="mcq", count=3), db=seeded_db))

    assert result["generated"] == 3 and result["question_type"] == "mcq"
    question = result["questions"][0]
    assert set(question) >= {"id", "type", "question", "options", "answer", "explanation", "citations"}
    citation = question["citations"][0]
    assert citation["document_id"] == "doc-1" and citation["filename"] == "mqtt.pdf" and citation["excerpt"]  # existing citation shape


def test_question_api_reports_a_shortfall_as_a_warning(seeded_db, monkeypatch) -> None:
    _fake_model(monkeypatch, [_tf()])
    result = asyncio.run(generate_question_set(QuestionGenerationRequest(document_ids=["doc-1"], question_type="true_false", count=3), db=seeded_db))
    assert result["generated"] == 1 and "Only 1 of 3" in result["warnings"][0]


@pytest.mark.parametrize(("overrides", "message"), [
    ({"question_type": "essay"}, "Unsupported question type"),
    ({"count": 0}, "between 1 and"),
    ({"count": 99}, "between 1 and"),
    ({"document_ids": ["doc-empty"]}, "no extractable text"),
    ({"selected_text": "Too short."}, "isn't enough text"),
])
def test_question_api_rejects_bad_requests_with_readable_messages(seeded_db, monkeypatch, overrides, message) -> None:
    _fake_model(monkeypatch, [_mcq()])
    request = QuestionGenerationRequest(**({"document_ids": ["doc-1"], "question_type": "mcq", "count": 3} | overrides))
    with pytest.raises(HTTPException) as error:
        asyncio.run(generate_question_set(request, db=seeded_db))
    assert error.value.status_code == 400 and message in error.value.detail


def test_question_api_fails_cleanly_when_no_valid_question_survives(seeded_db, monkeypatch) -> None:
    _fake_model(monkeypatch, [_mcq(answer="Not an option")])
    with pytest.raises(HTTPException) as error:
        asyncio.run(generate_question_set(QuestionGenerationRequest(document_ids=["doc-1"], count=2), db=seeded_db))
    assert error.value.status_code == 422


def test_question_api_maps_a_model_timeout_to_504(seeded_db, monkeypatch) -> None:
    import httpx

    async def _timeout(messages, **kwargs):
        try:
            raise httpx.ReadTimeout("slow")
        except httpx.ReadTimeout as exc:
            raise OllamaUnavailable("Could not reach Ollama") from exc
    monkeypatch.setattr(exam_module, "generate", _timeout)
    with pytest.raises(HTTPException) as error:
        asyncio.run(generate_question_set(QuestionGenerationRequest(document_ids=["doc-1"], count=2), db=seeded_db))
    assert error.value.status_code == 504 and "took too long" in error.value.detail


def test_question_api_respects_collection_scope(seeded_db, monkeypatch) -> None:
    prompts = _fake_model(monkeypatch, [_mcq()])
    seeded_db.add(Collection(id="col-1", name="Networking"))
    seeded_db.add(CollectionDocument(collection_id="col-1", document_id="doc-1"))
    seeded_db.commit()

    asyncio.run(generate_question_set(QuestionGenerationRequest(document_ids=["doc-1", "doc-2"], collection_id="col-1", count=1), db=seeded_db))
    assert all("Photosynthesis" not in prompt for prompt in prompts)

    with pytest.raises(HTTPException):
        asyncio.run(generate_question_set(QuestionGenerationRequest(document_ids=["doc-2"], collection_id="col-1", count=1), db=seeded_db))


def test_question_api_uses_the_selected_passage(seeded_db, monkeypatch) -> None:
    prompts = _fake_model(monkeypatch, [_mcq()])
    selection = "QoS level 2 guarantees that each message is delivered exactly once by using a four-step handshake between sender and receiver."
    asyncio.run(generate_question_set(QuestionGenerationRequest(document_ids=["doc-1"], count=1, selected_text=selection, selection_page=3), db=seeded_db))
    assert f"user-selected passage\n{selection}" in prompts[0]


def test_question_api_stick_to_document_prompt_forbids_outside_facts(seeded_db, monkeypatch) -> None:
    prompts = _fake_model(monkeypatch, [_mcq()])
    asyncio.run(generate_question_set(QuestionGenerationRequest(document_ids=["doc-1"], count=1), db=seeded_db))
    assert "Never use outside facts" in prompts[0]

    prompts = _fake_model(monkeypatch, [_mcq(source_ids=[])])
    result = asyncio.run(generate_question_set(QuestionGenerationRequest(document_ids=["doc-1"], count=1, grounding_mode="free"), db=seeded_db))
    assert "general knowledge" in prompts[0] and result["questions"][0]["grounded"] is False and result["warnings"]


def test_evaluate_api_grades_and_advances_session_progress() -> None:
    question = _mcq_question().model_dump()
    progress = ExamProgress(total_questions=2).model_dump()
    response = asyncio.run(evaluate_exam_answer(ExamEvaluationRequest(question=question, answer="Application", progress=progress)))
    assert response["evaluation"]["verdict"] == "correct"
    assert response["progress"]["answered_questions"] == 1 and response["progress"]["score"] == 1.0


@pytest.mark.parametrize(("payload", "message"), [
    ({"question": {"type": "essay"}, "answer": "x"}, "malformed or unsupported"),
    ({"question": "mcq", "answer": "x"}, None),
])
def test_evaluate_api_rejects_malformed_questions(payload, message) -> None:
    if message is None:
        with pytest.raises(ValidationError):
            ExamEvaluationRequest(**payload)
        return
    with pytest.raises(HTTPException) as error:
        asyncio.run(evaluate_exam_answer(ExamEvaluationRequest(**payload)))
    assert error.value.status_code == 400 and message in error.value.detail


def test_evaluate_api_rejects_an_empty_answer_and_bad_progress() -> None:
    question = _mcq_question().model_dump()
    with pytest.raises(HTTPException) as error:
        asyncio.run(evaluate_exam_answer(ExamEvaluationRequest(question=question, answer="  ")))
    assert "Choose or write an answer" in error.value.detail
    with pytest.raises(HTTPException) as error:
        asyncio.run(evaluate_exam_answer(ExamEvaluationRequest(question=question, answer="Application", progress={"total_questions": 0})))
    assert "Restart the exam" in error.value.detail
