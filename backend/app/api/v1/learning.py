from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

from app.api.v1.chat import _plan_for, _resolve_document_scope, build_citations, format_source_block, model_http_error
from app.core.config import get_settings
from app.db import get_db
from app.services import evidence, exam, writing_feedback
from app.services.learning import explanatory_plan, sample_document_sources
from app.services.ollama import OllamaUnavailable, generate
from app.services.retrieval import retrieve_with_plan
from app.services.retrieval_types import RetrievalMode
from app.services.vector_store import get_collection

# Group 6 EXAM / STUDY MODE, QUESTION GENERATION and WRITING EVALUATION.
# Unlike Tutor/Brainstorm (which are conversational and ride on /chat), these
# are stateless request/response actions: nothing here writes to the
# database. The exam session -- questions, answers, progress -- lives in the
# client and is sent back with each answer (see services/exam.py::ExamProgress).
# Document scope, routing, retrieval and citations are the same Group 1-5
# helpers /chat uses, imported rather than re-implemented.

router = APIRouter(prefix="/learning", tags=["learning"])

MIN_SOURCE_BUDGET = 6
MAX_SOURCE_BUDGET = 16
# Below this much passage text per requested question there is nothing
# substantive to examine -- better to say so than to let the model pad
# questions out of thin evidence (one selected sentence can carry one
# question; it can't carry ten).
MIN_EVIDENCE_CHARS = 80
MIN_EVIDENCE_CHARS_PER_QUESTION = 40


class QuestionGenerationRequest(BaseModel):
    document_ids: list[str] = Field(default_factory=list)
    collection_id: str | None = None
    grounding_mode: Literal["document", "free"] = "document"
    # A plain string (not a Literal) so an unsupported type gets a readable
    # 400 message instead of a raw validation error -- see
    # exam.normalize_requested_type.
    question_type: str = Field(default="mcq", max_length=40)
    count: int = 5
    topic: str | None = Field(default=None, max_length=300)
    selected_text: str | None = Field(default=None, max_length=8000)
    selection_page: int | None = None
    # Varies the document-wide sample so "New exam" isn't the same questions.
    seed: int | None = None


class ExamEvaluationRequest(BaseModel):
    question: dict
    answer: str | None = Field(default=None, max_length=4000)
    progress: dict | None = None


class WritingEvaluationRequest(BaseModel):
    answer: str = Field(default="", max_length=8000)
    question: str | None = Field(default=None, max_length=1000)
    topic: str | None = Field(default=None, max_length=300)
    document_ids: list[str] = Field(default_factory=list)
    collection_id: str | None = None
    grounding_mode: Literal["document", "free"] = "document"
    selected_text: str | None = Field(default=None, max_length=8000)
    selection_page: int | None = None


def _source_budget(count: int) -> int:
    return min(max(count + 2, MIN_SOURCE_BUDGET), MAX_SOURCE_BUDGET)


async def _question_sources(db: Session, request: QuestionGenerationRequest, document_ids: list[str] | None, budget: int) -> list[dict]:
    """A selection or topic uses the normal router + retrieval (so collection
    filtering, SELECTION and SECTION handling all apply unchanged); with
    neither, the exam covers the whole document via the evenly-spaced
    OVERVIEW sample."""
    selected_text = (request.selected_text or "").strip() or None
    topic = (request.topic or "").strip() or None
    if not selected_text and not topic:
        return sample_document_sources(db, document_ids, budget, request.seed)
    settings = get_settings()
    plan = explanatory_plan(
        _plan_for(db, topic or selected_text[:300], document_ids, selected_text, request.selection_page),
        document_wide=False, overview_top_k=settings.retrieval_overview_max_sources, default_top_k=budget,
    )
    if plan.mode != RetrievalMode.SELECTION:
        plan.top_k = budget
    plan.collection_id = request.collection_id
    return await retrieve_with_plan(db, get_collection(), plan)


@router.post("/questions")
async def generate_question_set(request: QuestionGenerationRequest, db: Session = Depends(get_db)):
    """Generate N structured, validated, citation-backed questions from the
    current document/collection scope. Used both for Exam Mode (the client
    stores the set as its session) and the standalone Question Bank."""
    try:
        question_type = exam.normalize_requested_type(request.question_type)
    except exam.UnsupportedQuestionType as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not 1 <= request.count <= exam.MAX_QUESTIONS:
        raise HTTPException(status_code=400, detail=f"Choose between 1 and {exam.MAX_QUESTIONS} questions.")

    document_ids = _resolve_document_scope(db, request)
    sources = await _question_sources(db, request, document_ids, _source_budget(request.count))
    topic = (request.topic or "").strip()
    strict_topic = bool(topic) and request.grounding_mode == "document" and not (request.selected_text or "").strip()
    topic_message = (
        f"The selected document doesn't appear to cover \"{topic}\". Try a different topic, leave the topic blank to "
        "cover the whole document, or switch to Go Freely."
    )
    if not sources:
        # With a topic, "found nothing" means the topic isn't in the document --
        # not that the document has no text (unless it really doesn't).
        default = topic_message if strict_topic else "The selected document has no extractable text to build questions from."
        raise HTTPException(status_code=400, detail=evidence.explain_empty_scope(db, document_ids, default))
    # Retrieval always returns the *closest* passages, even when the document
    # never mentions the topic -- without this a quiz "on quantum computing"
    # quietly became a quiz on whatever the nearest unrelated pages said.
    if strict_topic and not evidence.topic_is_covered(topic, [str(source.get("content") or "") for source in sources]):
        raise HTTPException(status_code=400, detail=topic_message)
    if sum(len(source.get("content") or "") for source in sources) < max(MIN_EVIDENCE_CHARS, MIN_EVIDENCE_CHARS_PER_QUESTION * request.count):
        raise HTTPException(
            status_code=400,
            detail="There isn't enough text in the selected content to write grounded questions. Select a longer passage or a broader topic.",
        )

    try:
        result = await exam.generate_questions(
            sources,
            question_type=question_type,
            count=request.count,
            grounding_mode=request.grounding_mode,
            build_citations=lambda batch_sources, ids: build_citations(db, batch_sources, ids),
            format_source=format_source_block,
            focus=(request.topic or "").strip() or None,
            seed=request.seed,
        )
    except OllamaUnavailable as exc:
        raise model_http_error(exc, "generating questions") from exc

    if not result.questions:
        raise HTTPException(
            status_code=422,
            detail="I couldn't generate valid, document-grounded questions from this content. Try again, choose a different topic, or ask for fewer questions.",
        )
    warnings = []
    if result.shortfall:
        warnings.append(f"Only {len(result.questions)} of {result.requested} questions passed validation, so the set is shorter than requested.")
    if any(not question.grounded for question in result.questions):
        warnings.append("Some questions draw on general knowledge rather than your documents (Go Freely); they are marked.")
    return {
        "questions": [question.model_dump() for question in result.questions],
        "question_type": question_type,
        "requested": result.requested,
        "generated": len(result.questions),
        "grounding_mode": request.grounding_mode,
        "warnings": warnings,
    }


@router.post("/exam/evaluate")
async def evaluate_exam_answer(request: ExamEvaluationRequest):
    """Grade one answer and (when the client sends its session progress)
    return the updated progress. MCQ/True-False are graded exactly;
    short answers against the question's key points."""
    try:
        question = exam.parse_question(request.question)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail="This question is malformed or unsupported and can't be graded.") from exc
    progress = None
    if request.progress is not None:
        try:
            progress = exam.ExamProgress.model_validate(request.progress)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail="The exam progress sent with this answer is invalid. Restart the exam.") from exc
    try:
        evaluation = await exam.evaluate_answer(question, request.answer or "")
    except exam.InvalidAnswer as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "evaluation": evaluation.model_dump(),
        "progress": exam.apply_evaluation(progress, evaluation).model_dump() if progress else None,
    }


@router.post("/writing/evaluate")
async def evaluate_writing(request: WritingEvaluationRequest, db: Session = Depends(get_db)):
    """Feedback on a student's written answer. `original_answer` in the
    response is the request's `answer` exactly as sent -- the suggested
    version is always a separate field, never a replacement."""
    if not request.answer.strip():
        raise HTTPException(status_code=400, detail="Write an answer before requesting feedback.")

    settings = get_settings()
    document_ids = _resolve_document_scope(db, request)
    selected_text = (request.selected_text or "").strip() or None
    query = (request.question or "").strip() or (request.topic or "").strip() or request.answer.strip()[:500]
    plan = explanatory_plan(
        _plan_for(db, query, document_ids, selected_text, request.selection_page),
        document_wide=False, overview_top_k=settings.retrieval_overview_max_sources, default_top_k=settings.retrieval_top_k,
    )
    plan.collection_id = request.collection_id
    sources = await retrieve_with_plan(db, get_collection(), plan)
    if not sources:
        # Nothing matched the answer's wording -- still judge it against what
        # the document covers rather than against nothing.
        sources = sample_document_sources(db, document_ids, settings.retrieval_top_k)
    if not sources and request.grounding_mode == "document":
        raise HTTPException(
            status_code=400,
            detail="There's no document text to check this answer against. Select a text-based PDF, or switch to Go Freely.",
        )

    context = "\n\n".join(format_source_block(source, index + 1) for index, source in enumerate(sources))
    prompt = writing_feedback.build_writing_prompt(request.answer, context, request.grounding_mode, request.question, len(sources))
    try:
        raw = await generate([{"role": "system", "content": writing_feedback.WRITING_SYSTEM_MESSAGE}, {"role": "user", "content": prompt}])
    except OllamaUnavailable as exc:
        raise model_http_error(exc, "reviewing your answer") from exc

    feedback = writing_feedback.parse_writing_feedback(raw, len(sources), request.grounding_mode)
    return {
        "original_answer": request.answer,
        "suggested_answer": feedback.suggested_answer,
        "feedback": feedback.to_dict(),
        "citations": build_citations(db, sources, feedback.source_ids),
        "grounding_mode": request.grounding_mode,
        "grounded": bool(sources),
    }
