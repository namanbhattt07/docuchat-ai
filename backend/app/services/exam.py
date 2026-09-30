import asyncio
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Annotated, Callable, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, field_validator, model_validator

from app.services.learning import (
    as_text,
    as_text_list,
    extract_json_object,
    strip_citation_markers,
    valid_source_ids,
)
from app.services.ollama import OllamaUnavailable, generate
from app.services.retrieval import terms

logger = logging.getLogger("docuchat.exam")

# Group 6 EXAM / STUDY MODE + QUESTION GENERATION. Three independent stages,
# each testable without a model:
#   1. generation   -- prompt per batch, then parse_generated_questions turns
#                      whatever the model returned into validated question
#                      objects (anything malformed/ungrounded is dropped, never
#                      "repaired" into a question with an invented answer).
#   2. grading      -- MCQ/True-False are graded deterministically; short
#                      answers are graded against the question's own key
#                      points + cited evidence (model, with a keyword fallback).
#   3. progress     -- a pure transition over session-local ExamProgress the
#                      client holds; nothing about an exam is ever persisted.

QuestionType = Literal["mcq", "true_false", "short_answer"]
QUESTION_TYPES: tuple[str, ...] = ("mcq", "true_false", "short_answer")
REQUESTABLE_TYPES: tuple[str, ...] = (*QUESTION_TYPES, "mixed")
TYPE_LABELS = {"mcq": "multiple-choice", "true_false": "true/false", "short_answer": "short-answer"}
MAX_QUESTIONS = 20
BATCH_SIZE = 5
MIN_MCQ_OPTIONS = 3
MAX_MCQ_OPTIONS = 6

_TYPE_ALIASES = {
    "mcq": "mcq", "multiple_choice": "mcq", "multiplechoice": "mcq", "multiple choice": "mcq", "choice": "mcq",
    "true_false": "true_false", "truefalse": "true_false", "true/false": "true_false", "tf": "true_false",
    "true or false": "true_false", "boolean": "true_false",
    "short_answer": "short_answer", "shortanswer": "short_answer", "short answer": "short_answer", "short": "short_answer",
    "open": "short_answer",
}
_OPTION_PREFIX = re.compile(r"^\s*(?:\(?[A-Fa-f][\)\.:]|\d[\)\.:])\s+")
_LETTER_ANSWER = re.compile(r"^\s*(?:option\s+)?\(?([A-Fa-f])\)?[\.\):]?\s*$", re.IGNORECASE)
# Students never see the passages, so a question that says "according to the
# passage" reads as broken -- the model is told not to, and this catches it
# when it does anyway.
_PASSAGE_REFERENCE = re.compile(r"\b(according to|in|from|based on) the (?:given |provided |above |supplied )?(passage|passages|text|excerpt|excerpts)\b", re.IGNORECASE)
_TRUE_VALUES = {"true", "t", "yes", "y", "1", "correct"}
_FALSE_VALUES = {"false", "f", "no", "n", "0", "incorrect"}


class UnsupportedQuestionType(ValueError):
    pass


class InvalidAnswer(ValueError):
    pass


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class _QuestionBase(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    explanation: str = ""
    # Indices into the passages of the batch that generated the question --
    # turned into full citation dicts (the same shape /chat returns) by the
    # API layer, so the frontend's citation UI works unchanged.
    source_ids: list[int] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)
    # False only under Go Freely, for a question the model could not tie to
    # a passage -- shown to the student as "general knowledge".
    grounded: bool = True


class MCQQuestion(_QuestionBase):
    type: Literal["mcq"] = "mcq"
    question: str = Field(min_length=5)
    options: list[str]
    answer: str

    @field_validator("options")
    @classmethod
    def _options_are_distinct(cls, options: list[str]) -> list[str]:
        cleaned = [option.strip() for option in options if option and option.strip()]
        if len({option.casefold() for option in cleaned}) != len(cleaned):
            raise ValueError("MCQ options must be distinct")
        if not MIN_MCQ_OPTIONS <= len(cleaned) <= MAX_MCQ_OPTIONS:
            raise ValueError(f"MCQ needs {MIN_MCQ_OPTIONS}-{MAX_MCQ_OPTIONS} options")
        return cleaned

    @model_validator(mode="after")
    def _answer_is_an_option(self) -> "MCQQuestion":
        if self.answer not in self.options:
            raise ValueError("MCQ answer must be exactly one of the options")
        return self


class TrueFalseQuestion(_QuestionBase):
    type: Literal["true_false"] = "true_false"
    statement: str = Field(min_length=5)
    answer: bool


class ShortAnswerQuestion(_QuestionBase):
    type: Literal["short_answer"] = "short_answer"
    question: str = Field(min_length=5)
    expected_answer: str = Field(min_length=1)
    key_points: list[str] = Field(min_length=1)


ExamQuestion = Annotated[Union[MCQQuestion, TrueFalseQuestion, ShortAnswerQuestion], Field(discriminator="type")]
_EXAM_QUESTION_ADAPTER: TypeAdapter = TypeAdapter(ExamQuestion)


def question_prompt(question) -> str:
    return question.statement if question.type == "true_false" else question.question


def correct_answer_text(question) -> str:
    if question.type == "true_false":
        return "True" if question.answer else "False"
    if question.type == "mcq":
        return question.answer
    return question.expected_answer


# ---------------------------------------------------------------------------
# Generation: planning, prompt, parsing
# ---------------------------------------------------------------------------


def normalize_requested_type(question_type: str) -> str:
    normalized = _TYPE_ALIASES.get(question_type.strip().lower(), question_type.strip().lower())
    if normalized not in REQUESTABLE_TYPES:
        raise UnsupportedQuestionType(
            f"Unsupported question type \"{question_type}\". Choose one of: MCQ, true/false, short answer, or mixed."
        )
    return normalized


def plan_type_counts(question_type: str, count: int) -> dict[str, int]:
    """How many of each type to ask for. "mixed" deals types round-robin, so
    10 mixed questions is 4 MCQ + 3 true/false + 3 short-answer."""
    if question_type != "mixed":
        return {question_type: count}
    counts = {kind: 0 for kind in QUESTION_TYPES}
    for index in range(count):
        counts[QUESTION_TYPES[index % len(QUESTION_TYPES)]] += 1
    return {kind: value for kind, value in counts.items() if value}


@dataclass
class GenerationBatch:
    type_counts: dict[str, int]
    sources: list[dict]

    @property
    def size(self) -> int:
        return sum(self.type_counts.values())


def plan_batches(type_counts: dict[str, int], sources: list[dict], batch_size: int = BATCH_SIZE) -> list[GenerationBatch]:
    """Split the request into evenly sized batches of at most `batch_size` questions and
    give each batch its own slice of the passages (round-robin, so every
    batch spans the document) -- smaller generations are far less likely to
    time out or come back malformed on a local model, and distinct passages
    per batch keep batches from writing the same question. With too few
    passages to split, every batch sees all of them and de-duplication
    handles any overlap.
    """
    # Deal types round-robin so every batch of a mixed exam gets a mix.
    remaining = dict(type_counts)
    queue: list[str] = []
    while any(remaining.values()):
        for kind in QUESTION_TYPES:
            if remaining.get(kind):
                queue.append(kind)
                remaining[kind] -= 1
    if not queue:
        return []
    # As even as possible (6 questions -> 3 + 3, not 5 + 1).
    batch_count = -(-len(queue) // batch_size)
    chunks = [queue[index::batch_count] for index in range(batch_count)]
    split = len(sources) >= 2 * len(chunks)
    batches = []
    for position, chunk in enumerate(chunks):
        counts: dict[str, int] = {}
        for kind in chunk:
            counts[kind] = counts.get(kind, 0) + 1
        batch_sources = sources[position::len(chunks)] if split else sources
        batches.append(GenerationBatch(type_counts=counts, sources=batch_sources))
    return batches


_SCHEMA_BY_TYPE = {
    "mcq": '{"type":"mcq","question":"...","options":["...","...","...","..."],"answer":"exact text of the correct option","explanation":"...","source_ids":[n]}',
    "true_false": '{"type":"true_false","statement":"a single declarative sentence","answer":true,"explanation":"...","source_ids":[n]}',
    "short_answer": '{"type":"short_answer","question":"...","expected_answer":"1-3 sentence model answer","key_points":["...","..."],"explanation":"what a good answer must include","source_ids":[n]}',
}
_RULES_BY_TYPE = {
    "mcq": (
        "Multiple-choice: exactly 4 options and exactly one correct answer; \"answer\" repeats the correct option's exact "
        "text. Distractors must be plausible but wrong according to the passages -- never use as a distractor anything "
        "the passages state is also true of the question's subject (for example another item from the same list)."
    ),
    "true_false": "True/false: one declarative \"statement\"; make roughly half of them false by changing a detail the passages state, so a false statement is contradicted by the passages rather than merely unmentioned.",
    "short_answer": "Short-answer: \"expected_answer\" is a concise model answer; \"key_points\" lists the 2-4 essential points a good answer must contain.",
}

EXAM_SYSTEM_MESSAGE = (
    "You are DocuChat, writing study and exam questions from the user's own document. Every question tests "
    "understanding of what the supplied passages actually say, and every answer must be verifiable from them."
)


def build_generation_prompt(batch: GenerationBatch, context: str, grounding_mode: str, focus: str | None = None, avoid: list[str] | None = None) -> str:
    wanted = ", ".join(f"{count} {TYPE_LABELS[kind]} (type \"{kind}\")" for kind, count in batch.type_counts.items())
    rules = "\n".join(f"- {_RULES_BY_TYPE[kind]}" for kind in batch.type_counts)
    schemas = ",".join(_SCHEMA_BY_TYPE[kind] for kind in batch.type_counts)
    grounding = (
        "- Base questions primarily on the passages and cite them in source_ids. You may add a question on closely "
        "related general knowledge about the same topic; give it an empty source_ids list."
        if grounding_mode == "free"
        else "- Every question must be answerable from the passages alone, and must cite in source_ids the passage(s) it is based on. Never use outside facts."
    )
    focus_line = f"- Focus on: {focus}\n" if focus else ""
    avoid_line = ("- Do not repeat or rephrase any of these existing questions:\n" + "\n".join(f"  * {item}" for item in avoid[:30]) + "\n") if avoid else ""
    return (
        f"Write exactly {batch.size} exam questions: {wanted}.\n"
        f"Rules:\n{grounding}\n{rules}\n"
        "- \"explanation\" says briefly why the answer is correct, referring to what the passage states.\n"
        "- Test concepts, definitions, mechanisms and relationships -- not trivia like page numbers, figure numbers or headings.\n"
        "- Each question must test a different fact or concept; spread the questions across different passages.\n"
        "- The student never sees the passages: phrase every question so it stands alone. Never write \"the passage\", "
        "\"the text\" or \"the excerpt\" (\"according to the document\" is fine).\n"
        f"{focus_line}{avoid_line}"
        f"Return exactly one JSON object, with no prose outside it:\n{{\"questions\":[{schemas}]}}\n"
        f"source_ids use passage numbers 1 to {len(batch.sources)}.\n\n"
        f"PASSAGES:\n{context}"
    )


def _normalize_type(value) -> str | None:
    return _TYPE_ALIASES.get(str(value or "").strip().lower().replace("-", "_"))


def _parse_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower().rstrip(".")
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return None


def _clean_option(value) -> str:
    return _OPTION_PREFIX.sub("", strip_citation_markers(as_text(value, 300))).strip()


def _resolve_mcq_answer(raw_answer, options: list[str]) -> str | None:
    """The model's "answer" as one of `options`, or None. Accepts the exact
    option text (the requested form), a letter ("B", "(b)", "Option B"), or
    an option text with its letter prefix still attached ("B) MQTT")."""
    if isinstance(raw_answer, int) and not isinstance(raw_answer, bool):
        return None  # 0- vs 1-based is ambiguous -- don't guess which option was meant
    text = as_text(raw_answer, 300)
    letter = _LETTER_ANSWER.match(text)
    if letter:
        index = ord(letter.group(1).upper()) - ord("A")
        return options[index] if index < len(options) else None
    cleaned = _clean_option(text).casefold()
    return next((option for option in options if option.casefold() == cleaned), None)


def _student_facing(text: str) -> str:
    return _PASSAGE_REFERENCE.sub(lambda match: f"{match.group(1)} the document", strip_citation_markers(text))


def _normalized_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


@dataclass
class ParseOutcome:
    questions: list = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)


def parse_generated_questions(
    raw: str,
    *,
    allowed_types: set[str],
    source_count: int,
    grounding_mode: str = "document",
    rng: random.Random | None = None,
    seen: set[str] | None = None,
) -> ParseOutcome:
    """Validate every question the model produced; malformed output never
    raises. A question is dropped (with a reason, for logging/diagnostics)
    when its type is unsupported or wasn't requested, a required field is
    missing, an MCQ answer isn't one of its options, it duplicates another
    question, or -- under Stick to Document -- it cites no real passage.
    MCQ options are shuffled so the correct answer isn't always "A".
    """
    rng = rng or random.Random()
    seen = seen if seen is not None else set()
    outcome = ParseOutcome()
    payload = extract_json_object(raw)
    items = payload.get("questions") if payload else None
    if not isinstance(items, list):
        outcome.dropped.append("model output contained no questions list")
        return outcome

    for item in items:
        if not isinstance(item, dict):
            outcome.dropped.append("non-object question entry")
            continue
        kind = _normalize_type(item.get("type"))
        if kind is None:
            outcome.dropped.append(f"unsupported question type {item.get('type')!r}")
            continue
        if kind not in allowed_types:
            outcome.dropped.append(f"unrequested question type {kind!r}")
            continue
        source_ids = valid_source_ids(item.get("source_ids") or item.get("sources") or [], source_count, 3)
        if not source_ids and grounding_mode != "free":
            outcome.dropped.append("question cites no passage")
            continue
        common = {
            "explanation": strip_citation_markers(as_text(item.get("explanation"), 1200)),
            "source_ids": source_ids,
            "grounded": bool(source_ids),
        }
        try:
            if kind == "mcq":
                options = [_clean_option(option) for option in (item.get("options") or []) if _clean_option(option)] if isinstance(item.get("options"), list) else []
                answer = _resolve_mcq_answer(item.get("answer"), options)
                if answer is None:
                    raise ValueError("answer is not one of the options")
                rng.shuffle(options)
                question = MCQQuestion(question=_student_facing(as_text(item.get("question"), 600)), options=options, answer=answer, **common)
            elif kind == "true_false":
                answer = _parse_bool(item.get("answer"))
                if answer is None:
                    raise ValueError("true/false answer is not a boolean")
                statement = _student_facing(as_text(item.get("statement") or item.get("question"), 600))
                question = TrueFalseQuestion(statement=statement, answer=answer, **common)
            else:
                expected = strip_citation_markers(as_text(item.get("expected_answer") or item.get("answer"), 1500))
                key_points = [strip_citation_markers(point) for point in as_text_list(item.get("key_points"), limit=6)]
                if not key_points and expected:
                    key_points = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", expected) if sentence.strip()][:4]
                question = ShortAnswerQuestion(
                    question=_student_facing(as_text(item.get("question"), 600)), expected_answer=expected, key_points=key_points, **common,
                )
        except (ValidationError, ValueError) as exc:
            outcome.dropped.append(f"invalid {kind}: {str(exc).splitlines()[0]}")
            continue

        key = _normalized_key(question_prompt(question))
        if key in seen:
            outcome.dropped.append("duplicate question")
            continue
        seen.add(key)
        outcome.questions.append(question)
    return outcome


@dataclass
class GenerationResult:
    questions: list
    requested: int
    dropped: list[str]
    failed_batches: int = 0

    @property
    def shortfall(self) -> int:
        return max(self.requested - len(self.questions), 0)


CitationBuilder = Callable[[list[dict], list[int]], list[dict]]
SourceFormatter = Callable[[dict, int], str]


async def generate_questions(
    sources: list[dict],
    *,
    question_type: str,
    count: int,
    grounding_mode: str,
    build_citations: CitationBuilder,
    format_source: SourceFormatter,
    focus: str | None = None,
    seed: int | None = None,
) -> GenerationResult:
    """Run every batch concurrently, validate each batch against its own
    passages (so source_ids -> citations can never point at a passage the
    batch didn't see), then make one top-up attempt for any shortfall.
    A batch that fails (model error or unusable output) only costs its own
    questions; the request fails only if *nothing* valid came back.
    """
    rng = random.Random(seed)
    type_counts = plan_type_counts(question_type, count)
    batches = plan_batches(type_counts, sources)
    seen: set[str] = set()
    result = GenerationResult(questions=[], requested=count, dropped=[])

    async def run(batch: GenerationBatch, avoid: list[str] | None = None) -> str | OllamaUnavailable:
        context = "\n\n".join(format_source(source, index + 1) for index, source in enumerate(batch.sources))
        prompt = build_generation_prompt(batch, context, grounding_mode, focus, avoid)
        try:
            return await generate([{"role": "system", "content": EXAM_SYSTEM_MESSAGE}, {"role": "user", "content": prompt}])
        except OllamaUnavailable as exc:  # one failed batch must not fail the whole set
            logger.warning("Question-generation batch failed: %s", exc)
            return exc

    def collect(batch: GenerationBatch, raw) -> None:
        if not isinstance(raw, str):
            result.failed_batches += 1
            return
        outcome = parse_generated_questions(
            raw, allowed_types=set(batch.type_counts), source_count=len(batch.sources),
            grounding_mode=grounding_mode, rng=rng, seen=seen,
        )
        result.dropped.extend(outcome.dropped)
        # Never keep more of a type than the batch asked for.
        remaining = dict(batch.type_counts)
        for question in outcome.questions:
            if remaining.get(question.type, 0) <= 0:
                continue
            remaining[question.type] -= 1
            question.citations = build_citations(batch.sources, question.source_ids) if question.source_ids else []
            result.questions.append(question)

    raws = await asyncio.gather(*(run(batch) for batch in batches))
    for batch, raw in zip(batches, raws):
        collect(batch, raw)

    errors = [raw for raw in raws if isinstance(raw, OllamaUnavailable)]
    if not result.questions and len(errors) == len(batches):
        raise errors[0]  # the model is down/timing out, not "no valid questions"

    if result.shortfall and result.questions:
        have: dict[str, int] = {}
        for question in result.questions:
            have[question.type] = have.get(question.type, 0) + 1
        missing = {kind: wanted - have.get(kind, 0) for kind, wanted in type_counts.items() if wanted - have.get(kind, 0) > 0}
        if missing:
            top_up = GenerationBatch(type_counts=missing, sources=sources[: max(BATCH_SIZE + 3, 8)])
            collect(top_up, await run(top_up, avoid=[question_prompt(question) for question in result.questions]))

    order = {kind: index for index, kind in enumerate(QUESTION_TYPES)}
    if question_type == "mixed":
        # Interleave for a mixed exam instead of all MCQs first.
        buckets: dict[str, list] = {kind: [q for q in result.questions if q.type == kind] for kind in QUESTION_TYPES}
        interleaved = []
        while any(buckets.values()):
            for kind in sorted(buckets, key=order.get):
                if buckets[kind]:
                    interleaved.append(buckets[kind].pop(0))
        result.questions = interleaved
    result.questions = result.questions[:count]
    return result


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


class AnswerEvaluation(BaseModel):
    question_id: str
    question_type: QuestionType
    verdict: Literal["correct", "partial", "incorrect"]
    points: float
    max_points: float = 1.0
    selected_answer: str
    correct_answer: str
    feedback: str
    explanation: str = ""
    key_points_covered: list[str] = Field(default_factory=list)
    key_points_missing: list[str] = Field(default_factory=list)
    # exact: deterministic (MCQ/TF). model: graded by the local model against
    # the key points. keyword_fallback: the model was unavailable or returned
    # nothing usable, so coverage was approximated by keyword overlap.
    method: Literal["exact", "model", "keyword_fallback"]
    citations: list[dict] = Field(default_factory=list)


def _require_answer(answer: str | None) -> str:
    if answer is None or not str(answer).strip():
        raise InvalidAnswer("Choose or write an answer before submitting.")
    return str(answer).strip()


def evaluate_objective(question, answer: str) -> AnswerEvaluation:
    """MCQ and True/False are graded deterministically -- no model call, so
    the verdict can never be "talked into" a different answer."""
    selected = _require_answer(answer)
    if question.type == "mcq":
        resolved = _resolve_mcq_answer(selected, question.options)
        if resolved is None:
            raise InvalidAnswer("That answer isn't one of this question's options.")
        correct = resolved == question.answer
        selected_display = resolved
    elif question.type == "true_false":
        parsed = _parse_bool(selected)
        if parsed is None:
            raise InvalidAnswer("Answer True or False.")
        correct = parsed == question.answer
        selected_display = "True" if parsed else "False"
    else:
        raise UnsupportedQuestionType("Short-answer questions are graded with evaluate_short_answer.")

    correct_text = correct_answer_text(question)
    feedback = "Correct!" if correct else f"Not quite — the correct answer is: {correct_text.rstrip('.')}."
    return AnswerEvaluation(
        question_id=question.id, question_type=question.type, verdict="correct" if correct else "incorrect",
        points=1.0 if correct else 0.0, selected_answer=selected_display, correct_answer=correct_text,
        feedback=feedback, explanation=question.explanation, method="exact", citations=question.citations,
    )


SHORT_ANSWER_SYSTEM_MESSAGE = (
    "You are DocuChat, grading a student's short answer fairly and encouragingly against a rubric taken from their "
    "own document. Judge meaning, not wording: a key point counts as covered if the student expresses the same idea."
)
KEYWORD_COVERAGE_THRESHOLD = 0.5


def build_short_answer_prompt(question: ShortAnswerQuestion, answer: str) -> str:
    points = "\n".join(f"{index}. {point}" for index, point in enumerate(question.key_points, start=1))
    evidence = "\n".join(f"- {citation.get('excerpt', '')}" for citation in question.citations if citation.get("excerpt")) or "- (none)"
    return (
        f"QUESTION: {question.question}\n"
        f"MODEL ANSWER: {question.expected_answer}\n"
        f"KEY POINTS:\n{points}\n"
        f"DOCUMENT EVIDENCE:\n{evidence}\n\n"
        f"STUDENT ANSWER: {answer}\n\n"
        "For each key point, decide whether the student's answer covers it. List any claim in the student's answer "
        "that contradicts the model answer or evidence. Then write 2-3 sentences of specific, encouraging feedback. "
        "Return exactly one JSON object, with no prose outside it:\n"
        '{"covered":[key-point-number,...],"incorrect_claims":["..."],"feedback":"..."}'
    )


def keyword_coverage(key_points: list[str], answer: str) -> list[bool]:
    """Deterministic fallback grader: a key point counts as covered when at
    least half of its content words appear in the answer (prefix match, so
    "encrypts" matches "encryption")."""
    answer_stems = {word[:5] for word in terms(answer)}
    covered = []
    for point in key_points:
        point_stems = [term[:5] for term in terms(point)]
        hits = sum(1 for stem in point_stems if stem in answer_stems)
        covered.append(bool(point_stems) and hits / len(point_stems) >= KEYWORD_COVERAGE_THRESHOLD)
    return covered


def _short_answer_verdict(covered_count: int, total: int, has_incorrect_claims: bool) -> tuple[str, float]:
    points = round(covered_count / total, 2) if total else 0.0
    if covered_count == total and not has_incorrect_claims:
        return "correct", points
    if covered_count == 0:
        return "incorrect", 0.0
    return "partial", points


async def evaluate_short_answer(question: ShortAnswerQuestion, answer: str) -> AnswerEvaluation:
    """Rubric (shown to the student): one point per question, split evenly
    across the question's key points; "correct" needs every key point and no
    claim that contradicts the document. The model only decides *which* key
    points are covered -- the score arithmetic is always done here."""
    selected = _require_answer(answer)
    total = len(question.key_points)
    method = "model"
    incorrect_claims: list[str] = []
    feedback = ""
    try:
        raw = await generate([
            {"role": "system", "content": SHORT_ANSWER_SYSTEM_MESSAGE},
            {"role": "user", "content": build_short_answer_prompt(question, selected)},
        ])
        payload = extract_json_object(raw)
    except OllamaUnavailable as exc:  # grading must still work with the model down
        logger.warning("Short-answer grading fell back to keyword coverage: %s", exc)
        payload = None

    covered_numbers = valid_source_ids(payload.get("covered") or [], total) if payload else []
    if payload is not None and isinstance(payload.get("covered"), list):
        covered_flags = [index + 1 in covered_numbers for index in range(total)]
        incorrect_claims = [strip_citation_markers(claim) for claim in as_text_list(payload.get("incorrect_claims"), limit=5)]
        feedback = strip_citation_markers(as_text(payload.get("feedback"), 1200))
    else:
        method = "keyword_fallback"
        covered_flags = keyword_coverage(question.key_points, selected)

    covered = [point for point, flag in zip(question.key_points, covered_flags) if flag]
    missing = [point for point, flag in zip(question.key_points, covered_flags) if not flag]
    verdict, points = _short_answer_verdict(len(covered), total, bool(incorrect_claims))
    if not feedback:
        feedback = (
            "Your answer covers all the key points." if not missing
            else f"Your answer covers {len(covered)} of {total} key points. Review the missing points below."
        )
    if incorrect_claims:
        feedback += " Check these claims against the document: " + "; ".join(incorrect_claims)
    if method == "keyword_fallback":
        feedback += " (Graded approximately by keyword match because the model's grading was unavailable.)"
    return AnswerEvaluation(
        question_id=question.id, question_type="short_answer", verdict=verdict, points=points, selected_answer=selected,
        correct_answer=question.expected_answer, feedback=feedback.strip(), explanation=question.explanation,
        key_points_covered=covered, key_points_missing=missing, method=method, citations=question.citations,
    )


async def evaluate_answer(question, answer: str) -> AnswerEvaluation:
    if question.type == "short_answer":
        return await evaluate_short_answer(question, answer)
    return evaluate_objective(question, answer)


def parse_question(payload: dict):
    """Re-validate a question the client sends back for grading -- the same
    schema generation produced, so a tampered/malformed question (e.g. an
    MCQ whose answer isn't an option) is rejected instead of graded."""
    return _EXAM_QUESTION_ADAPTER.validate_python(payload)


# ---------------------------------------------------------------------------
# Session-local progress
# ---------------------------------------------------------------------------


class ExamProgress(BaseModel):
    """Round-tripped by the client with every answer and never stored
    server-side -- closing the tab ends the exam, by design."""

    total_questions: int = Field(ge=1, le=MAX_QUESTIONS)
    answered_questions: int = Field(default=0, ge=0)
    correct_answers: int = Field(default=0, ge=0)
    partial_answers: int = Field(default=0, ge=0)
    score: float = Field(default=0.0, ge=0)
    percentage: float = Field(default=0.0, ge=0, le=100)
    completed: bool = False
    answered_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "ExamProgress":
        if self.answered_questions > self.total_questions or self.correct_answers + self.partial_answers > self.answered_questions:
            raise ValueError("Exam progress counts are inconsistent.")
        return self


def apply_evaluation(progress: ExamProgress, evaluation: AnswerEvaluation) -> ExamProgress:
    """Pure transition: returns a new progress object. Answering the same
    question twice (double-submit, retry after a network error) is a no-op,
    so the score can't be inflated."""
    if evaluation.question_id in progress.answered_ids:
        return progress
    answered = progress.answered_questions + 1
    score = round(progress.score + evaluation.points, 2)
    return ExamProgress(
        total_questions=progress.total_questions,
        answered_questions=answered,
        correct_answers=progress.correct_answers + (evaluation.verdict == "correct"),
        partial_answers=progress.partial_answers + (evaluation.verdict == "partial"),
        score=score,
        percentage=round(100 * score / answered, 1),
        completed=answered >= progress.total_questions,
        answered_ids=[*progress.answered_ids, evaluation.question_id],
    )
