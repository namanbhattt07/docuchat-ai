import re
from dataclasses import dataclass, field
from enum import Enum

from app.services.learning import (
    as_text,
    cited_ids_in,
    clean_model_output,
    extract_json_object,
    normalize_citation_markers,
    strip_citation_markers,
    valid_source_ids,
)

# Group 6 TUTOR MODE. Everything tutor-specific lives here -- the action
# catalog, the explicit session state, the persona/prompt builder and the
# response parser -- so api/v1/chat.py only orchestrates (scope, history,
# retrieval, citations, persistence are all the existing Group 1-5 code).
# Nothing here talks to the model or the database.


class TutorAction(str, Enum):
    EXPLAIN_SIMPLY = "explain_simply"
    EXPLAIN_DEEPLY = "explain_deeply"
    GIVE_EXAMPLE = "give_example"
    GIVE_ANALOGY = "give_analogy"
    WHY = "why"
    COMPARE = "compare"
    QUIZ_ME = "quiz_me"
    GIVE_HINT = "give_hint"
    REVISE = "revise"
    TEACH_FROM_BEGINNING = "teach_from_beginning"
    # Internal, never a UI button: the student's typed reply to a question
    # the tutor asked (see TutorSession.pending_question).
    CHECK_ANSWER = "check_answer"


@dataclass(frozen=True)
class TutorActionSpec:
    label: str
    instruction: str
    # Title of the tutor-generated section this action mostly produces.
    tutor_title: str = "Tutor's explanation"
    # False for actions that must not hand the student the answer (Quiz Me,
    # Give Hint): the cited "From the document" section is withheld from the
    # reply -- citations are still returned so the question stays traceable.
    reveals_answer: bool = True
    # The reply is only useful if it ends with a question for the student.
    requires_question: bool = False
    # "example"/"analogy": which tutor-generated illustration is expected.
    example_kind: str | None = None
    # Meaningless without something to act on (a topic, a selection, or an
    # open question) -- e.g. "Why?" about nothing.
    needs_context: bool = False


TUTOR_ACTIONS: dict[TutorAction, TutorActionSpec] = {
    TutorAction.EXPLAIN_SIMPLY: TutorActionSpec(
        label="Explain simply",
        instruction="Explain the current concept using simple language and a small example. Avoid jargon; define any term you must use.",
        example_kind="example",
    ),
    TutorAction.EXPLAIN_DEEPLY: TutorActionSpec(
        label="Explain deeply",
        instruction="Give a detailed technical explanation, including mechanism, assumptions and important details.",
    ),
    TutorAction.GIVE_EXAMPLE: TutorActionSpec(
        label="Give example",
        instruction="Give a concrete example grounded in the current document where possible, then say briefly what the example shows.",
        example_kind="example",
    ),
    TutorAction.GIVE_ANALOGY: TutorActionSpec(
        label="Give analogy",
        instruction="Use an intuitive analogy, then connect the analogy back to the actual concept, including where the analogy breaks down.",
        example_kind="analogy",
    ),
    TutorAction.WHY: TutorActionSpec(
        label="Why?",
        instruction="Explain why this concept/step exists and what problem it solves.",
        tutor_title="Why it matters",
        needs_context=True,
    ),
    TutorAction.COMPARE: TutorActionSpec(
        label="Compare",
        instruction=(
            "Compare the current concept with the most relevant related concept. If the selected passage names two "
            "concepts, compare those two. Cover similarities and the key differences."
        ),
        tutor_title="Comparison",
        needs_context=True,
    ),
    TutorAction.QUIZ_ME: TutorActionSpec(
        label="Quiz me",
        instruction=(
            "Do not immediately explain the answer. Ask one appropriate question and wait for the student's response. "
            "The question must be answerable from the passages. Leave \"document_points\" empty and keep "
            "\"tutor_explanation\" to at most one short encouraging sentence."
        ),
        tutor_title="Quiz",
        reveals_answer=False,
        requires_question=True,
    ),
    TutorAction.GIVE_HINT: TutorActionSpec(
        label="Give hint",
        instruction=(
            "Give a progressive hint without revealing the full answer. If there is a pending question, the hint is for "
            "that question: point the student toward the relevant idea, but do not state the answer itself. Leave "
            "\"document_points\" empty."
        ),
        tutor_title="Hint",
        reveals_answer=False,
        needs_context=True,
    ),
    TutorAction.REVISE: TutorActionSpec(
        label="Revise",
        instruction="Provide a concise revision summary of the current topic: the key points to remember, as a short bullet list.",
        tutor_title="Revision summary",
    ),
    TutorAction.TEACH_FROM_BEGINNING: TutorActionSpec(
        label="Teach from beginning",
        instruction=(
            "Start from the prerequisites and build the concept step-by-step. Number the steps. Only move to the next "
            "step once the previous one is explained."
        ),
        tutor_title="Step-by-step",
    ),
    TutorAction.CHECK_ANSWER: TutorActionSpec(
        label="Check my answer",
        instruction=(
            "The student is replying to the PENDING QUESTION you asked. Evaluate their answer against the passages: say "
            "what is correct, what is missing or wrong, then state the correct answer with citations in "
            "\"document_points\". Be encouraging and specific. If the student's message is not an attempt to answer "
            "(for example they ask for help or change the subject), respond to what they actually asked instead."
        ),
        tutor_title="Feedback on your answer",
    ),
}

TUTOR_ACTION_LABELS = {action.value: spec.label for action, spec in TUTOR_ACTIONS.items()}

TUTOR_PERSONA = (
    "You are DocuChat Tutor, a patient teaching assistant helping a student learn from their own document. Teach, "
    "don't just answer: explain step by step, start with the simplest correct explanation and only go deeper when "
    "asked, connect new ideas to what the student already saw, and ask the student a short question when that would "
    "check or deepen their understanding. When the student is clearly trying to work something out, guide them "
    "toward the answer instead of handing it over. Keep a warm, encouraging, concise tone."
)

_STICK_TO_DOCUMENT_RULE = (
    "GROUNDING: Stick to Document. The supplied passages are the only source of facts. \"document_points\" states "
    "what the passages say, citing every sentence with [n]. \"tutor_explanation\" re-explains those same facts in "
    "your own words -- it must not introduce facts the passages don't support. Examples and analogies are allowed "
    "only as clearly illustrative teaching aids. If the passages don't cover the request, return empty fields and an "
    "empty source_ids list rather than guessing."
)
_GO_FREELY_RULE = (
    "GROUNDING: Go Freely. Use the passages as the primary, authoritative source in \"document_points\" (cite every "
    "sentence with [n]). \"tutor_explanation\" may add general knowledge beyond the passages to help the student "
    "learn, but never with a [n] marker, since it does not come from the document."
)

MAX_TOPIC_CHARS = 80
INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I couldn't find enough in the selected document to teach this. Try selecting the relevant passage in the PDF, "
    "picking a different topic, or switching to Go Freely to allow general knowledge."
)
NO_QUESTION_MESSAGE = "I couldn't form a question grounded in the document for this topic. Try another topic or select a passage."


class TutorContextError(ValueError):
    """A tutor request that can't be served as asked -- surfaced to the
    student as a 400 with this exact, user-facing message."""


@dataclass
class TutorSession:
    """The explicit per-request tutor state (see TUTOR MODE). The client
    carries current_topic/pending_question between turns; recent_turns come
    from the conversation's persisted messages (Group 5 threads), so the
    tutor sees the same bounded history every other chat turn does.
    """

    question: str
    action: TutorAction | None = None
    current_topic: str | None = None
    selected_text: str | None = None
    selected_page: int | None = None
    document_ids: list[str] | None = None
    collection_id: str | None = None
    grounding_mode: str = "document"
    pending_question: str | None = None
    recent_turns: str = ""

    def __post_init__(self) -> None:
        self.current_topic = _clean_inline(self.current_topic)[:MAX_TOPIC_CHARS] or None
        self.selected_text = (self.selected_text or "").strip() or None
        self.pending_question = _clean_inline(self.pending_question) or None

    @property
    def spec(self) -> TutorActionSpec | None:
        return TUTOR_ACTIONS.get(self.action) if self.action else None

    @property
    def has_context(self) -> bool:
        return bool(self.current_topic or self.selected_text or self.pending_question)

    def validate(self) -> None:
        if self.action == TutorAction.CHECK_ANSWER and not self.pending_question:
            raise TutorContextError("There's no open tutor question to answer. Ask the tutor to quiz you first.")
        spec = self.spec
        if spec and spec.needs_context and not self.has_context:
            raise TutorContextError(
                f"\"{spec.label}\" needs something to work on. Ask the tutor about a concept first, type a topic before "
                "choosing the action, or select a passage in the PDF."
            )

    def retrieval(self) -> "TutorRetrieval":
        """What to search for. An action button carries no retrievable words
        of its own ("Why?"), so it searches for the session's topic (plus the
        open question, when there is one) instead -- which is what makes
        "Explain this simply" -> "Why?" -> "Give an example" stay on the same
        concept without an extra rewrite call. A typed follow-up still goes
        through the existing standalone-question rewrite.
        """
        if self.selected_text:
            return TutorRetrieval(query=" ".join(part for part in (self.current_topic, self.question) if part))
        if self.action == TutorAction.CHECK_ANSWER:
            return TutorRetrieval(query=" ".join(part for part in (self.current_topic, self.pending_question) if part))
        if self.action:
            if self.current_topic:
                extra = self.pending_question if self.action == TutorAction.GIVE_HINT else None
                return TutorRetrieval(query=" ".join(part for part in (self.current_topic, extra) if part))
            return TutorRetrieval(query=None, document_wide=True)
        return TutorRetrieval(query=None, needs_rewrite=True)

    def history_for_rewrite(self) -> str:
        """recent_turns plus the current topic, so the standalone-question
        rewrite resolves "this"/"it" to the topic being taught even when the
        last message excerpt was truncated."""
        topic_line = f"CURRENT TUTOR TOPIC: {self.current_topic}\n" if self.current_topic else ""
        return f"{self.recent_turns}{topic_line}"


@dataclass
class TutorRetrieval:
    query: str | None
    document_wide: bool = False
    needs_rewrite: bool = False


def build_tutor_system_message(grounding_mode: str) -> str:
    return f"{TUTOR_PERSONA} {_GO_FREELY_RULE if grounding_mode == 'free' else _STICK_TO_DOCUMENT_RULE}"


_TUTOR_SCHEMA = (
    '{"topic":"short name of the concept being taught (2-8 words)",'
    '"document_points":"what the passages say that matters here, citing each sentence with [n]",'
    '"tutor_explanation":"your own step-by-step teaching, no [n] markers",'
    '"example_kind":"example | analogy | none",'
    '"example":"an illustrative example or analogy you made up, or empty string",'
    '"check_question":"one short question for the student, or empty string",'
    '"source_ids":[source-number,...]}'
)


def build_tutor_prompt(session: TutorSession, context: str, max_sources: int, format_instruction: str = "") -> str:
    spec = session.spec
    state_lines = [f"- Current topic: {session.current_topic or 'not set yet -- infer it from the request'}"]
    if session.selected_text:
        page = f" (page {session.selected_page})" if session.selected_page else ""
        state_lines.append(
            f"- The student selected a passage in the PDF{page}. It is SOURCE 1 below: teach from it first, and treat "
            "\"this\"/\"it\" as referring to it."
        )
    if session.pending_question:
        state_lines.append(f"- PENDING QUESTION you asked the student: \"{session.pending_question}\"")
    if session.current_topic and not session.selected_text:
        state_lines.append("- \"this\", \"it\" and \"that\" in the student's message refer to the current topic unless they clearly name something else.")

    if spec:
        task = f"TUTOR ACTION -- {spec.label}: {spec.instruction}"
    else:
        task = (
            "TUTOR ACTION -- Answer the student's message as a tutor: explain the idea step by step, simple first. If "
            "they seem to be working something out, guide them with a hint or a question rather than giving the whole "
            "answer."
        )
    if spec and spec.example_kind:
        task += f" Put the {spec.example_kind} in \"example\" and set \"example_kind\" to \"{spec.example_kind}\"."
    if spec and spec.requires_question:
        task += " \"check_question\" is required."
    else:
        task += " You may end with one short \"check_question\" when it would check understanding; otherwise leave it empty."

    # An action button acts on the stored topic, whose retrieval can surface
    # merely-adjacent passages when the document never covers it -- the
    # model must say so rather than teach from them. (A typed question sets
    # its own focus, and a selection *is* the evidence, so neither needs it.)
    topic_check = ""
    if spec and session.current_topic and not session.selected_text:
        fallback = (
            "teach it from general knowledge in tutor_explanation" if session.grounding_mode == "free"
            else "say in tutor_explanation that the document doesn't cover it"
        )
        topic_check = (
            f"\nFirst check that the passages actually discuss \"{session.current_topic}\". If none of them do, do not "
            f"teach it from loosely related passages: return empty document_points, an empty source_ids list, and {fallback}."
        )
    format_line = f"\nPRESENTATION (applies to tutor_explanation): {format_instruction}" if format_instruction else ""
    return (
        f"{session.recent_turns}"
        "TUTOR SESSION\n" + "\n".join(state_lines) + "\n\n"
        f"{task}{format_line}{topic_check}\n\n"
        "Only \"document_points\" may contain [n] citation markers, and only for passages you actually used. "
        f"Return exactly one JSON object, with no prose outside it:\n{_TUTOR_SCHEMA}\n"
        f"Use 0 to {max_sources} source_ids.\n\n"
        f"PASSAGES:\n{context}\n\n"
        f"STUDENT: {session.question}"
    )


@dataclass
class TutorSection:
    kind: str  # document | tutor | example | analogy | question | notice
    title: str
    content: str
    note: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "title": self.title, "content": self.content, "note": self.note}


@dataclass
class TutorResponse:
    topic: str
    action: TutorAction | None
    sections: list[TutorSection] = field(default_factory=list)
    source_ids: list[int] = field(default_factory=list)
    check_question: str = ""
    insufficient_evidence: bool = False
    grounding_mode: str = "document"

    @property
    def awaiting_answer(self) -> bool:
        return bool(self.check_question)

    def to_markdown(self) -> str:
        blocks = []
        for section in self.sections:
            note = f"_{section.note}_\n\n" if section.note else ""
            blocks.append(f"### {section.title}\n\n{note}{section.content}")
        return "\n\n".join(blocks)

    def metadata(self) -> dict:
        return {
            "topic": self.topic,
            "action": self.action.value if self.action else None,
            "action_label": TUTOR_ACTIONS[self.action].label if self.action else None,
            "sections": [section.to_dict() for section in self.sections],
            "awaiting_answer": self.awaiting_answer,
            "pending_question": self.check_question or None,
            "grounded": bool(self.source_ids),
            "insufficient_evidence": self.insufficient_evidence,
            "grounding_mode": self.grounding_mode,
        }


def _clean_inline(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _fallback_topic(session: TutorSession) -> str:
    if session.current_topic:
        return session.current_topic
    if session.selected_text:
        words = session.selected_text.split()
        return " ".join(words[:8]) + ("…" if len(words) > 8 else "")
    if session.action is None:
        return _clean_inline(session.question)[:MAX_TOPIC_CHARS]
    return "This document"


def parse_tutor_response(raw: str, session: TutorSession, source_count: int, max_sources: int) -> TutorResponse:
    """Turn the model's JSON into labelled sections, enforcing the grounding
    contract the prompt asked for rather than trusting it: only
    "document_points" keeps [n] markers, every cited id must point at a real
    passage, and under Stick to Document a reply with no verified evidence is
    replaced by a plain "couldn't find it" notice instead of being shown.
    Malformed output falls back the same way chat.py's grounded_response does
    (markers found in free text still count as evidence).
    """
    spec = session.spec
    free = session.grounding_mode == "free"
    payload = extract_json_object(raw)
    if payload is None:
        text = normalize_citation_markers(clean_model_output(raw))
        payload = {"document_points": text, "source_ids": cited_ids_in(text)} if cited_ids_in(text) else {"tutor_explanation": text}

    document_points = normalize_citation_markers(as_text(payload.get("document_points")))
    tutor_explanation = strip_citation_markers(as_text(payload.get("tutor_explanation")))
    example = strip_citation_markers(as_text(payload.get("example")))
    example_kind = str(payload.get("example_kind") or "").strip().lower()
    check_question = strip_citation_markers(as_text(payload.get("check_question"), 600))
    if session.action and session.current_topic:
        # An action (or answering the tutor's question) keeps teaching the
        # same topic; only a typed question or the first turn lets the model
        # name a new one.
        topic = session.current_topic
    else:
        topic = _clean_inline(as_text(payload.get("topic"), 200))[:MAX_TOPIC_CHARS] or _fallback_topic(session)

    source_ids = valid_source_ids([*(payload.get("source_ids") or []), *cited_ids_in(document_points)], source_count, max_sources)
    document_points = re.sub(r"\[(\d+)\]", lambda match: match.group(0) if int(match.group(1)) in source_ids else "", document_points).strip()
    if document_points:
        for source_id in source_ids:
            if f"[{source_id}]" not in document_points:
                document_points += f" [{source_id}]"

    response = TutorResponse(topic=topic, action=session.action, grounding_mode=session.grounding_mode)
    if not source_ids and not free:
        response.sections = [TutorSection("notice", "Not enough evidence", INSUFFICIENT_EVIDENCE_MESSAGE)]
        response.insufficient_evidence = True
        return response
    if spec and spec.requires_question and not check_question:
        response.sections = [TutorSection("notice", "No question generated", NO_QUESTION_MESSAGE)]
        response.insufficient_evidence = not source_ids
        return response

    tutor_note = (
        "Generated by the tutor — may include general knowledge beyond your documents."
        if free
        else "Generated by the tutor — explains the document's points in simpler terms."
    )
    sections: list[TutorSection] = []
    reveals = spec.reveals_answer if spec else True
    if reveals:
        if document_points and source_ids:
            sections.append(TutorSection("document", "From the document", document_points))
        elif free:
            sections.append(TutorSection("notice", "From the document", "Your selected documents don't cover this directly."))
            response.insufficient_evidence = True
    if tutor_explanation:
        sections.append(TutorSection("tutor", spec.tutor_title if spec else "Tutor's explanation", tutor_explanation, tutor_note))
    if example and example_kind != "none":
        kind = "analogy" if (example_kind == "analogy" or (spec and spec.example_kind == "analogy")) else "example"
        sections.append(TutorSection(
            kind,
            "Analogy" if kind == "analogy" else "Example",
            example,
            f"{'Analogy' if kind == 'analogy' else 'Illustrative example'} generated by the tutor — not taken from the document.",
        ))
    if check_question:
        sections.append(TutorSection("question", "Your turn", check_question, "Reply in the chat — the tutor will check your answer."))
    if not sections:
        sections.append(TutorSection("notice", "Not enough evidence", INSUFFICIENT_EVIDENCE_MESSAGE))
        response.insufficient_evidence = True

    response.sections = sections
    response.source_ids = source_ids
    response.check_question = check_question
    return response
