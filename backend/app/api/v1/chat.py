import json
import re
from typing import Literal
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db import get_db
from app.models import Collection, Conversation, Message
from app.services import brainstorm, calculation, evidence, extraction, templates, tutor, visual_qa, visuals
from app.services import collections as collections_service
from app.services.citations import locate_passage_bbox
from app.services.learning import explanatory_plan, merge_sources, sample_document_sources
from app.services.ollama import OllamaUnavailable, generate
from app.services.query_router import FormatMode, route, route_format
from app.services.retrieval import list_known_sections, retrieve_overview, retrieve_with_plan, rewrite_standalone_question
from app.services.providers import get_vision_provider, vision_capability
from app.services.retrieval_types import RetrievalMode, RetrievalPlan
from app.services.vector_store import get_collection

router = APIRouter(prefix="/chat", tags=["chat"])

HISTORY_TURNS = 3
HISTORY_EXCERPT_CHARS = 300

# Local "thinking" models sometimes cite a source as "[source-1]" instead of
# the requested "[1]" -- normalize before checking what's already cited, or
# the force-append step below double-cites it (e.g. "...[source-1] [1]").
CITATION_WORD_MARKER = re.compile(r"\[\s*source[\s_-]?(\d+)\s*\]", re.IGNORECASE)
DUPLICATE_MARKER = re.compile(r"(\[\d+\])(?:\s*\1)+")

# RetrievalMode -> a sentence telling the model what kind of passage set
# it's looking at, so it explains/synthesizes appropriately instead of
# treating every mode's passages like a single closest-match lookup.
_MODE_CONTEXT_SENTENCES: dict[RetrievalMode, str] = {
    RetrievalMode.OVERVIEW: (
        "The passages below are representative excerpts sampled evenly across the whole document, not just the "
        "single closest match to the question. Synthesize a coherent answer that draws on as many of them as are "
        "relevant."
    ),
    RetrievalMode.SECTION: "The passages below are drawn from the specific section of the document the question refers to.",
    RetrievalMode.PAGE: "The passages below are drawn from the specific page of the document the question refers to.",
    RetrievalMode.SELECTION: (
        "The first passage is the exact text the user selected in the document; the remaining passages are "
        "supporting context from elsewhere in the document. Prioritize explaining the selected passage itself."
    ),
    RetrievalMode.MULTI_DOCUMENT: (
        "The passages below are drawn from multiple documents the user selected. When documents disagree or cover "
        "different ground, say so explicitly and attribute each claim to its source document by filename."
    ),
    RetrievalMode.FACT: (
        "Answer using ONLY the supplied passages, but explain it naturally and clearly rather than just quoting "
        "fragments -- connect related details across the passages the way a good tutor would. Do not infer facts "
        "the passages don't support."
    ),
}

# FormatMode -> a presentation instruction, layered on top of the mode
# context sentence above. Deliberately independent of RetrievalMode (see
# IMPORTANT -- SEPARATE RETRIEVAL FROM PRESENTATION): the same SECTION
# passages can be explained normally, or asked for as bullets, a table, etc.
_FORMAT_SENTENCES: dict[FormatMode, str] = {
    FormatMode.DEFAULT: "",
    FormatMode.BULLETS: (
        "Format the answer as a concise Markdown bullet list (one idea per bullet, using \"- \"). Do not add a "
        "bullet for anything the passages don't support."
    ),
    FormatMode.TABLE: (
        "Format the answer as a single valid Markdown table with a header row and one row per item being "
        "compared. Only include a row/column the passages actually support -- if a specific cell's value isn't "
        "stated in the passages, write \"Not stated\" instead of inventing it. You may add one short sentence "
        "above the table for context, but the comparison itself must be the table."
    ),
    FormatMode.SECTIONED: (
        "Format the answer using Markdown headings (## Heading) with a short paragraph or bullets under each -- "
        "one heading per distinct section/topic the passages actually cover. Do not invent a heading the passages "
        "don't support."
    ),
    FormatMode.SUMMARY: (
        "Write a concise, structured summary -- a short paragraph, or a few bullets for the most important points -- "
        "noticeably more condensed than a full explanatory answer, while still citing every claim."
    ),
}


# Selection-mode-only (Group 4): which action the user picked in the PDF
# viewer's selection menu, layered on top of the SELECTION mode context
# sentence above (see SELECTION-BASED AI / SELECTION UX). Not a dict lookup
# on request.selection_action directly in the prompt builder below, so a
# plain SELECTION request with no action (still supported, e.g. asking a
# normal question while text happens to be selected) gets no extra sentence.
_SELECTION_ACTION_INSTRUCTIONS: dict[str, str] = {
    "explain": "Explain the selected passage in simple, clear language.",
    "summarize": "Summarize only the selected passage -- do not summarize the whole document.",
    "analyze": "Analyze the selected passage and explain its implications or important details.",
    "rewrite": "Rewrite the selected passage in different words while preserving its exact meaning.",
}
# A fixed, deterministic lead-in prepended to the final answer (never left to
# the model's own phrasing) so the chat UI always visibly indicates a
# selection-based answer, per SELECTION UX's "Chat should clearly indicate
# that the response is based on selected text."
_SELECTION_ACTION_LEAD_IN: dict[str, str] = {
    "explain": "Based on the selected passage:",
    "summarize": "Summary of the selected passage:",
    "analyze": "Analysis of the selected passage:",
    "rewrite": "Rewritten passage:",
}
_DEFAULT_SELECTION_LEAD_IN = "Based on the selected passage:"


def build_instructions(mode: RetrievalMode, format_mode: FormatMode, max_sources: int, extra_instruction: str = "") -> str:
    """The JSON-schema + citation-rule instructions sent to the model,
    parametrized by retrieval mode (what kind of passages it has) and
    format mode (how to present the answer) -- see build_format_instructions
    tests for the format-vs-retrieval independence this is built to keep.
    `extra_instruction` layers on a further, request-specific instruction
    (currently only the SELECTION action verb) without needing its own mode.
    LOCATION/CALCULATION/EXTRACTION never actually reach this function (each
    has its own deterministic or differently-schemed prompt in ask()), so
    `.get(..., "")` rather than a bare lookup keeps this total over the whole
    enum instead of a KeyError trap for any future caller.
    """
    body = " ".join(part for part in (_MODE_CONTEXT_SENTENCES.get(mode, ""), _FORMAT_SENTENCES[format_mode], extra_instruction) if part)
    schema = (
        '{"answer":"well-explained answer with numeric citation markers in square brackets, e.g. [1] -- never the '
        'word \\"source\\" inside the brackets","source_ids":[source-number,...]}'
    )
    return (
        f"{body} Return exactly one JSON object, with no prose outside it:\n{schema}\n"
        f"Cite every passage you actually drew on, using 1 to {max_sources} source_ids. If none of the passages "
        'are usable, return {"answer":"I couldn\'t find a supported answer in the selected document.","source_ids":[]}.'
    )


def build_free_instructions(mode: RetrievalMode, format_mode: FormatMode, max_sources: int, extra_instruction: str = "") -> str:
    """Group 5 GO FREELY grounding: the same passage-grounded contract as
    build_instructions, plus one extra, clearly separate field for
    supplemental general knowledge (see GROUNDING CONTROL). The model is
    told explicitly that only "answer" may carry a [n] citation -- never
    "additional_context" -- so a free-knowledge sentence can never be
    displayed as if it came from the document (see GROUNDING RULES / GO
    FREELY: "Never attach a document citation to a free-knowledge statement").
    """
    body = " ".join(part for part in (_MODE_CONTEXT_SENTENCES.get(mode, ""), _FORMAT_SENTENCES[format_mode], extra_instruction) if part)
    schema = (
        '{"answer":"well-explained answer grounded in the passages, with numeric citation markers in square '
        'brackets, e.g. [1] -- never the word \\"source\\" inside the brackets","source_ids":[source-number,...],'
        '"additional_context":"optional supplemental general knowledge the passages do NOT cover, with no '
        'citation markers at all -- empty string if the passages already fully answer the question"}'
    )
    return (
        f"{body} Use the supplied passages as your primary and authoritative evidence for \"answer\". Only if they "
        'are genuinely insufficient to fully answer the question, you may supplement with your own general '
        'knowledge -- but that supplemental knowledge belongs ONLY in "additional_context", never inside "answer", '
        'and must never carry a [n] citation marker since it does not come from the document. Return exactly one '
        f"JSON object, with no prose outside it:\n{schema}\n"
        f'Cite every passage you actually drew on in "answer", using 1 to {max_sources} source_ids. If none of the '
        'passages are usable and no reasonable general knowledge applies either, return {"answer":"I couldn\'t '
        'find a supported answer in the selected documents.","source_ids":[],"additional_context":""}.'
    )


def grounded_response_with_context(raw: str, source_count: int, max_sources: int = 3) -> tuple[str, list[int], str]:
    """Go Freely counterpart to grounded_response -- kept as a separate
    function (rather than a flag on grounded_response) so Stick to
    Document's parsing, and every existing test of it, stays untouched.
    Returns (answer, used_source_ids, additional_context); additional_context
    is always de-cited and is only ever non-empty when the caller actually
    asked for supplemental knowledge (see build_free_instructions).
    """
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    additional_context = ""
    try:
        payload = json.loads(cleaned[cleaned.find("{"):cleaned.rfind("}") + 1])
        answer = str(payload["answer"]).strip()
        source_ids = [int(value) for value in payload["source_ids"]]
        additional_context = str(payload.get("additional_context") or "").strip()
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        answer = cleaned
        source_ids = [int(value) for value in re.findall(r"\[(\d+)\]", cleaned)]

    answer = CITATION_WORD_MARKER.sub(r"[\1]", answer)
    if not source_ids:
        source_ids = [int(value) for value in re.findall(r"\[(\d+)\]", answer)]
    source_ids = list(dict.fromkeys(source_id for source_id in source_ids if 1 <= source_id <= source_count))[:max_sources]
    # Never trust a citation marker the model may have left inside the
    # supplemental sentence -- that would misrepresent free knowledge as
    # document evidence (see GROUNDING RULES / GO FREELY).
    additional_context = re.sub(r"\[\d+\]", "", additional_context).strip()
    # A model sometimes "fills" the field with markup debris (a lone ``` fence,
    # "-", "..."). Nothing readable means nothing to add, so no empty
    # "Additional context" heading -- the same rule as an empty string.
    if not re.search(r"\w", additional_context):
        additional_context = ""

    if not answer or not source_ids:
        # No verified document evidence -- "answer" can't be trusted to
        # actually be document-grounded, so it is dropped. Supplemental
        # knowledge (if any) still survives, clearly labelled, since Go
        # Freely explicitly permits answering from general knowledge when
        # the documents fall short.
        if additional_context:
            return "I couldn't find enough evidence in the selected documents to answer this directly.", [], additional_context
        return "I couldn't verify a source-grounded answer from the retrieved document passages.", [], ""

    for source_id in source_ids:
        if f"[{source_id}]" not in answer:
            answer += f" [{source_id}]"
    answer = DUPLICATE_MARKER.sub(r"\1", answer)
    return answer, source_ids, additional_context


def compose_grounded_answer(answer: str, additional_context: str) -> str:
    """Go Freely presentation (see GROUNDING ANSWER PRESENTATION): the
    "From your documents" / "Additional context" heading split only appears
    when there is actually supplemental knowledge to separate out -- a fully
    document-grounded answer renders identically to Stick to Document's, per
    "If no external knowledge was necessary, there is no need to add an
    unnecessary section."
    """
    if not additional_context:
        return answer
    return f"### From your documents\n\n{answer}\n\n### Additional context\n\n{additional_context}"


def format_source_block(source: dict, index: int) -> str:
    if source.get("source_type") == "selection":
        return f"SOURCE {index} | {source.get('filename') or 'this document'} | user-selected passage\n{source['content']}"
    origin = " | OCR text (may contain recognition errors)" if source.get("source_type") == "ocr" else ""
    return f"SOURCE {index} | {source['filename']} | page {source['page_number']}{origin}\n{source['content']}"


def grounded_system_message(use_free_grounding: bool) -> str:
    """The persona + grounding rule shared by normal chat answers and prompt
    templates (Stick to Document vs Go Freely)."""
    return (
        "You are DocuChat, a knowledgeable and friendly study partner helping the user understand their own "
        "document. Explain things clearly and naturally, the way a good tutor would -- connect ideas across the "
        "passages instead of just repeating bullet points verbatim. "
        + (
            "You may supplement the document's evidence with your own general knowledge when the passages "
            "genuinely don't cover something, but that supplemental knowledge must stay clearly separate from the "
            "document-grounded answer and must never be cited as if it came from the document."
            if use_free_grounding
            else (
                "Stay strictly grounded in the supplied passages: never state a fact they don't support, and cite "
                "every claim with a [n] marker. If the passages genuinely don't cover the question, say so plainly "
                "instead of guessing."
            )
        )
    )


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_ids: list[str] = Field(default_factory=list)
    conversation_id: str | None = None
    # SELECTION-mode retrieval plumbing (see RetrievalMode.SELECTION).
    selected_text: str | None = None
    selection_page: int | None = None
    # Group 4: which action the PDF viewer's selection menu triggered
    # (Explain/Summarize/Analyze/Rewrite). Only meaningful alongside
    # selected_text -- see the validation in ask() below. Group 5 reuses this
    # same field/pipeline for citation/quote reuse (see frontend
    # askSelectionAction call sites) rather than adding a parallel one.
    selection_action: Literal["explain", "summarize", "analyze", "rewrite"] | None = None
    # Group 5 multi-document collections: when set, `document_ids` is
    # validated server-side against this collection's real membership before
    # retrieval ever runs (see _resolve_document_scope) -- never trusted
    # blindly from the frontend (see SECURITY / DATA ISOLATION).
    collection_id: str | None = None
    # Group 5 grounding control (see GROUNDING CONTROL). Persisted only for
    # the duration of this one request, per "Grounding mode can be
    # request-level if that matches the current architecture."
    grounding_mode: Literal["document", "free"] = "document"
    # Group 5 threads/follow-ups: the prior message (usually an assistant
    # turn) the user hit "Reply" on, so that turn's context is pulled into
    # history even if the conversation has since moved on (see
    # _recent_history's anchor_message_id).
    reply_to_message_id: str | None = None
    # Group 6 learning modes. "chat" (the default) is the unchanged Group 1-5
    # pipeline; "tutor" and "brainstorm" share its scope validation, history,
    # routing, retrieval, citations and persistence, and only swap the
    # prompt/parser (see services/tutor.py, services/brainstorm.py).
    mode: Literal["chat", "tutor", "brainstorm", "template"] = "chat"
    # Group 7: which prompt template to apply (mode="template" only) -- a
    # built-in id ("builtin:case-brief") or a custom template's uuid.
    template_id: str | None = Field(default=None, max_length=80)
    # Tutor-only session state, carried by the client between turns (see
    # services/tutor.py::TutorSession). selected_text/selection_page above
    # double as the tutor's selected passage -- same Group 4 plumbing.
    tutor_action: Literal[
        "explain_simply", "explain_deeply", "give_example", "give_analogy", "why", "compare",
        "quiz_me", "give_hint", "revise", "teach_from_beginning", "check_answer",
    ] | None = None
    current_topic: str | None = Field(default=None, max_length=300)
    pending_question: str | None = Field(default=None, max_length=1000)


class AbstractRequest(BaseModel):
    document_ids: list[str] = Field(default_factory=list)
    conversation_id: str | None = None


def grounded_response(raw: str, source_count: int, max_sources: int = 3) -> tuple[str, list[int]]:
    """Accept only source-bound output; never display unverified citations."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    try:
        payload = json.loads(cleaned[cleaned.find("{"):cleaned.rfind("}") + 1])
        answer = str(payload["answer"]).strip()
        source_ids = [int(value) for value in payload["source_ids"]]
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        answer = cleaned
        source_ids = [int(value) for value in re.findall(r"\[(\d+)\]", cleaned)]

    answer = CITATION_WORD_MARKER.sub(r"[\1]", answer)
    if not source_ids:
        source_ids = [int(value) for value in re.findall(r"\[(\d+)\]", answer)]
    source_ids = list(dict.fromkeys(source_id for source_id in source_ids if 1 <= source_id <= source_count))[:max_sources]
    if not answer or not source_ids:
        return "I couldn't verify a source-grounded answer from the retrieved document passages.", []
    for source_id in source_ids:
        if f"[{source_id}]" not in answer:
            answer += f" [{source_id}]"
    answer = DUPLICATE_MARKER.sub(r"\1", answer)
    return answer, source_ids


def build_citations(db: Session, sources: list[dict], used_source_ids: list[int]) -> list[dict]:
    """Attach a bbox to every cited source, reusing the chunk's own bbox
    first (the Group 1 behavior) and only falling back to text-based passage
    location (see services/citations.py) when a chunk has none. `bbox_source`
    tells the frontend whether the highlight is exact, best-effort, or
    (rarely) not locatable at all, so it never claims precision it doesn't
    have.
    """
    citations = []
    for index in used_source_ids:
        source = sources[index - 1]
        bbox = source.get("bbox")
        if bbox:
            bbox_source = "exact"
        else:
            bbox = locate_passage_bbox(db, source["document_id"], source["page_number"], source["content"])
            bbox_source = "approximate" if bbox else "unavailable"
        citations.append({
            "index": index,
            "document_id": source["document_id"],
            "filename": source["filename"],
            "page_number": source["page_number"],
            "excerpt": source["content"][:280],
            "section": source.get("section"),
            "bbox": bbox,
            "bbox_source": bbox_source,
            # "ocr" when the passage came from recognized (scanned) text --
            # the page number is still the real PDF page either way.
            "source_type": source.get("source_type") or "text",
        })
    return citations


def build_location_answer(topic: str, sources: list[dict]) -> str:
    """Deterministic, no LLM call -- a LOCATION query wants a short pointer
    list, not a synthesized explanation (see LOCATION QUERY RESPONSE: "Do
    not generate a long explanation unless the user asks for one"). Every
    entry maps 1:1 to a citation built the normal way (see build_citations),
    so "clickable, jumps to the page, highlights the passage" comes from the
    existing citation UI/navigation -- nothing new to build or maintain.
    """
    lines = [f"### {topic}", ""]
    for source in sources:
        page = source.get("page_number")
        section = source.get("section")
        lines.append(f"- **{section}** — Page {page}" if section else f"- Page {page}")
    return "\n".join(lines)


def build_location_not_found_answer(topic: str) -> str:
    return f"I couldn't find where the document discusses \"{topic}\"."


_CALCULATION_OPERATION_LABELS: dict[calculation.CalculationOperation, str] = {
    calculation.CalculationOperation.DIFFERENCE: "Difference",
    calculation.CalculationOperation.RATIO: "Ratio",
    calculation.CalculationOperation.PERCENTAGE: "Percentage",
    calculation.CalculationOperation.PERCENTAGE_INCREASE: "Percentage increase",
    calculation.CalculationOperation.PERCENTAGE_DECREASE: "Percentage decrease",
    calculation.CalculationOperation.AVERAGE: "Average",
    calculation.CalculationOperation.COMPARISON: "Comparison",
    calculation.CalculationOperation.SUM: "Sum",
}
_CALCULATION_PERCENT_OPERATIONS = {
    calculation.CalculationOperation.PERCENTAGE,
    calculation.CalculationOperation.PERCENTAGE_INCREASE,
    calculation.CalculationOperation.PERCENTAGE_DECREASE,
}


def build_calculation_answer(outcome: calculation.CalculationOutcome) -> str:
    """Deterministic, no LLM call -- see FORMULA & DATA-ANALYSIS SUPPORT /
    CRITICAL RULE: the numbers and the arithmetic this renders were already
    computed entirely in services/calculation.py; this function only ever
    formats them, so a model can never alter, "round helpfully," or
    recompute a value on the way to the user.
    """
    assert outcome.result is not None and outcome.inputs is not None
    result = outcome.result
    rows = ["| Value | Amount | Source |", "|---|---|---|"]
    for value in outcome.inputs:
        marker = f"[{value.source_index}]" if value.source_index else "—"
        unit_suffix = value.unit if value.unit and value.unit not in ("%", "usd") else ""
        rows.append(f"| {value.label.capitalize()} | {value.value:g}{unit_suffix} | {marker} |")

    label = _CALCULATION_OPERATION_LABELS.get(result.operation, result.operation.value)
    unit_suffix = "%" if result.operation in _CALCULATION_PERCENT_OPERATIONS else ""
    summary = f"**{label}: {result.result:g}{unit_suffix}**"
    steps = "\n".join(f"- {step}" for step in result.steps)
    return "\n".join(rows) + "\n\n" + summary + "\n\n" + steps


EXTRACTION_SYSTEM_MESSAGE = (
    "You are DocuChat, extracting structured data from the user's document. Stay strictly grounded in the supplied "
    "passages -- never state a fact they don't support, and never invent a value for a field the passages don't "
    "state; use null for it instead."
)


def _persist_turn(db: Session, conversation: Conversation, question: str, answer: str, citations: list[dict], reply_to_message_id: str | None = None) -> dict:
    """Shared tail for every /chat branch -- saves both Message rows (with
    explicit, pre-known ids so the response can hand the assistant message's
    id back to the frontend for the Group 5 "Reply" affordance) and returns
    the same payload shape every branch has always returned, plus message_id.
    """
    assistant_message_id = str(uuid4())
    db.add(conversation)  # a new conversation is only stored once it has a turn (see ask())
    db.add_all([
        Message(id=str(uuid4()), conversation_id=conversation.id, role="user", content=question, reply_to_message_id=reply_to_message_id),
        Message(id=assistant_message_id, conversation_id=conversation.id, role="assistant", content=answer, citations_json=json.dumps(citations)),
    ])
    db.commit()
    return {"conversation_id": conversation.id, "answer": answer, "citations": citations, "message_id": assistant_message_id}


def _recent_history(db: Session, conversation_id: str, anchor_message_id: str | None = None) -> str:
    """Prior turns, given as context only -- never as citable evidence -- so
    follow-ups like "what about the second point" can resolve what "second
    point" refers to instead of being answered as if in isolation.

    Group 5 threads/follow-ups: when the user explicitly hit "Reply" on an
    older turn (`anchor_message_id`), that message is folded into the window
    even if it has scrolled outside the normal most-recent-N-turns cutoff --
    otherwise "Reply" would be indistinguishable from just asking a new
    question in an already-moved-on conversation.
    """
    messages = list(db.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(HISTORY_TURNS * 2)
    ).all())
    if anchor_message_id and not any(message.id == anchor_message_id for message in messages):
        anchor = db.get(Message, anchor_message_id)
        if anchor and anchor.conversation_id == conversation_id:
            messages.append(anchor)
    if not messages:
        return ""
    ordered = sorted(messages, key=lambda message: message.created_at)
    lines = [f"{message.role.upper()}: {message.content[:HISTORY_EXCERPT_CHARS]}" for message in ordered]
    return "PRIOR CONVERSATION (context only, not evidence -- do not cite this section):\n" + "\n".join(lines) + "\n\n"


def _resolve_document_scope(db: Session, request: "ChatRequest") -> list[str] | None:
    """Group 5 multi-document collections: resolves the eligible document
    ids retrieval must use. When a collection is active, the submitted
    document_ids are validated against that collection's real membership
    server-side (never trusted blindly -- see SECURITY / DATA ISOLATION), and
    an empty selection is a controlled 400, not a silent fall-back to the
    whole collection (see EMPTY SELECTION). Outside a collection, behavior is
    unchanged from before Group 5.
    """
    if not request.collection_id:
        return request.document_ids or None
    if not db.get(Collection, request.collection_id):
        raise HTTPException(status_code=404, detail="Collection not found.")
    member_ids = collections_service.member_document_ids(db, request.collection_id)
    scoped = [document_id for document_id in request.document_ids if document_id in member_ids]
    if not scoped:
        raise HTTPException(status_code=400, detail="No documents selected in this collection.")
    return scoped


def _plan_for(db: Session, question: str, document_ids: list[str] | None, selected_text: str | None, selection_page: int | None) -> RetrievalPlan:
    """The query-router call every /chat mode shares, with every top-k taken
    from settings. SECTION classification is resolved against the document's
    *actual* headings when we have them (e.g. "Sensors" instead of a raw
    guessed phrase) -- see query_router._match_section.
    """
    settings = get_settings()
    return route(
        question,
        document_ids,
        selected_text=selected_text,
        selection_page=selection_page,
        known_sections=list_known_sections(db, document_ids),
        default_top_k=settings.retrieval_top_k,
        overview_top_k=settings.retrieval_overview_max_sources,
        section_top_k=settings.retrieval_section_max_sources,
        page_top_k=settings.retrieval_page_max_sources,
        selection_top_k=settings.retrieval_selection_max_sources,
        multi_document_top_k=settings.retrieval_multi_document_max_sources,
        location_top_k=settings.retrieval_location_max_sources,
        extraction_top_k=settings.retrieval_extraction_max_sources,
        calculation_top_k=settings.retrieval_calculation_max_sources,
    )


def model_http_error(exc: OllamaUnavailable, activity: str) -> HTTPException:
    """Group 6: distinguishes "the model took too long" (504 -- worth retrying
    with a smaller request) from "the model isn't running" (503). The
    learning-mode requests are the longest generations in the app, so a
    timeout there must not read as "start Ollama".
    """
    if isinstance(exc.__cause__, httpx.TimeoutException):
        return HTTPException(status_code=504, detail=f"The local model took too long while {activity}. Try again, or ask for less at once.")
    return HTTPException(status_code=503, detail=f"Start Ollama and pull the configured Qwen model before {activity}.")


def _resolve_template(db: Session, request: "ChatRequest") -> templates.TemplateSpec:
    """A template request that can't be served is rejected before a
    conversation row is created for it (same rule as tutor requests)."""
    if not request.template_id:
        raise HTTPException(status_code=400, detail="Choose a template to apply.")
    try:
        return templates.get_template(db, request.template_id)
    except templates.TemplateNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _tutor_session(request: "ChatRequest", document_ids: list[str] | None) -> tutor.TutorSession:
    session = tutor.TutorSession(
        question=request.question,
        action=tutor.TutorAction(request.tutor_action) if request.tutor_action else None,
        current_topic=request.current_topic,
        selected_text=request.selected_text,
        selected_page=request.selection_page,
        document_ids=document_ids,
        collection_id=request.collection_id,
        grounding_mode=request.grounding_mode,
        pending_question=request.pending_question,
    )
    try:
        session.validate()
    except tutor.TutorContextError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return session


def _tutor_turn_text(session: tutor.TutorSession) -> str:
    """What gets persisted as the user's side of a tutor turn -- an action
    button's canned label alone ("Why?") would be meaningless in history, so
    the topic it acted on is folded in."""
    spec = session.spec
    if spec and session.action != tutor.TutorAction.CHECK_ANSWER:
        target = "the selected passage" if session.selected_text else session.current_topic
        return f"[Tutor · {spec.label}] {target}" if target else f"[Tutor · {spec.label}]"
    return session.question


async def _ask_tutor(db: Session, request: "ChatRequest", session: tutor.TutorSession, conversation: Conversation, document_ids: list[str] | None) -> dict:
    """Group 6 TUTOR MODE on top of the shared /chat pipeline: same router,
    retrieval, citations and message persistence; only the retrieval query,
    prompt and parser are tutor-specific (services/tutor.py)."""
    settings = get_settings()
    retrieval = session.retrieval()
    if retrieval.needs_rewrite:
        rewrite_history = session.history_for_rewrite()
        query = await rewrite_standalone_question(rewrite_history, request.question) if rewrite_history else request.question
    else:
        query = retrieval.query or request.question
    plan = explanatory_plan(
        _plan_for(db, query, document_ids, session.selected_text, session.selected_page),
        document_wide=retrieval.document_wide,
        overview_top_k=settings.retrieval_overview_max_sources,
        default_top_k=settings.retrieval_top_k,
    )
    plan.collection_id = request.collection_id

    sources = await retrieve_with_plan(db, get_collection(), plan)
    if not sources:
        raise HTTPException(status_code=400, detail=evidence.explain_empty_scope(db, document_ids, "Upload a text-based PDF before starting a tutor session."))

    max_sources = settings.retrieval_overview_max_sources if plan.mode in (RetrievalMode.OVERVIEW, RetrievalMode.MULTI_DOCUMENT) else 4
    context = "\n\n".join(format_source_block(source, i + 1) for i, source in enumerate(sources))
    prompt = tutor.build_tutor_prompt(session, context, max_sources, _FORMAT_SENTENCES[route_format(request.question)])
    try:
        raw = await generate([{"role": "system", "content": tutor.build_tutor_system_message(session.grounding_mode)}, {"role": "user", "content": prompt}])
    except OllamaUnavailable as exc:
        raise model_http_error(exc, "starting a tutor session") from exc

    parsed = tutor.parse_tutor_response(raw, session, len(sources), max_sources)
    citations = build_citations(db, sources, parsed.source_ids)
    result = _persist_turn(db, conversation, _tutor_turn_text(session), parsed.to_markdown(), citations, reply_to_message_id=request.reply_to_message_id)
    return {**result, "mode": "tutor", "tutor": parsed.metadata()}


# Retrieval modes whose question may be diverted to the visual path -- the
# plain "answer a question about the document" modes. LOCATION / CALCULATION /
# EXTRACTION / SELECTION have their own deterministic behavior and never are.
_VISUAL_ELIGIBLE_MODES = {RetrievalMode.FACT, RetrievalMode.OVERVIEW, RetrievalMode.SECTION, RetrievalMode.PAGE, RetrievalMode.MULTI_DOCUMENT}


async def _maybe_answer_visually(
    db: Session, request: "ChatRequest", conversation: Conversation, plan: RetrievalPlan, sources: list[dict], document_ids: list[str] | None,
) -> dict | None:
    """Group 7 visual Q&A gate. Returns None (-> the normal grounded answer)
    unless the question is about a chart/figure/diagram AND a recorded visual
    element exists on a relevant page. Then: an honest "not supported" reply
    with page/figure citations and the page's extracted text -- or, only when a
    local vision model is verified installed, that model's interpretation of
    the rendered page (see services/visual_qa.py)."""
    if plan.mode not in _VISUAL_ELIGIBLE_MODES or request.selected_text or not visual_qa.looks_visual(request.question):
        return None
    targets = visual_qa.resolve_targets(db, document_ids, request.question, sources)
    if not targets:
        return None
    capability = await vision_capability()
    provider = await get_vision_provider() if capability.supported else None
    outcome = await visual_qa.answer_visual_question(db, request.question, targets, sources, provider, capability.reason, visuals.stored_pdf_path)
    citations = build_citations(db, outcome.evidence, list(range(1, len(outcome.evidence) + 1)))
    result = _persist_turn(db, conversation, request.question, outcome.answer, citations, reply_to_message_id=request.reply_to_message_id)
    return {**result, "visual": outcome.metadata}


BRAINSTORM_MAX_SOURCES = 10
BRAINSTORM_RELEVANT_SOURCES = 6


async def _ask_brainstorm(db: Session, request: "ChatRequest", conversation: Conversation, history: str, document_ids: list[str] | None) -> dict:
    """Group 6 BRAINSTORMING MODE: evidence first, ideas second. Retrieval is
    the normal routed plan (relevance), topped up with document-wide coverage
    (breadth) unless the request is already pinned to a selection, section or
    page -- brainstorming "project ideas from this paper" needs more of the
    paper than the few passages closest to that sentence.
    """
    settings = get_settings()
    selected_text = (request.selected_text or "").strip() or None
    query = await rewrite_standalone_question(history, request.question) if history else request.question
    plan = explanatory_plan(
        _plan_for(db, query, document_ids, selected_text, request.selection_page),
        document_wide=False,
        overview_top_k=settings.retrieval_overview_max_sources,
        default_top_k=settings.retrieval_top_k,
    )
    plan.collection_id = request.collection_id
    relevant = await retrieve_with_plan(db, get_collection(), plan)
    if plan.mode in (RetrievalMode.SELECTION, RetrievalMode.SECTION, RetrievalMode.PAGE):
        sources = relevant[:BRAINSTORM_MAX_SOURCES]
    else:
        coverage = sample_document_sources(db, document_ids, BRAINSTORM_MAX_SOURCES - BRAINSTORM_RELEVANT_SOURCES)
        sources = merge_sources(relevant[:BRAINSTORM_RELEVANT_SOURCES], coverage, BRAINSTORM_MAX_SOURCES)
    if not sources:
        raise HTTPException(status_code=400, detail=evidence.explain_empty_scope(db, document_ids, "Upload a text-based PDF before brainstorming."))

    context = "\n\n".join(format_source_block(source, i + 1) for i, source in enumerate(sources))
    prompt = brainstorm.build_brainstorm_prompt(request.question, context, history, len(sources))
    try:
        raw = await generate([{"role": "system", "content": brainstorm.build_brainstorm_system_message(request.grounding_mode)}, {"role": "user", "content": prompt}])
    except OllamaUnavailable as exc:
        raise model_http_error(exc, "brainstorming") from exc

    parsed = brainstorm.parse_brainstorm_response(raw, len(sources), request.grounding_mode)
    citations = build_citations(db, sources, parsed.source_ids)
    result = _persist_turn(db, conversation, request.question, parsed.to_markdown(), citations, reply_to_message_id=request.reply_to_message_id)
    return {**result, "mode": "brainstorm", "brainstorm": parsed.metadata()}


TEMPLATE_MAX_SOURCES = 14
TEMPLATE_RELEVANT_SOURCES = 6


async def _ask_template(db: Session, request: "ChatRequest", conversation: Conversation, template: templates.TemplateSpec, document_ids: list[str] | None) -> dict:
    """Group 7 PROMPT TEMPLATES on top of the shared /chat pipeline. Nothing
    here is a second RAG implementation: evidence is the normal hybrid
    retrieval (for the template's subject) topped up with the same
    document-wide coverage sampler Brainstorm and Exam use; the answer goes
    through the same grounded_response / grounded_response_with_context
    parsing, so an unverifiable answer yields the existing insufficient-
    evidence reply, citations are only ever real retrieved passages, and
    Stick to Document / Go Freely behave exactly as in chat.
    """
    settings = get_settings()
    plan = RetrievalPlan(
        mode=RetrievalMode.FACT,
        query=templates.retrieval_query(template, request.question),
        document_ids=document_ids,
        collection_id=request.collection_id,
        top_k=settings.retrieval_top_k,
    )
    relevant = await retrieve_with_plan(db, get_collection(), plan)
    coverage = sample_document_sources(db, document_ids, TEMPLATE_MAX_SOURCES - TEMPLATE_RELEVANT_SOURCES)
    sources = merge_sources(relevant[:TEMPLATE_RELEVANT_SOURCES], coverage, TEMPLATE_MAX_SOURCES)
    if not sources:
        raise HTTPException(status_code=400, detail=evidence.explain_empty_scope(db, document_ids, "Upload a text-based PDF before applying a template."))
    unfit = templates.applicability_problem(db, template, document_ids)
    if unfit:
        answer = f"The “{template.name}” template can't be applied to this document. {unfit} Try a different template, such as Executive Summary or Study Guide."
        result = _persist_turn(db, conversation, request.question, answer, [], reply_to_message_id=request.reply_to_message_id)
        return {**result, "mode": "template", "template": {"id": template.id, "name": template.name}}

    use_free_grounding = request.grounding_mode == "free"
    build = build_free_instructions if use_free_grounding else build_instructions
    instructions = build(RetrievalMode.OVERVIEW, FormatMode.DEFAULT, TEMPLATE_MAX_SOURCES, templates.build_template_instruction(template))
    context = "\n\n".join(format_source_block(source, i + 1) for i, source in enumerate(sources))
    prompt = f"{instructions}\n\nPASSAGES:\n{context}\n\nREQUEST: {request.question}"
    try:
        raw_answer = await generate([{"role": "system", "content": grounded_system_message(use_free_grounding)}, {"role": "user", "content": prompt}])
    except OllamaUnavailable as exc:
        raise model_http_error(exc, "applying a template") from exc

    if use_free_grounding:
        answer, used_source_ids, additional_context = grounded_response_with_context(raw_answer, len(sources), TEMPLATE_MAX_SOURCES)
    else:
        answer, used_source_ids = grounded_response(raw_answer, len(sources), TEMPLATE_MAX_SOURCES)
        additional_context = ""
    citations = build_citations(db, sources, used_source_ids)
    result = _persist_turn(db, conversation, request.question, compose_grounded_answer(answer, additional_context), citations, reply_to_message_id=request.reply_to_message_id)
    return {**result, "mode": "template", "template": {"id": template.id, "name": template.name}}


@router.get("/conversations")
def list_conversations(db: Session = Depends(get_db)):
    items = db.scalars(select(Conversation).order_by(Conversation.created_at.desc())).all()
    return [{"id": item.id, "title": item.title, "created_at": item.created_at} for item in items]


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, db: Session = Depends(get_db)):
    conversation = db.get(Conversation, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    messages = db.scalars(select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at)).all()
    return {
        "id": conversation.id,
        "title": conversation.title,
        "messages": [
            {
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "citations": json.loads(message.citations_json or "[]"),
                "reply_to_message_id": message.reply_to_message_id,
            }
            for message in messages
        ],
    }


@router.post("")
async def ask(request: ChatRequest, db: Session = Depends(get_db)):
    if request.selection_action and not (request.selected_text and request.selected_text.strip()):
        raise HTTPException(status_code=400, detail="A text selection is required for this action.")

    settings = get_settings()
    document_ids = _resolve_document_scope(db, request)
    # Group 6: a tutor request that can't be served (e.g. "Why?" with no
    # topic) is rejected before a conversation row is ever created for it.
    tutor_session = _tutor_session(request, document_ids) if request.mode == "tutor" else None
    template = _resolve_template(db, request) if request.mode == "template" else None
    conversation = db.get(Conversation, request.conversation_id) if request.conversation_id else None
    history = _recent_history(db, conversation.id, request.reply_to_message_id) if conversation else ""
    if not conversation:
        title = (
            f"Tutor: {request.current_topic or request.question}" if tutor_session
            else f"Template: {template.name}" if template
            else request.question
        )
        # Not added to the session yet: a request rejected below (nothing to
        # search, a model outage) must not leave an empty conversation behind.
        # _persist_turn stores it together with its first turn.
        conversation = Conversation(id=str(uuid4()), title=title[:70])

    if tutor_session:
        tutor_session.recent_turns = history
        return await _ask_tutor(db, request, tutor_session, conversation, document_ids)
    if request.mode == "brainstorm":
        return await _ask_brainstorm(db, request, conversation, history, document_ids)
    if template:
        return await _ask_template(db, request, conversation, template, document_ids)

    # Retrieval needs a self-contained query -- "what are its type" carries
    # no signal about what "its" is. Generation still uses the user's literal
    # wording below, so history stays faithful to what was actually typed.
    retrieval_question = await rewrite_standalone_question(history, request.question) if history else request.question

    plan = _plan_for(db, retrieval_question, document_ids, request.selected_text, request.selection_page)
    plan.collection_id = request.collection_id
    # Presentation is classified from the user's literal wording, not the
    # (possibly rewritten) retrieval query -- "explain that in bullet
    # points" carries formatting intent even though it has no retrievable
    # signal of its own (see IMPORTANT -- SEPARATE RETRIEVAL FROM PRESENTATION).
    format_mode = route_format(request.question)

    sources = await retrieve_with_plan(db, get_collection(), plan)
    visual_response = await _maybe_answer_visually(db, request, conversation, plan, sources, document_ids)
    if visual_response is not None:
        return visual_response
    if not sources:
        # LOCATION treats "nothing found" as a legitimate, expected answer
        # (the topic just isn't in the document) rather than the generic
        # "no text extracted from this PDF" error every other mode gives --
        # see LOCATION QUERY RESPONSE / missing result.
        if plan.mode == RetrievalMode.LOCATION:
            topic = plan.location_topic or plan.query
            return _persist_turn(db, conversation, request.question, build_location_not_found_answer(topic), [], reply_to_message_id=request.reply_to_message_id)
        if plan.mode == RetrievalMode.PAGE and plan.page is not None:
            # "What is on page 99?" / a blank page: a definite answer from the
            # document's own structure, not an error and not a guess.
            reply = evidence.page_absence_reply(db, document_ids, plan.page)
            if reply:
                return _persist_turn(db, conversation, request.question, reply, [], reply_to_message_id=request.reply_to_message_id)
        raise HTTPException(status_code=400, detail=evidence.explain_empty_scope(db, document_ids, "Upload a text-based PDF before asking a question."))

    if plan.mode == RetrievalMode.LOCATION:
        topic = plan.location_topic or plan.query
        answer = build_location_answer(topic, sources)
        citations = build_citations(db, sources, list(range(1, len(sources) + 1)))
        return _persist_turn(db, conversation, request.question, answer, citations, reply_to_message_id=request.reply_to_message_id)

    if plan.mode == RetrievalMode.CALCULATION:
        outcome = calculation.compute(request.question, sources)
        if outcome.success:
            answer = build_calculation_answer(outcome)
            used_ids = sorted({value.source_index for value in outcome.inputs} - {None}) if outcome.inputs else []
            citations = build_citations(db, sources, used_ids) if used_ids else []
        else:
            answer = outcome.message or "I couldn't verify a source-grounded calculation from the retrieved document passages."
            citations = []
        return _persist_turn(db, conversation, request.question, answer, citations, reply_to_message_id=request.reply_to_message_id)

    if plan.mode == RetrievalMode.EXTRACTION:
        extraction_request = extraction.parse_extraction_request(request.question)
        extraction_context = "\n\n".join(format_source_block(source, i + 1) for i, source in enumerate(sources))
        extraction_instructions = extraction.build_extraction_instructions(extraction_request.entity_type, extraction_request.fields, len(sources))
        extraction_prompt = f"{extraction_instructions}\n\nPASSAGES:\n{extraction_context}\n\nTASK: Extract the requested structured data."
        try:
            raw_extraction = await generate([{"role": "system", "content": EXTRACTION_SYSTEM_MESSAGE}, {"role": "user", "content": extraction_prompt}])
        except OllamaUnavailable as exc:
            raise HTTPException(status_code=503, detail="Start Ollama and pull the configured Qwen model before chatting.") from exc
        items = extraction.parse_extraction_response(raw_extraction, extraction_request.fields, len(sources), sources)
        if items:
            answer = extraction.render_extraction_markdown(extraction_request.entity_type, items, extraction_request.fields, format_mode)
            used_ids = extraction.used_source_ids(items)
            citations = build_citations(db, sources, used_ids) if used_ids else []
        else:
            answer = "I couldn't extract structured data for that request from the selected document."
            citations = []
        return _persist_turn(db, conversation, request.question, answer, citations, reply_to_message_id=request.reply_to_message_id)

    max_sources = settings.retrieval_overview_max_sources if plan.mode in (RetrievalMode.OVERVIEW, RetrievalMode.MULTI_DOCUMENT) else 3
    context = "\n\n".join(format_source_block(source, i + 1) for i, source in enumerate(sources))
    follow_up_note = (
        "This is a follow-up in an ongoing conversation. If the user is asking you to elaborate, go deeper, or "
        "explain further, you must surface new information from the passages beyond what you already said -- do "
        "not just repeat a prior answer verbatim.\n\n"
        if history
        else ""
    )
    selection_instruction = (
        _SELECTION_ACTION_INSTRUCTIONS.get(request.selection_action, "")
        if plan.mode == RetrievalMode.SELECTION and request.selection_action
        else ""
    )
    # Group 5 GROUNDING CONTROL: Stick to Document (default) keeps the exact
    # Group 1-4 prompt/parsing pipeline untouched below; Go Freely swaps in
    # the instructions/parser that allow a clearly separate, never-cited
    # "additional_context" field (see build_free_instructions).
    use_free_grounding = request.grounding_mode == "free"
    instructions = (
        build_free_instructions(plan.mode, format_mode, max_sources, selection_instruction)
        if use_free_grounding
        else build_instructions(plan.mode, format_mode, max_sources, selection_instruction)
    )
    prompt = f"{history}{follow_up_note}{instructions}\n\nPASSAGES:\n{context}\n\nQUESTION: {request.question}"
    system_message = grounded_system_message(use_free_grounding)
    try:
        raw_answer = await generate([{"role": "system", "content": system_message}, {"role": "user", "content": prompt}])
    except OllamaUnavailable as exc:
        raise HTTPException(status_code=503, detail="Start Ollama and pull the configured Qwen model before chatting.") from exc

    if use_free_grounding:
        answer, used_source_ids, additional_context = grounded_response_with_context(raw_answer, len(sources), max_sources)
    else:
        answer, used_source_ids = grounded_response(raw_answer, len(sources), max_sources)
        additional_context = ""

    if plan.mode == RetrievalMode.SELECTION and used_source_ids:
        lead_in = _SELECTION_ACTION_LEAD_IN.get(request.selection_action, _DEFAULT_SELECTION_LEAD_IN)
        answer = f"{lead_in}\n\n{answer}"
    final_answer = compose_grounded_answer(answer, additional_context)
    citations = build_citations(db, sources, used_source_ids)
    return _persist_turn(db, conversation, request.question, final_answer, citations, reply_to_message_id=request.reply_to_message_id)


ABSTRACT_SYSTEM_MESSAGE = (
    "You are DocuChat, generating a structured high-level abstract of the user's document. Stay strictly grounded "
    "in the supplied passages -- never state a fact they don't support -- and cite every claim with a [n] marker."
)

ABSTRACT_OUTLINE = "# Abstract\n## Main Topic\n## Key Ideas\n## Important Sections\n## Key Terms / Concepts"


@router.post("/abstract")
async def abstract(request: AbstractRequest, db: Session = Depends(get_db)):
    """A document-level action, not a normal question -- so it always uses
    OVERVIEW retrieval directly (document-wide coverage) rather than routing
    the way a typed question would (see ABSTRACT ACTION: "must NOT simply
    ask FACT retrieval to summarize its top 5 chunks"). Still flows through
    the same grounded_response/build_citations/Message pipeline as /chat, so
    citations and conversation history behave identically.
    """
    settings = get_settings()
    conversation = db.get(Conversation, request.conversation_id) if request.conversation_id else None
    if not conversation:
        # Not added to the session yet, same as ask(): a request that fails
        # below (nothing to search, a model outage) must not leave an empty
        # conversation behind. _persist_turn stores it with its first turn.
        conversation = Conversation(id=str(uuid4()), title="Document abstract")

    document_ids = request.document_ids or None
    max_sources = settings.retrieval_overview_max_sources
    sources = retrieve_overview(db, document_ids, limit=max_sources)
    if not sources:
        raise HTTPException(status_code=400, detail="Upload a text-based PDF before generating an abstract.")

    context = "\n\n".join(format_source_block(source, i + 1) for i, source in enumerate(sources))
    instructions = f"""The passages below are representative excerpts sampled evenly across the whole document, covering its structure end to end. Write a structured abstract following this Markdown outline -- omit a section only if the passages truly give you nothing for it, and never invent content to fill a section:

{ABSTRACT_OUTLINE}

Cite every claim with a numeric marker in square brackets, e.g. [1] -- never the word "source" inside the brackets. Return exactly one JSON object, with no prose outside it:
{{"answer":"the full Markdown abstract above, as one string with literal \\n newlines","source_ids":[source-number,...]}}
Use 1 to {max_sources} source_ids, covering as much of the document as genuinely supports a claim. If the passages are unusable, return {{"answer":"I couldn't generate a supported abstract from this document.","source_ids":[]}}."""
    prompt = f"{instructions}\n\nPASSAGES:\n{context}\n\nTASK: Generate the abstract."
    try:
        raw_answer = await generate([{"role": "system", "content": ABSTRACT_SYSTEM_MESSAGE}, {"role": "user", "content": prompt}])
    except OllamaUnavailable as exc:
        raise HTTPException(status_code=503, detail="Start Ollama and pull the configured Qwen model before generating an abstract.") from exc
    answer, used_source_ids = grounded_response(raw_answer, len(sources), max_sources)
    citations = build_citations(db, sources, used_source_ids)
    result = _persist_turn(db, conversation, "Generate an abstract of this document.", answer, citations)
    return {"conversation_id": result["conversation_id"], "answer": answer, "citations": citations}
