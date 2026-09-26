import json
import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db import get_db
from app.models import Conversation, Message
from app.services.ollama import OllamaUnavailable, generate
from app.services.retrieval import is_broad_query, retrieve, retrieve_overview
from app.services.vector_store import get_collection

router = APIRouter(prefix="/chat", tags=["chat"])

HISTORY_TURNS = 3
HISTORY_EXCERPT_CHARS = 300

# Local "thinking" models sometimes cite a source as "[source-1]" instead of
# the requested "[1]" -- normalize before checking what's already cited, or
# the force-append step below double-cites it (e.g. "...[source-1] [1]").
CITATION_WORD_MARKER = re.compile(r"\[\s*source[\s_-]?(\d+)\s*\]", re.IGNORECASE)
DUPLICATE_MARKER = re.compile(r"(\[\d+\])(?:\s*\1)+")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
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


def _recent_history(db: Session, conversation_id: str) -> str:
    """Prior turns, given as context only -- never as citable evidence -- so
    follow-ups like "what about the second point" can resolve what "second
    point" refers to instead of being answered as if in isolation.
    """
    messages = db.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(HISTORY_TURNS * 2)
    ).all()
    if not messages:
        return ""
    lines = [f"{message.role.upper()}: {message.content[:HISTORY_EXCERPT_CHARS]}" for message in reversed(messages)]
    return "PRIOR CONVERSATION (context only, not evidence -- do not cite this section):\n" + "\n".join(lines) + "\n\n"


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
    return {"id": conversation.id, "title": conversation.title, "messages": [{"id": message.id, "role": message.role, "content": message.content, "citations": json.loads(message.citations_json or "[]")} for message in messages]}


@router.post("")
async def ask(request: ChatRequest, db: Session = Depends(get_db)):
    settings = get_settings()
    conversation = db.get(Conversation, request.conversation_id) if request.conversation_id else None
    history = _recent_history(db, conversation.id) if conversation else ""
    if not conversation:
        conversation = Conversation(title=request.question[:70])
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    broad = is_broad_query(request.question)
    if broad:
        sources = retrieve_overview(db, request.document_ids or None)
        max_sources = settings.retrieval_overview_max_sources
    else:
        sources = await retrieve(db, get_collection(), request.question, request.document_ids or None)
        max_sources = 3

    if not sources:
        raise HTTPException(status_code=400, detail="Upload a text-based PDF before asking a question.")

    context = "\n\n".join(f"SOURCE {i + 1} | {source['filename']} | page {source['page_number']}\n{source['content']}" for i, source in enumerate(sources))
    if broad:
        instructions = f"""The passages below are representative excerpts sampled evenly across the whole document, not just the single closest match to the question. Synthesize a coherent answer that draws on as many of them as are relevant. Return exactly one JSON object, with no prose outside it:
{{"answer":"synthesized answer with numeric citation markers in square brackets, e.g. [1] -- never the word \"source\" inside the brackets","source_ids":[source-number,...]}}
Cite every passage you actually drew on, up to {max_sources} source_ids. If none of the passages are usable, return {{"answer":"I couldn't find a supported answer in the selected document.","source_ids":[]}}."""
    else:
        instructions = f"""Answer using ONLY the supplied passages. Do not infer missing facts. Select a source only when it directly supports the answer. Return exactly one JSON object, with no prose outside it:
{{"answer":"short factual answer with numeric citation markers in square brackets, e.g. [1] -- never the word \"source\" inside the brackets","source_ids":[source-number]}}
Use 1 to {max_sources} source_ids. If the passages do not directly answer the question, return {{"answer":"I couldn't find a supported answer in the selected document.","source_ids":[]}}."""
    prompt = f"{history}{instructions}\n\nPASSAGES:\n{context}\n\nQUESTION: {request.question}"
    try:
        raw_answer = await generate([{"role": "system", "content": "You are a strict evidence-grounded research assistant. Never use outside knowledge."}, {"role": "user", "content": prompt}])
    except OllamaUnavailable as exc:
        raise HTTPException(status_code=503, detail="Start Ollama and pull the configured Qwen model before chatting.") from exc
    answer, used_source_ids = grounded_response(raw_answer, len(sources), max_sources)
    citations = [{"index": index, "document_id": sources[index - 1]["document_id"], "filename": sources[index - 1]["filename"], "page_number": sources[index - 1]["page_number"], "excerpt": sources[index - 1]["content"][:280]} for index in used_source_ids]
    db.add_all([Message(conversation_id=conversation.id, role="user", content=request.question), Message(conversation_id=conversation.id, role="assistant", content=answer, citations_json=json.dumps(citations))])
    db.commit()
    return {"conversation_id": conversation.id, "answer": answer, "citations": citations}
