import re
from collections import Counter, defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import Chunk, Document
from app.services.ollama import OllamaUnavailable, embed, generate

STOP_WORDS = {"a", "an", "and", "are", "about", "the", "of", "for", "in", "is", "it", "to", "was", "what", "which", "with", "this", "that"}

# Questions that ask about the document as a whole rather than a specific
# fact. A narrow top-k similarity search can never answer these well: there
# is no single passage that "is" a summary, so the closest-matching chunks
# are usually unrelated technical details, and the model correctly (but
# unhelpfully) refuses to answer from them. See `retrieve_overview`.
BROAD_QUERY_PATTERN = re.compile(
    r"\b("
    r"brief( me)?|summar(y|ize|ise)|overview|outline|"
    r"main idea|main point|key (point|takeaway)s?|"
    r"what('?s| is) (this|the) (document|pdf|file) about|"
    r"what (is|does) (this|the) (document|pdf|file) (cover|contain)|"
    r"tell me about this (document|pdf|file)|"
    r"topics covered|table of contents|tl;?dr|"
    r"give (me )?(a |an )?(summary|overview|brief|breakdown)"
    r")\b",
    re.IGNORECASE,
)


def is_broad_query(question: str) -> bool:
    return bool(BROAD_QUERY_PATTERN.search(question))


async def rewrite_standalone_question(history: str, question: str) -> str:
    """Follow-ups like "what are its type" or "explain them in detail" carry
    no retrievable signal on their own. Left as-is, retrieval matches
    whatever is generically similar to the leftover words, hands the model
    real-but-irrelevant passages, and it answers confidently from those --
    which is what actually caused the "hallucinations" seen in testing, not
    the model inventing facts. Resolving references against the prior turns
    before searching fixes it at the source. Retrieval-only: the caller
    still uses the user's literal question for generation and history.
    """
    prompt = (
        f"{history}"
        "Rewrite ONLY the follow-up question below as a fully standalone "
        "question with every pronoun and reference resolved using the prior "
        'conversation (e.g. "what are its type" -> "what are the types of '
        'sensors"). If it is already standalone, return it unchanged. Reply '
        "with ONLY the rewritten question -- no quotes, no explanation, no "
        "prefix.\n\n"
        f"FOLLOW-UP QUESTION: {question}"
    )
    try:
        raw = await generate([{"role": "user", "content": prompt}], think=False)
    except OllamaUnavailable:
        return question
    rewritten = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip().strip('"').strip()
    return rewritten or question


def terms(text: str) -> list[str]:
    return [term for term in re.findall(r"[a-z0-9]+", text.lower()) if len(term) > 2 and term not in STOP_WORDS]


def lexical_score(question: str, content: str, query_terms: list[str] | None = None) -> float:
    query_terms = terms(question) if query_terms is None else query_terms
    if not query_terms:
        return 0.0
    lowered = content.lower()
    if not any(term in lowered for term in query_terms):
        return 0.0
    counts = Counter(terms(content))
    score = sum(min(counts[term], 2) for term in query_terms)
    for size in (2, 3):
        for index in range(len(query_terms) - size + 1):
            phrase = " ".join(query_terms[index:index + size])
            if phrase in lowered:
                score += size * 2
    return score


def _load_candidate_chunks(db: Session, document_ids: list[str] | None) -> dict[str, dict]:
    statement = select(Chunk, Document.filename).join(Document, Chunk.document_id == Document.id)
    if document_ids:
        statement = statement.where(Chunk.document_id.in_(document_ids))
    rows = db.execute(statement).all()
    return {
        chunk.id: {
            "chunk_id": chunk.id,
            "content": chunk.content,
            "document_id": chunk.document_id,
            "filename": filename,
            "page_number": chunk.page_number,
        }
        for chunk, filename in rows
    }


async def retrieve(db: Session, collection, question: str, document_ids: list[str] | None, limit: int | None = None) -> list[dict]:
    """Fuse semantic and lexical evidence, favouring exact phrase matches."""
    settings = get_settings()
    limit = limit or settings.retrieval_top_k
    records = _load_candidate_chunks(db, document_ids)
    if not records:
        return []

    query_text = "Instruct: Given a document question, retrieve the passage that directly answers it.\nQuery: " + question
    vector = (await embed([query_text]))[0]
    where = {"document_id": {"$in": document_ids}} if document_ids else None
    result = collection.query(
        query_embeddings=[vector],
        n_results=min(settings.retrieval_candidate_pool, len(records)),
        where=where,
        include=["metadatas"],
    )
    scores: dict[str, float] = {}
    for rank, chunk_id in enumerate(result["ids"][0], start=1):
        if chunk_id in records:
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (60 + rank)

    query_terms = terms(question)
    lexical = sorted(
        ((lexical_score(question, item["content"], query_terms), chunk_id) for chunk_id, item in records.items()),
        reverse=True,
    )
    for rank, (score, chunk_id) in enumerate(lexical, start=1):
        if score <= 0:
            break
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (60 + rank) + min(score, 18) / 100

    return [{**records[chunk_id], "score": score} for chunk_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]]


def retrieve_overview(db: Session, document_ids: list[str] | None, limit: int | None = None) -> list[dict]:
    """For "summarize/brief me on this" style questions: no single passage IS
    the answer, so instead of ranking by similarity to the literal question
    text (which starves the model of coverage), sample chunks spread evenly
    across every page so the model actually sees the whole document. This is
    synchronous and skips embedding/vector search entirely, so it is also
    faster than the normal path.
    """
    settings = get_settings()
    limit = limit or settings.retrieval_overview_max_sources
    records = _load_candidate_chunks(db, document_ids)
    if not records:
        return []

    by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for item in records.values():
        by_page[(item["document_id"], item["page_number"])].append(item)
    for chunks in by_page.values():
        chunks.sort(key=lambda item: item["chunk_id"])

    pages = sorted(by_page.keys())
    selected: list[dict] = []
    round_index = 0
    while len(selected) < limit and round_index < max(len(chunks) for chunks in by_page.values()):
        for page in pages:
            if round_index < len(by_page[page]):
                selected.append(by_page[page][round_index])
                if len(selected) >= limit:
                    break
        round_index += 1
    return selected
