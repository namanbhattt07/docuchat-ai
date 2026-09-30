import logging
import re
from collections import Counter, defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import Chunk, Document
from app.services import evidence, keyword_index
from app.services.ollama import OllamaUnavailable, embed, generate
from app.services.retrieval_types import RetrievalMode, RetrievalPlan

logger = logging.getLogger("docuchat.retrieval")

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
            "section": chunk.section,
            "bbox": chunk.bbox,
            "start_offset": chunk.start_offset,
            "end_offset": chunk.end_offset,
            "source_type": chunk.source_type,
        }
        for chunk, filename in rows
    }


async def _vector_candidates(collection, question: str, document_ids: list[str] | None, pool_size: int, records: dict[str, dict]) -> list[str]:
    """Chunk ids ordered best-to-worst by cosine similarity. Ranks (not raw
    scores) are what feed the fusion step -- see `fuse_scores`.
    """
    query_text = "Instruct: Given a document question, retrieve the passage that directly answers it.\nQuery: " + question
    vector = (await embed([query_text]))[0]
    where = {"document_id": {"$in": document_ids}} if document_ids else None
    result = collection.query(
        query_embeddings=[vector],
        n_results=min(pool_size, len(records)),
        where=where,
        include=["metadatas"],
    )
    return [chunk_id for chunk_id in result["ids"][0] if chunk_id in records]


def _keyword_candidates(db: Session, records: dict[str, dict], query_terms: list[str], document_ids: list[str] | None, pool_size: int, question: str) -> list[str]:
    """Chunk ids ordered best-to-worst by lexical relevance. Backed by
    SQLite FTS5/bm25 (see services/keyword_index.py) when available; on a
    non-sqlite `database_url` this degrades to the pure-Python overlap+phrase
    scorer below (`lexical_score`) rather than losing the keyword signal
    entirely.
    """
    if keyword_index.is_available(db):
        hits = keyword_index.search(db, query_terms, document_ids, pool_size)
        return [chunk_id for chunk_id, _ in hits if chunk_id in records]
    scored = sorted(
        ((lexical_score(question, item["content"], query_terms), chunk_id) for chunk_id, item in records.items()),
        reverse=True,
    )
    return [chunk_id for score, chunk_id in scored if score > 0][:pool_size]


def fuse_scores(vector_ids: list[str], keyword_ids: list[str], vector_weight: float, keyword_weight: float, rrf_k: int) -> dict[str, float]:
    """Reciprocal rank fusion. Each list contributes weight / (k + rank),
    regardless of that source's native score scale -- cosine similarity and
    bm25 aren't comparable numbers, so fusing by *rank* rather than
    normalizing two incompatible scales is what keeps this explainable and
    tunable via just the two weights.
    """
    fused: dict[str, float] = defaultdict(float)
    for rank, chunk_id in enumerate(vector_ids, start=1):
        fused[chunk_id] += vector_weight / (rrf_k + rank)
    for rank, chunk_id in enumerate(keyword_ids, start=1):
        fused[chunk_id] += keyword_weight / (rrf_k + rank)
    return dict(fused)


def _offset_overlap(a: dict, b: dict) -> float:
    a_start, a_end = a.get("start_offset") or 0, a.get("end_offset") or 0
    b_start, b_end = b.get("start_offset") or 0, b.get("end_offset") or 0
    if a_end <= a_start or b_end <= b_start:
        return 1.0 if a["content"] == b["content"] else 0.0
    overlap = max(0, min(a_end, b_end) - max(a_start, b_start))
    shorter = min(a_end - a_start, b_end - b_start)
    return overlap / shorter if shorter else 0.0


def deduplicate(sources: list[dict], overlap_ratio: float) -> list[dict]:
    """Collapse near-duplicate chunks -- the same passage reached via both
    the vector and keyword paths with slightly different chunk boundaries --
    to the single higher-scored copy. Scoped to the same document+page so
    two genuinely different chunks that merely share a page are never
    dropped (see DEDUPLICATION).
    """
    kept: list[dict] = []
    for source in sorted(sources, key=lambda item: item["score"], reverse=True):
        duplicate_of = next(
            (
                existing for existing in kept
                if existing["document_id"] == source["document_id"]
                and existing["page_number"] == source["page_number"]
                and _offset_overlap(existing, source) >= overlap_ratio
            ),
            None,
        )
        if duplicate_of is None:
            kept.append(source)
    return kept


def select_diverse(sources: list[dict], limit: int, group_key=None) -> list[dict]:
    """Round-robin the highest-scored remaining candidate from each group
    (by default, document+page) instead of taking the top N by score alone,
    so five near-tied chunks from one page can't crowd out every other page
    (see DIVERSITY-AWARE SELECTION). Falls back to plain top-N once there
    are no more distinct groups left to draw from.
    """
    if group_key is None:
        group_key = lambda item: (item["document_id"], item["page_number"])  # noqa: E731
    if len(sources) <= limit:
        return sorted(sources, key=lambda item: item["score"], reverse=True)

    grouped: dict = defaultdict(list)
    for source in sources:
        grouped[group_key(source)].append(source)
    for group in grouped.values():
        group.sort(key=lambda item: item["score"], reverse=True)

    selected: list[dict] = []
    round_index = 0
    keys = list(grouped.keys())
    while len(selected) < limit and any(round_index < len(grouped[key]) for key in keys):
        ordering = sorted(
            (key for key in keys if round_index < len(grouped[key])),
            key=lambda key: grouped[key][round_index]["score"],
            reverse=True,
        )
        for key in ordering:
            selected.append(grouped[key][round_index])
            if len(selected) >= limit:
                break
        round_index += 1
    return selected


def _log_debug(plan: RetrievalPlan, vector_ids: list[str], keyword_ids: list[str], sources: list[dict]) -> None:
    """Internal observability only (see RETRIEVAL DEBUGGING). Gated behind
    DEBUG level and never includes chunk content -- ids, documents and page
    numbers are enough to diagnose a bad retrieval without logging document
    text.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug(
        "retrieval mode=%s query=%r vector_candidates=%s keyword_candidates=%s final_chunks=%s documents=%s pages=%s",
        plan.mode.value, plan.query[:120], vector_ids[:10], keyword_ids[:10],
        [source["chunk_id"] for source in sources],
        sorted({source["document_id"] for source in sources}),
        sorted({source["page_number"] for source in sources if source["page_number"] is not None}),
    )


async def _hybrid_retrieve(db: Session, collection, plan: RetrievalPlan, group_key=None) -> list[dict]:
    """The shared vector+keyword->fusion->dedup->diversity pipeline behind
    FACT and MULTI_DOCUMENT (see RETRIEVAL ARCHITECTURE UPGRADE). Also
    lazily backfills the keyword index for any document ingested before
    Group 3, the same "index on first use" pattern as TOC/PageText.
    """
    settings = get_settings()
    records = _load_candidate_chunks(db, plan.document_ids)
    if not records:
        return []
    for document_id in {item["document_id"] for item in records.values()}:
        keyword_index.ensure_indexed(db, document_id)

    pool = settings.retrieval_candidate_pool
    query_terms = terms(plan.query)
    vector_ids = await _vector_candidates(collection, plan.query, plan.document_ids, pool, records)
    keyword_ids = _keyword_candidates(db, records, query_terms, plan.document_ids, pool, plan.query)
    fused = fuse_scores(vector_ids, keyword_ids, settings.retrieval_vector_weight, settings.retrieval_keyword_weight, settings.retrieval_rrf_k)

    sources = [{**records[chunk_id], "score": score} for chunk_id, score in fused.items()]
    sources = deduplicate(sources, settings.retrieval_dedup_overlap_ratio)
    if plan.diversity:
        sources = select_diverse(sources, plan.top_k, group_key=group_key)
    else:
        sources = sorted(sources, key=lambda item: item["score"], reverse=True)[: plan.top_k]

    _log_debug(plan, vector_ids, keyword_ids, sources)
    return sources


async def retrieve_fact(db: Session, collection, plan: RetrievalPlan) -> list[dict]:
    """Precise, small-context retrieval for direct factual questions (see
    FACT retrieval mode)."""
    return await _hybrid_retrieve(db, collection, plan)


async def retrieve_multi_document(db: Session, collection, plan: RetrievalPlan) -> list[dict]:
    """Same hybrid core as FACT, but diversity groups by document instead of
    by page -- otherwise the single best-matching document could crowd out
    every other selected document (see MULTI_DOCUMENT retrieval mode)."""
    return await _hybrid_retrieve(db, collection, plan, group_key=lambda item: item["document_id"])


def retrieve_page(db: Session, plan: RetrievalPlan) -> list[dict]:
    """For "what is discussed on page N" -- purely structural, no
    embedding/vector search needed since the target page is already known.
    Reading-order chunks from that page, padded with the immediate
    neighbours only if the page itself has too little content to be a
    useful context (see PAGE retrieval mode)."""
    records = _load_candidate_chunks(db, plan.document_ids)
    if not records:
        return []
    on_page = [item for item in records.values() if item["page_number"] == plan.page]
    if not on_page:
        # Nothing on that page (it doesn't exist, or holds no text). Its
        # neighbours are *not* that page: returning them would let the model
        # describe page 7's content as if it were page 8.
        return []
    if len(on_page) < 2:
        on_page += [item for item in records.values() if item["page_number"] is not None and plan.page is not None and abs(item["page_number"] - plan.page) == 1]
    on_page.sort(key=lambda item: (item["page_number"] != plan.page, item.get("start_offset", 0)))
    return [{**item, "score": 1.0 if item["page_number"] == plan.page else 0.5} for item in on_page[: plan.top_k]]


async def retrieve_section(db: Session, collection, plan: RetrievalPlan) -> list[dict]:
    """Chunks tagged with the requested heading (see extract_page_blocks'
    `section` labelling), in reading order. Falls back to a scoped hybrid
    search using the section phrase as the query when no chunk carries that
    heading verbatim -- an uncertain section match must still answer
    something, never an empty context (see SECTION retrieval mode)."""
    records = _load_candidate_chunks(db, plan.document_ids)
    if not records:
        return []
    section_needle = (plan.section or "").lower()
    matched = [item for item in records.values() if item.get("section") and section_needle in item["section"].lower()]
    if matched:
        matched.sort(key=lambda item: (item["document_id"], item["page_number"], item.get("start_offset", 0)))
        return [{**item, "score": 1.0} for item in matched[: plan.top_k]]

    fallback_plan = RetrievalPlan(
        mode=RetrievalMode.FACT,
        query=f"{plan.section} {plan.query}".strip(),
        document_ids=plan.document_ids,
        top_k=plan.top_k,
        diversity=False,
    )
    return await retrieve_fact(db, collection, fallback_plan)


async def retrieve_selection(db: Session, collection, plan: RetrievalPlan) -> list[dict]:
    """The user's selected text is the primary source (always ranked and
    cited first), plus a few hybrid-retrieved supporting chunks for context.
    No UI surfaces text selection yet (that lands in a later group) -- this
    only ever runs when a caller explicitly passes `selected_text`, so it is
    inert today and exists purely so that workflow has somewhere to plug in
    later (see SELECTION retrieval mode)."""
    selected_text = plan.selected_text or ""
    document_id = (plan.document_ids or [None])[0]
    filename = None
    if document_id:
        records = _load_candidate_chunks(db, [document_id])
        if records:
            filename = next(iter(records.values()))["filename"]

    selection_source = {
        "chunk_id": "selection-0",
        "content": selected_text,
        "document_id": document_id,
        "filename": filename,
        "page_number": plan.page,
        "section": None,
        "bbox": None,
        "start_offset": 0,
        "end_offset": len(selected_text),
        "source_type": "selection",
        "score": float("inf"),
    }

    remaining = max(plan.top_k - 1, 0)
    supporting: list[dict] = []
    if remaining and selected_text:
        supporting_plan = RetrievalPlan(
            mode=RetrievalMode.FACT, query=selected_text, document_ids=plan.document_ids,
            top_k=remaining, diversity=False,
        )
        supporting = [
            item for item in await retrieve_fact(db, collection, supporting_plan)
            if item["content"] != selected_text
        ]
    return [selection_source] + supporting


def retrieve_location(db: Session, plan: RetrievalPlan) -> list[dict]:
    """Literal keyword lookup, not semantic search -- a LOCATION query asks
    where a term is actually mentioned, not what's conceptually similar to
    it, so this deliberately skips vector search and ranks purely by
    keyword/FTS5 relevance (see LOCATION QUERY FLOW). A topic that never
    appears verbatim anywhere correctly returns no results rather than
    confidently pointing at a page that merely seems related.
    """
    records = _load_candidate_chunks(db, plan.document_ids)
    if not records:
        return []
    for document_id in {item["document_id"] for item in records.values()}:
        keyword_index.ensure_indexed(db, document_id)

    settings = get_settings()
    query_terms = terms(plan.query)
    keyword_ids = _keyword_candidates(db, records, query_terms, plan.document_ids, settings.retrieval_candidate_pool, plan.query)
    if not keyword_ids:
        return []

    # The keyword query is OR-ed (so a phrase match ranks first), which means a
    # chunk mentioning only *one* word of "quantum sensors" comes back too. A
    # location answer names pages, so a page has to be about the whole topic:
    # keep only chunks that contain enough of its terms.
    keyword_ids = [chunk_id for chunk_id in keyword_ids if evidence.covers_terms(query_terms, records[chunk_id]["content"])]
    if not keyword_ids:
        return []

    total = len(keyword_ids)
    sources = [{**records[chunk_id], "score": float(total - rank)} for rank, chunk_id in enumerate(keyword_ids)]
    sources = deduplicate(sources, settings.retrieval_dedup_overlap_ratio)
    return select_diverse(sources, plan.top_k)


async def retrieve_extraction(db: Session, collection, plan: RetrievalPlan) -> list[dict]:
    """Same hybrid core as FACT, just with far more sources -- the entities a
    batch-extraction request wants (e.g. every protocol) can be scattered
    anywhere across the document, so a normal FACT-sized top_k would starve
    the model of coverage the same way it would for OVERVIEW (see BATCH /
    STRUCTURED EXTRACTION).
    """
    return await _hybrid_retrieve(db, collection, plan)


async def retrieve_with_plan(db: Session, collection, plan: RetrievalPlan) -> list[dict]:
    """Single dispatch point from a RetrievalPlan to the mode-specific
    retrieval function -- see RetrievalMode for the full list. CALCULATION
    has no dedicated function: it needs the same passage-finding behavior as
    FACT (the numbers just live in whichever passages are relevant), so it
    falls through to the same default branch as FACT itself.
    """
    if plan.mode == RetrievalMode.OVERVIEW:
        return retrieve_overview(db, plan.document_ids, limit=plan.top_k)
    if plan.mode == RetrievalMode.PAGE:
        return retrieve_page(db, plan)
    if plan.mode == RetrievalMode.SECTION:
        return await retrieve_section(db, collection, plan)
    if plan.mode == RetrievalMode.SELECTION:
        return await retrieve_selection(db, collection, plan)
    if plan.mode == RetrievalMode.MULTI_DOCUMENT:
        return await retrieve_multi_document(db, collection, plan)
    if plan.mode == RetrievalMode.LOCATION:
        return retrieve_location(db, plan)
    if plan.mode == RetrievalMode.EXTRACTION:
        return await retrieve_extraction(db, collection, plan)
    return await retrieve_fact(db, collection, plan)


def list_known_sections(db: Session, document_ids: list[str] | None, limit: int = 200) -> list[str]:
    """Distinct section headings actually present in the chunk data, used by
    the query router to resolve a fuzzy section reference (e.g. "the
    Sensors section") against real document headings instead of guessing
    blind."""
    statement = select(Chunk.section).where(Chunk.section.is_not(None)).distinct()
    if document_ids:
        statement = statement.where(Chunk.document_id.in_(document_ids))
    return [row[0] for row in db.execute(statement.limit(limit)).all() if row[0]]


async def retrieve(db: Session, collection, question: str, document_ids: list[str] | None, limit: int | None = None) -> list[dict]:
    """Backward-compatible entry point kept for existing callers -- a plain
    FACT-mode hybrid retrieval (vector+keyword fusion, deduplicated, no
    diversity re-ranking) built on the same `_hybrid_retrieve` core the
    query-router-driven path uses. New code should prefer
    `retrieve_with_plan`.
    """
    settings = get_settings()
    plan = RetrievalPlan(mode=RetrievalMode.FACT, query=question, document_ids=document_ids, top_k=limit or settings.retrieval_top_k, diversity=False)
    return await _hybrid_retrieve(db, collection, plan)


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
