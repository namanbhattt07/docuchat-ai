import asyncio
import json
import logging
import re
from dataclasses import replace
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.models import Chunk, Document, DocumentImage, PageText
from app.services import keyword_index, visuals
from app.services.document_model import (
    PAGE_FAILED,
    PAGE_NEEDS_OCR,
    SOURCE_OCR,
    SOURCE_TEXT,
    DocumentChunk,
    DocumentPage,
    DocumentProvider,
    PageBlock,
    PageIndexEntry,
)
from app.services.ollama import embed
from app.services.pdf_provider import document_provider_for, extract_page_blocks
from app.services.suggestions import generate_suggested_questions

# Format-independent ingestion: chunking, embedding, indexing and the status
# of the document. Anything that has to know how a particular file format is
# parsed lives behind a DocumentProvider (services/pdf_provider.py for PDF);
# `PageBlock`, `PageIndexEntry` and `extract_page_blocks` are re-exported here
# because existing callers and tests import them from this module.
__all__ = ["PageBlock", "PageIndexEntry", "extract_page_blocks"]

logger = logging.getLogger("docuchat.ingestion")

INTERRUPTED_DETAIL = "Processing was interrupted because the app stopped. Delete this document and upload it again."

# Documents that are already `ready` (searchable, chattable) but whose
# suggested questions are still being written by the model -- see
# process_document. In-memory on purpose: it only has to cover the seconds
# between the two commits, and if the process dies in that window the
# questions simply stay unset and are backfilled on first request.
_suggestions_in_flight: set[str] = set()


def suggestions_pending(document_id: str) -> bool:
    return document_id in _suggestions_in_flight


async def embed_in_batches(texts: list[str], batch_size: int = 16, concurrency: int = 4) -> list[list[float]]:
    """Send chunk-embedding requests to Ollama in parallel (bounded by a
    semaphore) instead of one at a time -- a dense PDF can mean 80+ batches,
    and awaiting them sequentially was the dominant cost of ingestion.
    """
    batches = [texts[start:start + batch_size] for start in range(0, len(texts), batch_size)]
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run(batch: list[str]) -> list[list[float]]:
        async with semaphore:
            return await embed(batch)

    results = await asyncio.gather(*(run(batch) for batch in batches))
    vectors: list[list[float]] = []
    for result in results:
        vectors.extend(result)
    return vectors


def _chunk_spans(cleaned: str, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    """Character offsets for chunking an already whitespace-collapsed
    string, snapped to a sentence boundary when one is available. Factored
    out of `chunk_text` so page-aware chunking (`chunk_page`) can reuse the
    exact same boundaries while also tracking where each chunk came from.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    length = len(cleaned)
    while start < length:
        end = min(length, start + chunk_size)
        if end < length:
            boundary = cleaned.rfind(". ", start, end)
            if boundary > start + 300:
                end = boundary + 1
        spans.append((start, end))
        if end >= length:
            # Reached the end of the text -- without this, the next `start`
            # (end - overlap) can land *before* the current chunk started
            # whenever the final chunk is shorter than `overlap`, and the
            # loop then limps forward one character at a time, emitting
            # dozens of near-duplicate tail chunks instead of stopping.
            break
        start = max(end - overlap, start + 1)
    return spans


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    return [cleaned[start:end] for start, end in _chunk_spans(cleaned, chunk_size, overlap)]


def _serialize_blocks(blocks: list[PageBlock]) -> str:
    serialized = []
    for block in blocks:
        item = {"start": block.start, "end": block.end, "bbox": list(block.bbox)}
        if block.source_type != SOURCE_TEXT:
            item["source"] = block.source_type
        serialized.append(item)
    return json.dumps(serialized)


def _page_text_rows(document_id: str, pages: list[DocumentPage], *, classified: bool = True) -> list[PageText]:
    """`classified=False` (the lazy backfill of documents ingested before
    per-page status existed) leaves status NULL -- "never classified" --
    instead of stamping a status that was not actually determined."""
    return [
        PageText(
            id=f"{document_id}-{entry.page_number}",
            document_id=document_id,
            page_number=entry.page_number,
            content=entry.text,
            blocks_json=_serialize_blocks(entry.blocks),
            status=entry.status if classified else None,
            source_type=entry.source_type if classified else None,
            status_detail=entry.status_detail if classified else None,
            ocr_confidence=entry.ocr_confidence if classified else None,
        )
        for entry in pages
    ]


# A heuristic-heading candidate that is just a folio/page number or a lone
# roman numeral (e.g. a running page-number stamp PyMuPDF's font-size check
# mistook for a heading) -- filtered out before it can pollute the TOC.
_NOISE_HEADING_PATTERN = re.compile(r"^[\divxlcdm.\-\s]{1,6}$", re.IGNORECASE)


def build_toc(native_toc: list, heading_candidates: list[tuple[int, str, float]]) -> list[dict]:
    """Prefer the PDF's own bookmarks; only fall back to the heuristic
    heading list (built from the same is_heading flags used for chunk
    `section` labels -- see extract_page_blocks) when there is no native
    outline at all. Never mixes the two, so the UI never has to reason about
    a partially-native, partially-guessed tree.
    """
    if native_toc:
        entries = []
        for index, item in enumerate(native_toc):
            if len(item) < 3:
                continue
            level, title, page = item[0], item[1], item[2]
            title = (title or "").strip()
            if not title or not isinstance(page, int) or page < 1:
                continue
            entries.append({"id": f"native-{index}", "title": title, "page": page, "level": max(1, int(level)), "source": "native"})
        return entries

    # Running headers/footers repeat the same heading text on every
    # consecutive page -- collapse those runs to a single entry (their first
    # page) instead of listing the same "Chapter 3" dozens of times.
    deduped: list[tuple[int, str, float]] = []
    last_title: str | None = None
    for page, title, size in heading_candidates:
        title = title.strip()
        if not title or _NOISE_HEADING_PATTERN.match(title) or title == last_title:
            continue
        deduped.append((page, title, size))
        last_title = title
    if not deduped:
        return []

    # Bucket the distinct font sizes seen among headings into up to 3 levels
    # (largest = level 1) -- reuses the size the is_heading check already
    # computed rather than guessing depth from indentation or numbering.
    sizes = sorted({round(size, 1) for _, _, size in deduped}, reverse=True)

    def level_for(size: float) -> int:
        rounded = round(size, 1)
        return min(sizes.index(rounded) + 1, 3) if rounded in sizes else 3

    return [
        {"id": f"heuristic-{index}", "title": title, "page": page, "level": level_for(size), "source": "heuristic"}
        for index, (page, title, size) in enumerate(deduped)
    ]


def _union_bbox(boxes: list[tuple[float, float, float, float]]) -> list[float]:
    return [
        round(min(box[0] for box in boxes), 2),
        round(min(box[1] for box in boxes), 2),
        round(max(box[2] for box in boxes), 2),
        round(max(box[3] for box in boxes), 2),
    ]


def chunk_page(page_text: str, blocks: list[PageBlock], chunk_size: int, overlap: int) -> list[dict]:
    """Chunk one page's already-assembled text, attaching each chunk's
    offsets, the bounding box of every block it overlaps (unioned, so a
    renderer can draw a single highlight rectangle), and the nearest
    preceding heading as its "section" label.
    """
    if not page_text:
        return []
    pieces: list[dict] = []
    for start, end in _chunk_spans(page_text, chunk_size, overlap):
        overlapping = [block for block in blocks if block.start < end and block.end > start]
        bbox = _union_bbox([block.bbox for block in overlapping]) if overlapping else None
        # The most recent heading whose start falls within this chunk's
        # span (not strictly before the chunk's own start) -- a chunk that
        # is small enough to contain a subheading and the text under it
        # should be tagged with that subheading, not left blank just
        # because the heading isn't *before* the chunk begins.
        section: str | None = None
        for block in blocks:
            if block.start > end:
                break
            if block.is_heading:
                section = block.text
        pieces.append({
            "content": page_text[start:end],
            "start_offset": start,
            "end_offset": end,
            "bbox": bbox,
            "section": section,
            # Any OCR'd block in the chunk makes it an OCR chunk: its text
            # carries recognition uncertainty a pure-text chunk doesn't.
            "source_type": SOURCE_OCR if any(block.source_type == SOURCE_OCR for block in overlapping) else SOURCE_TEXT,
        })
    return pieces


def chunk_document_page(page: DocumentPage, chunk_size: int, overlap: int) -> list[DocumentChunk]:
    """chunk_page, expressed in the format-independent DocumentChunk type."""
    return [
        DocumentChunk(
            page_number=page.page_number,
            chunk_index=index,
            content=piece["content"],
            start_offset=piece["start_offset"],
            end_offset=piece["end_offset"],
            bbox=piece["bbox"],
            section=piece["section"],
            source_type=piece["source_type"],
        )
        for index, piece in enumerate(chunk_page(page.text, page.blocks, chunk_size, overlap))
    ]


def _document_exists(db: Session, document_id: str) -> bool:
    return db.execute(select(Document.id).where(Document.id == document_id)).first() is not None


def discard_partial_index(db: Session, collection, document_id: str) -> None:
    """Remove everything ingestion may have written for a document that will
    never be usable (deleted mid-processing, or interrupted). Best effort: each
    store is cleaned independently so one failure can't strand the others."""
    for cleanup in (
        lambda: collection.delete(where={"document_id": document_id}),
        lambda: keyword_index.delete_document(db, document_id),
        lambda: db.execute(delete(Chunk).where(Chunk.document_id == document_id)),
        lambda: db.execute(delete(PageText).where(PageText.document_id == document_id)),
        lambda: db.execute(delete(DocumentImage).where(DocumentImage.document_id == document_id)),
        lambda: visuals.delete_render_cache(document_id),
    ):
        try:
            cleanup()
        except Exception:  # noqa: BLE001
            logger.exception("Could not fully clean up the partial index of document %s", document_id)
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        db.rollback()


def recover_interrupted_documents(db: Session, get_collection) -> int:
    """Called once at startup. A document still `processing` when the app
    starts can only be one whose ingestion died with the previous process (the
    work runs in-process, in the background) -- left alone it would show a
    spinner forever. It is marked failed, with what to do about it, and any
    partial index is discarded. Returns how many were recovered."""
    stuck = db.scalars(select(Document).where(Document.status == "processing")).all()
    if not stuck:
        return 0
    for document in stuck:
        document.status = "failed"
        document.status_detail = INTERRUPTED_DETAIL
    db.commit()
    collection = get_collection()  # opened only when there is something to clean up
    for document in stuck:
        discard_partial_index(db, collection, document.id)
    return len(stuck)


def _report(db: Session, document: Document, detail: str) -> None:
    """Publish the current ingestion stage. `status_detail` is what the UI
    polls while a document is `processing` ("Extracting text", "OCR
    processing page 3 (2 of 4)", "Indexing", ...)."""
    document.status_detail = detail[:500]
    db.commit()


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _page_list(numbers: list[int], limit: int = 8) -> str:
    shown = ", ".join(str(number) for number in numbers[:limit])
    return shown + (f" and {len(numbers) - limit} more" if len(numbers) > limit else "")


async def _resolve_ocr_pages(db: Session, document: Document, file_path: Path, provider: DocumentProvider, pages: list[DocumentPage], settings: Settings) -> list[DocumentPage]:
    """OCR exactly the pages the scan flagged -- never the whole file -- one at
    a time on a worker thread (the database session stays on this thread), and
    report progress between pages. The returned list keeps every page at its
    real position, so page numbers (and therefore citations) are unchanged."""
    pending = [page for page in pages if page.status == PAGE_NEEDS_OCR]
    if not pending:
        return pages

    _report(db, document, f"OCR required on {_plural(len(pending), 'page')}")
    available, reason = provider.ocr_availability()
    resolved: dict[int, DocumentPage] = {}
    if not available:
        for page in pending:
            resolved[page.page_number] = replace(page, status=PAGE_FAILED, status_detail=(reason or "OCR is unavailable.")[:300])
    else:
        budget = min(len(pending), settings.ocr_max_pages)
        for position, page in enumerate(pending, start=1):
            if position > budget:
                resolved[page.page_number] = replace(
                    page, status=PAGE_FAILED,
                    status_detail=f"Skipped: OCR is limited to {settings.ocr_max_pages} scanned pages per document.",
                )
                continue
            _report(db, document, f"OCR processing page {page.page_number} ({position} of {budget})")
            resolved[page.page_number] = await asyncio.to_thread(provider.recognize_page, file_path, page)
    return [resolved.get(page.page_number, page) for page in pages]


def _final_status(pages: list[DocumentPage], has_chunks: bool) -> tuple[str, str | None]:
    """The document-level outcome, derived from the explicit per-page
    statuses. Failed OCR is reported as such -- never as "empty"."""
    failed = [page for page in pages if page.status == PAGE_FAILED]
    reason = next((page.status_detail for page in failed if page.status_detail), None)
    if failed:
        detail = f"Some pages could not be processed with OCR (page {_page_list([page.page_number for page in failed])})."
        if reason:
            detail += f" {reason}"
        return ("ready" if has_chunks else "failed"), detail[:500]
    if has_chunks:
        return "ready", None
    return "empty", "No readable text was found: every page is blank, or is an image OCR could not read any text from."


async def process_document(db: Session, document: Document, file_path: Path, collection, provider: DocumentProvider | None = None) -> None:
    settings = get_settings()
    provider = provider or document_provider_for(file_path)

    _report(db, document, "Extracting text")
    scan = await asyncio.to_thread(provider.scan, file_path)
    document.page_count = scan.page_count
    pages = await _resolve_ocr_pages(db, document, file_path, provider, scan.pages, settings)

    _report(db, document, "Indexing")
    records: list[Chunk] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int]] = []
    ids: list[str] = []
    heading_candidates: list[tuple[int, str, float]] = []
    for entry in pages:
        for piece in chunk_document_page(entry, settings.chunk_size, settings.chunk_overlap):
            chunk_id = piece.chunk_id(document.id)
            records.append(Chunk(
                id=chunk_id,
                document_id=document.id,
                page_number=piece.page_number,
                content=piece.content,
                section=piece.section,
                start_offset=piece.start_offset,
                end_offset=piece.end_offset,
                bbox=piece.bbox,
                source_type=piece.source_type,
            ))
            ids.append(chunk_id)
            documents.append(piece.content)
            metadatas.append({"document_id": document.id, "filename": document.filename, "page_number": entry.page_number})
        heading_candidates.extend((entry.page_number, block.text, block.font_size) for block in entry.blocks if block.is_heading)

    # Persisted alongside the chunks (not derived from them) so document
    # search and TOC can be served straight from the database -- see
    # services/search.py -- without ever re-opening this PDF.
    document.toc_json = json.dumps(build_toc(scan.native_toc, heading_candidates))
    db.add_all(_page_text_rows(document.id, pages))
    # Figure metadata (references only -- no pixel data); OCR'd pages may add
    # "Figure N" captions that only exist as recognized text.
    db.add_all(visuals.image_rows(document.id, [region for entry in pages for region in entry.images]))
    document.visuals_indexed = 1

    if records:
        vectors = await embed_in_batches(documents, concurrency=settings.ollama_embed_concurrency)
        if not _document_exists(db, document.id):
            # Deleted while it was being processed: writing its vectors now
            # would leave orphans nothing could ever clean up.
            db.rollback()
            return
        collection.add(ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas)
        db.add_all(records)
        keyword_index.index_chunks(db, document.id, ids, documents, [record.page_number for record in records])

    document.status, document.status_detail = _final_status(pages, has_chunks=bool(records))
    if not records:
        document.suggested_questions_json = json.dumps([])
        db.commit()
        return

    # The document is usable the moment its chunks are indexed, so the status is
    # committed *before* the suggested questions are generated. That LLM call
    # takes several seconds (measured ~6s for a 7-page PDF, ~13s for 300 pages),
    # and awaiting it first left every upload showing "Indexing" long after it
    # was searchable. The id is registered first so the suggestions endpoint can
    # tell "still being written" from "never generated".
    document_id = document.id  # read now: the commit expires `document`, and the user may delete it while suggestions are written
    _suggestions_in_flight.add(document_id)
    try:
        db.commit()
        await _store_suggested_questions(db, document, pages, settings.suggested_questions_count)
    finally:
        _suggestions_in_flight.discard(document_id)


async def _store_suggested_questions(db: Session, document: Document, pages: list[DocumentPage], count: int) -> None:
    """Best-effort and isolated from the status logic on purpose: a
    suggestion-generation failure (Ollama down, bad output) must never turn a
    successfully-ingested document into a failed one (see SUGGESTED QUESTIONS /
    Generation strategy)."""
    try:
        questions = await generate_suggested_questions([entry.text for entry in pages], count)
    except Exception:  # noqa: BLE001
        questions = []
    try:
        document.suggested_questions_json = json.dumps(questions)
        db.commit()
    except Exception:  # noqa: BLE001 -- the document is already ready; leave the questions unset so they are backfilled lazily
        db.rollback()


def ensure_document_index(db: Session, document: Document, file_path: Path, provider: DocumentProvider | None = None) -> list[dict]:
    """Return the cached TOC, backfilling it (and the page-search index) for
    documents ingested before this column/table existed. A no-op PDF re-parse
    for every document ingested going forward, since process_document
    already populates toc_json and PageText in the same pass.
    """
    if document.toc_json is not None:
        return json.loads(document.toc_json)

    provider = provider or document_provider_for(file_path)
    scan = provider.scan(file_path)  # native extraction only -- a backfill never triggers OCR
    heading_candidates = [
        (entry.page_number, block.text, block.font_size)
        for entry in scan.pages
        for block in entry.blocks
        if block.is_heading
    ]
    toc_entries = build_toc(scan.native_toc, heading_candidates)
    document.toc_json = json.dumps(toc_entries)
    db.query(PageText).filter(PageText.document_id == document.id).delete()
    db.add_all(_page_text_rows(document.id, scan.pages, classified=False))
    db.commit()
    return toc_entries
