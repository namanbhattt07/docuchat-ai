import asyncio
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db import SessionLocal, get_db
from app.models import Chunk, Document, DocumentImage, PageText
from app.services import keyword_index, visuals
from app.services.document_model import DocumentUnreadable
from app.services.documents import discard_partial_index, ensure_document_index, process_document, suggestions_pending
from app.services.ollama import OllamaUnavailable
from app.services.pdf_provider import document_provider_for
from app.services.search import search_pages
from app.services.suggestions import generate_suggested_questions
from app.services.vector_store import get_collection

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger("docuchat.ingestion")

UPLOAD_READ_CHUNK_BYTES = 1024 * 1024
# Per the PDF spec the "%PDF-" header may follow up to 1024 bytes of junk.
PDF_MAGIC = b"%PDF-"
PDF_MAGIC_WINDOW = 1024


PAGE_STATUS_KEYS = ("text", "ocr", "empty", "failed", "unknown")
# Cached page/figure previews are immutable for a given (document, page, dpi).
_IMAGE_HEADERS = {"Cache-Control": "private, max-age=3600"}


def _page_status_counts(db: Session) -> dict[str, dict[str, int]]:
    """Per-document tally of explicit page statuses (one grouped query for the
    whole library). NULL status = a page indexed before per-page status
    existed, reported as "unknown" rather than guessed."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(PAGE_STATUS_KEYS, 0))
    for document_id, page_status, total in db.execute(
        select(PageText.document_id, PageText.status, func.count()).group_by(PageText.document_id, PageText.status)
    ).all():
        counts[document_id][page_status or "unknown"] += total
    return counts


@router.get("")
def list_documents(db: Session = Depends(get_db)):
    documents = db.scalars(select(Document).order_by(Document.created_at.desc())).all()
    summaries = _page_status_counts(db)
    return [
        {
            "id": item.id,
            "filename": item.filename,
            "page_count": item.page_count,
            "status": item.status,
            "status_detail": item.status_detail,
            "created_at": item.created_at,
            "page_summary": summaries.get(item.id) or dict.fromkeys(PAGE_STATUS_KEYS, 0),
        }
        for item in documents
    ]


def _failure_detail(exc: Exception) -> str:
    """What the person sees when ingestion fails -- a reason and a next step,
    never a stack trace."""
    if isinstance(exc, DocumentUnreadable):
        return str(exc)
    if isinstance(exc, OllamaUnavailable):
        return (
            "The local AI models aren't available, so this document couldn't be indexed. Start Ollama with the "
            f"embedding model, then delete this document and upload it again. ({str(exc)[:200]})"
        )
    return "This PDF could not be processed."


def _record_failure(db: Session, document_id: str, file_path: Path, exc: Exception) -> None:
    document = db.get(Document, document_id)
    if document is None:
        # Deleted while it was being processed: nothing to report, but make
        # sure none of its partial index outlives it.
        discard_partial_index(db, get_collection(), document_id)
        return
    document.status = "failed"
    document.status_detail = _failure_detail(exc)[:500]
    db.commit()
    if not isinstance(exc, OllamaUnavailable):
        file_path.unlink(missing_ok=True)


async def _run_ingestion(document_id: str, file_path: Path) -> None:
    """Runs after the upload response has already been sent, on its own DB
    session -- the request-scoped session from `get_db` is closed by then.
    """
    db = SessionLocal()
    try:
        document = db.get(Document, document_id)
        if document is None:
            return
        try:
            await process_document(db, document, file_path, get_collection())
        except Exception as exc:  # noqa: BLE001 -- whatever went wrong, the document must leave "processing"
            if not isinstance(exc, (DocumentUnreadable, OllamaUnavailable)):
                logger.exception("Ingestion failed for document %s", document_id)
            # Half-written rows (page text, figures, TOC) from the failed
            # attempt must not be committed along with the failed status, and
            # a session left needing a rollback would fail this very update.
            db.rollback()
            _record_failure(db, document_id, file_path, exc)
    finally:
        db.close()


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type != "application/pdf" and not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")

    settings = get_settings()
    upload_dir = Path(settings.upload_directory)
    upload_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    document = Document(filename=file.filename or "Untitled PDF")
    db.add(document)
    db.commit()
    db.refresh(document)

    destination = upload_dir / f"{document.id}.pdf"
    hasher = hashlib.sha256()
    size = 0
    try:
        with destination.open("wb") as buffer:
            while chunk := await file.read(UPLOAD_READ_CHUNK_BYTES):
                if size == 0 and PDF_MAGIC not in chunk[:PDF_MAGIC_WINDOW]:
                    # A renamed text/image/zip file: say so now, not after a
                    # "processing" spinner ends in a generic failure.
                    raise HTTPException(status_code=415, detail="This file isn't a PDF. Only PDF files are supported.")
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(status_code=413, detail=f"PDF exceeds the {settings.max_upload_size_mb} MB upload limit.")
                hasher.update(chunk)
                buffer.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="This file is empty.")
    except HTTPException:
        destination.unlink(missing_ok=True)
        db.delete(document)
        db.commit()
        raise

    content_hash = hasher.hexdigest()
    duplicate = db.scalar(select(Document).where(Document.content_hash == content_hash, Document.status != "failed"))
    if duplicate:
        destination.unlink(missing_ok=True)
        db.delete(document)
        db.commit()
        raise HTTPException(status_code=409, detail=f'This PDF is already in your library as "{duplicate.filename}".')

    document.content_hash = content_hash
    db.commit()

    background_tasks.add_task(_run_ingestion, document.id, destination)
    return {
        "id": document.id,
        "filename": document.filename,
        "page_count": document.page_count,
        "status": document.status,
        "status_detail": document.status_detail,
    }


@router.get("/{document_id}/file")
def get_document_file(document_id: str, download: bool = False, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")
    path = _require_stored_pdf(document_id)
    # Starlette's FileResponse honors Range requests on its own, which is
    # what lets the viewer start rendering a large PDF before the whole
    # file has downloaded instead of blocking on one giant fetch. The
    # frontend and backend run on different origins/ports, so the anchor
    # `download` attribute alone can't force a save dialog -- an explicit
    # `attachment` disposition (via ?download=1) is what actually does it.
    disposition = "attachment" if download else "inline"
    return FileResponse(path, media_type="application/pdf", filename=document.filename, content_disposition_type=disposition)


def _require_stored_pdf(document_id: str) -> Path:
    path = Path(get_settings().upload_directory) / f"{document_id}.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="The stored PDF for this document is missing.")
    return path


def _require_indexable(document: Document) -> None:
    """The TOC / page-search endpoints backfill their cache from the PDF when it
    is missing. For a document still being ingested that cache is *about to be
    written by the ingestion itself*: backfilling it here raced that write, and
    the loser was the ingestion (UNIQUE constraint on page_text -> the document
    ended "failed"). So there is nothing to serve yet, and a failed document
    has nothing usable to backfill from."""
    if document.status == "processing":
        raise HTTPException(status_code=409, detail="This document is still being processed. Try again once it shows Ready.")
    if document.status == "failed":
        raise HTTPException(status_code=409, detail="This document could not be processed, so it has no contents or text to search.")


@router.get("/{document_id}/toc")
def get_document_toc(document_id: str, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")
    _require_indexable(document)
    if document.toc_json is None:
        # Only documents ingested before Group 2 (or mid-ingestion) reach
        # here -- everything else is served straight from the cached column
        # below without touching the PDF at all.
        ensure_document_index(db, document, _require_stored_pdf(document_id))
    return {"document_id": document.id, "title": document.filename, "items": json.loads(document.toc_json)}


@router.get("/{document_id}/search")
def search_document(document_id: str, q: str = Query(..., min_length=1, max_length=300), db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")
    _require_indexable(document)
    if document.toc_json is None:
        ensure_document_index(db, document, _require_stored_pdf(document_id))  # backfills PageText for documents indexed before Group 2
    pages = db.scalars(select(PageText).where(PageText.document_id == document_id)).all()
    matches = search_pages(pages, q)
    return {
        "query": q,
        "total": len(matches),
        "matches": [{"index": index + 1, **match} for index, match in enumerate(matches)],
    }


@router.get("/{document_id}/suggested-questions")
async def get_suggested_questions(document_id: str, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")
    if document.suggested_questions_json is None:
        if suggestions_pending(document.id):
            # The document became ready before its questions were written (see
            # process_document); generating them here too would duplicate the
            # model call. The client asks again shortly.
            return {"document_id": document.id, "questions": [], "pending": True}
        # Only documents ingested before Group 3 (or whose generation was cut
        # short) reach here -- everything else is served straight from the
        # cache `process_document` populates. Mirrors the TOC backfill pattern
        # in get_document_toc.
        questions: list[str] = []
        if document.status == "ready":
            pdf_path = _require_stored_pdf(document_id)
            pages = (await asyncio.to_thread(document_provider_for(pdf_path).scan, pdf_path)).pages
            try:
                questions = await generate_suggested_questions([entry.text for entry in pages], get_settings().suggested_questions_count)
            except Exception:  # noqa: BLE001 -- never fail the request over a suggestion
                questions = []
        document.suggested_questions_json = json.dumps(questions)
        db.commit()
    return {"document_id": document.id, "questions": json.loads(document.suggested_questions_json)}


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")
    get_collection().delete(where={"document_id": document_id})
    keyword_index.delete_document(db, document_id)
    path = Path(get_settings().upload_directory) / f"{document_id}.pdf"
    if path.exists():
        path.unlink()
    for chunk in db.scalars(select(Chunk).where(Chunk.document_id == document_id)).all():
        db.delete(chunk)
    for page in db.scalars(select(PageText).where(PageText.document_id == document_id)).all():
        db.delete(page)
    for image in db.scalars(select(DocumentImage).where(DocumentImage.document_id == document_id)).all():
        db.delete(image)
    visuals.delete_render_cache(document_id)
    db.delete(document)
    db.commit()


# -- Group 7: OCR page status, page images, figures ---------------------------


def _require_document(db: Session, document_id: str) -> Document:
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")
    return document


def _image_urls(document_id: str, image: dict) -> dict:
    base = f"/api/v1/documents/{document_id}"
    return {
        **image,
        "preview_url": f"{base}/images/{image['id']}/preview",
        "page_image_url": f"{base}/pages/{image['page_number']}/image",
    }


@router.get("/{document_id}/pages")
def get_document_pages(document_id: str, db: Session = Depends(get_db)):
    """Explicit per-page processing status (text / ocr / empty / failed /
    unknown) plus whether each page carries figures -- what the UI uses to
    say a page was OCR'd, is genuinely blank, or failed OCR."""
    document = _require_document(db, document_id)
    pages = db.scalars(select(PageText).where(PageText.document_id == document_id).order_by(PageText.page_number)).all()
    figure_counts: dict[int, int] = defaultdict(int)
    for page_number, count in db.execute(
        select(DocumentImage.page_number, func.count()).where(DocumentImage.document_id == document_id).group_by(DocumentImage.page_number)
    ).all():
        figure_counts[page_number] = count
    return {
        "document_id": document.id,
        "page_count": document.page_count,
        "pages": [
            {
                "page_number": page.page_number,
                "status": page.status or "unknown",
                "source_type": page.source_type,
                "status_detail": page.status_detail,
                "ocr_confidence": page.ocr_confidence,
                "char_count": len(page.content or ""),
                "figure_count": figure_counts[page.page_number],
            }
            for page in pages
        ],
    }


@router.get("/{document_id}/pages/{page_number}/image")
def get_page_image(document_id: str, page_number: int, dpi: int | None = Query(default=None, ge=30, le=600), db: Session = Depends(get_db)):
    """PNG render of one page, produced on first request and cached under
    data/processed. `dpi` is clamped server-side."""
    document = _require_document(db, document_id)
    if not 1 <= page_number <= (document.page_count or 0):
        raise HTTPException(status_code=404, detail=f"Page {page_number} does not exist in this document.")
    path = _require_stored_pdf(document_id)
    rendered = visuals.render_page_image(document_id, path, page_number, dpi)
    return FileResponse(rendered, media_type="image/png", headers=_IMAGE_HEADERS)


@router.get("/{document_id}/images")
def get_document_images(document_id: str, page: int | None = Query(default=None, ge=1), db: Session = Depends(get_db)):
    """Figure / image metadata for the document (optionally one page). Records
    references and boxes only -- pixels are served by the preview endpoints."""
    document = _require_document(db, document_id)
    if not document.visuals_indexed and document.status == "ready":
        visuals.ensure_visuals_indexed(db, document, _require_stored_pdf(document_id))
    items = [_image_urls(document_id, visuals.serialize_image(row)) for row in visuals.list_images(db, document_id, page)]
    return {"document_id": document_id, "total": len(items), "images": items}


@router.get("/{document_id}/images/{image_id}/preview")
def get_image_preview(document_id: str, image_id: str, db: Session = Depends(get_db)):
    _require_document(db, document_id)
    image = db.scalar(select(DocumentImage).where(DocumentImage.id == image_id, DocumentImage.document_id == document_id))
    if not image:
        raise HTTPException(status_code=404, detail="Image not found.")
    rendered = visuals.render_figure_preview(document_id, _require_stored_pdf(document_id), image)
    return FileResponse(rendered, media_type="image/png", headers=_IMAGE_HEADERS)
