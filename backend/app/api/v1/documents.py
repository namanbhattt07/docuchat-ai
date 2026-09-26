import hashlib
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db import SessionLocal, get_db
from app.models import Chunk, Document
from app.services.documents import process_document
from app.services.ollama import OllamaUnavailable
from app.services.vector_store import get_collection

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_READ_CHUNK_BYTES = 1024 * 1024


@router.get("")
def list_documents(db: Session = Depends(get_db)):
    documents = db.scalars(select(Document).order_by(Document.created_at.desc())).all()
    return [
        {
            "id": item.id,
            "filename": item.filename,
            "page_count": item.page_count,
            "status": item.status,
            "status_detail": item.status_detail,
            "created_at": item.created_at,
        }
        for item in documents
    ]


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
        except OllamaUnavailable as exc:
            document.status = "failed"
            document.status_detail = str(exc)
            db.commit()
        except Exception:
            document.status = "failed"
            document.status_detail = "This PDF could not be processed."
            db.commit()
            file_path.unlink(missing_ok=True)
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
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(status_code=413, detail=f"PDF exceeds the {settings.max_upload_size_mb} MB upload limit.")
                hasher.update(chunk)
                buffer.write(chunk)
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


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")
    get_collection().delete(where={"document_id": document_id})
    path = Path(get_settings().upload_directory) / f"{document_id}.pdf"
    if path.exists():
        path.unlink()
    for chunk in db.scalars(select(Chunk).where(Chunk.document_id == document_id)).all():
        db.delete(chunk)
    db.delete(document)
    db.commit()
