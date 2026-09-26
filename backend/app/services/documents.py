import asyncio
from pathlib import Path
import re

import fitz
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import Chunk, Document
from app.services.ollama import embed


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


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        if end < len(cleaned):
            boundary = cleaned.rfind(". ", start, end)
            if boundary > start + 300:
                end = boundary + 1
        chunks.append(cleaned[start:end])
        if end >= len(cleaned):
            # Reached the end of the text -- without this, the next `start`
            # (end - overlap) can land *before* the current chunk started
            # whenever the final chunk is shorter than `overlap`, and the
            # loop then limps forward one character at a time, emitting
            # dozens of near-duplicate tail chunks instead of stopping.
            break
        start = max(end - overlap, start + 1)
    return chunks


async def process_document(db: Session, document: Document, file_path: Path, collection) -> None:
    settings = get_settings()
    pdf = fitz.open(file_path)
    records: list[Chunk] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int]] = []
    ids: list[str] = []
    for page_index, page in enumerate(pdf):
        page_text = page.get_text("text")
        for chunk_index, content in enumerate(chunk_text(page_text, settings.chunk_size, settings.chunk_overlap)):
            chunk_id = f"{document.id}-{page_index + 1}-{chunk_index}"
            records.append(Chunk(id=chunk_id, document_id=document.id, page_number=page_index + 1, content=content))
            ids.append(chunk_id)
            documents.append(content)
            metadatas.append({"document_id": document.id, "filename": document.filename, "page_number": page_index + 1})
    document.page_count = len(pdf)
    pdf.close()

    if records:
        vectors = await embed_in_batches(documents, concurrency=settings.ollama_embed_concurrency)
        collection.add(ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas)
        db.add_all(records)
        document.status = "ready"
        document.status_detail = None
    else:
        # No extractable text (e.g. a scanned/image-only PDF). Marking this
        # "ready" was misleading -- the document looked usable but every
        # question against it would 400 with an unrelated-looking error.
        document.status = "empty"
        document.status_detail = "No selectable text was found in this PDF (it may be a scanned image). OCR is not supported yet."
    db.commit()
