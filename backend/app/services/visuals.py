import shutil
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import Document, DocumentImage
from app.services.document_model import DocumentProvider, ImageRegion
from app.services.pdf_provider import document_provider_for

# Group 7 image / figure handling (deliberately limited scope): detect which
# pages carry figures, keep their metadata, and render page/figure previews on
# demand. Nothing here interprets an image -- see services/visual_qa.py for
# the (honestly gated) question-answering side.

# Fixed render sizes the API may serve for a figure preview / page image. A
# closed set (plus clamping) keeps the on-disk cache bounded and means a
# request can never ask for an arbitrary-size bitmap.
FIGURE_PREVIEW_DPI = 110
FIGURE_PREVIEW_PAD_PT = 6.0


def image_row_id(document_id: str, region: ImageRegion) -> str:
    return f"{document_id}-p{region.page_number}-{region.kind}-{region.image_index}"


def image_rows(document_id: str, regions: list[ImageRegion]) -> list[DocumentImage]:
    return [
        DocumentImage(
            id=image_row_id(document_id, region),
            document_id=document_id,
            page_number=region.page_number,
            image_index=region.image_index,
            kind=region.kind,
            label=region.label,
            caption=region.caption,
            bbox=[round(value, 2) for value in region.bbox],
            width=region.width,
            height=region.height,
            xref=region.xref,
        )
        for region in regions
    ]


def serialize_image(row: DocumentImage) -> dict:
    return {
        "id": row.id,
        "document_id": row.document_id,
        "page_number": row.page_number,
        "kind": row.kind,
        "label": row.label,
        "caption": row.caption,
        "bbox": row.bbox,
        "width": row.width,
        "height": row.height,
        "xref": row.xref,
    }


def list_images(db: Session, document_id: str, page: int | None = None) -> list[DocumentImage]:
    statement = select(DocumentImage).where(DocumentImage.document_id == document_id)
    if page is not None:
        statement = statement.where(DocumentImage.page_number == page)
    return list(db.scalars(statement.order_by(DocumentImage.page_number, DocumentImage.kind, DocumentImage.image_index)).all())


def ensure_visuals_indexed(db: Session, document: Document, path: Path, provider: DocumentProvider | None = None) -> None:
    """Record figure metadata for a document ingested before it existed. New
    documents get it during ingestion, so this is a one-time, metadata-only
    pass per older document (no OCR, no embeddings, no chunk changes)."""
    if document.visuals_indexed:
        return
    provider = provider or document_provider_for(path)
    regions = provider.list_images(path)
    for existing in db.scalars(select(DocumentImage).where(DocumentImage.document_id == document.id)).all():
        db.delete(existing)
    db.add_all(image_rows(document.id, regions))
    document.visuals_indexed = 1
    db.commit()


# -- rendering ---------------------------------------------------------------


def stored_pdf_path(document_id: str) -> Path:
    return Path(get_settings().upload_directory) / f"{document_id}.pdf"


def _processed_root() -> Path:
    return Path(get_settings().processed_directory)


def _document_cache_dir(document_id: str) -> Path:
    return _processed_root() / document_id


def clamp_dpi(dpi: int | None) -> int:
    settings = get_settings()
    return max(50, min(dpi or settings.page_render_default_dpi, settings.page_render_max_dpi))


def render_page_image(document_id: str, pdf_path: Path, page_number: int, dpi: int | None = None, provider: DocumentProvider | None = None) -> Path:
    """PNG of one full page, rendered on first request and cached on disk.
    The path is derived only from the (already looked-up) document id, an
    integer page and a clamped integer dpi -- never from request text."""
    dpi = clamp_dpi(dpi)
    target = _document_cache_dir(document_id) / "pages" / f"{int(page_number)}@{dpi}.png"
    if not target.exists():
        provider = provider or document_provider_for(pdf_path)
        data = provider.render_page(pdf_path, int(page_number), dpi=dpi)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return target


def render_figure_preview(document_id: str, pdf_path: Path, image: DocumentImage, provider: DocumentProvider | None = None) -> Path:
    """PNG of just one figure's region (padded slightly). Falls back to the
    whole page when the figure has no locatable box -- e.g. a caption-derived
    entry whose artwork is vector graphics the app cannot outline."""
    if image.kind != "image" or not image.bbox:
        return render_page_image(document_id, pdf_path, image.page_number, None, provider)
    target = _document_cache_dir(document_id) / "images" / f"{image.id}@{FIGURE_PREVIEW_DPI}.png"
    if not target.exists():
        provider = provider or document_provider_for(pdf_path)
        x0, y0, x1, y1 = image.bbox
        clip = (x0 - FIGURE_PREVIEW_PAD_PT, y0 - FIGURE_PREVIEW_PAD_PT, x1 + FIGURE_PREVIEW_PAD_PT, y1 + FIGURE_PREVIEW_PAD_PT)
        data = provider.render_page(pdf_path, image.page_number, dpi=FIGURE_PREVIEW_DPI, clip=clip)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return target


def delete_render_cache(document_id: str) -> None:
    cache_dir = _document_cache_dir(document_id)
    root = _processed_root().resolve()
    # Belt and braces: only ever remove something that is really inside the
    # processed directory, whatever the id looked like.
    if cache_dir.exists() and root in cache_dir.resolve().parents:
        shutil.rmtree(cache_dir, ignore_errors=True)
