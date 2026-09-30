import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import Chunk, Document, DocumentImage
from app.services import visuals
from app.services.providers import ProviderUnavailable, VisionProvider

# Group 7 visual Q&A -- deliberately honest. The default text model (qwen3:8b)
# cannot see images, so a question about a chart/diagram is only "answered"
# from pixels when a real local vision model is configured AND installed
# (providers/registry.py::vision_capability). Otherwise the response says so
# plainly and still gives the user what the app genuinely has: the page
# citation, a preview reference to the figure, its caption, and the text that
# was extracted from that page. It never says "the chart shows X" from text
# alone.

UNSUPPORTED_SUFFIX = "but visual question answering is not supported by the current model configuration."

# A visual question needs a visual noun AND an intent to look at / interpret
# it. Requiring both keeps ordinary text questions ("what is a graph
# database?") on the normal grounded-answer path; the caller additionally
# requires that a figure really exists on a relevant page before diverting.
_VISUAL_NOUN = re.compile(
    r"\b(charts?|graphs?|plots?|diagrams?|figure(?!\s+out)s?|fig\.?|images?|pictures?|photos?|photographs?|illustrations?|"
    r"infographics?|schematics?|flow ?charts?|screenshots?|visuals?|histograms?|bar chart|pie chart|curves?)\b",
    re.IGNORECASE,
)
_VISUAL_INTENT = re.compile(
    r"\b(show(?:s|n|ing)?|depict(?:s|ed|ing)?|illustrat(?:e|es|ed|ing)|display(?:s|ed|ing)?|look(?:s)? like|"
    r"represent(?:s|ed|ing)?|trend|axis|axes|legend|bars?|slices?|colou?rs?|describe|interpret|explain|"
    r"summari[sz]e|what(?:'s| is| are) (?:in|on|inside)|plotted|indicate(?:s|d)?|highest|lowest|peak|read|see|visible)\b",
    re.IGNORECASE,
)
_TEXT_QUESTION = re.compile(r"\b(captions?|list of figures|table of figures|figure numbers?|how many figures)\b", re.IGNORECASE)
_FIGURE_LABEL = re.compile(r"\b(figure|fig\.?|chart|diagram|graph|plot)\s+(\d+(?:[.\-]\d+)*[a-z]?)\b", re.IGNORECASE)
_PAGE_REFERENCE = re.compile(r"\bpage\s+(\d{1,5})\b", re.IGNORECASE)
_CANONICAL_KIND = {"fig": "Figure", "figure": "Figure", "chart": "Chart", "diagram": "Diagram", "graph": "Graph", "plot": "Plot"}

MAX_TARGET_PAGES = 3
EXCERPT_CHARS = 420


def looks_visual(question: str) -> bool:
    return bool(_VISUAL_NOUN.search(question) and _VISUAL_INTENT.search(question) and not _TEXT_QUESTION.search(question))


def explicit_label(question: str) -> str | None:
    match = _FIGURE_LABEL.search(question)
    if not match:
        return None
    return f"{_CANONICAL_KIND[match.group(1).lower().rstrip('.')]} {match.group(2)}"


def explicit_page(question: str) -> int | None:
    match = _PAGE_REFERENCE.search(question)
    return int(match.group(1)) if match else None


@dataclass
class VisualTarget:
    document_id: str
    filename: str
    page_number: int
    figures: list[DocumentImage] = field(default_factory=list)


@dataclass
class VisualAnswer:
    """Everything /chat needs to persist and return a visual response."""

    answer: str
    # Citation-shaped source dicts, in [n] order (see chat.build_citations).
    evidence: list[dict]
    metadata: dict


def resolve_targets(db: Session, document_ids: list[str] | None, question: str, sources: list[dict]) -> list[VisualTarget]:
    """Which pages the question is about *and* carry a recorded visual element.
    An explicitly named figure/page wins; otherwise the pages the normal
    retrieval already found for this question are used (no second search).
    Returns [] when no relevant page has a figure -- the caller then answers
    normally rather than pretending there is something to look at."""
    statement = select(DocumentImage, Document.filename).join(Document, DocumentImage.document_id == Document.id)
    if document_ids:
        statement = statement.where(DocumentImage.document_id.in_(document_ids))
    rows = db.execute(statement.order_by(DocumentImage.page_number, DocumentImage.kind, DocumentImage.image_index)).all()
    by_page: dict[tuple[str, int], list[tuple[DocumentImage, str]]] = {}
    for image, filename in rows:
        by_page.setdefault((image.document_id, image.page_number), []).append((image, filename))

    label, page = explicit_label(question), explicit_page(question)
    if label:
        wanted = label.lower()
        keys = [key for key, items in by_page.items() if any((image.label or "").lower() == wanted for image, _ in items)]
    elif page:
        keys = [key for key in by_page if key[1] == page]
    else:
        keys = []
        for source in sources:
            key = (source["document_id"], source["page_number"])
            if key in by_page and key not in keys:
                keys.append(key)
    return [
        VisualTarget(document_id=key[0], filename=by_page[key][0][1], page_number=key[1], figures=[image for image, _ in by_page[key]])
        for key in keys[:MAX_TARGET_PAGES]
    ]


def _page_text_evidence(db: Session, target: VisualTarget, sources: list[dict]) -> dict | None:
    """One extracted-text passage from the target page: the retrieved chunk if
    retrieval already found one there, else the page's first chunk."""
    for source in sources:
        if source["document_id"] == target.document_id and source["page_number"] == target.page_number:
            return source
    chunk = db.scalar(
        select(Chunk).where(Chunk.document_id == target.document_id, Chunk.page_number == target.page_number).order_by(Chunk.start_offset)
    )
    if chunk is None:
        return None
    return {
        "chunk_id": chunk.id, "document_id": chunk.document_id, "filename": target.filename, "page_number": chunk.page_number,
        "content": chunk.content, "section": chunk.section, "bbox": chunk.bbox, "source_type": chunk.source_type,
    }


def _figure_evidence(target: VisualTarget) -> tuple[dict, str]:
    """The figure itself as a citation (its box when known, so "Open in PDF"
    lands on the figure; else the page) plus a one-line description for the
    answer. Only what was actually detected is described: a caption's own
    words, or the fact that an embedded image exists -- never its content."""
    raster = next((figure for figure in target.figures if figure.kind == "image"), None)
    captioned = next((figure for figure in target.figures if figure.caption), None)
    anchor = raster or captioned or target.figures[0]
    if captioned:
        content = captioned.caption
        description = f"**{captioned.label}** — {_excerpt(captioned.caption)}" if captioned.label else _excerpt(captioned.caption)
    else:
        size = f" ({raster.width}×{raster.height} px)" if raster and raster.width and raster.height else ""
        content = f"Embedded image on page {target.page_number}{size}"
        description = f"Embedded image{size}"
    evidence = {
        "chunk_id": anchor.id, "document_id": target.document_id, "filename": target.filename, "page_number": target.page_number,
        "content": content, "section": None, "bbox": anchor.bbox, "source_type": "figure",
    }
    return evidence, description


def _page_phrase(targets: list[VisualTarget]) -> str:
    multiple_documents = len({target.document_id for target in targets}) > 1
    parts = [f"page {target.page_number}" + (f" of {target.filename}" if multiple_documents else "") for target in targets]
    noun = "a visual element" if len(targets) == 1 and len(targets[0].figures) == 1 else "visual elements"
    listed = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f" and {parts[-1]}"
    return f"{noun} on {listed}"


def _excerpt(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned if len(cleaned) <= EXCERPT_CHARS else cleaned[: EXCERPT_CHARS - 1].rstrip() + "…"


def _metadata(targets: list[VisualTarget], *, supported: bool, model: str | None, reason: str) -> dict:
    return {
        "supported": supported,
        "model": model,
        "reason": reason,
        "targets": [
            {
                "document_id": target.document_id,
                "filename": target.filename,
                "page_number": target.page_number,
                "page_image_url": f"/api/v1/documents/{target.document_id}/pages/{target.page_number}/image",
                "figures": [
                    {
                        "id": figure.id,
                        "kind": figure.kind,
                        "label": figure.label,
                        "caption": figure.caption,
                        "bbox": figure.bbox,
                        "preview_url": f"/api/v1/documents/{target.document_id}/images/{figure.id}/preview",
                    }
                    for figure in target.figures
                ],
            }
            for target in targets
        ],
    }


def build_answer(
    db: Session,
    targets: list[VisualTarget],
    sources: list[dict],
    *,
    interpretation: str | None = None,
    model: str | None = None,
    reason: str,
    failure: str | None = None,
) -> VisualAnswer:
    """The visual response with its evidence. `interpretation` is text a vision
    model produced from the page image (and only then); without it the
    response is the honest "not supported" notice plus real extracted text."""
    evidence: list[dict] = []
    lines: list[str] = []

    if interpretation is not None:
        lines.append(f"**Vision model interpretation ({model})** — generated from the rendered page image; it may be inaccurate.\n\n{interpretation.strip()}")
    else:
        lines.append(f"This document contains {_page_phrase(targets)}, {UNSUPPORTED_SUFFIX}")
        if failure:
            lines.append(f"_{failure}_")

    lines.append("**Where to look**")
    for target in targets:
        figure_evidence, description = _figure_evidence(target)
        evidence.append(figure_evidence)
        lines.append(f"- {description} — page {target.page_number} [{len(evidence)}]")

    extracted: list[tuple[VisualTarget, dict]] = []
    for target in targets:
        passage = _page_text_evidence(db, target, sources)
        if passage:
            extracted.append((target, passage))
    if extracted:
        lines.append("**Text extracted from the same page** (this is the page's text, not an interpretation of the image)")
        for target, passage in extracted:
            evidence.append(passage)
            lines.append(f"> {_excerpt(passage['content'])} [{len(evidence)}] _(page {target.page_number})_")
    else:
        lines.append("_No text could be extracted from this page._")

    return VisualAnswer(answer="\n\n".join(lines), evidence=evidence, metadata=_metadata(targets, supported=interpretation is not None, model=model, reason=reason))


async def answer_visual_question(
    db: Session,
    question: str,
    targets: list[VisualTarget],
    sources: list[dict],
    provider: VisionProvider | None,
    unsupported_reason: str,
    stored_pdf_path: Callable[[str], Path],
) -> VisualAnswer:
    """Route a visual question: vision model when one is really available,
    otherwise the honest unsupported response. Only the first target page is
    sent to the model (one bounded call per question)."""
    if provider is None:
        return build_answer(db, targets, sources, reason=unsupported_reason)

    first = targets[0]
    page_text = _page_text_evidence(db, first, sources)
    try:
        pdf_path: Path = stored_pdf_path(first.document_id)
        image_path = visuals.render_page_image(first.document_id, pdf_path, first.page_number, get_settings().page_render_default_dpi)
        interpretation = await provider.answer(image_path.read_bytes(), question, page_text["content"] if page_text else "")
    except (ProviderUnavailable, OSError, IndexError) as exc:
        return build_answer(db, targets, sources, reason=unsupported_reason, failure=f"A vision model ({provider.model}) is configured but could not be used: {exc}")
    if not interpretation or not interpretation.strip():
        return build_answer(db, targets, sources, reason=unsupported_reason, failure=f"The vision model ({provider.model}) returned no answer.")
    return build_answer(db, targets, sources, interpretation=interpretation, model=provider.model, reason="Answered by the configured local vision model.")
