import json
import re

from app.models import PageText

# In-document keyword search. Deliberately independent of retrieval.py /
# vector_store.py -- this never touches the LLM or the embedding index, it
# just scans the page text PageText already cached at ingestion time (see
# services/documents.py::process_document), so it stays fast and responsive
# even on large PDFs without re-parsing anything.

SNIPPET_RADIUS = 60


def normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def bbox_for_range(blocks: list[dict], start: int, end: int) -> list[float] | None:
    overlapping = [block["bbox"] for block in blocks if block["start"] < end and block["end"] > start]
    if not overlapping:
        return None
    return [
        round(min(box[0] for box in overlapping), 2),
        round(min(box[1] for box in overlapping), 2),
        round(max(box[2] for box in overlapping), 2),
        round(max(box[3] for box in overlapping), 2),
    ]


def search_pages(pages: list[PageText], query: str) -> list[dict]:
    """Case-insensitive phrase search across every cached page, in page
    order. Returns one entry per match with the page, character offsets, a
    best-effort bbox (via the page's stored block spans), and a short
    surrounding snippet for display.
    """
    normalized = normalize_query(query)
    if not normalized:
        return []
    pattern = re.compile(re.escape(normalized), re.IGNORECASE)

    matches: list[dict] = []
    for page in sorted(pages, key=lambda item: item.page_number):
        content = page.content or ""
        if not content:
            continue
        blocks = json.loads(page.blocks_json or "[]")
        for occurrence in pattern.finditer(content):
            start, end = occurrence.start(), occurrence.end()
            snippet_start = max(0, start - SNIPPET_RADIUS)
            snippet_end = min(len(content), end + SNIPPET_RADIUS)
            snippet = ("…" if snippet_start > 0 else "") + content[snippet_start:snippet_end] + ("…" if snippet_end < len(content) else "")
            matches.append({
                "page": page.page_number,
                "start_offset": start,
                "end_offset": end,
                "bbox": bbox_for_range(blocks, start, end),
                "snippet": snippet,
            })
    return matches
