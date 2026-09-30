import difflib
import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import PageText
from app.services.search import bbox_for_range, normalize_query

# Hardens Group 1's citation -> bbox highlight for the (rare) case a chunk
# was persisted without a bbox. Reuses the same PageText cache document
# search reads from, so this never re-parses the PDF either.

# A fallback match shorter than this (in characters) is treated as noise --
# a couple of accidentally-shared words rather than the actual passage.
MIN_FALLBACK_MATCH_CHARS = 20


def locate_passage_bbox(db: Session, document_id: str, page_number: int, text: str) -> list[float] | None:
    page = db.scalar(select(PageText).where(PageText.document_id == document_id, PageText.page_number == page_number))
    if not page or not page.content:
        return None

    content = page.content
    needle = normalize_query(text)
    if not needle:
        return None

    lowered_content = content.lower()
    lowered_needle = needle.lower()

    index = lowered_content.find(lowered_needle)
    if index != -1:
        start, end = index, index + len(needle)
    else:
        # The needle (e.g. a chunk that spans a paragraph break the page's
        # normalization collapsed differently) doesn't appear verbatim --
        # fall back to the longest common run instead of failing outright,
        # which tolerates small formatting differences and text that
        # originally spanned multiple lines.
        matcher = difflib.SequenceMatcher(None, lowered_content, lowered_needle, autojunk=False)
        match = matcher.find_longest_match(0, len(lowered_content), 0, len(lowered_needle))
        min_length = max(MIN_FALLBACK_MATCH_CHARS, int(len(lowered_needle) * 0.5))
        if match.size < min_length:
            return None
        start, end = match.a, match.a + match.size

    blocks = json.loads(page.blocks_json or "[]")
    return bbox_for_range(blocks, start, end)
