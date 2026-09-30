import json
import logging
import re

from app.services.ollama import OllamaUnavailable, generate

logger = logging.getLogger("docuchat.suggestions")

# Generated once at ingestion time (see services/documents.py::process_document)
# and cached on Document.suggested_questions_json -- never on every page
# load/chat open. A page-count-proportional sample (not the whole document)
# keeps the one extra LLM call this costs small and bounded even for a
# multi-hundred-page PDF.
MAX_SAMPLE_PAGES = 14
CHARS_PER_PAGE = 500
MAX_EXCERPT_CHARS = 7000


def _sample_pages(page_texts: list[str]) -> list[tuple[int, str]]:
    """Evenly-spaced page indices (1-based) so the sample spans the whole
    document instead of just the opening pages -- the same "coverage over
    proximity" idea as retrieve_overview, applied to raw page text since no
    chunk/embedding data exists yet this early in ingestion.
    """
    non_empty = [(index, text) for index, text in enumerate(page_texts, start=1) if text.strip()]
    if not non_empty:
        return []
    if len(non_empty) <= MAX_SAMPLE_PAGES:
        return non_empty
    step = len(non_empty) / MAX_SAMPLE_PAGES
    picked_indices = {int(i * step) for i in range(MAX_SAMPLE_PAGES)}
    return [non_empty[i] for i in sorted(picked_indices)]


def _build_excerpt(page_texts: list[str]) -> str:
    sampled = _sample_pages(page_texts)
    parts: list[str] = []
    total = 0
    for page_number, text in sampled:
        snippet = re.sub(r"\s+", " ", text).strip()[:CHARS_PER_PAGE]
        if not snippet:
            continue
        piece = f"[Page {page_number}] {snippet}"
        if total + len(piece) > MAX_EXCERPT_CHARS:
            break
        parts.append(piece)
        total += len(piece)
    return "\n\n".join(parts)


def _parse_questions(raw: str, count: int) -> list[str]:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    try:
        payload = json.loads(cleaned[cleaned.find("["):cleaned.rfind("]") + 1])
        questions = [str(item).strip() for item in payload]
    except (ValueError, json.JSONDecodeError):
        # Fall back to one-question-per-line if the model didn't return
        # clean JSON (still common enough with local models to be worth
        # handling rather than discarding a usable answer).
        questions = [re.sub(r'^[\-\d.\)\s"]+|["\s]+$', "", line) for line in cleaned.splitlines()]
    unique: list[str] = []
    for question in questions:
        question = question.strip().strip('"')
        if question and question not in unique and len(question) <= 200:
            unique.append(question)
    return unique[:count]


async def generate_suggested_questions(page_texts: list[str], count: int) -> list[str]:
    """Best-effort only: any failure (Ollama down, malformed output) returns
    an empty list rather than raising, so a suggestion failure can never
    fail document ingestion (see SUGGESTED QUESTIONS / Generation strategy).
    """
    excerpt = _build_excerpt(page_texts)
    if not excerpt:
        return []
    prompt = (
        f"Based ONLY on the document excerpts below (sampled from across the whole document), write exactly "
        f"{count} distinct, specific questions a reader could ask and have answered directly from this document. "
        "Avoid generic questions like \"What is this document about?\" -- prefer questions about specific "
        "concepts, definitions, comparisons, or named sections that actually appear in the excerpts. Return ONLY "
        f"a JSON array of {count} question strings, no prose outside it.\n\nEXCERPTS:\n{excerpt}"
    )
    try:
        raw = await generate([{"role": "user", "content": prompt}], think=False)
    except OllamaUnavailable as exc:
        logger.warning("Suggested-question generation skipped: %s", exc)
        return []
    try:
        return _parse_questions(raw, count)
    except Exception:  # noqa: BLE001 -- never let a parsing edge case fail ingestion
        logger.warning("Suggested-question generation returned unparsable output")
        return []
