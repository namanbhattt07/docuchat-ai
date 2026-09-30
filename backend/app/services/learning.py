import json
import random
import re

from sqlalchemy.orm import Session

from app.services.retrieval import retrieve_overview
from app.services.retrieval_types import RetrievalMode, RetrievalPlan

# Group 6 (Tutor / Exam / Brainstorm): small, dependency-free helpers every
# learning-mode service shares -- kept here instead of copied into tutor.py,
# exam.py, brainstorm.py and writing_feedback.py. Nothing in this module
# talks to the model or builds a prompt; it only parses/validates what came
# back and picks which existing retrieval output to hand the model.

_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_TRAILING_COMMA_PATTERN = re.compile(r",\s*([}\]])")
_CITATION_MARKER_PATTERN = re.compile(r"\s*\[(?:\s*source[\s_-]?)?\d+\]", re.IGNORECASE)
_WORD_CITATION_PATTERN = re.compile(r"\[\s*source[\s_-]?(\d+)\s*\]", re.IGNORECASE)
_GROUPED_CITATION_PATTERN = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)+)\s*\]")

# A passage shorter than this is almost always a title page, running header
# or TOC fragment -- useless as the basis for an exam question.
MIN_SOURCE_CHARS = 120
# Upper bound on how many chunks the coverage sampler considers -- far above
# any real document here (a 200-page PDF is a few hundred chunks), it only
# exists so a pathological library can't turn one exam request into an
# unbounded in-memory sort.
OVERVIEW_CANDIDATE_CAP = 5000


def clean_model_output(raw: str) -> str:
    return _THINK_PATTERN.sub("", raw or "").strip()


def extract_json_object(raw: str) -> dict | None:
    """Best-effort parse of the single JSON object a learning-mode prompt
    asks for. Local models commonly wrap it in a ```json fence, add prose
    around it, or leave a trailing comma -- each is tolerated here rather
    than failing the whole request. Returns None (never raises) when nothing
    usable is found, so every caller has one explicit "malformed" branch.
    """
    cleaned = _CODE_FENCE_PATTERN.sub("", clean_model_output(raw))
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    candidate = cleaned[start:end + 1]
    for attempt in (candidate, _TRAILING_COMMA_PATTERN.sub(r"\1", candidate)):
        try:
            payload = json.loads(attempt)
        except (ValueError, json.JSONDecodeError):
            continue
        return payload if isinstance(payload, dict) else None
    return None


def normalize_citation_markers(text: str) -> str:
    """"[source-1]" -> "[1]" (the same normalization chat.py applies), and a
    grouped "[2, 8]" -> "[2] [8]" so each id is validated and cited on its
    own rather than left as a marker no citation matches."""
    text = _WORD_CITATION_PATTERN.sub(r"[\1]", text or "")
    return _GROUPED_CITATION_PATTERN.sub(lambda match: " ".join(f"[{part.strip()}]" for part in match.group(1).split(",")), text)


def strip_citation_markers(text: str) -> str:
    """Tutor explanations, examples and generated ideas never come from the
    document, so a [n] marker inside one would misrepresent it as evidence
    (the Group 5 GO FREELY citation contract) -- always removed.
    """
    return _CITATION_MARKER_PATTERN.sub("", text or "").strip()


def cited_ids_in(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"\[(\d+)\]", normalize_citation_markers(text))]


def valid_source_ids(values, source_count: int, max_sources: int | None = None) -> list[int]:
    """Keep only integer ids inside 1..source_count, de-duplicated in order.
    Anything else (strings that aren't numbers, out-of-range ids, nested
    junk) is dropped rather than trusted -- a citation must always point at
    a passage that was actually shown to the model.
    """
    if not isinstance(values, list):
        values = [values] if values is not None else []
    ids: list[int] = []
    for value in values:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= number <= source_count and number not in ids:
            ids.append(number)
    return ids[:max_sources] if max_sources else ids


def as_text(value, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = "\n".join(f"- {as_text(item)}" for item in value if as_text(item))
    return str(value).strip()[:limit]


def as_text_list(value, limit: int = 12, item_limit: int = 600) -> list[str]:
    if isinstance(value, str):
        value = [line for line in re.split(r"\n+", value) if line.strip()]
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        text = re.sub(r"^[\-\*•\d.\)\s]+", "", as_text(item, item_limit)).strip()
        if text and text not in items:
            items.append(text)
    return items[:limit]


def sample_document_sources(db: Session, document_ids: list[str] | None, limit: int, seed: int | None = None) -> list[dict]:
    """Document-wide coverage for exam/brainstorm generation, built on the
    existing OVERVIEW retrieval rather than a new retriever: take
    retrieve_overview's full page-by-page candidate set, drop the passages
    too short to test anything, then keep `limit` of them spaced evenly by
    position across the document. A plain retrieve_overview(limit=N) walks
    pages in order and stops after N, which for a long PDF means every
    question would come from the front matter. `seed` shifts the sampling
    window so "New exam" produces a different question set, while staying
    reproducible in tests.
    """
    candidates = retrieve_overview(db, document_ids, limit=OVERVIEW_CANDIDATE_CAP)
    substantive = [source for source in candidates if len((source.get("content") or "").strip()) >= MIN_SOURCE_CHARS]
    pool = substantive or candidates
    if len(pool) <= limit:
        return pool
    pool = sorted(pool, key=lambda item: (item["document_id"], item["page_number"] or 0, item.get("start_offset") or 0))
    step = len(pool) / limit
    offset = random.Random(seed).random() * step if seed is not None else 0.0
    picked = sorted({min(int(offset + index * step), len(pool) - 1) for index in range(limit)})
    return [pool[index] for index in picked]


def merge_sources(primary: list[dict], secondary: list[dict], limit: int) -> list[dict]:
    """Relevance-first merge (primary), topped up with coverage (secondary),
    de-duplicated by chunk id -- used where a request needs both "what's
    closest to the question" and "what else the document says".
    """
    merged: list[dict] = []
    seen: set[str] = set()
    for source in [*primary, *secondary]:
        key = source.get("chunk_id") or f"{source.get('document_id')}:{source.get('page_number')}:{source.get('content', '')[:40]}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(source)
        if len(merged) >= limit:
            break
    return merged


def explanatory_plan(plan: RetrievalPlan, *, document_wide: bool, overview_top_k: int, default_top_k: int) -> RetrievalPlan:
    """Tutor and Brainstorm always *explain or build on* content, so Group
    4's answer-shaped modes (a LOCATION page list, deterministic
    CALCULATION, EXTRACTION tables) are re-planned as ordinary FACT retrieval
    over the same query and scope. `document_wide` (e.g. "Teach from
    beginning" with no topic yet) uses OVERVIEW coverage instead.
    SELECTION/SECTION/PAGE/MULTI_DOCUMENT plans are kept exactly as the
    router produced them.
    """
    if plan.mode == RetrievalMode.SELECTION:
        return plan
    if document_wide:
        plan.mode, plan.top_k, plan.diversity = RetrievalMode.OVERVIEW, overview_top_k, True
    elif plan.mode in (RetrievalMode.LOCATION, RetrievalMode.CALCULATION, RetrievalMode.EXTRACTION):
        plan.mode, plan.top_k, plan.diversity, plan.location_topic = RetrievalMode.FACT, default_top_k, True, None
    return plan
