import re

from app.services import calculation, extraction
from app.services.retrieval import is_broad_query
from app.services.retrieval_types import FormatMode, RetrievalMode, RetrievalPlan

__all__ = ["FormatMode", "RetrievalMode", "RetrievalPlan", "route", "route_format"]

# Group 3's query router: replaces the old "is_broad_query-only" check with
# a real retrieval-mode + answer-format classifier. Deliberately no LLM call
# here -- every signal below is a cheap regex/structural check over the
# question text (and, optionally, the document's own known section
# headings), so routing costs microseconds and never depends on Ollama
# being up. See RETRIEVAL ARCHITECTURE UPGRADE / QUERY ROUTER in the Group 3
# brief for the full rationale.

PAGE_PATTERN = re.compile(r"\bpage\s+(\d{1,4})\b", re.IGNORECASE)

# "the Sensors section", "the MQTT section" -- the clean case: an explicit
# heading name followed by the word "section". "the" is deliberately
# mandatory (not optional): with an optional prefix, re.search is free to
# anchor the match at the *start* of the question instead of at the real
# "the" right before the heading -- e.g. "What does the Sensors section
# say?" would then lazily capture "What does the Sensors" instead of just
# "Sensors", because an empty optional match at position 0 is tried before
# the engine ever reaches the actual "the".
SECTION_SUFFIX_PATTERN = re.compile(r"\bthe\s+([a-z][a-z0-9\-/ ]{1,45}?)\s+section\b", re.IGNORECASE)
_SECTION_CUE_PATTERN = re.compile(r"\bsection\b|\bchapter\b", re.IGNORECASE)
# "what does chapter 3 discuss", "what does the Sensors section cover" --
# catches chapter/number references and phrasing SECTION_SUFFIX_PATTERN
# doesn't (no literal word "section").
SECTION_DISCUSS_PATTERN = re.compile(
    r"\bwhat\s+(?:does|do)\s+(?:the\s+)?([a-z0-9][a-z0-9\-/ ]{1,45}?)\s+(?:discuss|say|cover|talk about|explain)\b",
    re.IGNORECASE,
)
# Common section names that read naturally without the word "section"
# ("what does the introduction say", not "the introduction section say").
GENERIC_SECTION_NAMES = (
    "introduction", "conclusion", "abstract", "background", "methodology",
    "methods", "results", "discussion", "references", "appendix", "preface",
)
GENERIC_SECTION_PATTERN = re.compile(r"\bthe\s+(" + "|".join(GENERIC_SECTION_NAMES) + r")\b", re.IGNORECASE)

# Group 4: LOCATION queries ask *where* something is, not what it means --
# "Where does the document discuss ZigBee?" should point at a page, not
# explain ZigBee. Each pattern's captured group is the bare topic phrase
# (e.g. "ZigBee"), used both as the keyword-search query and the results
# heading instead of the whole, noisier question.
_LOCATION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bwhere\s+is\s+the\s+definition\s+of\s+(.+?)[\?\.!]*$", re.IGNORECASE),
    re.compile(r"\bwhere\s+is\s+(.+?)\s+(?:explained|defined|discussed|mentioned|described|located|found|covered)\b", re.IGNORECASE),
    re.compile(
        r"\bwhere\s+(?:does|do)\s+(?:the\s+)?(?:document|pdf|file|text)?\s*"
        r"(?:discuss(?:es)?|mention(?:s)?|talk(?:s)?\s+about|cover(?:s)?|explain(?:s)?|describe(?:s)?)\s+(.+?)[\?\.!]*$",
        re.IGNORECASE,
    ),
    re.compile(r"\bwhich\s+page[s]?\s+(?:discusses?|mentions?|covers?|explains?|talks?\s+about|describes?)\s+(.+?)[\?\.!]*$", re.IGNORECASE),
    re.compile(r"\bwhere\s+can\s+i\s+find\s+(.+?)[\?\.!]*$", re.IGNORECASE),
    # Lowest priority: a bare "where is X" with no trailing verb (e.g. "Where
    # is the introduction?") -- checked last so a more specific pattern above
    # (which also captures a cleaner topic phrase) always wins when it applies.
    re.compile(r"\bwhere\s+is\s+(?:the\s+)?(.+?)[\?\.!]*$", re.IGNORECASE),
]
_LOCATION_TRAILING_CLAUSE_PATTERN = re.compile(r"\s+in\s+(?:this|the)\s+(?:document|pdf|file|text)$", re.IGNORECASE)


def is_location_query(question: str) -> bool:
    return any(pattern.search(question) for pattern in _LOCATION_PATTERNS)


def extract_location_topic(question: str) -> str | None:
    for pattern in _LOCATION_PATTERNS:
        match = pattern.search(question)
        if not match:
            continue
        topic = match.group(1).strip().strip("?.! ")
        topic = _LOCATION_TRAILING_CLAUSE_PATTERN.sub("", topic).strip()
        return topic or None
    return None


_TABLE_PATTERN = re.compile(r"\btabl(e|es|ular|ulate)\b|\bcompar(e|ison|ing|isons)\b|\bversus\b|\bvs\.?\b", re.IGNORECASE)
_BULLETS_PATTERN = re.compile(r"\bbullet|\bpoint[- ]?wise\b|\bin points\b|\blist (them|these|it)\b|\bas a list\b", re.IGNORECASE)
_SECTIONED_PATTERN = re.compile(r"section[- ]?wise|section by section|by section|chapter[- ]?wise|breakdown by section", re.IGNORECASE)
_SUMMARY_PATTERN = re.compile(r"\bsummar(y|ize|ise)\b|\bbrief(ly)?\b|\btl;?dr\b|\bconcise(ly)?\b|\bshort(er)? version\b", re.IGNORECASE)


def route_format(question: str) -> FormatMode:
    """Answer *presentation* only -- never touches retrieval. Checked in
    order of specificity: an explicit "table"/"compare" instruction should
    win even if the question also happens to say "briefly".
    """
    if _TABLE_PATTERN.search(question):
        return FormatMode.TABLE
    if _BULLETS_PATTERN.search(question):
        return FormatMode.BULLETS
    if _SECTIONED_PATTERN.search(question):
        return FormatMode.SECTIONED
    if _SUMMARY_PATTERN.search(question):
        return FormatMode.SUMMARY
    return FormatMode.DEFAULT


def _match_section(question: str, known_sections: list[str] | None) -> str | None:
    """Two separate questions, answered in order: (1) is this question about
    a *section* at all (a structural cue -- "section"/"chapter", a "what
    does X discuss" phrasing, or a common generic heading like
    "introduction")? (2) if so, *which* section -- resolved against the
    document's real headings first, since that is far more reliable than
    any regex extraction over free-form phrasing.
    """
    generic_match = GENERIC_SECTION_PATTERN.search(question)
    discuss_match = SECTION_DISCUSS_PATTERN.search(question)
    has_signal = bool(_SECTION_CUE_PATTERN.search(question) or generic_match or discuss_match)
    if not has_signal:
        return None

    if known_sections:
        lowered_question = question.lower()
        matches = [heading for heading in known_sections if heading and heading.lower() in lowered_question]
        if matches:
            return max(matches, key=len)  # the most specific (longest) heading actually named

    if generic_match:
        return generic_match.group(1)
    suffix_match = SECTION_SUFFIX_PATTERN.search(question)
    if suffix_match:
        return suffix_match.group(1).strip()
    if discuss_match:
        return discuss_match.group(1).strip()
    # A structural cue fired ("section"/"chapter" appears) but no heading
    # phrase could be extracted or resolved -- fall through to the router's
    # normal OVERVIEW/FACT fallback rather than fabricating a section filter
    # from the whole question.
    return None


def route(
    question: str,
    document_ids: list[str] | None = None,
    *,
    selected_text: str | None = None,
    selection_page: int | None = None,
    known_sections: list[str] | None = None,
    default_top_k: int = 8,
    overview_top_k: int = 10,
    section_top_k: int = 8,
    page_top_k: int = 8,
    selection_top_k: int = 6,
    multi_document_top_k: int = 12,
    location_top_k: int = 6,
    extraction_top_k: int = 20,
    calculation_top_k: int = 6,
) -> RetrievalPlan:
    """Classify a question into a RetrievalPlan. Checked most-structural
    (hardest to get wrong) first: an explicit text selection or multiple
    selected documents is a fact about the *request*, not an inference over
    the question text, so those win over anything the wording implies.
    Falls back to FACT -- never an empty/uncertain plan -- when nothing
    matches, per "ROUTER MUST NOT BREAK EXISTING QUESTIONS".
    """
    document_ids = document_ids or []

    if selected_text and selected_text.strip():
        return RetrievalPlan(
            mode=RetrievalMode.SELECTION, query=question, document_ids=document_ids or None,
            selected_text=selected_text, page=selection_page, top_k=selection_top_k, diversity=False,
        )

    if len(document_ids) > 1:
        return RetrievalPlan(
            mode=RetrievalMode.MULTI_DOCUMENT, query=question, document_ids=document_ids,
            top_k=multi_document_top_k, diversity=True,
        )

    page_match = PAGE_PATTERN.search(question)
    if page_match:
        return RetrievalPlan(
            mode=RetrievalMode.PAGE, query=question, document_ids=document_ids or None,
            page=int(page_match.group(1)), top_k=page_top_k, diversity=False,
        )

    # LOCATION and EXTRACTION are checked next -- both are explicit,
    # structural cues ("where does X discuss Y", "... for each Z") that are
    # just as reliable as an explicit page number, and neither should be
    # shadowed by an incidental "section"/percentage word elsewhere in the
    # same question (see the ordering rationale for CALCULATION below).
    if is_location_query(question):
        topic = extract_location_topic(question) or question
        return RetrievalPlan(
            mode=RetrievalMode.LOCATION, query=topic, document_ids=document_ids or None,
            location_topic=topic, top_k=location_top_k, diversity=True,
        )

    if extraction.is_extraction_query(question):
        return RetrievalPlan(
            mode=RetrievalMode.EXTRACTION, query=question, document_ids=document_ids or None,
            top_k=extraction_top_k, diversity=True,
        )

    # CALCULATION is checked before SECTION so an explicit "percentage
    # increase" question still gets deterministic math even when it also
    # happens to name a section (e.g. "the percentage increase in the
    # Sensors section") -- the calculation pipeline still retrieves
    # normally, it just never hands the arithmetic to the LLM afterward.
    if calculation.is_calculation_query(question):
        return RetrievalPlan(
            mode=RetrievalMode.CALCULATION, query=question, document_ids=document_ids or None,
            top_k=calculation_top_k, diversity=True,
        )

    section = _match_section(question, known_sections)
    if section:
        return RetrievalPlan(
            mode=RetrievalMode.SECTION, query=question, document_ids=document_ids or None,
            section=section, top_k=section_top_k, diversity=False,
        )

    if is_broad_query(question):
        return RetrievalPlan(
            mode=RetrievalMode.OVERVIEW, query=question, document_ids=document_ids or None,
            top_k=overview_top_k, diversity=True,
        )

    return RetrievalPlan(
        mode=RetrievalMode.FACT, query=question, document_ids=document_ids or None,
        top_k=default_top_k, diversity=True,
    )
