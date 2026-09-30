from dataclasses import dataclass
from enum import Enum

# Shared, dependency-free types for Group 3's retrieval architecture.
# Split out from query_router.py so both query_router (which classifies a
# question into one of these) and retrieval (which executes one) can import
# the types without importing each other.


class RetrievalMode(str, Enum):
    FACT = "fact"
    OVERVIEW = "overview"
    SECTION = "section"
    PAGE = "page"
    SELECTION = "selection"
    MULTI_DOCUMENT = "multi_document"
    # Group 4 additions -- see query_router.route() for the classification
    # rules and services/{calculation,extraction}.py for what each mode does
    # once retrieval hands its sources to api/v1/chat.py.
    LOCATION = "location"
    CALCULATION = "calculation"
    EXTRACTION = "extraction"


class FormatMode(str, Enum):
    DEFAULT = "default"
    BULLETS = "bullets"
    TABLE = "table"
    SECTIONED = "sectioned"
    SUMMARY = "summary"


@dataclass
class RetrievalPlan:
    """What to retrieve and how -- deliberately separate from FormatMode
    (how to *present* the answer). A query router that conflated the two
    couldn't express "the Sensors section, in bullet points": SECTION
    retrieval with BULLETS formatting.
    """

    mode: RetrievalMode
    query: str
    document_ids: list[str] | None = None
    # Group 5: which collection (if any) `document_ids` was scoped from --
    # carried through purely for observability/debugging (see retrieval.py's
    # _log_debug); retrieval itself only ever needs the already-resolved
    # document_ids, never the collection id.
    collection_id: str | None = None
    section: str | None = None
    page: int | None = None
    selected_text: str | None = None
    top_k: int = 8
    diversity: bool = True
    # LOCATION only: the bare topic phrase extracted from the question (e.g.
    # "ZigBee" out of "Where does the document discuss ZigBee?") -- used as
    # both the keyword-search query and the heading of the deterministic
    # results list in api/v1/chat.py, instead of searching/labelling with the
    # whole, noisier question text.
    location_topic: str | None = None
