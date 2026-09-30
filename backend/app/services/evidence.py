import math
import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document, PageText

# Group 8 -- deterministic "is there actually evidence for this?" checks that
# run *before* (or instead of) asking a model. The model is a poor judge of its
# own ignorance: handed unrelated passages it will sometimes answer from them
# anyway, so wherever the question can be checked against the document by plain
# code (does that page exist? does the topic appear at all?), it is -- and the
# answer is the existing "not enough evidence" behaviour, never an invention.

# A topic/location phrase with this many significant terms must have this
# fraction of them in the passage for the passage to count as being *about* it
# (two-word topics need both, three need two, four need three ...).
TERM_COVERAGE = 0.6

_WORD = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {"a", "an", "and", "are", "about", "the", "of", "for", "in", "is", "it", "to", "was", "what", "which", "with", "this", "that", "how", "does", "do", "on"}
_SUFFIXES = ("ing", "ers", "er", "ions", "ion", "es", "ed", "ly", "s")


def _stem(word: str) -> str:
    """Just enough normalisation that "sensors" matches "sensor" and "computing"
    matches "computer" -- while "quantum" still does not match "quantity". Not a
    real stemmer (that would blur exactly the terms keyword search exists for)."""
    if word.isdigit():
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def significant_terms(text: str) -> list[str]:
    """Lower-cased content words of a topic or question, in order, without repeats."""
    terms = [term for term in _WORD.findall(text.lower()) if len(term) > 2 and term not in _STOP_WORDS]
    return list(dict.fromkeys(terms))


def required_term_count(term_count: int) -> int:
    return max(1, math.ceil(term_count * TERM_COVERAGE)) if term_count else 0


def covers_terms(terms: list[str], text: str) -> bool:
    """Does `text` contain enough of `terms` to be about them? No terms means
    nothing to check, which counts as covered."""
    if not terms:
        return True
    stems = {_stem(word) for word in _WORD.findall(text.lower())}
    return sum(1 for term in terms if _stem(term) in stems) >= required_term_count(len(terms))


def topic_is_covered(topic: str, passages: list[str]) -> bool:
    """Whether any retrieved passage is actually about `topic`. Vector search
    always returns its nearest neighbours -- for a topic the document never
    mentions those are simply unrelated text -- so relevance has to be checked
    separately before anything is generated 'about' it."""
    terms = significant_terms(topic)
    return not terms or any(covers_terms(terms, passage) for passage in passages)


# -- scope / page checks -------------------------------------------------------


def _scoped_documents(db: Session, document_ids: list[str] | None) -> list[Document]:
    statement = select(Document).order_by(Document.created_at)
    if document_ids:
        statement = statement.where(Document.id.in_(document_ids))
    return list(db.scalars(statement).all())


def _quoted(documents: list[Document]) -> str:
    names = [f'"{document.filename}"' for document in documents[:3]]
    return ", ".join(names) + (f" and {len(documents) - 3} more" if len(documents) > 3 else "")


def explain_empty_scope(db: Session, document_ids: list[str] | None, default: str) -> str:
    """Why a question found nothing to work from. When the selected documents
    simply aren't usable yet (still processing, failed, blank) say *that* --
    "Upload a text-based PDF" is wrong advice for a document that is mid-upload.
    Falls back to `default` when the documents are fine."""
    documents = _scoped_documents(db, document_ids)
    if not documents:
        return "There are no documents to search yet. Upload a PDF first." if not document_ids else default
    if any(document.status == "ready" for document in documents):
        return default
    problems: list[str] = []
    processing = [document for document in documents if document.status == "processing"]
    failed = [document for document in documents if document.status == "failed"]
    empty = [document for document in documents if document.status == "empty"]
    if processing:
        problems.append(f"{_quoted(processing)} {'is' if len(processing) == 1 else 'are'} still being processed. Try again once it shows Ready.")
    if failed:
        reason = failed[0].status_detail or "processing failed"
        problems.append(f"{_quoted(failed)} could not be processed ({reason.rstrip('.')}). Delete it and upload it again.")
    if empty:
        problems.append(f"{_quoted(empty)} has no extractable text (it is blank, or a scan the OCR could not read).")
    return " ".join(problems) or default


def page_absence_reply(db: Session, document_ids: list[str] | None, page: int) -> str | None:
    """The honest answer to "what is on page N" when retrieval found no text
    there: the page doesn't exist, or it exists but holds no readable text.
    None when the scope isn't a usable, ready document (the caller then reports
    that instead)."""
    documents = [document for document in _scoped_documents(db, document_ids) if document.status == "ready"]
    if not documents:
        return None
    longest = max(document.page_count or 0 for document in documents)
    if page < 1 or page > longest:
        if len(documents) == 1:
            count = documents[0].page_count or 0
            return f'"{documents[0].filename}" has {count} page{"" if count == 1 else "s"}, so there is no page {page}.'
        return f"None of the selected documents has a page {page} (the longest has {longest} pages)."

    rows = db.scalars(select(PageText).where(PageText.document_id.in_([document.id for document in documents]), PageText.page_number == page)).all()
    failed = next((row for row in rows if row.status == "failed"), None)
    if failed:
        return f"Page {page} could not be read: {failed.status_detail or 'OCR failed on this page'}"
    return f"Page {page} has no readable text, so there is nothing on it to answer from."
