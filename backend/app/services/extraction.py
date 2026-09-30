import json
import re
from dataclasses import dataclass

from app.services.evidence import covers_terms, significant_terms
from app.services.retrieval_types import FormatMode

# Group 4's batch/structured extraction. Split cleanly into three testable
# stages, per the brief's "this should make the system testable" requirement:
#   1. parse_extraction_request  -- pure regex, no LLM: what entity type and
#      fields is the user asking for?
#   2. (in api/v1/chat.py) the LLM identifies the entities and fills the
#      fields, grounded strictly in the retrieved passages.
#   3. parse_extraction_response + render_extraction_markdown -- pure, no
#      LLM: validate every claimed source_id, force "Not stated" for
#      anything not actually claimed, and deterministically render the table/
#      bullets/sections. A missing field is NEVER inferred or fabricated --
#      only stage 2 (the LLM) can populate a value, and stage 3 only ever
#      downgrades an unconvincing answer to "Not stated," never invents one.

_FOR_EACH_INTENT_PATTERN = re.compile(r"\bfor\s+each\b|\bfor\s+every\b", re.IGNORECASE)
_FOR_EACH_ENTITY_PATTERN = re.compile(r"\bfor\s+(?:each|every)\s+([a-z][a-z0-9\-/]*(?:\s+[a-z][a-z0-9\-/]*)?)", re.IGNORECASE)
_BULLET_LINE_PATTERN = re.compile(r"^\s*[\-\*•]\s*(.+?)\s*$", re.MULTILINE)
_BEFORE_FOR_EACH_PATTERN = re.compile(r"^(.*?)\bfor\s+(?:each|every)\b", re.IGNORECASE | re.DOTALL)
_LEADING_VERB_PATTERN = re.compile(r"^(extract|give me|give|list|provide|find|show me|show)\s+", re.IGNORECASE)
_LEADING_ARTICLE_PATTERN = re.compile(r"^(the\s+following\s*:?\s*|the\s+|following\s*:?\s*)", re.IGNORECASE)


def is_extraction_query(question: str) -> bool:
    return bool(_FOR_EACH_INTENT_PATTERN.search(question))


def _strip_leading(phrase: str) -> str:
    phrase = _LEADING_VERB_PATTERN.sub("", phrase.strip())
    phrase = _LEADING_ARTICLE_PATTERN.sub("", phrase.strip())
    return phrase.strip(" :")


def _split_field_list(phrase: str) -> list[str]:
    phrase = re.sub(r"\s+and\s+", ", ", phrase, flags=re.IGNORECASE)
    return [part.strip() for part in phrase.split(",") if part.strip()]


@dataclass
class ExtractionRequest:
    entity_type: str
    fields: list[str]


def parse_extraction_request(question: str) -> ExtractionRequest:
    """Best-effort, regex-only parse of "extract X for each Y" -- never
    raises; an unparseable field list just falls back to a single generic
    "details" field so the request can still proceed instead of failing.
    """
    entity_match = _FOR_EACH_ENTITY_PATTERN.search(question)
    entity_type = entity_match.group(1).strip() if entity_match else "item"

    bullets = [bullet for bullet in _BULLET_LINE_PATTERN.findall(question) if bullet]
    if bullets:
        fields = bullets
    else:
        before_match = _BEFORE_FOR_EACH_PATTERN.search(question)
        phrase = _strip_leading(before_match.group(1)) if before_match else ""
        fields = _split_field_list(phrase) if phrase else []
    if not fields:
        fields = ["details"]
    return ExtractionRequest(entity_type=entity_type, fields=fields)


def build_extraction_instructions(entity_type: str, fields: list[str], max_sources: int) -> str:
    field_schema = ", ".join(f'"{field}":{{"value":"<value or null>","source_id":<source-number or null>}}' for field in fields)
    schema = f'{{"items":[{{"entity":"<name of the {entity_type}>","fields":{{{field_schema}}}}}]}}'
    field_list = ", ".join(fields)
    return (
        f"Identify every distinct {entity_type} the passages below actually describe, and extract these fields for "
        f'each one: {field_list}. Use ONLY what the passages state -- never infer, guess, or fabricate a value. If a '
        f'field is not stated for a given {entity_type}, set its "value" to null and its "source_id" to null. Every '
        f"non-null value must include the source_id (1 to {max_sources}) of the passage it came from. Return exactly "
        f"one JSON object, with no prose outside it:\n{schema}"
    )


_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
_NOT_STATED_VALUES = {"", "none", "null", "n/a", "na", "not stated", "not applicable", "unknown", "not mentioned"}


@dataclass
class ExtractedField:
    value: str
    source_id: int | None


@dataclass
class ExtractionItem:
    entity: str
    fields: dict[str, ExtractedField]


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_UNVERIFIABLE_WORDS = {"yes", "no", "true", "false"}


def _numbers_in(text: str) -> set[str]:
    return {match.replace(",", "").rstrip(".") for match in _NUMBER.findall(text)}


def _supported_by(value: str, passage: str) -> bool:
    """Is a value the model claims to have read really in the passage it cites?
    Every number it states must occur there, and a wordy value must share most
    of its content words with it. A model that fills a field from its own
    memory (a country's population, a capital city) and pins a real-looking
    [n] on it fails this, because that passage says nothing of the sort."""
    numbers = _numbers_in(value)
    if numbers:
        return numbers <= _numbers_in(passage)
    terms = significant_terms(value)
    if not terms or set(terms) <= _UNVERIFIABLE_WORDS:
        return True  # a bare yes/no can't be checked against wording
    return covers_terms(terms, passage)


def _verify_against_sources(items: list[ExtractionItem], sources: list[dict]) -> list[ExtractionItem]:
    """Fail closed on what the model claims: a value its cited passage doesn't
    support becomes "Not stated", and an item with no supported value left --
    or whose name appears in none of the passages at all -- is dropped, since
    nothing shows it exists in the document."""
    passages = [str(source.get("content") or "") for source in sources]
    verified: list[ExtractionItem] = []
    for item in items:
        for name, cell in item.fields.items():
            if cell.source_id is not None and not _supported_by(cell.value, passages[cell.source_id - 1]):
                item.fields[name] = ExtractedField(value="Not stated", source_id=None)
        has_evidence = any(cell.source_id is not None for cell in item.fields.values())
        # Lenient on purpose: any one word of the entity's name in any passage.
        # A model's label ("Bluetooth LE") may paraphrase the document's, but a
        # name that shares no word with the evidence at all was made up.
        entity_terms = significant_terms(item.entity)
        named_in_evidence = not entity_terms or any(covers_terms([term], passage) for term in entity_terms for passage in passages)
        if has_evidence and named_in_evidence:
            verified.append(item)
    return verified


def parse_extraction_response(raw: str, fields: list[str], source_count: int, sources: list[dict] | None = None) -> list[ExtractionItem]:
    """Accept only source-bound values -- mirrors chat.py's grounded_response
    contract: any source_id outside 1..source_count is dropped (never
    displayed as if verified), and any field the model didn't actually
    supply a value for becomes "Not stated" rather than left absent.

    When the `sources` the model was shown are given, every claimed value is
    also checked against the passage it cites (see `_verify_against_sources`).
    """
    cleaned = _THINK_PATTERN.sub("", raw).strip()
    try:
        payload = json.loads(cleaned[cleaned.find("{"):cleaned.rfind("}") + 1])
        raw_items = payload.get("items", [])
        if not isinstance(raw_items, list):
            raw_items = []
    except (ValueError, AttributeError, json.JSONDecodeError):
        raw_items = []

    items: list[ExtractionItem] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        entity = str(raw_item.get("entity") or "").strip()
        if not entity:
            continue
        raw_fields = raw_item.get("fields")
        raw_fields = raw_fields if isinstance(raw_fields, dict) else {}

        parsed_fields: dict[str, ExtractedField] = {}
        for field in fields:
            raw_field = raw_fields.get(field)
            value, source_id = None, None
            if isinstance(raw_field, dict):
                value, source_id = raw_field.get("value"), raw_field.get("source_id")
            elif isinstance(raw_field, str):
                value = raw_field

            value_text = str(value).strip() if value is not None else ""
            if value_text.lower() in _NOT_STATED_VALUES:
                value_text = ""
            try:
                source_id = int(source_id) if source_id is not None else None
            except (TypeError, ValueError):
                source_id = None
            if source_id is not None and not (1 <= source_id <= source_count):
                source_id = None
            if not value_text:
                value_text, source_id = "Not stated", None  # never cite a source for a value we aren't claiming
            parsed_fields[field] = ExtractedField(value=value_text, source_id=source_id)
        items.append(ExtractionItem(entity=entity, fields=parsed_fields))
    return _verify_against_sources(items, sources) if sources is not None else items


def used_source_ids(items: list[ExtractionItem]) -> list[int]:
    ids = {field.source_id for item in items for field in item.fields.values() if field.source_id is not None}
    return sorted(ids)


def _title_case(label: str) -> str:
    return " ".join(word.capitalize() for word in label.split())


def _cell_text(item: ExtractionItem, field: str) -> str:
    cell = item.fields.get(field)
    if cell is None:
        return "Not stated"
    marker = f" [{cell.source_id}]" if cell.source_id is not None else ""
    return f"{cell.value}{marker}"


def render_extraction_markdown(entity_type: str, items: list[ExtractionItem], fields: list[str], format_mode: FormatMode) -> str:
    """Deterministic rendering only -- the LLM's job ended at
    parse_extraction_response; nothing here can introduce a value the model
    didn't supply. Retrieval intent (that this is an extraction at all) and
    presentation format are kept independent, same as the rest of Group 3/4:
    the same `items` render as a table, bullets, or sections purely based on
    `format_mode`.
    """
    if format_mode == FormatMode.BULLETS:
        lines = []
        for item in items:
            lines.append(f"- **{item.entity}**")
            lines.extend(f"  - {_title_case(field)}: {_cell_text(item, field)}" for field in fields)
        return "\n".join(lines)

    if format_mode == FormatMode.SECTIONED:
        sections = []
        for item in items:
            body = "\n".join(f"- {_title_case(field)}: {_cell_text(item, field)}" for field in fields)
            sections.append(f"## {item.entity}\n{body}")
        return "\n\n".join(sections)

    header = "| " + " | ".join([_title_case(entity_type), *(_title_case(field) for field in fields)]) + " |"
    separator = "|" + "---|" * (len(fields) + 1)
    rows = ["| " + " | ".join([item.entity, *(_cell_text(item, field) for field in fields)]) + " |" for item in items]
    return "\n".join([header, separator, *rows])
