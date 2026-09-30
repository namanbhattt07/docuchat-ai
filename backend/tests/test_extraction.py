import json

from app.services.extraction import (
    ExtractedField,
    ExtractionItem,
    is_extraction_query,
    parse_extraction_request,
    parse_extraction_response,
    render_extraction_markdown,
    used_source_ids,
)
from app.services.retrieval_types import FormatMode


# ---------------------------------------------------------------------------
# Intent + field/entity parsing -- pure regex, no LLM
# ---------------------------------------------------------------------------


def test_is_extraction_query_detects_for_each_phrasing() -> None:
    assert is_extraction_query("Extract the specifications for each sensor.") is True
    assert is_extraction_query("Give me the name, year, author and key finding for each paper.") is True
    assert is_extraction_query("What is the role of sensors in IoT?") is False


def test_parse_extraction_request_reads_bullet_list_fields() -> None:
    question = "Extract the following for each protocol:\n- frequency\n- range\n- topology\n- data rate"
    request = parse_extraction_request(question)
    assert request.entity_type == "protocol"
    assert request.fields == ["frequency", "range", "topology", "data rate"]


def test_parse_extraction_request_reads_comma_separated_fields() -> None:
    request = parse_extraction_request("Give me the name, year, author and key finding for each paper.")
    assert request.entity_type == "paper"
    assert request.fields == ["name", "year", "author", "key finding"]


def test_parse_extraction_request_reads_single_field_phrase() -> None:
    request = parse_extraction_request("Extract the specifications for each sensor.")
    assert request.entity_type == "sensor"
    assert request.fields == ["specifications"]


def test_parse_extraction_request_falls_back_to_generic_field_when_unparseable() -> None:
    request = parse_extraction_request("List stuff for each thing")
    assert request.entity_type == "thing"
    assert request.fields  # never empty


# ---------------------------------------------------------------------------
# LLM response parsing/validation -- grounded_response-style tests, no LLM
# ---------------------------------------------------------------------------


def _raw(items: list[dict]) -> str:
    return json.dumps({"items": items})


def test_parse_extraction_response_reads_valid_items() -> None:
    raw = _raw([
        {"entity": "ZigBee", "fields": {"frequency": {"value": "2.4 GHz", "source_id": 1}, "range": {"value": "10-100m", "source_id": 2}}},
    ])
    items = parse_extraction_response(raw, fields=["frequency", "range"], source_count=2)
    assert len(items) == 1
    assert items[0].entity == "ZigBee"
    assert items[0].fields["frequency"].value == "2.4 GHz"
    assert items[0].fields["frequency"].source_id == 1


def test_parse_extraction_response_replaces_missing_field_with_not_stated() -> None:
    raw = _raw([{"entity": "ZigBee", "fields": {"frequency": {"value": "2.4 GHz", "source_id": 1}}}])
    items = parse_extraction_response(raw, fields=["frequency", "range"], source_count=1)
    assert items[0].fields["range"].value == "Not stated"
    assert items[0].fields["range"].source_id is None


def test_parse_extraction_response_normalizes_null_style_values_to_not_stated() -> None:
    raw = _raw([{"entity": "ZigBee", "fields": {"range": {"value": "null", "source_id": 1}}}])
    items = parse_extraction_response(raw, fields=["range"], source_count=1)
    assert items[0].fields["range"].value == "Not stated"
    assert items[0].fields["range"].source_id is None  # never cite a source for a value we're not claiming


def test_parse_extraction_response_drops_source_ids_outside_valid_range() -> None:
    raw = _raw([{"entity": "ZigBee", "fields": {"frequency": {"value": "2.4 GHz", "source_id": 99}}}])
    items = parse_extraction_response(raw, fields=["frequency"], source_count=2)
    assert items[0].fields["frequency"].source_id is None
    assert items[0].fields["frequency"].value == "2.4 GHz"  # value itself is kept, just un-cited


def test_parse_extraction_response_handles_multiple_entities_across_pages() -> None:
    raw = _raw([
        {"entity": "ZigBee", "fields": {"frequency": {"value": "2.4 GHz", "source_id": 1}}},
        {"entity": "Bluetooth", "fields": {"frequency": {"value": "2.4 GHz", "source_id": 2}}},
    ])
    items = parse_extraction_response(raw, fields=["frequency"], source_count=2)
    assert [item.entity for item in items] == ["ZigBee", "Bluetooth"]
    assert items[0].fields["frequency"].source_id == 1
    assert items[1].fields["frequency"].source_id == 2


def test_parse_extraction_response_skips_items_without_an_entity_name() -> None:
    raw = _raw([{"entity": "", "fields": {}}, {"entity": "ZigBee", "fields": {}}])
    items = parse_extraction_response(raw, fields=["frequency"], source_count=1)
    assert [item.entity for item in items] == ["ZigBee"]


def test_parse_extraction_response_returns_empty_list_for_malformed_json() -> None:
    items = parse_extraction_response("not json at all", fields=["frequency"], source_count=1)
    assert items == []


def test_parse_extraction_response_strips_think_block() -> None:
    raw = "<think>reasoning</think>" + _raw([{"entity": "ZigBee", "fields": {}}])
    items = parse_extraction_response(raw, fields=["frequency"], source_count=1)
    assert items[0].entity == "ZigBee"


def test_used_source_ids_dedupes_and_sorts_across_items() -> None:
    items = [
        ExtractionItem("ZigBee", {"frequency": ExtractedField("2.4 GHz", 2), "range": ExtractedField("10m", 1)}),
        ExtractionItem("Bluetooth", {"frequency": ExtractedField("2.4 GHz", 2), "range": ExtractedField("Not stated", None)}),
    ]
    assert used_source_ids(items) == [1, 2]


# ---------------------------------------------------------------------------
# Deterministic rendering -- no LLM, table/bullets/sections + "Not stated"
# ---------------------------------------------------------------------------


def _items() -> list[ExtractionItem]:
    return [
        ExtractionItem("ZigBee", {"frequency": ExtractedField("2.4 GHz", 1), "range": ExtractedField("Not stated", None)}),
        ExtractionItem("Bluetooth", {"frequency": ExtractedField("2.4 GHz", 2), "range": ExtractedField("10m", 2)}),
    ]


def test_render_extraction_markdown_table_includes_citation_markers() -> None:
    markdown = render_extraction_markdown("protocol", _items(), ["frequency", "range"], FormatMode.TABLE)
    assert "| Protocol | Frequency | Range |" in markdown
    assert "2.4 GHz [1]" in markdown
    assert "Not stated" in markdown
    assert "10m [2]" in markdown


def test_render_extraction_markdown_never_attaches_a_marker_to_not_stated() -> None:
    markdown = render_extraction_markdown("protocol", _items(), ["frequency", "range"], FormatMode.TABLE)
    assert "Not stated [" not in markdown


def test_render_extraction_markdown_bullets_format() -> None:
    markdown = render_extraction_markdown("protocol", _items(), ["frequency", "range"], FormatMode.BULLETS)
    assert "- **ZigBee**" in markdown
    assert "  - Frequency: 2.4 GHz [1]" in markdown
    assert "  - Range: Not stated" in markdown


def test_render_extraction_markdown_sectioned_format() -> None:
    markdown = render_extraction_markdown("protocol", _items(), ["frequency", "range"], FormatMode.SECTIONED)
    assert "## ZigBee" in markdown
    assert "## Bluetooth" in markdown


def test_render_extraction_markdown_default_format_renders_as_table() -> None:
    markdown = render_extraction_markdown("protocol", _items(), ["frequency", "range"], FormatMode.DEFAULT)
    assert markdown.startswith("| Protocol |")
