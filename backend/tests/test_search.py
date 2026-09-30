import json

from app.services.search import bbox_for_range, normalize_query, search_pages


class FakePage:
    def __init__(self, page_number: int, content: str, blocks: list[dict] | None = None) -> None:
        self.page_number = page_number
        self.content = content
        self.blocks_json = json.dumps(blocks or [])


def test_normalize_query_collapses_whitespace() -> None:
    assert normalize_query("  machine   learning\n\n") == "machine learning"


def test_search_pages_is_case_insensitive() -> None:
    pages = [FakePage(1, "Machine Learning is a subfield of AI.")]

    matches = search_pages(pages, "machine learning")

    assert len(matches) == 1
    assert matches[0]["page"] == 1


def test_search_pages_matches_phrases_not_just_single_words() -> None:
    pages = [FakePage(1, "deep learning and machine learning are related but distinct fields.")]

    matches = search_pages(pages, "machine learning")

    assert len(matches) == 1
    assert matches[0]["start_offset"] == pages[0].content.lower().index("machine learning")


def test_search_pages_finds_matches_across_multiple_pages_in_order() -> None:
    pages = [
        FakePage(2, "second page mentions overfitting once."),
        FakePage(1, "first page mentions overfitting twice, overfitting is bad."),
    ]

    matches = search_pages(pages, "overfitting")

    assert [match["page"] for match in matches] == [1, 1, 2]


def test_search_pages_returns_empty_for_blank_query() -> None:
    pages = [FakePage(1, "some content here")]

    assert search_pages(pages, "   ") == []


def test_search_pages_returns_empty_when_no_matches() -> None:
    pages = [FakePage(1, "nothing relevant in here")]

    assert search_pages(pages, "machine learning") == []


def test_search_pages_attaches_bbox_from_overlapping_blocks() -> None:
    content = "Intro text. Machine learning is powerful."
    start = content.index("Machine learning")
    end = start + len("Machine learning")
    blocks = [{"start": start - 2, "end": end + 5, "bbox": [10.0, 20.0, 200.0, 40.0]}]
    pages = [FakePage(1, content, blocks)]

    matches = search_pages(pages, "machine learning")

    assert matches[0]["bbox"] == [10.0, 20.0, 200.0, 40.0]


def test_search_pages_bbox_is_none_without_overlapping_blocks() -> None:
    pages = [FakePage(1, "machine learning shows up with no block metadata", [])]

    matches = search_pages(pages, "machine learning")

    assert matches[0]["bbox"] is None


def test_bbox_for_range_unions_overlapping_blocks() -> None:
    blocks = [
        {"start": 0, "end": 10, "bbox": [0.0, 0.0, 50.0, 10.0]},
        {"start": 8, "end": 20, "bbox": [40.0, 5.0, 90.0, 15.0]},
    ]

    assert bbox_for_range(blocks, 5, 12) == [0.0, 0.0, 90.0, 15.0]


def test_bbox_for_range_returns_none_when_nothing_overlaps() -> None:
    blocks = [{"start": 0, "end": 5, "bbox": [0.0, 0.0, 10.0, 10.0]}]

    assert bbox_for_range(blocks, 20, 30) is None
