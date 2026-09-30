from app.services.documents import PageBlock, build_toc


def test_build_toc_prefers_native_bookmarks() -> None:
    native = [[1, "Chapter One", 1], [2, "Section 1.1", 2]]

    toc = build_toc(native, heading_candidates=[(1, "Ignored heuristic heading", 24.0)])

    assert [item["source"] for item in toc] == ["native", "native"]
    assert toc[0] == {"id": "native-0", "title": "Chapter One", "page": 1, "level": 1, "source": "native"}
    assert toc[1]["level"] == 2


def test_build_toc_skips_blank_native_titles_and_invalid_pages() -> None:
    native = [[1, "  ", 1], [1, "Real Title", 0], [1, "Good Entry", 3]]

    toc = build_toc(native, heading_candidates=[])

    assert len(toc) == 1
    assert toc[0]["title"] == "Good Entry"


def test_build_toc_falls_back_to_heuristic_when_no_native_toc() -> None:
    candidates = [(1, "Introduction", 24.0), (2, "Background", 24.0), (3, "Method", 18.0)]

    toc = build_toc(native_toc=[], heading_candidates=candidates)

    assert [item["source"] for item in toc] == ["heuristic", "heuristic", "heuristic"]
    assert toc[0]["level"] == 1
    assert toc[1]["level"] == 1
    assert toc[2]["level"] == 2  # smaller font size than the level-1 headings


def test_build_toc_collapses_repeated_running_headers() -> None:
    candidates = [(1, "Chapter 3", 20.0), (2, "Chapter 3", 20.0), (3, "Chapter 3", 20.0), (4, "Chapter 4", 20.0)]

    toc = build_toc(native_toc=[], heading_candidates=candidates)

    assert [item["title"] for item in toc] == ["Chapter 3", "Chapter 4"]
    assert toc[0]["page"] == 1


def test_build_toc_filters_noise_headings_like_folio_numbers() -> None:
    candidates = [(1, "12", 24.0), (1, "iv", 24.0), (2, "Real Heading", 24.0)]

    toc = build_toc(native_toc=[], heading_candidates=candidates)

    assert [item["title"] for item in toc] == ["Real Heading"]


def test_build_toc_returns_empty_list_when_nothing_usable() -> None:
    assert build_toc(native_toc=[], heading_candidates=[]) == []


def test_extract_page_blocks_records_font_size_for_toc_levels() -> None:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Big Heading", fontsize=26)
    page.insert_text((72, 110), "Smaller body copy that is definitely not a heading at all.", fontsize=10)

    from app.services.documents import extract_page_blocks

    _, blocks = extract_page_blocks(page)
    doc.close()

    heading = next(block for block in blocks if block.is_heading)
    assert heading.font_size >= 26

    # Default for hand-built PageBlock instances (existing test fixtures)
    # stays 0.0 so pre-Group-2 test literals keep working unmodified.
    assert PageBlock(text="x", bbox=(0, 0, 1, 1), start=0, end=1, is_heading=False).font_size == 0.0
