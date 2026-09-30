import fitz

from app.services.documents import PageBlock, chunk_page, extract_page_blocks


def test_chunk_page_empty_text_returns_empty_list() -> None:
    assert chunk_page("", [], chunk_size=900, overlap=150) == []


def test_chunk_page_unions_bbox_of_overlapping_blocks() -> None:
    text = "Heading text. Body paragraph continues here."
    blocks = [
        PageBlock(text="Heading text.", bbox=(10.0, 10.0, 100.0, 20.0), start=0, end=13, is_heading=True),
        PageBlock(text="Body paragraph continues here.", bbox=(10.0, 25.0, 150.0, 40.0), start=14, end=45, is_heading=False),
    ]

    chunks = chunk_page(text, blocks, chunk_size=900, overlap=150)

    assert len(chunks) == 1
    assert chunks[0]["bbox"] == [10.0, 10.0, 150.0, 40.0]
    assert chunks[0]["start_offset"] == 0
    assert chunks[0]["end_offset"] == len(text)


def test_chunk_page_attaches_nearest_preceding_heading() -> None:
    text = "Intro. Section One. Detail about the first section."
    blocks = [
        PageBlock(text="Intro.", bbox=(0, 0, 10, 10), start=0, end=6, is_heading=False),
        PageBlock(text="Section One.", bbox=(0, 12, 10, 22), start=7, end=19, is_heading=True),
        PageBlock(text="Detail about the first section.", bbox=(0, 24, 10, 34), start=20, end=52, is_heading=False),
    ]

    chunks = chunk_page(text, blocks, chunk_size=900, overlap=150)

    assert chunks[0]["section"] == "Section One."


def test_chunk_page_splits_long_text_with_offsets_advancing() -> None:
    text = "sentence one. " * 100
    blocks = [PageBlock(text=text.strip(), bbox=(0, 0, 10, 10), start=0, end=len(text), is_heading=False)]

    chunks = chunk_page(text, blocks, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert all(chunk["end_offset"] > chunk["start_offset"] for chunk in chunks)
    assert chunks[1]["start_offset"] < chunks[0]["end_offset"]  # overlap holds


def test_extract_page_blocks_detects_heading_by_font_size() -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Chapter One", fontsize=24)
    page.insert_text((72, 110), "This is a regular paragraph of body text describing something in detail.", fontsize=10)

    page_text, blocks = extract_page_blocks(page)
    doc.close()

    assert "Chapter One" in page_text
    heading_texts = [block.text for block in blocks if block.is_heading]
    assert any("Chapter One" in text for text in heading_texts)
    assert all(block.bbox is not None for block in blocks)


def test_extract_page_blocks_empty_page_returns_no_blocks() -> None:
    doc = fitz.open()
    page = doc.new_page()

    page_text, blocks = extract_page_blocks(page)
    doc.close()

    assert page_text == ""
    assert blocks == []
