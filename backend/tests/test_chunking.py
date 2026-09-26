from app.services.documents import chunk_text


def test_chunk_text_returns_empty_list_for_blank_input() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_chunk_text_respects_size_limit() -> None:
    text = "sentence one. " * 100
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert len(chunks) > 1


def test_chunk_text_snaps_to_sentence_boundary_when_available() -> None:
    text = "A" * 320 + ". " + "B" * 700
    chunks = chunk_text(text, chunk_size=900, overlap=150)
    assert chunks[0].endswith(".")


def test_consecutive_chunks_overlap() -> None:
    text = "word " * 400
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert chunks[0][-50:] in chunks[1]


def test_whitespace_is_collapsed() -> None:
    chunks = chunk_text("line one\n\nline   two", chunk_size=900, overlap=150)
    assert chunks == ["line one line two"]
