from app.api.v1.chat import grounded_response


def test_parses_valid_json_response() -> None:
    raw = '{"answer":"Overfitting means memorizing noise. [1]","source_ids":[1]}'

    answer, source_ids = grounded_response(raw, source_count=2)

    assert source_ids == [1]
    assert "[1]" in answer


def test_normalizes_word_style_markers_instead_of_double_citing() -> None:
    """Regression test: the model sometimes writes "[source-1]" instead of
    "[1]". The old code only checked for the literal "[1]" before
    force-appending a citation marker, so it appended a second one, producing
    "...[source-1] [1]" in the UI -- which reads as a glitch/hallucination
    even though the underlying fact was correct.
    """
    raw = '{"answer":"I chose top-volume SKUs for a richer signal [source-1]","source_ids":[1]}'

    answer, source_ids = grounded_response(raw, source_count=1)

    assert "[source-1]" not in answer.lower()
    assert answer.count("[1]") == 1
    assert source_ids == [1]


def test_falls_back_to_regex_when_json_is_missing() -> None:
    raw = "The answer is clearly stated here [1] and confirmed again [2]."

    answer, source_ids = grounded_response(raw, source_count=2)

    assert source_ids == [1, 2]


def test_refuses_when_no_valid_sources_are_cited() -> None:
    raw = '{"answer":"Something outside the provided passages.","source_ids":[]}'

    answer, source_ids = grounded_response(raw, source_count=2)

    assert source_ids == []
    assert "couldn't verify" in answer.lower() or "couldn’t verify" in answer.lower()


def test_caps_sources_to_max_sources() -> None:
    raw = '{"answer":"broad summary [1][2][3][4]","source_ids":[1,2,3,4]}'

    answer, source_ids = grounded_response(raw, source_count=4, max_sources=2)

    assert source_ids == [1, 2]


def test_strips_think_block_before_parsing() -> None:
    raw = '<think>internal reasoning here</think>{"answer":"Fact from doc [1]","source_ids":[1]}'

    answer, source_ids = grounded_response(raw, source_count=1)

    assert "internal reasoning" not in answer
    assert source_ids == [1]


def test_drops_source_ids_outside_valid_range() -> None:
    raw = '{"answer":"Answer citing a source that does not exist [9]","source_ids":[9]}'

    answer, source_ids = grounded_response(raw, source_count=2)

    assert source_ids == []
