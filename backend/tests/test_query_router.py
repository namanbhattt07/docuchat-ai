import pytest

from app.services.query_router import FormatMode, RetrievalMode, route, route_format


@pytest.mark.parametrize(
    "question",
    [
        "What is IoT?",
        "What are the applications of IoT?",
        "Explain sensors.",
        "Why is MQTT used?",
        "What is a neural network?",
        "What is the role of sensors in IoT?",
    ],
)
def test_route_classifies_plain_factual_questions_as_fact(question: str) -> None:
    plan = route(question)
    assert plan.mode == RetrievalMode.FACT
    assert plan.diversity is True


@pytest.mark.parametrize(
    "question",
    [
        "Summarize this document",
        "Give me a brief summary.",
        "Brief me on this PDF.",
        "What is the main idea of this document?",
    ],
)
def test_route_classifies_broad_questions_as_overview(question: str) -> None:
    plan = route(question)
    assert plan.mode == RetrievalMode.OVERVIEW


def test_route_classifies_page_reference_as_page() -> None:
    plan = route("What is discussed on page 10?")
    assert plan.mode == RetrievalMode.PAGE
    assert plan.page == 10
    assert plan.diversity is False


def test_route_classifies_named_section_as_section() -> None:
    plan = route("What does the Sensors section say?", known_sections=["Sensors", "Introduction"])
    assert plan.mode == RetrievalMode.SECTION
    assert plan.section == "Sensors"


def test_route_classifies_chapter_discussion_as_section() -> None:
    plan = route("What does chapter 3 discuss?")
    assert plan.mode == RetrievalMode.SECTION


def test_route_classifies_generic_section_name_without_document_headings() -> None:
    plan = route("What does the introduction say?")
    assert plan.mode == RetrievalMode.SECTION
    assert plan.section == "introduction"


def test_route_prefers_page_over_section_when_both_present() -> None:
    plan = route("What does page 5 of the Sensors section say?")
    assert plan.mode == RetrievalMode.PAGE
    assert plan.page == 5


def test_route_classifies_selected_text_as_selection_regardless_of_wording() -> None:
    plan = route("What is IoT?", selected_text="MQTT is a lightweight publish/subscribe protocol.")
    assert plan.mode == RetrievalMode.SELECTION
    assert plan.selected_text == "MQTT is a lightweight publish/subscribe protocol."


def test_route_classifies_multiple_documents_as_multi_document() -> None:
    plan = route("What are the differences between these two documents?", document_ids=["doc-1", "doc-2"])
    assert plan.mode == RetrievalMode.MULTI_DOCUMENT
    assert plan.document_ids == ["doc-1", "doc-2"]


@pytest.mark.parametrize(
    ("question", "expected_topic"),
    [
        ("Where does the document discuss ZigBee?", "ZigBee"),
        ("Where is TCP/IP explained?", "TCP/IP"),
        ("Which page mentions MQTT?", "MQTT"),
        ("Where does the document talk about sensors?", "sensors"),
        ("Where is the definition of latency?", "latency"),
    ],
)
def test_route_classifies_location_queries_and_extracts_the_topic(question: str, expected_topic: str) -> None:
    plan = route(question)
    assert plan.mode == RetrievalMode.LOCATION
    assert plan.location_topic == expected_topic


def test_route_classifies_extraction_queries() -> None:
    plan = route("Extract the following for each protocol:\n- frequency\n- range")
    assert plan.mode == RetrievalMode.EXTRACTION


def test_route_classifies_extraction_for_comma_field_list() -> None:
    plan = route("Give me the name, year, author and key finding for each paper.")
    assert plan.mode == RetrievalMode.EXTRACTION


@pytest.mark.parametrize(
    "question",
    [
        "What is the percentage increase from 100 to 125?",
        "What is the difference between 80000 and 50000?",
        "What is the average of 10, 20 and 30?",
    ],
)
def test_route_classifies_calculation_queries(question: str) -> None:
    plan = route(question)
    assert plan.mode == RetrievalMode.CALCULATION


def test_route_location_takes_priority_over_generic_section_wording() -> None:
    # "Where is X" is a stronger, more explicit structural signal than the
    # generic "the introduction/conclusion/..." section-name heuristic.
    plan = route("Where is the introduction?")
    assert plan.mode == RetrievalMode.LOCATION


def test_route_page_still_wins_over_location_when_an_explicit_page_number_is_present() -> None:
    plan = route("Where on page 5 does it discuss ZigBee?")
    assert plan.mode == RetrievalMode.PAGE


def test_route_falls_back_to_fact_for_ambiguous_query() -> None:
    plan = route("hmm okay")
    assert plan.mode == RetrievalMode.FACT


def test_route_never_returns_none_document_ids_as_missing_plan() -> None:
    # A genuinely uncertain classification must still retrieve *something*
    # -- never an empty plan/mode.
    plan = route("")
    assert plan.mode == RetrievalMode.FACT


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Explain in bullet points.", FormatMode.BULLETS),
        ("Give me bullet points on this.", FormatMode.BULLETS),
        ("Make a table.", FormatMode.TABLE),
        ("Compare X and Y in a table.", FormatMode.TABLE),
        ("Compare these.", FormatMode.TABLE),
        ("Give me section-wise notes.", FormatMode.SECTIONED),
        ("Explain section by section.", FormatMode.SECTIONED),
        ("Summarize this.", FormatMode.SUMMARY),
        ("Explain briefly.", FormatMode.SUMMARY),
        ("What is IoT?", FormatMode.DEFAULT),
    ],
)
def test_route_format_classifies_presentation_intent(question: str, expected: FormatMode) -> None:
    assert route_format(question) == expected


def test_route_format_is_independent_of_retrieval_mode() -> None:
    plan = route("Give me the Sensors section in bullet points.", known_sections=["Sensors"])
    assert plan.mode == RetrievalMode.SECTION
    assert route_format("Give me the Sensors section in bullet points.") == FormatMode.BULLETS
