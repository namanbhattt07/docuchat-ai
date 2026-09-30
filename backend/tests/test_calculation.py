import pytest

from app.services.calculation import (
    CalculationError,
    CalculationOperation,
    average,
    comparison,
    compute,
    detect_operation,
    difference,
    extract_labeled_values_from_sources,
    extract_two_values_from_question,
    is_calculation_query,
    percentage,
    percentage_decrease,
    percentage_increase,
    ratio,
    total_sum,
)


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "What is the percentage increase from 100 to 125?",
        "What is the percentage decrease in cost?",
        "What is the difference between 80000 and 50000?",
        "What is the ratio of revenue to cost?",
        "What is the average of these values?",
        "Revenue increased from 100 to 125, what is the growth rate?",
    ],
)
def test_is_calculation_query_detects_calculation_intent(question: str) -> None:
    assert is_calculation_query(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "What is IoT?",
        "Explain sensors.",
        "What is the difference between ZigBee and Bluetooth?",  # conceptual, not numeric
        "Compare these protocols.",
    ],
)
def test_is_calculation_query_rejects_non_numeric_conceptual_questions(question: str) -> None:
    assert is_calculation_query(question) is False


def test_is_calculation_query_requires_digit_for_weak_keywords() -> None:
    # "average" alone with no digit and no strong phrase is too ambiguous.
    assert is_calculation_query("What is the average rainfall?") is False
    assert is_calculation_query("What is the average of 10 and 20?") is True


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the percentage increase from 100 to 125?", CalculationOperation.PERCENTAGE_INCREASE),
        ("What is the percentage decrease from 125 to 100?", CalculationOperation.PERCENTAGE_DECREASE),
        ("What percentage of 80000 is 20000?", CalculationOperation.PERCENTAGE),
        ("What is the ratio of 80 to 20?", CalculationOperation.RATIO),
        ("What is the average of 10, 20 and 30?", CalculationOperation.AVERAGE),
        ("What is the sum of 10 and 20?", CalculationOperation.SUM),
        ("Compare 100 and 200.", CalculationOperation.COMPARISON),
        ("What is the difference between 100 and 50?", CalculationOperation.DIFFERENCE),
    ],
)
def test_detect_operation_identifies_the_requested_operation(question: str, expected: CalculationOperation) -> None:
    assert detect_operation(question) == expected


# ---------------------------------------------------------------------------
# Deterministic math
# ---------------------------------------------------------------------------


def test_percentage_increase_basic() -> None:
    result = percentage_increase(100, 125)
    assert result.result == pytest.approx(25.0)


def test_percentage_decrease_basic() -> None:
    result = percentage_decrease(125, 100)
    assert result.result == pytest.approx(20.0)


def test_percentage_increase_raises_on_zero_old_value() -> None:
    with pytest.raises(CalculationError):
        percentage_increase(0, 50)


def test_percentage_decrease_raises_on_zero_old_value() -> None:
    with pytest.raises(CalculationError):
        percentage_decrease(0, 50)


def test_ratio_basic() -> None:
    result = ratio(80, 20)
    assert result.result == pytest.approx(4.0)


def test_ratio_raises_on_zero_denominator() -> None:
    with pytest.raises(CalculationError):
        ratio(10, 0)


def test_percentage_basic() -> None:
    result = percentage(20000, 80000)
    assert result.result == pytest.approx(25.0)


def test_percentage_raises_on_zero_whole() -> None:
    with pytest.raises(CalculationError):
        percentage(10, 0)


def test_difference_basic() -> None:
    result = difference(80000, 50000)
    assert result.result == pytest.approx(30000)


def test_difference_handles_negative_values() -> None:
    result = difference(-10, 5)
    assert result.result == pytest.approx(-15)


def test_average_basic() -> None:
    result = average([10, 20, 30])
    assert result.result == pytest.approx(20.0)


def test_average_handles_decimal_values() -> None:
    result = average([10.5, 20.25])
    assert result.result == pytest.approx(15.375)


def test_average_raises_on_empty_list() -> None:
    with pytest.raises(CalculationError):
        average([])


def test_sum_basic() -> None:
    result = total_sum([10, 20, 30])
    assert result.result == pytest.approx(60)


def test_sum_raises_on_empty_list() -> None:
    with pytest.raises(CalculationError):
        total_sum([])


def test_comparison_reports_signed_difference() -> None:
    result = comparison(100, 60)
    assert result.result == pytest.approx(40)
    assert any("%" in step for step in result.steps)


def test_comparison_handles_equal_values_without_dividing_by_zero() -> None:
    result = comparison(0, 0)
    assert result.result == pytest.approx(0)


# ---------------------------------------------------------------------------
# Value extraction
# ---------------------------------------------------------------------------


def test_extract_two_values_from_question_handles_from_to_phrasing() -> None:
    pair = extract_two_values_from_question("Revenue increased from 100 to 125, what is the percentage increase?")
    assert pair is not None
    old, new = pair
    assert old.value == pytest.approx(100)
    assert new.value == pytest.approx(125)


def test_extract_two_values_from_question_handles_between_phrasing() -> None:
    pair = extract_two_values_from_question("What is the difference between 80000 and 50000?")
    assert pair is not None
    a, b = pair
    assert a.value == pytest.approx(80000)
    assert b.value == pytest.approx(50000)


def test_extract_two_values_from_question_returns_none_when_no_numbers_present() -> None:
    assert extract_two_values_from_question("What is the percentage increase in revenue?") is None


def test_extract_labeled_values_from_sources_finds_labeled_numbers() -> None:
    sources = [{"content": "Revenue was 80000 in Q1. Cost was 50000 in Q1."}]
    values = extract_labeled_values_from_sources("What is the profit margin given revenue and cost?", sources)
    labels = {value.label: value.value for value in values}
    assert labels.get("revenue") == pytest.approx(80000)
    assert labels.get("cost") == pytest.approx(50000)
    assert all(value.source_index == 1 for value in values)


def test_chapter_section_and_page_numbers_are_locators_not_quantities() -> None:
    """Found in the browser on a 300-page report: 'Chapter 122: Crop Rotation' and
    'Section 122.1 covers...' were averaged in as if they were costs."""
    sources = [{"content": "Chapter 122: Crop Rotation. Section 122.1 discusses crop rotation. See page 7. The cost of crop rotation was 854 dollars per hectare."}]

    values = extract_labeled_values_from_sources("What is the average cost for crop rotation?", sources)

    assert [value.value for value in values] == [pytest.approx(854)]


def test_an_average_over_a_pile_of_scattered_numbers_is_refused_with_guidance_not_computed() -> None:
    sources = [{"content": f"The cost of crop rotation was {100 + number} dollars."} for number in range(1, 13)]

    outcome = compute("What is the average cost for crop rotation?", sources)

    assert outcome.success is False and outcome.result is None and outcome.inputs is None
    assert "12 different numbers" in outcome.message
    assert "average of 84 and 91" in outcome.message  # tells the person how to get an answer


def test_a_handful_of_source_values_is_still_averaged() -> None:
    sources = [{"content": "The cost of crop rotation was 100 dollars in one season. The cost was 200 dollars in another."}]

    outcome = compute("What is the average cost for crop rotation?", sources)

    assert outcome.success is True and outcome.result.result == pytest.approx(150)


def test_extract_labeled_values_from_sources_returns_empty_when_nothing_matches() -> None:
    sources = [{"content": "ZigBee operates in the 2.4 GHz band."}]
    values = extract_labeled_values_from_sources("What is the average of revenue and cost?", sources)
    assert values == []


# ---------------------------------------------------------------------------
# compute() orchestration -- the seam api/v1/chat.py actually calls
# ---------------------------------------------------------------------------


def test_compute_succeeds_using_numbers_in_the_question_directly() -> None:
    outcome = compute("What is the percentage increase from 100 to 125?", sources=[])
    assert outcome.success is True
    assert outcome.operation == CalculationOperation.PERCENTAGE_INCREASE
    assert outcome.result.result == pytest.approx(25.0)


def test_compute_falls_back_to_source_passages_when_question_has_no_numbers() -> None:
    sources = [{"content": "Revenue was 80000 last year. Cost was 50000 last year."}]
    outcome = compute("What is the difference between revenue and cost?", sources)
    assert outcome.success is True
    assert outcome.result.result == pytest.approx(30000)


def test_compute_returns_insufficient_evidence_when_no_values_found() -> None:
    sources = [{"content": "ZigBee is a wireless protocol with no numeric specifications mentioned here."}]
    outcome = compute("What is the percentage increase in adoption?", sources)
    assert outcome.success is False
    assert outcome.message is not None


def test_compute_handles_division_by_zero_gracefully() -> None:
    outcome = compute("What is the percentage increase from 0 to 50?", sources=[])
    assert outcome.success is False
    assert "zero" in (outcome.message or "").lower()


def test_compute_handles_missing_second_value() -> None:
    sources: list[dict] = []
    outcome = compute("What is the ratio of revenue to cost?", sources)
    assert outcome.success is False


def test_compute_average_from_question_numbers() -> None:
    outcome = compute("What is the average of 10, 20 and 30?", sources=[])
    assert outcome.success is True
    assert outcome.result.result == pytest.approx(20.0)


def test_compute_rejects_incompatible_units() -> None:
    outcome = compute("What is the ratio between 10 kg and 5 m?", sources=[])
    assert outcome.success is False
    assert "unit" in (outcome.message or "").lower()


def test_extract_two_values_from_question_captures_units() -> None:
    pair = extract_two_values_from_question("What is the ratio between 10 kg and 5 m?")
    assert pair is not None
    a, b = pair
    assert a.unit == "kg"
    assert b.unit == "m"
