import re
from dataclasses import dataclass
from enum import Enum

from app.services.retrieval import terms

# Group 4's deterministic calculation engine. The one rule this module exists
# to enforce: an LLM must never be trusted to do arithmetic (see FORMULA &
# DATA-ANALYSIS SUPPORT / CRITICAL RULE). Every numeric operation below is
# plain Python math -- api/v1/chat.py never sends numbers to Ollama and asks
# it to compute anything; at most it asks the model to *phrase* a result this
# module already computed. That also makes every operation here testable
# without a live model.


class CalculationOperation(str, Enum):
    DIFFERENCE = "difference"
    RATIO = "ratio"
    PERCENTAGE = "percentage"
    PERCENTAGE_INCREASE = "percentage_increase"
    PERCENTAGE_DECREASE = "percentage_decrease"
    AVERAGE = "average"
    COMPARISON = "comparison"
    SUM = "sum"


class CalculationError(Exception):
    """Raised for a well-formed but mathematically invalid request (division
    by zero, an empty average, incompatible units) -- always caught by the
    caller and turned into a grounded "insufficient evidence" response,
    never allowed to bubble up as a 500.
    """


# Checked before the weak pattern below: these phrases imply a computation is
# wanted regardless of whether the question itself contains any digits (the
# values usually live in the retrieved document passages, not the question).
_STRONG_CALC_PATTERN = re.compile(
    r"percentage\s+(increase|decrease)|percent\s+(increase|decrease)|"
    r"\bgrowth\s+rate\b|\bratio\s+of\b|\baverage\s+of\b|"
    r"\bhow\s+much\s+(more|less|higher|lower)\b|\bpercent(age)?\s+change\b",
    re.IGNORECASE,
)
# A bare calculation word only counts as calculation intent when the question
# also contains a digit -- otherwise "the difference between IoT and MQTT" (a
# purely conceptual, non-numeric question) would wrongly be routed here.
_WEAK_CALC_PATTERN = re.compile(
    r"\b(percentage|percent|ratio|average|difference|sum|total|"
    r"increase[ds]?|decrease[ds]?)\b",
    re.IGNORECASE,
)
_HAS_DIGIT = re.compile(r"\d")
# A weak keyword with no digit in the question is still calculation intent
# when it also names a measurable quantity -- "the difference between
# revenue and cost" should compute (the numbers live in the document), while
# "the difference between ZigBee and Bluetooth" should not (there is nothing
# to subtract; it wants a normal explanatory FACT answer). Named entities
# like "ZigBee" simply never appear in this vocabulary, so the check stays a
# cheap set-intersection rather than any real entity typing.
_QUANTITY_NOUNS = {
    "revenue", "cost", "costs", "price", "prices", "sales", "profit", "margin",
    "amount", "count", "weight", "distance", "speed", "rate", "score", "value",
    "values", "size", "temperature", "age", "salary", "budget", "expense",
    "expenses", "income", "population", "rating", "quantity", "number",
    "duration", "length", "width", "height", "volume", "capacity",
    "throughput", "latency", "bandwidth", "frequency", "voltage", "current",
    "power", "efficiency", "accuracy", "growth", "loss", "gain", "return",
}

_OPERATION_PATTERNS: list[tuple[CalculationOperation, re.Pattern]] = [
    (CalculationOperation.PERCENTAGE_INCREASE, re.compile(r"percentage\s+increase|percent\s+increase|\bincrease[ds]?\b", re.IGNORECASE)),
    (CalculationOperation.PERCENTAGE_DECREASE, re.compile(r"percentage\s+decrease|percent\s+decrease|\bdecrease[ds]?\b", re.IGNORECASE)),
    (CalculationOperation.PERCENTAGE, re.compile(r"\bpercentage\b|\bpercent\b|%", re.IGNORECASE)),
    (CalculationOperation.RATIO, re.compile(r"\bratio\b", re.IGNORECASE)),
    (CalculationOperation.AVERAGE, re.compile(r"\baverage\b|\bmean\b", re.IGNORECASE)),
    (CalculationOperation.SUM, re.compile(r"\bsum\b|\btotal\b", re.IGNORECASE)),
    # DIFFERENCE is checked before the COMPARISON catch-all -- "the
    # difference between X and Y" should compute a plain subtraction, not
    # the richer comparison (larger/smaller + relative %) COMPARISON gives.
    (CalculationOperation.DIFFERENCE, re.compile(r"\bdifference\b", re.IGNORECASE)),
    (CalculationOperation.COMPARISON, re.compile(r"\bcompare\b|\bcomparison\b|\bhow\s+much\s+(more|less|higher|lower)\b", re.IGNORECASE)),
]


def is_calculation_query(question: str) -> bool:
    if _STRONG_CALC_PATTERN.search(question):
        return True
    if not _WEAK_CALC_PATTERN.search(question):
        return False
    if _HAS_DIGIT.search(question):
        return True
    return bool(set(terms(question)) & _QUANTITY_NOUNS)


def detect_operation(question: str) -> CalculationOperation:
    """Which operation the question is asking for. Checked most-specific
    first (percentage increase/decrease before bare percentage, which is
    before the generic difference/comparison catch-all) so "percentage
    increase" is never misread as a plain difference.
    """
    for operation, pattern in _OPERATION_PATTERNS:
        if pattern.search(question):
            return operation
    return CalculationOperation.DIFFERENCE


# ---------------------------------------------------------------------------
# Deterministic math. Every function below raises CalculationError instead of
# ever returning a silently-wrong number (a ZeroDivisionError, a NaN, etc.).
# ---------------------------------------------------------------------------


@dataclass
class CalculationResult:
    operation: CalculationOperation
    result: float
    steps: list[str]


def difference(a: float, b: float) -> CalculationResult:
    result = a - b
    return CalculationResult(CalculationOperation.DIFFERENCE, result, [f"{a:g} - {b:g} = {result:g}"])


def ratio(a: float, b: float) -> CalculationResult:
    if b == 0:
        raise CalculationError("Cannot compute a ratio with a denominator of zero.")
    result = a / b
    return CalculationResult(CalculationOperation.RATIO, result, [f"{a:g} / {b:g} = {result:g}"])


def percentage(part: float, whole: float) -> CalculationResult:
    if whole == 0:
        raise CalculationError("Cannot compute a percentage of a whole that is zero.")
    result = part / whole * 100
    return CalculationResult(CalculationOperation.PERCENTAGE, result, [f"({part:g} / {whole:g}) * 100 = {result:g}%"])


def percentage_increase(old: float, new: float) -> CalculationResult:
    if old == 0:
        raise CalculationError("Cannot compute a percentage increase from a starting value of zero.")
    result = (new - old) / old * 100
    return CalculationResult(CalculationOperation.PERCENTAGE_INCREASE, result, [f"(({new:g} - {old:g}) / {old:g}) * 100 = {result:g}%"])


def percentage_decrease(old: float, new: float) -> CalculationResult:
    if old == 0:
        raise CalculationError("Cannot compute a percentage decrease from a starting value of zero.")
    result = (old - new) / old * 100
    return CalculationResult(CalculationOperation.PERCENTAGE_DECREASE, result, [f"(({old:g} - {new:g}) / {old:g}) * 100 = {result:g}%"])


def average(values: list[float]) -> CalculationResult:
    if not values:
        raise CalculationError("Cannot compute an average of zero values.")
    result = sum(values) / len(values)
    joined = " + ".join(f"{value:g}" for value in values)
    return CalculationResult(CalculationOperation.AVERAGE, result, [f"({joined}) / {len(values)} = {result:g}"])


def total_sum(values: list[float]) -> CalculationResult:
    if not values:
        raise CalculationError("Cannot compute a sum of zero values.")
    result = sum(values)
    joined = " + ".join(f"{value:g}" for value in values)
    return CalculationResult(CalculationOperation.SUM, result, [f"{joined} = {result:g}"])


def comparison(a: float, b: float) -> CalculationResult:
    """Which of the two values is larger, and by how much/what percentage.
    `result` is the signed difference (a - b) so callers can tell direction;
    the percentage-of-the-smaller-value figure lives in `steps` since it has
    no single natural "the result" slot the way the other operations do.
    """
    diff = a - b
    steps = [f"{a:g} vs {b:g} -> difference = {diff:g}"]
    smaller = min(abs(a), abs(b))
    if smaller != 0:
        pct = abs(diff) / smaller * 100
        steps.append(f"|{diff:g}| / {smaller:g} * 100 = {pct:g}% relative difference")
    return CalculationResult(CalculationOperation.COMPARISON, diff, steps)


# ---------------------------------------------------------------------------
# Value extraction -- regex-based, never an LLM. Two sources of numbers are
# tried in order: the question itself (the user already gave the numbers,
# e.g. "difference between 80000 and 50000"), then the retrieved document
# passages (the more common case -- the numbers live in the PDF).
# ---------------------------------------------------------------------------


@dataclass
class ExtractedValue:
    value: float
    label: str
    unit: str | None
    source_index: int | None  # 1-based index into the sources list, or None when parsed straight from the question
    raw_text: str


_NUMBER_TOKEN = r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?"
_UNIT_WORDS = r"(?:kg|g|km|m|cm|mm|mi|ft|GHz|MHz|Kbps|Mbps|Gbps|USD|INR|EUR|%|hours?|hrs?|days?|years?)"
# The negative lookbehind keeps this from matching the "1" inside a token
# like "Q1" or "v2" as if it were a standalone numeric value.
_NUMBER_WITH_UNIT_PATTERN = re.compile(rf"(?<![A-Za-z0-9])({_NUMBER_TOKEN})\s?({_UNIT_WORDS})?", re.IGNORECASE)

# Deliberately excludes generic calculation vocabulary ("percentage",
# "average", ...) and stop words -- those describe the *operation*, not the
# thing being measured, so they would only add noise to label matching.
_CALC_VOCAB = {
    "percentage", "percent", "ratio", "average", "mean", "difference", "sum",
    "total", "increase", "increased", "decrease", "decreased", "compare",
    "comparison", "much", "more", "less", "higher", "lower", "growth", "rate",
    "change",
    # connectives of the calculation phrasing itself ("difference between X and
    # Y", "from X to Y"), not things being measured
    "between", "from", "versus", "compared", "than",
}


_YEAR_LIKE = re.compile(r"^(?:19|20)\d{2}$")


# "Chapter 122", "Section 122.1", "Page 7", "Figure 3": document structure, not quantities.
_LOCATOR_BEFORE = re.compile(r"\b(?:chapter|section|page|pages|figure|fig|table|step|part|item|appendix|no)\.?\s*$", re.IGNORECASE)

# An average/sum whose operands the question does not name is gathered from the
# retrieved passages; past this many it is no longer "the" figure the person
# meant but a pile of unrelated numbers from all over the document.
_MAX_SOURCE_OPERANDS = 10


def _is_year_like(raw_number: str, unit: str | None) -> bool:
    """A bare four-digit number in the calendar-year range ("2022"). In prose or
    a question it names *when*, not *how much* -- treating it as an operand made
    "percentage increase in revenue from 2022 to 2023" compute 0.049% and
    "total cost in 2022 and 2023" answer 4045. Anything carrying a currency
    sign, percent sign, unit, comma or decimal is a quantity, not a year."""
    return not unit and bool(_YEAR_LIKE.match(raw_number.strip()))


def _clean_number(raw: str) -> float | None:
    cleaned = raw.replace(",", "").replace("$", "").replace("%", "").strip()
    if not cleaned or cleaned in {"-", "+", "."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _unit_of(raw_number: str, unit: str | None) -> str | None:
    if unit:
        return unit.lower()
    if "%" in raw_number:
        return "%"
    if "$" in raw_number:
        return "usd"
    return None


# "from X to Y" (percentage change phrasing), "between X and Y", "X and Y" /
# "X vs Y" (comparison/ratio/difference phrasing) -- covers the phrasings
# used throughout the Group 4 brief's own examples. Each number optionally
# captures a trailing unit word (e.g. "10 kg") so a direct "between 10 kg and
# 5 m" style question still carries its units into the compatibility check,
# not just numbers extracted from retrieved document passages.
_NUM_UNIT = rf"(?<![A-Za-z0-9])({_NUMBER_TOKEN})\s?({_UNIT_WORDS})?"
_FROM_TO_PATTERN = re.compile(rf"from\s+{_NUM_UNIT}\s+to\s+{_NUM_UNIT}", re.IGNORECASE)
_BETWEEN_PATTERN = re.compile(rf"between\s+{_NUM_UNIT}\s+and\s+{_NUM_UNIT}", re.IGNORECASE)
_AND_OR_VS_PATTERN = re.compile(rf"{_NUM_UNIT}\s*(?:,|and|vs\.?|versus)\s*{_NUM_UNIT}", re.IGNORECASE)


def extract_two_values_from_question(question: str) -> tuple[ExtractedValue, ExtractedValue] | None:
    """Direct extraction when the user already stated the numbers in the
    question itself -- no retrieval/document lookup needed. Tries the most
    semantically specific pattern first ("from X to Y" implies old->new
    order, which matters for percentage increase/decrease).
    """
    for pattern, labels in (
        (_FROM_TO_PATTERN, ("old value", "new value")),
        (_BETWEEN_PATTERN, ("first value", "second value")),
        (_AND_OR_VS_PATTERN, ("first value", "second value")),
    ):
        match = pattern.search(question)
        if not match:
            continue
        raw_a, unit_a, raw_b, unit_b = match.group(1), match.group(2), match.group(3), match.group(4)
        value_a, value_b = _clean_number(raw_a), _clean_number(raw_b)
        if value_a is None or value_b is None:
            continue
        if _is_year_like(raw_a, unit_a) and _is_year_like(raw_b, unit_b) and _label_terms(question):
            # "revenue from 2022 to 2023": the years frame the question; the
            # amounts to compare are in the document.
            continue
        return (
            ExtractedValue(value_a, labels[0], _unit_of(raw_a, unit_a), None, raw_a),
            ExtractedValue(value_b, labels[1], _unit_of(raw_b, unit_b), None, raw_b),
        )
    return None


def extract_all_values_from_question(question: str) -> list[ExtractedValue]:
    """Every number literally present in the question -- used for AVERAGE/SUM
    when the user lists the values directly (e.g. "average of 10, 20, 30").
    """
    values: list[ExtractedValue] = []
    names_a_quantity = bool(_label_terms(question))
    for match in _NUMBER_WITH_UNIT_PATTERN.finditer(question):
        raw_number, unit = match.group(1), match.group(2)
        value = _clean_number(raw_number)
        if value is None or (names_a_quantity and _is_year_like(raw_number, unit)):
            continue
        values.append(ExtractedValue(value, f"value {len(values) + 1}", _unit_of(raw_number, unit), None, raw_number))
    return values


def _label_terms(question: str) -> list[str]:
    # Words only: "2022" says when, not what is being measured, and letting it
    # act as a label made every line mentioning that year look relevant.
    return [term for term in terms(question) if term not in _CALC_VOCAB and not term.isdigit()]


def extract_labeled_values_from_sources(question: str, sources: list[dict]) -> list[ExtractedValue]:
    """Scan the retrieved passages for numbers near one of the question's own
    key nouns (e.g. "revenue", "cost") -- the label is whichever of those
    terms appears closest before the number on the same line, so the two
    values a comparison needs come back distinguishable rather than as two
    anonymous numbers.
    """
    label_terms = _label_terms(question)
    if not label_terms:
        return []

    found: list[ExtractedValue] = []
    for source_index, source in enumerate(sources, start=1):
        content = source.get("content", "")
        for line in re.split(r"(?<=[.!?])\s+|\n", content):
            lowered = line.lower()
            matching_terms = [term for term in label_terms if term in lowered]
            if not matching_terms:
                continue
            for match in _NUMBER_WITH_UNIT_PATTERN.finditer(line):
                raw_number, unit = match.group(1), match.group(2)
                value = _clean_number(raw_number)
                if value is None or _is_year_like(raw_number, unit) or _LOCATOR_BEFORE.search(line[:match.start()]):
                    continue
                # The label term whose occurrence is closest to (and before)
                # this number's position in the line -- the most reliable
                # cheap heuristic for "which noun does this number belong to."
                preceding = [term for term in matching_terms if lowered.rfind(term, 0, match.start()) != -1]
                label = preceding[-1] if preceding else matching_terms[0]
                found.append(ExtractedValue(value, label, _unit_of(raw_number, unit), source_index, match.group(0).strip()))
    return found


def find_paired_values_in_same_sentence(question: str, sources: list[dict]) -> tuple[ExtractedValue, ExtractedValue] | None:
    """A much stronger signal than extract_labeled_values_from_sources' flat
    list: two numbers that appear together in the *same* sentence as one of
    the question's key nouns (e.g. "Data rates of 20 kbps and up to 250
    kbps") are almost certainly the pair the question is actually asking
    about, whereas the first two label-matching numbers found anywhere
    across several retrieved passages can accidentally pair unrelated
    figures from two different sentences. Tried before that flatter fallback
    in compute() for exactly this reason.
    """
    label_terms = _label_terms(question)
    if not label_terms:
        return None

    for source_index, source in enumerate(sources, start=1):
        content = source.get("content", "")
        for line in re.split(r"(?<=[.!?])\s+|\n", content):
            lowered = line.lower()
            if not any(term in lowered for term in label_terms):
                continue
            matches = list(_NUMBER_WITH_UNIT_PATTERN.finditer(line))
            if len(matches) < 2:
                continue
            values = [(m, _clean_number(m.group(1))) for m in matches]
            values = [(m, v) for m, v in values if v is not None and not _is_year_like(m.group(1), m.group(2))]
            if len(values) < 2:
                continue
            # Prefer numbers that carry a unit (e.g. "20 kbps") over bare
            # integers -- a sentence can easily contain unrelated bare
            # numbers (a standard's version number, a list/group number)
            # alongside the real quantity, and a unit is the cheapest signal
            # that a number is an actual measured value rather than an
            # identifier.
            with_unit = [(m, v) for m, v in values if _unit_of(m.group(1), m.group(2))]
            if len(with_unit) >= 2:
                values = with_unit
            (match_a, value_a), (match_b, value_b) = values[0], values[1]
            return (
                ExtractedValue(value_a, "first value", _unit_of(match_a.group(1), match_a.group(2)), source_index, match_a.group(0).strip()),
                ExtractedValue(value_b, "second value", _unit_of(match_b.group(1), match_b.group(2)), source_index, match_b.group(0).strip()),
            )
    return None


_INCOMPATIBLE_UNIT_MESSAGE = "Cannot combine values with incompatible units ({0} and {1}) without an explicit conversion."


def _check_unit_compatibility(a: ExtractedValue, b: ExtractedValue) -> None:
    if a.unit and b.unit and a.unit != b.unit:
        raise CalculationError(_INCOMPATIBLE_UNIT_MESSAGE.format(a.unit, b.unit))


@dataclass
class CalculationOutcome:
    success: bool
    operation: CalculationOperation | None = None
    inputs: list[ExtractedValue] | None = None
    result: CalculationResult | None = None
    message: str | None = None


def _pair_across_labels(labeled: list[ExtractedValue]) -> tuple[ExtractedValue, ExtractedValue] | None:
    """The first value of each of two different labels, or None when the values
    found all belong to one quantity."""
    if not labeled:
        return None
    first = labeled[0]
    other = next((value for value in labeled[1:] if value.label != first.label), None)
    return (first, other) if other else None


def _pair_from_labeled(labeled: list[ExtractedValue]) -> tuple[ExtractedValue, ExtractedValue] | None:
    """The first two labelled numbers -- the last resort, used when everything
    found belongs to a single quantity."""
    return (labeled[0], labeled[1]) if len(labeled) >= 2 else None


def compute(question: str, sources: list[dict]) -> CalculationOutcome:
    """Orchestrates detection -> extraction -> deterministic math for a
    CALCULATION-mode chat request. Never raises -- every failure mode
    (insufficient evidence, incompatible units, division by zero) comes back
    as `success=False` with a human-readable `message` so the caller can
    return a grounded refusal instead of a 500.
    """
    operation = detect_operation(question)

    if operation in (CalculationOperation.AVERAGE, CalculationOperation.SUM):
        values = extract_all_values_from_question(question)
        if len(values) < 2:
            values = extract_labeled_values_from_sources(question, sources)
            if len(values) > _MAX_SOURCE_OPERANDS:
                return CalculationOutcome(success=False, message=(
                    f"That would mean combining {len(values)} different numbers from across the document, which isn't one well-defined figure. "
                    'Give me the values to use (for example "the average of 84 and 91"), or ask for the figure directly '
                    '(for example "What does the document say the cost was in season 12?").'
                ))
        if len(values) < 2:
            return CalculationOutcome(success=False, message="I couldn't find at least two numeric values in the document to compute that.")
        try:
            plain_values = [value.value for value in values]
            result = average(plain_values) if operation == CalculationOperation.AVERAGE else total_sum(plain_values)
        except CalculationError as exc:
            return CalculationOutcome(success=False, message=str(exc))
        return CalculationOutcome(success=True, operation=operation, inputs=values, result=result)

    pair = extract_two_values_from_question(question)
    if pair is None:
        labeled = extract_labeled_values_from_sources(question, sources)
        # A question naming two different quantities ("revenue" vs "cost") wants
        # one value of each. That must win over pairing two numbers from one
        # sentence, which for such a question is two values of the *same*
        # quantity (e.g. this year's and last year's revenue).
        pair = _pair_across_labels(labeled) or find_paired_values_in_same_sentence(question, sources) or _pair_from_labeled(labeled)
    if pair is None:
        return CalculationOutcome(success=False, message="I couldn't find the numeric values this calculation needs in the selected document.")

    value_a, value_b = pair
    try:
        _check_unit_compatibility(value_a, value_b)
        if operation == CalculationOperation.PERCENTAGE_INCREASE:
            result = percentage_increase(value_a.value, value_b.value)
        elif operation == CalculationOperation.PERCENTAGE_DECREASE:
            result = percentage_decrease(value_a.value, value_b.value)
        elif operation == CalculationOperation.PERCENTAGE:
            result = percentage(value_a.value, value_b.value)
        elif operation == CalculationOperation.RATIO:
            result = ratio(value_a.value, value_b.value)
        elif operation == CalculationOperation.COMPARISON:
            result = comparison(value_a.value, value_b.value)
        else:
            result = difference(value_a.value, value_b.value)
    except CalculationError as exc:
        return CalculationOutcome(success=False, message=str(exc))

    return CalculationOutcome(success=True, operation=operation, inputs=[value_a, value_b], result=result)
