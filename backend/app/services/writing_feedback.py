import re
from dataclasses import dataclass, field

from app.services.learning import (
    as_text,
    as_text_list,
    cited_ids_in,
    clean_model_output,
    extract_json_object,
    normalize_citation_markers,
    strip_citation_markers,
    valid_source_ids,
)

# Group 6 WRITING EVALUATION. The student's original text is never edited,
# trimmed or re-rendered by this module -- the API layer echoes the request
# field back untouched -- and the model's rewrite lives in its own field.
# The rubric is qualitative on purpose: there is no reliable basis for an
# objective numeric grade of free writing, so none is invented.

RUBRIC_CRITERIA: dict[str, str] = {
    "factual_grounding": "Factual grounding",
    "completeness": "Completeness",
    "clarity": "Clarity",
    "structure": "Structure",
    "terminology": "Terminology",
}
# Only these two depend on having document evidence to compare against.
EVIDENCE_CRITERIA = {"factual_grounding", "completeness"}
RATINGS = ("strong", "adequate", "needs_work", "not_assessed")
_RATING_ALIASES = {
    "strong": "strong", "good": "strong", "excellent": "strong", "high": "strong",
    "adequate": "adequate", "ok": "adequate", "fair": "adequate", "medium": "adequate", "moderate": "adequate",
    "needs_work": "needs_work", "needs work": "needs_work", "weak": "needs_work", "poor": "needs_work", "low": "needs_work",
    "not_assessed": "not_assessed", "n/a": "not_assessed", "not assessed": "not_assessed",
}

WRITING_SYSTEM_MESSAGE = (
    "You are DocuChat, a supportive writing tutor giving feedback on a student's answer using their own document as "
    "the reference. Be specific and constructive, never harsh, and address the student directly as \"you\". Preserve "
    "the student's voice in your suggested version."
)

_SCHEMA = (
    '{"summary":"1-2 sentence overall impression",'
    '"strengths":["..."],'
    '"improvements":["specific, actionable suggestion"],'
    '"missing_points":[{"point":"important point from the passages the answer leaves out","source_ids":[n]}],'
    '"factual_issues":[{"issue":"a claim in the answer the passages contradict, and the correction","source_ids":[n]}],'
    '"rubric":{"factual_grounding":{"rating":"strong|adequate|needs_work","comment":"..."},'
    '"completeness":{...},"clarity":{...},"structure":{...},"terminology":{...}},'
    '"suggested_answer":"an improved version of the answer, citing passages with [n] where it adds document facts"}'
)


def build_writing_prompt(answer: str, context: str, grounding_mode: str, prompt_text: str | None, source_count: int) -> str:
    grounding = (
        "Judge factual accuracy and completeness against the passages first. You may use general knowledge for "
        "anything the passages don't cover, but only passage-based points may carry [n] citations."
        if grounding_mode == "free"
        else "Judge factual accuracy and completeness ONLY against the passages. Do not add facts from outside them -- "
        "neither in the feedback nor in the suggested version."
    )
    question_line = f"QUESTION THE STUDENT WAS ANSWERING: {prompt_text}\n" if prompt_text else ""
    evidence = f"PASSAGES:\n{context}\n\n" if context else "PASSAGES: (none available -- rate factual_grounding and completeness as not_assessed)\n\n"
    return (
        f"{question_line}"
        "Give feedback on the student's answer below, covering factual grounding, missing important points, clarity, "
        f"structure, terminology and completeness. {grounding} Then write a suggested improved version that keeps "
        "the student's own structure and voice where they work.\n"
        f"Return exactly one JSON object, with no prose outside it:\n{_SCHEMA}\n"
        f"source_ids use passage numbers 1 to {max(source_count, 1)}.\n\n"
        f"{evidence}"
        f"STUDENT ANSWER:\n{answer}"
    )


@dataclass
class CitedPoint:
    text: str
    source_ids: list[int]

    def to_dict(self, key: str) -> dict:
        return {key: self.text, "source_ids": self.source_ids}


@dataclass
class RubricEntry:
    criterion: str
    label: str
    rating: str
    comment: str

    def to_dict(self) -> dict:
        return {"criterion": self.criterion, "label": self.label, "rating": self.rating, "comment": self.comment}


@dataclass
class WritingFeedback:
    suggested_answer: str = ""
    summary: str = ""
    strengths: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    missing_points: list[CitedPoint] = field(default_factory=list)
    factual_issues: list[CitedPoint] = field(default_factory=list)
    rubric: list[RubricEntry] = field(default_factory=list)
    # False when the model's output couldn't be parsed as the requested
    # object -- its raw text is still shown as the summary, clearly marked.
    structured: bool = True

    @property
    def source_ids(self) -> list[int]:
        ids: list[int] = []
        for point in [*self.missing_points, *self.factual_issues]:
            ids.extend(source_id for source_id in point.source_ids if source_id not in ids)
        ids.extend(source_id for source_id in cited_ids_in(self.suggested_answer) if source_id not in ids)
        return ids

    def to_dict(self) -> dict:
        return {
            "suggested_answer": self.suggested_answer,
            "summary": self.summary,
            "strengths": self.strengths,
            "improvements": self.improvements,
            "missing_points": [point.to_dict("point") for point in self.missing_points],
            "factual_issues": [point.to_dict("issue") for point in self.factual_issues],
            "rubric": [entry.to_dict() for entry in self.rubric],
            "structured": self.structured,
        }


def _cited_points(value, key: str, source_count: int, has_evidence: bool, require_citation: bool) -> list[CitedPoint]:
    """Missing points / factual issues are claims about what the document
    says, so under Stick to Document one without a verified citation is
    dropped rather than shown as if the document supported it."""
    points: list[CitedPoint] = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, str):
            item = {key: item}
        if not isinstance(item, dict):
            continue
        text = normalize_citation_markers(as_text(item.get(key) or item.get("text"), 600))
        ids = valid_source_ids([*(item.get("source_ids") or []), *cited_ids_in(text)], source_count, 3) if has_evidence else []
        text = strip_citation_markers(text)
        if text and (ids or not require_citation):
            points.append(CitedPoint(text=text, source_ids=ids))
    return points[:8]


def _rating(value) -> str:
    return _RATING_ALIASES.get(str(value or "").strip().lower().replace("-", "_"), "not_assessed")


def parse_writing_feedback(raw: str, source_count: int, grounding_mode: str = "document") -> WritingFeedback:
    has_evidence = source_count > 0
    require_citation = grounding_mode != "free"
    payload = extract_json_object(raw)
    if payload is None:
        text = strip_citation_markers(clean_model_output(raw))
        return WritingFeedback(
            summary=text[:3000] or "The model didn't return any feedback. Please try again.",
            structured=False,
            rubric=[RubricEntry(key, label, "not_assessed", "") for key, label in RUBRIC_CRITERIA.items()],
        )

    suggested = normalize_citation_markers(as_text(payload.get("suggested_answer"), 8000))
    if has_evidence:
        suggested = re.sub(r"\s*\[(\d+)\]", lambda m: m.group(0) if 1 <= int(m.group(1)) <= source_count else "", suggested)
    else:
        suggested = strip_citation_markers(suggested)

    raw_rubric = payload.get("rubric") if isinstance(payload.get("rubric"), dict) else {}
    rubric = []
    for key, label in RUBRIC_CRITERIA.items():
        entry = raw_rubric.get(key)
        rating, comment = (_rating(entry.get("rating")), strip_citation_markers(as_text(entry.get("comment"), 400))) if isinstance(entry, dict) else (_rating(entry), "")
        if key in EVIDENCE_CRITERIA and not has_evidence:
            # No passages were available to check facts against -- a rating
            # here would be a grade without evidence behind it.
            rating, comment = "not_assessed", "No document evidence was available to check this against."
        rubric.append(RubricEntry(key, label, rating, comment))

    return WritingFeedback(
        suggested_answer=suggested.strip(),
        summary=strip_citation_markers(as_text(payload.get("summary"), 1000)),
        strengths=[strip_citation_markers(item) for item in as_text_list(payload.get("strengths"), limit=6)],
        improvements=[strip_citation_markers(item) for item in as_text_list(payload.get("improvements"), limit=8)],
        missing_points=_cited_points(payload.get("missing_points"), "point", source_count, has_evidence, require_citation),
        factual_issues=_cited_points(payload.get("factual_issues"), "issue", source_count, has_evidence, require_citation),
        rubric=rubric,
    )
