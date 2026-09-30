from dataclasses import dataclass, field

from app.services.learning import (
    as_text,
    cited_ids_in,
    extract_json_object,
    normalize_citation_markers,
    strip_citation_markers,
    valid_source_ids,
)

# Group 6 BRAINSTORMING MODE. The one rule this module exists to enforce:
# generated ideas must never be presented as if they came from the document.
# The model is asked for two separate lists, and the parser (not the prompt)
# guarantees the split -- an "evidence" point without a verified citation is
# dropped, and [n] markers are always stripped out of generated ideas.

IDEA_KINDS = ("project", "application", "research_question", "hypothesis", "extension", "idea")
MAX_EVIDENCE_POINTS = 8
MAX_IDEAS = 8

BRAINSTORM_PERSONA = (
    "You are DocuChat, acting as a creative brainstorming partner for the user's document. You first establish what "
    "the document actually says, then generate new ideas from it. You keep these strictly separate: facts from the "
    "document are cited, while ideas you generate are clearly your own and are never cited as if the document said them."
)

_STICK_RULE = (
    "GROUNDING: Stick to Document. \"evidence\" may only contain points the passages state, each with its "
    "source_ids. Every idea must build on that evidence (list the evidence source ids it builds on in \"builds_on\") "
    "and must not rely on outside facts. If the passages give you nothing to build on, return empty lists."
)
_FREE_RULE = (
    "GROUNDING: Go Freely. \"evidence\" may only contain points the passages state, each with its source_ids. Ideas "
    "may also draw on your general knowledge; use \"builds_on\" only when an idea extends a specific evidence point."
)

_BRAINSTORM_SCHEMA = (
    '{"evidence":[{"point":"a fact or finding the passages state","source_ids":[source-number,...]}],'
    '"ideas":[{"title":"short idea name","description":"one or two sentences","kind":"project | application | '
    'research_question | hypothesis | extension","builds_on":[source-number,...]}]}'
)

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I couldn't find enough document evidence to brainstorm from. Try a more specific request, select a passage in "
    "the PDF, or switch to Go Freely to allow ideas from general knowledge."
)
FREE_NO_EVIDENCE_NOTICE = "The selected documents don't directly support these ideas — they come from general knowledge."


def build_brainstorm_system_message(grounding_mode: str) -> str:
    return f"{BRAINSTORM_PERSONA} {_FREE_RULE if grounding_mode == 'free' else _STICK_RULE}"


def build_brainstorm_prompt(request_text: str, context: str, history: str, max_sources: int, idea_count: int = 5, focus: str | None = None) -> str:
    focus_line = f"FOCUS: {focus}\n" if focus else ""
    return (
        f"{history}"
        f"{focus_line}"
        f"TASK: Brainstorm in response to the user's request. First list 2-{MAX_EVIDENCE_POINTS} key points from the "
        f"passages that are relevant to it (\"evidence\"), then generate about {idea_count} distinct, concrete ideas "
        "(\"ideas\") -- projects, applications, open research questions, hypotheses or extensions, whichever the "
        "request asks for. Ideas must be your own suggestions, not restatements of the evidence. Never put a [n] "
        "marker inside an idea's text.\n"
        f"Return exactly one JSON object, with no prose outside it:\n{_BRAINSTORM_SCHEMA}\n"
        f"source_ids and builds_on use source numbers 1 to {max_sources}.\n\n"
        f"PASSAGES:\n{context}\n\n"
        f"USER REQUEST: {request_text}"
    )


@dataclass
class EvidencePoint:
    point: str
    source_ids: list[int]

    def to_dict(self) -> dict:
        return {"point": self.point, "source_ids": self.source_ids}


@dataclass
class Idea:
    title: str
    description: str
    kind: str
    builds_on: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"title": self.title, "description": self.description, "kind": self.kind, "builds_on": self.builds_on}


@dataclass
class BrainstormResult:
    evidence: list[EvidencePoint] = field(default_factory=list)
    ideas: list[Idea] = field(default_factory=list)
    notice: str = ""
    insufficient_evidence: bool = False
    grounding_mode: str = "document"

    @property
    def source_ids(self) -> list[int]:
        """Every source id shown anywhere in the result -- the citations the
        API layer builds (evidence first, then anything an idea builds on)."""
        ids: list[int] = []
        for point in self.evidence:
            ids.extend(source_id for source_id in point.source_ids if source_id not in ids)
        for idea in self.ideas:
            ids.extend(source_id for source_id in idea.builds_on if source_id not in ids)
        return ids

    def to_markdown(self) -> str:
        blocks = []
        if self.evidence:
            lines = [f"- {point.point} {' '.join(f'[{i}]' for i in point.source_ids)}".rstrip() for point in self.evidence]
            blocks.append("### Document evidence\n\n_Points the document actually states._\n\n" + "\n".join(lines))
        if self.ideas:
            lines = []
            for idea in self.ideas:
                builds = f" _(builds on {', '.join(f'[{i}]' for i in idea.builds_on)})_" if idea.builds_on else ""
                lines.append(f"- **{idea.title}** — {idea.description}{builds}")
            blocks.append("### Generated ideas\n\n_Suggestions generated by the assistant — not claims made by the document._\n\n" + "\n".join(lines))
        if self.notice:
            blocks.append(self.notice)
        return "\n\n".join(blocks)

    def metadata(self) -> dict:
        return {
            "evidence": [point.to_dict() for point in self.evidence],
            "ideas": [idea.to_dict() for idea in self.ideas],
            "notice": self.notice,
            "insufficient_evidence": self.insufficient_evidence,
            "grounding_mode": self.grounding_mode,
        }


def _normalize_kind(value) -> str:
    kind = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return kind if kind in IDEA_KINDS else "idea"


def parse_brainstorm_response(raw: str, source_count: int, grounding_mode: str) -> BrainstormResult:
    free = grounding_mode == "free"
    payload = extract_json_object(raw) or {}
    raw_evidence = payload.get("evidence") if isinstance(payload.get("evidence"), list) else []
    raw_ideas = payload.get("ideas") if isinstance(payload.get("ideas"), list) else []

    evidence: list[EvidencePoint] = []
    for item in raw_evidence:
        if isinstance(item, str):
            item = {"point": item}
        if not isinstance(item, dict):
            continue
        text = normalize_citation_markers(as_text(item.get("point"), 600))
        ids = valid_source_ids([*(item.get("source_ids") or []), *cited_ids_in(text)], source_count)
        text = strip_citation_markers(text)
        # Evidence with no verified passage behind it is exactly the thing
        # that would blur "document says" with "model says" -- dropped.
        if text and ids and all(existing.point != text for existing in evidence):
            evidence.append(EvidencePoint(point=text, source_ids=ids))
    evidence = evidence[:MAX_EVIDENCE_POINTS]
    evidence_ids = {source_id for point in evidence for source_id in point.source_ids}

    ideas: list[Idea] = []
    for item in raw_ideas:
        if isinstance(item, str):
            item = {"title": item}
        if not isinstance(item, dict):
            continue
        title = strip_citation_markers(as_text(item.get("title"), 160))
        description = strip_citation_markers(as_text(item.get("description"), 800))
        if not title and not description:
            continue
        if not title:
            title, description = description, ""
        # builds_on may only reference passages the evidence list actually
        # cites -- an idea can't claim to build on a passage the user never saw.
        builds_on = [source_id for source_id in valid_source_ids(item.get("builds_on") or [], source_count) if source_id in evidence_ids]
        ideas.append(Idea(title=title, description=description, kind=_normalize_kind(item.get("kind")), builds_on=builds_on))
    ideas = ideas[:MAX_IDEAS]

    result = BrainstormResult(evidence=evidence, ideas=ideas, grounding_mode=grounding_mode)
    if not evidence:
        result.insufficient_evidence = True
        if free and ideas:
            result.notice = FREE_NO_EVIDENCE_NOTICE
        else:
            result.ideas = []
            result.notice = INSUFFICIENT_EVIDENCE_MESSAGE
    elif not ideas:
        result.notice = "I found relevant evidence but couldn't generate ideas from it this time. Try rephrasing the request."
    return result
