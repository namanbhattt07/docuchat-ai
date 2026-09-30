from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Chunk, PromptTemplate

# Group 7 custom prompt templates. A template is only three things a person
# writes -- a name, a description, and an instruction (plus an optional output
# format). There is no template language: the instruction is plain text that
# is handed to the model together with the document's retrieved passages, via
# the same retrieval / grounding / citation pipeline /chat already uses
# (see api/v1/chat.py::_ask_template). Built-in templates are constants here,
# not database rows, so they can never be edited, deleted, or lost in a
# migration; only user-created templates are stored.

BUILTIN_PREFIX = "builtin:"

MAX_NAME = 80
MAX_DESCRIPTION = 300
MAX_INSTRUCTION = 2000
MAX_OUTPUT_FORMAT = 1000

NOT_COVERED = "Not covered in the selected document(s)."


class TemplateError(ValueError):
    """Invalid template input (blank/oversized field, duplicate name)."""


class TemplateNotFound(LookupError):
    pass


class TemplateReadOnly(PermissionError):
    pass


@dataclass(frozen=True)
class TemplateSpec:
    id: str
    name: str
    description: str
    instruction: str
    output_format: str | None
    builtin: bool
    created_at: datetime | None = None
    updated_at: datetime | None = None
    # Only for a template written for one *kind* of document (a Case Brief needs
    # a legal case). `fit_terms` is the vocabulary such a document cannot avoid;
    # a document containing fewer than `fit_min_terms` distinct ones of them is
    # not that kind of document. Checked in code because the model, told to say
    # so, instead fills a Case Brief's "Rule of law" section for a sensor report.
    fit_terms: tuple[str, ...] = ()
    fit_min_terms: int = 2
    fit_problem: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "instruction": self.instruction,
            "output_format": self.output_format,
            "builtin": self.builtin,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @property
    def sections(self) -> list[str]:
        """Section headings named by the output format ("## Facts" -> "Facts")."""
        if not self.output_format:
            return []
        return [line.lstrip("#").strip() for line in self.output_format.splitlines() if line.strip().startswith("#")]


def _format(*headings: str) -> str:
    return "\n".join(f"## {heading}" for heading in headings)


def _builtin(slug: str, name: str, description: str, instruction: str, *headings: str, **fit) -> TemplateSpec:
    return TemplateSpec(BUILTIN_PREFIX + slug, name, description, instruction, _format(*headings), builtin=True, **fit)


BUILTIN_TEMPLATES: tuple[TemplateSpec, ...] = (
    _builtin(
        "research-paper-review", "Research Paper Review",
        "Review a paper's question, method, findings, limits and conclusions.",
        "Review the document as a research paper. State what it sets out to answer, how it went about it, what it "
        "found, what limits the findings, and what it concludes, then add critical observations on how well the "
        "evidence supports the claims.",
        "Research question", "Methodology", "Key findings", "Limitations", "Conclusions", "Critical observations",
    ),
    _builtin(
        "case-brief", "Case Brief",
        "Brief a legal case: facts, issue, rule, analysis and conclusion.",
        "Write a case brief of the document. Summarise the material facts, the legal issue presented, the rule of law "
        "applied, the court's or author's reasoning, and the outcome.",
        "Facts", "Issue", "Rule", "Analysis", "Conclusion",
        fit_terms=(
            "court", "plaintiff", "defendant", "appellant", "respondent", "petitioner", "judgment", "ruling", "statute",
            "appeal", "jury", "verdict", "tort", "judge", "holding", "liable", "liability", "lawsuit", "litigation",
            "counsel", "testimony", "plaintiffs", "defendants", "tribunal", "damages",
        ),
        fit_problem="It doesn't read like a legal case: no court, parties, ruling or similar legal language was found in it.",
    ),
    _builtin(
        "lecture-notes", "Lecture Notes",
        "Turn the material into organised notes with definitions and revision points.",
        "Turn the document into clear lecture notes a student could revise from: the main concepts, the definitions it "
        "gives, its important examples, the key takeaways, and short revision points.",
        "Main concepts", "Definitions", "Important examples", "Key takeaways", "Revision points",
    ),
    _builtin(
        "executive-summary", "Executive Summary",
        "A short decision-oriented summary: objective, findings, evidence, implications.",
        "Write an executive summary of the document for a busy decision-maker: the objective, the main findings, the "
        "most important supporting evidence, the implications, and the conclusion. Be concise.",
        "Objective", "Main findings", "Important evidence", "Implications", "Conclusion",
    ),
    _builtin(
        "study-guide", "Study Guide",
        "Topics, definitions, key concepts, likely exam questions and quick revision.",
        "Create a study guide from the document: the topics it covers, its definitions, the most important concepts, "
        "potential exam questions a student should be able to answer from it, and a quick revision list.",
        "Topics", "Definitions", "Important concepts", "Potential exam questions", "Quick revision",
    ),
    _builtin(
        "technical-documentation-review", "Technical Documentation Review",
        "Review technical docs: purpose, architecture, components, APIs, dependencies.",
        "Review the document as technical documentation: its purpose, the architecture it describes, the components, "
        "the APIs or interfaces it exposes, its dependencies, and the important implementation details.",
        "Purpose", "Architecture", "Components", "APIs/interfaces", "Dependencies", "Important implementation details",
    ),
)

_BUILTIN_BY_ID = {template.id: template for template in BUILTIN_TEMPLATES}


def _spec(row: PromptTemplate) -> TemplateSpec:
    return TemplateSpec(
        id=row.id, name=row.name, description=row.description or "", instruction=row.instruction,
        output_format=row.output_format or None, builtin=False, created_at=row.created_at, updated_at=row.updated_at,
    )


def list_templates(db: Session) -> list[TemplateSpec]:
    custom = db.scalars(select(PromptTemplate).order_by(PromptTemplate.created_at, PromptTemplate.name)).all()
    return [*BUILTIN_TEMPLATES, *(_spec(row) for row in custom)]


def get_template(db: Session, template_id: str) -> TemplateSpec:
    builtin = _BUILTIN_BY_ID.get(template_id)
    if builtin:
        return builtin
    row = db.get(PromptTemplate, template_id)
    if row is None:
        raise TemplateNotFound("Template not found.")
    return _spec(row)


def _clean(value: str | None, label: str, limit: int, *, required: bool) -> str | None:
    text = (value or "").strip()
    if not text:
        if required:
            raise TemplateError(f"{label} is required.")
        return None
    if len(text) > limit:
        raise TemplateError(f"{label} must be {limit} characters or fewer.")
    return text


def _validated(db: Session, name: str, description: str | None, instruction: str, output_format: str | None, *, exclude_id: str | None = None) -> dict:
    cleaned_name = _clean(name, "Name", MAX_NAME, required=True)
    taken = {template.name.lower() for template in BUILTIN_TEMPLATES}
    taken |= {row.name.lower() for row in db.scalars(select(PromptTemplate).where(PromptTemplate.id != (exclude_id or ""))).all()}
    if cleaned_name.lower() in taken:
        raise TemplateError(f'A template named "{cleaned_name}" already exists.')
    return {
        "name": cleaned_name,
        "description": _clean(description, "Description", MAX_DESCRIPTION, required=False) or "",
        "instruction": _clean(instruction, "Instruction", MAX_INSTRUCTION, required=True),
        "output_format": _clean(output_format, "Output format", MAX_OUTPUT_FORMAT, required=False),
    }


def create_template(db: Session, name: str, description: str | None, instruction: str, output_format: str | None = None) -> TemplateSpec:
    row = PromptTemplate(**_validated(db, name, description, instruction, output_format))
    db.add(row)
    db.commit()
    db.refresh(row)
    return _spec(row)


def _custom_row(db: Session, template_id: str) -> PromptTemplate:
    if template_id in _BUILTIN_BY_ID:
        raise TemplateReadOnly("Built-in templates can't be edited or deleted. Duplicate one as a custom template instead.")
    row = db.get(PromptTemplate, template_id)
    if row is None:
        raise TemplateNotFound("Template not found.")
    return row


def update_template(db: Session, template_id: str, name: str, description: str | None, instruction: str, output_format: str | None = None) -> TemplateSpec:
    row = _custom_row(db, template_id)
    for field, value in _validated(db, name, description, instruction, output_format, exclude_id=row.id).items():
        setattr(row, field, value)
    db.commit()
    db.refresh(row)
    return _spec(row)


def delete_template(db: Session, template_id: str) -> None:
    db.delete(_custom_row(db, template_id))
    db.commit()


# -- prompt assembly -----------------------------------------------------------


def retrieval_query(template: TemplateSpec, request_text: str) -> str:
    """What to search the document for: the template's own subject matter (its
    name and section headings) plus whatever the person asked, so the relevant
    half of the evidence targets the sections the template will fill in."""
    return " ".join(part for part in (template.name, *template.sections, request_text) if part).strip()


def build_template_instruction(template: TemplateSpec) -> str:
    """The template-specific part of the prompt, layered onto chat's standard
    JSON-schema + citation instructions (chat.build_instructions). Requiring
    "Not covered ..." for an unsupported section is what keeps a structured
    template from inventing content to fill its headings."""
    if template.output_format:
        layout = (
            "Write the \"answer\" as Markdown using exactly these sections, in this order, each as its own heading, "
            f"and no others:\n{template.output_format}\n"
        )
    else:
        layout = "Organise the \"answer\" as Markdown with clear headings. "
    return (
        f"TEMPLATE \"{template.name}\": {template.instruction} {layout}"
        f"Under each section write only what the passages support, citing with [n]. Material that is merely related "
        f"to a heading, but is not what that heading asks for, does not cover it. If the passages do not cover a "
        f"section, write \"{NOT_COVERED}\" under its heading instead of guessing or stretching other content to fit. "
        f"If the document is not the kind of document this template is written for, say so in one sentence at the "
        f"top and mark every section it cannot fill as not covered."
    )


def applicability_problem(db: Session, template: TemplateSpec, document_ids: list[str] | None) -> str | None:
    """Why this template can't sensibly be applied to the selected document(s),
    or None. Only templates that declare `fit_terms` are ever refused, and only
    when the document contains too few of them (see TemplateSpec)."""
    if not template.fit_terms:
        return None
    present = 0
    for term in template.fit_terms:
        statement = select(func.count()).select_from(Chunk).where(func.lower(Chunk.content).contains(term))
        if document_ids:
            statement = statement.where(Chunk.document_id.in_(document_ids))
        if db.scalar(statement):
            present += 1
            if present >= template.fit_min_terms:
                return None
    return template.fit_problem or "The document doesn't look like the kind of material this template is written for."
