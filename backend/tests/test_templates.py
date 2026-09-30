import asyncio
import json

import pytest
from fakes import FakeEmbeddingProvider, FakeLLMProvider
from fastapi import HTTPException
from ingest import RecordingCollection
from sqlalchemy import select

import app.api.v1.chat as chat_module
import app.api.v1.templates as templates_api
from app.api.v1.chat import ChatRequest, ask
from app.models import Chunk, Collection, Conversation, Document, PromptTemplate
from app.services import collections as collections_service
from app.services import templates
from app.services.providers import ProviderUnavailable, use_providers

# ---------------------------------------------------------------------------
# Group 7 / 29: prompt templates -- the built-in set, custom-template CRUD,
# and (most importantly) that applying one goes through the *existing*
# retrieval / grounding / citation pipeline rather than a new one.
# ---------------------------------------------------------------------------

EXPECTED_SECTIONS = {
    "Research Paper Review": ["Research question", "Methodology", "Key findings", "Limitations", "Conclusions", "Critical observations"],
    "Case Brief": ["Facts", "Issue", "Rule", "Analysis", "Conclusion"],
    "Lecture Notes": ["Main concepts", "Definitions", "Important examples", "Key takeaways", "Revision points"],
    "Executive Summary": ["Objective", "Main findings", "Important evidence", "Implications", "Conclusion"],
    "Study Guide": ["Topics", "Definitions", "Important concepts", "Potential exam questions", "Quick revision"],
    "Technical Documentation Review": ["Purpose", "Architecture", "Components", "APIs/interfaces", "Dependencies", "Important implementation details"],
}


# ---------- built-in templates ----------


def test_the_six_required_builtin_templates_exist_with_the_specified_sections() -> None:
    by_name = {template.name: template for template in templates.BUILTIN_TEMPLATES}

    assert set(by_name) == set(EXPECTED_SECTIONS)
    for name, sections in EXPECTED_SECTIONS.items():
        assert by_name[name].sections == sections, name


def test_builtin_templates_are_well_formed() -> None:
    ids = [template.id for template in templates.BUILTIN_TEMPLATES]
    assert len(ids) == len(set(ids))
    for template in templates.BUILTIN_TEMPLATES:
        assert template.builtin is True
        assert template.id.startswith(templates.BUILTIN_PREFIX)
        assert template.description.strip() and len(template.description) <= templates.MAX_DESCRIPTION
        assert template.instruction.strip() and len(template.instruction) <= templates.MAX_INSTRUCTION
        assert len(template.name) <= templates.MAX_NAME
        assert template.output_format and all(line.startswith("## ") for line in template.output_format.splitlines())


def test_a_template_prompt_names_its_sections_and_forbids_inventing_content() -> None:
    template = templates.get_template(None, "builtin:case-brief")

    prompt = templates.build_template_instruction(template)

    assert "Case Brief" in prompt
    for heading in ("## Facts", "## Issue", "## Rule", "## Analysis", "## Conclusion"):
        assert heading in prompt
    assert templates.NOT_COVERED in prompt
    assert "citing with [n]" in prompt
    # related-but-off-topic material must not be stretched to fill a heading,
    # and a wrong-kind document is called out rather than force-fitted
    assert "merely related" in prompt and "not the kind of document" in prompt


def test_retrieval_query_targets_the_templates_subject_and_the_request() -> None:
    template = templates.get_template(None, "builtin:research-paper-review")
    query = templates.retrieval_query(template, "focus on the sample size")
    for word in ("Research Paper Review", "Methodology", "Limitations", "sample size"):
        assert word in query


def test_the_builtin_lookup_needs_no_database() -> None:
    assert templates.get_template(None, "builtin:study-guide").name == "Study Guide"
    with pytest.raises(templates.TemplateNotFound):
        templates.get_template(_NoDb(), "does-not-exist")


class _NoDb:
    def get(self, model, key):
        return None


# ---------- custom template CRUD ----------


def test_create_list_update_delete_custom_template(db_session) -> None:
    created = templates.create_template(db_session, "  Meeting Minutes ", "Minutes of a meeting", "Summarise decisions and owners.", "## Decisions\n## Owners")

    assert created.builtin is False and created.name == "Meeting Minutes"
    listing = templates.list_templates(db_session)
    assert [t.name for t in listing[:6]] == [t.name for t in templates.BUILTIN_TEMPLATES]  # built-ins first, untouched
    assert listing[-1].id == created.id and created.sections == ["Decisions", "Owners"]

    updated = templates.update_template(db_session, created.id, "Meeting Minutes v2", "", "Summarise decisions.", None)
    assert updated.name == "Meeting Minutes v2" and updated.output_format is None and updated.description == ""
    assert templates.get_template(db_session, created.id).instruction == "Summarise decisions."

    templates.delete_template(db_session, created.id)
    assert db_session.scalars(select(PromptTemplate)).all() == []
    with pytest.raises(templates.TemplateNotFound):
        templates.get_template(db_session, created.id)


def test_custom_template_only_needs_name_and_instruction(db_session) -> None:
    created = templates.create_template(db_session, "Quick", None, "Explain simply.")
    assert created.description == "" and created.output_format is None


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"name": "  ", "instruction": "x"}, "Name is required"),
    ({"name": "N", "instruction": "   "}, "Instruction is required"),
    ({"name": "N" * 81, "instruction": "x"}, "Name must be 80"),
    ({"name": "N", "instruction": "x" * 2001}, "Instruction must be 2000"),
    ({"name": "N", "instruction": "x", "description": "d" * 301}, "Description must be 300"),
    ({"name": "N", "instruction": "x", "output_format": "o" * 1001}, "Output format must be 1000"),
])
def test_custom_template_validation(db_session, kwargs, message) -> None:
    with pytest.raises(templates.TemplateError, match=message):
        templates.create_template(db_session, kwargs["name"], kwargs.get("description"), kwargs["instruction"], kwargs.get("output_format"))
    assert db_session.scalars(select(PromptTemplate)).all() == []


def test_template_names_must_be_unique_across_builtin_and_custom(db_session) -> None:
    with pytest.raises(templates.TemplateError, match="already exists"):
        templates.create_template(db_session, "case brief", None, "x")  # built-in name, case-insensitive
    first = templates.create_template(db_session, "Mine", None, "x")
    with pytest.raises(templates.TemplateError, match="already exists"):
        templates.create_template(db_session, "MINE", None, "y")
    # renaming to its own current name is fine
    assert templates.update_template(db_session, first.id, "Mine", "now with a description", "x").description == "now with a description"
    other = templates.create_template(db_session, "Other", None, "x")
    with pytest.raises(templates.TemplateError, match="already exists"):
        templates.update_template(db_session, other.id, "Mine", None, "x")


def test_builtin_templates_cannot_be_edited_or_deleted(db_session) -> None:
    with pytest.raises(templates.TemplateReadOnly):
        templates.update_template(db_session, "builtin:case-brief", "Renamed", None, "x")
    with pytest.raises(templates.TemplateReadOnly):
        templates.delete_template(db_session, "builtin:case-brief")
    assert templates.get_template(db_session, "builtin:case-brief").name == "Case Brief"


def test_updating_or_deleting_a_missing_template_is_not_found(db_session) -> None:
    with pytest.raises(templates.TemplateNotFound):
        templates.update_template(db_session, "nope", "N", None, "x")
    with pytest.raises(templates.TemplateNotFound):
        templates.delete_template(db_session, "nope")


# ---------- API layer ----------


def test_api_list_marks_builtins_and_returns_custom_templates(db_session) -> None:
    templates_api.create_template(templates_api.TemplateBody(name="Mine", instruction="Do it."), db=db_session)

    listing = templates_api.list_templates(db=db_session)["templates"]

    assert [t["builtin"] for t in listing] == [True] * 6 + [False]
    assert listing[-1]["name"] == "Mine" and listing[-1]["output_format"] is None


def test_api_maps_errors_to_http_statuses(db_session) -> None:
    def status_of(call):
        with pytest.raises(HTTPException) as exc_info:
            call()
        return exc_info.value.status_code

    body = templates_api.TemplateBody(name="Case Brief", instruction="x")
    assert status_of(lambda: templates_api.create_template(body, db=db_session)) == 409                                   # duplicate name
    assert status_of(lambda: templates_api.create_template(templates_api.TemplateBody(name=" ", instruction="x"), db=db_session)) == 422
    assert status_of(lambda: templates_api.update_template("builtin:case-brief", templates_api.TemplateBody(name="Z", instruction="x"), db=db_session)) == 403
    assert status_of(lambda: templates_api.delete_template("builtin:case-brief", db=db_session)) == 403
    assert status_of(lambda: templates_api.delete_template("missing", db=db_session)) == 404
    assert status_of(lambda: templates_api.update_template("missing", templates_api.TemplateBody(name="Z", instruction="x"), db=db_session)) == 404


# ---------- applying a template through /chat ----------

PAPER_PASSAGES = {
    ("paper", 1): "The study asks whether battery-powered sensor gateways can sustain telemetry for eighteen months without maintenance visits.",
    ("paper", 4): "Methodology: forty gateways were deployed across three buildings and their duty cycles were logged for a full year of operation.",
    ("paper", 9): "Key findings: median battery life reached eighteen months, although gateways near heavy machinery failed roughly twice as often.",
    ("outside", 2): (
        "This unrelated document describes the catering menus for the annual company dinner, including the vegetarian "
        "options and the dessert table, and should never influence the analysis of the research paper."
    ),
}


class NoVectorHits(RecordingCollection):
    pass


@pytest.fixture()
def seeded(db_session, monkeypatch):
    monkeypatch.setattr(chat_module, "get_collection", lambda: NoVectorHits())
    db_session.add_all([Document(id="paper", filename="paper.pdf", status="ready"), Document(id="outside", filename="outside.pdf", status="ready")])
    db_session.add_all([
        Chunk(id=f"{doc}-{page}-0", document_id=doc, page_number=page, content=text, start_offset=0, end_offset=len(text))
        for (doc, page), text in PAPER_PASSAGES.items()
    ])
    db_session.commit()
    return db_session


def _apply(db, template_id, *, llm=None, question="Apply the template", document_ids=("paper",), **extra):
    llm = llm or FakeLLMProvider(json.dumps({"answer": "## Facts\nGateways were deployed. [1]\n\n## Issue\nNot covered in the selected document(s).", "source_ids": [1]}))
    with use_providers(llm=llm, embedding=FakeEmbeddingProvider()):
        result = asyncio.run(ask(ChatRequest(question=question, document_ids=list(document_ids), mode="template", template_id=template_id, **extra), db=db))
    return result, llm


def _prompt(llm) -> str:
    return llm.calls[0]["messages"][-1]["content"]


def test_applying_a_builtin_template_returns_a_structured_grounded_answer(seeded) -> None:
    answer = "## Research question\nWhether gateways can run for eighteen months. [1]\n\n## Methodology\nNot covered in the selected document(s)."
    result, llm = _apply(
        seeded, "builtin:research-paper-review", question="Apply the “Research Paper Review” template",
        llm=FakeLLMProvider(json.dumps({"answer": answer, "source_ids": [1]})),
    )

    assert result["mode"] == "template"
    assert result["template"] == {"id": "builtin:research-paper-review", "name": "Research Paper Review"}
    assert result["answer"].startswith("## Research question")            # Markdown headings survive for the UI to render
    assert "Not covered in the selected document(s)." in result["answer"]
    assert result["citations"] and result["citations"][0]["document_id"] == "paper"
    prompt = _prompt(llm)
    for heading in ("## Research question", "## Methodology", "## Key findings", "## Limitations", "## Conclusions", "## Critical observations"):
        assert heading in prompt
    assert "Apply the “Research Paper Review” template" in prompt


def test_the_template_prompt_contains_only_the_selected_documents_passages(seeded) -> None:
    _, llm = _apply(seeded, "builtin:research-paper-review")
    prompt = _prompt(llm)

    assert "sustain telemetry for eighteen months" in prompt and "forty gateways" in prompt
    assert "catering menus" not in prompt                    # `outside` was not selected
    assert "paper.pdf" in prompt and "SOURCE 1" in prompt


def test_template_evidence_covers_the_whole_document_not_just_the_closest_passage(seeded) -> None:
    _, llm = _apply(seeded, "builtin:executive-summary")
    prompt = _prompt(llm)
    assert "page 1" in prompt and "page 4" in prompt and "page 9" in prompt


def test_citations_are_only_real_retrieved_passages(seeded) -> None:
    hallucinated = FakeLLMProvider(json.dumps({"answer": "## Facts\nInvented. [1] [7]", "source_ids": [1, 7, 99]}))

    result, _ = _apply(seeded, "builtin:research-paper-review", llm=hallucinated)

    assert [citation["index"] for citation in result["citations"]] == [1]  # 7 and 99 don't exist among the sources


def test_template_with_no_verifiable_evidence_returns_the_existing_insufficient_evidence_reply(seeded) -> None:
    ungrounded = FakeLLMProvider(json.dumps({"answer": "## Facts\nSome confident but uncited claims.", "source_ids": []}))

    result, _ = _apply(seeded, "builtin:research-paper-review", llm=ungrounded)

    assert result["answer"] == "I couldn't verify a source-grounded answer from the retrieved document passages."
    assert result["citations"] == []


def test_template_on_a_document_with_no_text_is_a_controlled_400(db_session, monkeypatch) -> None:
    monkeypatch.setattr(chat_module, "get_collection", lambda: NoVectorHits())
    db_session.add(Document(id="empty", filename="scan.pdf", status="empty"))
    db_session.commit()

    with pytest.raises(HTTPException) as exc_info:
        _apply(db_session, "builtin:research-paper-review", document_ids=("empty",))
    # names the actual problem (a blank/unreadable document) rather than generic upload advice
    assert exc_info.value.status_code == 400 and "no extractable text" in exc_info.value.detail and "scan.pdf" in exc_info.value.detail


def test_a_template_request_without_a_valid_template_is_rejected_before_creating_a_conversation(seeded) -> None:
    for template_id, expected in ((None, 400), ("does-not-exist", 404)):
        with pytest.raises(HTTPException) as exc_info:
            _apply(seeded, template_id)
        assert exc_info.value.status_code == expected
    assert seeded.scalars(select(Conversation)).all() == []


def test_stick_to_document_prompt_forbids_general_knowledge_and_go_freely_allows_it_separately(seeded) -> None:
    _, stick = _apply(seeded, "builtin:study-guide")
    assert "additional_context" not in _prompt(stick)
    assert "Stay strictly grounded" in stick.calls[0]["messages"][0]["content"]

    free_reply = FakeLLMProvider(json.dumps({
        "answer": "## Topics\nBattery life. [1]", "source_ids": [1], "additional_context": "Lithium cells lose capacity in the cold.",
    }))
    result, free = _apply(seeded, "builtin:study-guide", llm=free_reply, grounding_mode="free")

    assert "additional_context" in _prompt(free)
    assert "### From your documents" in result["answer"] and "### Additional context" in result["answer"]
    assert "Lithium cells" in result["answer"]
    assert "[" not in result["answer"].split("### Additional context")[1]  # free knowledge is never cited as the document


def test_templates_respect_collection_scope_and_validate_membership(seeded) -> None:
    collection = Collection(id="col-1", name="Study set")
    seeded.add(collection)
    seeded.commit()
    collections_service.add_document(seeded, "col-1", "paper")

    result, llm = _apply(seeded, "builtin:lecture-notes", document_ids=("paper", "outside"), collection_id="col-1")

    assert "catering menus" not in _prompt(llm)              # `outside` isn't a member -> filtered server-side
    assert {citation["document_id"] for citation in result["citations"]} == {"paper"}
    with pytest.raises(HTTPException) as exc_info:           # nothing selected in the collection -> controlled 400
        _apply(seeded, "builtin:lecture-notes", document_ids=("outside",), collection_id="col-1")
    assert exc_info.value.status_code == 400


def test_templates_work_across_multiple_selected_documents(seeded) -> None:
    def cite_one_source_from_each_document(messages):
        import re

        numbers = {name: int(number) for number, name in re.findall(r"SOURCE (\d+) \| (\S+\.pdf)", messages[-1]["content"])[::-1]}
        return json.dumps({"answer": f"## Topics\nBatteries [{numbers['paper.pdf']}] and dinners [{numbers['outside.pdf']}].", "source_ids": [numbers["paper.pdf"], numbers["outside.pdf"]]})

    result, _ = _apply(seeded, "builtin:study-guide", llm=FakeLLMProvider(cite_one_source_from_each_document), document_ids=("paper", "outside"))

    assert {citation["document_id"] for citation in result["citations"]} == {"paper", "outside"}
    assert {citation["filename"] for citation in result["citations"]} == {"paper.pdf", "outside.pdf"}


def test_a_custom_template_is_applied_with_its_own_instruction_and_format(seeded) -> None:
    custom = templates.create_template(seeded, "Risk Register", "Lists risks", "List every operational risk the document mentions.", "## Risks\n## Mitigations")

    result, llm = _apply(seeded, custom.id)

    prompt = _prompt(llm)
    assert "Risk Register" in prompt and "List every operational risk" in prompt
    assert "## Risks" in prompt and "## Mitigations" in prompt
    assert result["template"]["name"] == "Risk Register"


def test_a_custom_template_without_an_output_format_still_gets_structured_headings(seeded) -> None:
    custom = templates.create_template(seeded, "Plain", None, "Explain the document simply.")
    _, llm = _apply(seeded, custom.id)
    assert "Organise the \"answer\" as Markdown with clear headings." in _prompt(llm)


def test_a_deleted_custom_template_can_no_longer_be_applied(seeded) -> None:
    custom = templates.create_template(seeded, "Temp", None, "x")
    templates.delete_template(seeded, custom.id)
    with pytest.raises(HTTPException) as exc_info:
        _apply(seeded, custom.id)
    assert exc_info.value.status_code == 404


def test_provider_outage_while_applying_a_template_is_a_503(seeded) -> None:
    with pytest.raises(HTTPException) as exc_info:
        _apply(seeded, "builtin:research-paper-review", llm=FakeLLMProvider(ProviderUnavailable("down")))
    assert exc_info.value.status_code == 503


def test_template_turn_is_persisted_and_titled_like_other_modes(seeded) -> None:
    from app.models import Message

    result, _ = _apply(seeded, "builtin:research-paper-review", question="Research Paper Review: focus on the methodology")

    conversation = seeded.get(Conversation, result["conversation_id"])
    assert conversation.title == "Template: Research Paper Review"
    messages = seeded.scalars(select(Message).where(Message.conversation_id == conversation.id).order_by(Message.created_at)).all()
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].content == "Research Paper Review: focus on the methodology"
    assert json.loads(messages[1].citations_json)[0]["document_id"] == "paper"
    assert result["message_id"] == messages[1].id


def test_ordinary_chat_is_unaffected_by_the_template_field(seeded) -> None:
    llm = FakeLLMProvider(json.dumps({"answer": "Eighteen months. [1]", "source_ids": [1]}))
    with use_providers(llm=llm, embedding=FakeEmbeddingProvider()):
        result = asyncio.run(ask(ChatRequest(question="How long does the battery life last?", document_ids=["paper"]), db=seeded))
    assert "mode" not in result and "template" not in result
    assert "TEMPLATE" not in llm.calls[0]["messages"][-1]["content"]
