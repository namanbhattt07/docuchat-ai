from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import get_db
from app.services import templates as templates_service
from app.services.templates import TemplateError, TemplateNotFound, TemplateReadOnly

# Group 7 prompt templates: list + custom-template CRUD. *Applying* a template
# is not a route here -- it rides on POST /chat (mode="template") so it shares
# that endpoint's scope validation, retrieval, grounding, citations and
# message persistence instead of duplicating any of it.

router = APIRouter(prefix="/templates", tags=["templates"])


class TemplateBody(BaseModel):
    # Length limits are enforced (with friendly messages) by the service; the
    # schema only bounds the payload so a huge body is rejected early.
    name: str = Field(max_length=500)
    description: str | None = Field(default=None, max_length=5000)
    instruction: str = Field(max_length=20000)
    output_format: str | None = Field(default=None, max_length=10000)


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, TemplateNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, TemplateReadOnly):
        return HTTPException(status_code=403, detail=str(exc))
    duplicate = "already exists" in str(exc)
    return HTTPException(status_code=409 if duplicate else 422, detail=str(exc))


@router.get("")
def list_templates(db: Session = Depends(get_db)):
    return {"templates": [template.to_dict() for template in templates_service.list_templates(db)]}


@router.post("", status_code=status.HTTP_201_CREATED)
def create_template(body: TemplateBody, db: Session = Depends(get_db)):
    try:
        return templates_service.create_template(db, body.name, body.description, body.instruction, body.output_format).to_dict()
    except (TemplateError, TemplateNotFound, TemplateReadOnly) as exc:
        raise _http_error(exc) from exc


@router.put("/{template_id}")
def update_template(template_id: str, body: TemplateBody, db: Session = Depends(get_db)):
    try:
        return templates_service.update_template(db, template_id, body.name, body.description, body.instruction, body.output_format).to_dict()
    except (TemplateError, TemplateNotFound, TemplateReadOnly) as exc:
        raise _http_error(exc) from exc


@router.delete("/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_template(template_id: str, db: Session = Depends(get_db)):
    try:
        templates_service.delete_template(db, template_id)
    except (TemplateNotFound, TemplateReadOnly) as exc:
        raise _http_error(exc) from exc
