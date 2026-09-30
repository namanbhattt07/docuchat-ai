import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.health import router as health_router
from app.api.v1.documents import router as documents_router
from app.api.v1.chat import router as chat_router
from app.api.v1.collections import router as collections_router
from app.api.v1.learning import router as learning_router
from app.api.v1.system import router as system_router
from app.api.v1.templates import router as templates_router
from app.core.config import get_settings
from app.db import Base, SessionLocal, engine, ensure_schema
from app.services.documents import recover_interrupted_documents
from app.services.ollama import close_client
from app.services.providers import ProviderUnavailable
from app.services.vector_store import get_collection
from app import models  # noqa: F401 -- registers ORM tables on Base.metadata

settings = get_settings()
logger = logging.getLogger("docuchat.startup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    ensure_schema()
    with SessionLocal() as db:
        recovered = recover_interrupted_documents(db, get_collection)
    if recovered:
        logger.warning("Marked %d document(s) that were still processing at shutdown as failed.", recovered)
    yield
    await close_client()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Local-first PDF RAG application API.",
    lifespan=lifespan,
)

@app.exception_handler(ProviderUnavailable)
async def provider_unavailable_handler(request: Request, exc: ProviderUnavailable) -> JSONResponse:
    """The last line of defence for a model outage that no endpoint caught (a
    query embedding during retrieval, for one). Without it the error surfaces as
    an unhandled 500, which Starlette sends *outside* the CORS middleware -- the
    browser then reports a bare "Failed to fetch" instead of the reason. Handled
    here it is a normal JSON error the UI can show, with CORS headers intact."""
    if isinstance(exc.__cause__, httpx.TimeoutException):
        return JSONResponse(status_code=504, content={"detail": "The local model took too long to respond. Try again, or ask for less at once."})
    return JSONResponse(
        status_code=503,
        content={"detail": "The local AI models aren't available. Start Ollama and make sure the configured models are pulled."},
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # The PDF viewer runs on another origin and pdf.js can only read these if
    # they are exposed; without them it can't tell ranges are supported.
    expose_headers=["Accept-Ranges", "Content-Range", "Content-Length"],
)

app.include_router(health_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(collections_router, prefix="/api/v1")
app.include_router(learning_router, prefix="/api/v1")
app.include_router(system_router, prefix="/api/v1")
app.include_router(templates_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.backend_host, port=settings.backend_port, reload=True)
