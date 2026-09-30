from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import get_settings

settings = get_settings()
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_schema() -> None:
    """Add columns introduced after the first release directly with ALTER TABLE.

    There is no migration tool (Alembic) in this project yet, so this keeps
    existing local databases usable across upgrades instead of requiring
    users to delete their data/docuchat.db. Safe to call on every startup --
    a no-op once the columns already exist.
    """
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    column_additions = {
        "documents": {
            "content_hash": "VARCHAR(64)",
            "status_detail": "VARCHAR(500)",
            "toc_json": "TEXT",
            "suggested_questions_json": "TEXT",
            # Group 7 -- see models.Document.visuals_indexed.
            "visuals_indexed": "INTEGER DEFAULT 0",
        },
        "page_text": {
            # Group 7 explicit OCR / page-processing status -- see models.PageText.
            "status": "VARCHAR(20)",
            "source_type": "VARCHAR(20)",
            "status_detail": "VARCHAR(300)",
            "ocr_confidence": "FLOAT",
        },
        "chunks": {
            "section": "VARCHAR(300)",
            "start_offset": "INTEGER DEFAULT 0",
            "end_offset": "INTEGER DEFAULT 0",
            "bbox": "JSON",
            "source_type": "VARCHAR(20) DEFAULT 'text'",
        },
        "messages": {
            # Group 5 threads/follow-ups -- see models.Message.reply_to_message_id.
            "reply_to_message_id": "VARCHAR(36)",
        },
    }
    with engine.begin() as connection:
        for table, additions in column_additions.items():
            if table not in table_names:
                continue
            existing = {column["name"] for column in inspector.get_columns(table)}
            for column, ddl_type in additions.items():
                if column not in existing:
                    connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}"))

    # Group 3's lexical/keyword retrieval layer (SQLite FTS5), additive and
    # sqlite-only -- see services/keyword_index.py. Skipped entirely on a
    # non-sqlite database_url; hybrid retrieval falls back to vector-only in
    # that case rather than failing.
    from app.services.keyword_index import ensure_keyword_schema

    ensure_keyword_schema(engine)
