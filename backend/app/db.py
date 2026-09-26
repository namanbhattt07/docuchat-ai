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
    if "documents" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("documents")}
    additions = {"content_hash": "VARCHAR(64)", "status_detail": "VARCHAR(500)"}
    with engine.begin() as connection:
        for column, ddl_type in additions.items():
            if column not in existing:
                connection.execute(text(f"ALTER TABLE documents ADD COLUMN {column} {ddl_type}"))
