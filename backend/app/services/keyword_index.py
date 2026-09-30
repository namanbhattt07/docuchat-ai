from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

# The lexical half of hybrid retrieval (see services/retrieval.py). Kept in
# its own module/table rather than bolted onto `chunks` because FTS5 is a
# SQLite-only virtual table -- isolating it here means the rest of the app
# never has to know it exists, and a non-sqlite `database_url` degrades to
# vector-only retrieval instead of failing (see `is_available`).

FTS_TABLE = "chunk_fts"

_CREATE_SQL = (
    f"CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE} USING fts5("
    "chunk_id UNINDEXED, document_id UNINDEXED, page_number UNINDEXED, content, "
    "tokenize='unicode61 remove_diacritics 2')"
)


def _is_sqlite(bind) -> bool:
    return bind.dialect.name == "sqlite"


def ensure_keyword_schema(engine: Engine) -> None:
    """Create the FTS5 index once. Safe to call on every startup.

    `unicode61` (no stemming) is deliberate: Group 3's brief is exact terms,
    abbreviations, identifiers and proper nouns (e.g. "MQTT") -- a stemmer
    would blur exactly the queries this layer exists to rescue.
    """
    if not _is_sqlite(engine):
        return
    with engine.begin() as connection:
        connection.execute(text(_CREATE_SQL))


def is_available(db: Session) -> bool:
    return _is_sqlite(db.get_bind())


def _ensure_table(db: Session) -> None:
    """`CREATE VIRTUAL TABLE IF NOT EXISTS` is a cheap catalog check, so
    calling it before every operation (rather than trusting `main.py`'s
    startup hook already ran) means this module never depends on init
    order -- important for tests and scripts that build a bare sqlite
    engine directly instead of going through the app's lifespan.
    """
    db.execute(text(_CREATE_SQL))


def index_chunks(db: Session, document_id: str, chunk_ids: list[str], contents: list[str], page_numbers: list[int]) -> None:
    """Insert one FTS row per chunk. Called once at ingestion time (see
    services/documents.py::process_document) right after the same chunks are
    embedded into the vector store, so both retrieval paths come online
    together.
    """
    if not is_available(db) or not chunk_ids:
        return
    _ensure_table(db)
    db.execute(
        text(f"INSERT INTO {FTS_TABLE}(chunk_id, document_id, page_number, content) VALUES (:chunk_id, :document_id, :page_number, :content)"),
        [
            {"chunk_id": chunk_id, "document_id": document_id, "page_number": page_number, "content": content}
            for chunk_id, content, page_number in zip(chunk_ids, contents, page_numbers)
        ],
    )


def delete_document(db: Session, document_id: str) -> None:
    if not is_available(db):
        return
    _ensure_table(db)
    db.execute(text(f"DELETE FROM {FTS_TABLE} WHERE document_id = :document_id"), {"document_id": document_id})


def ensure_indexed(db: Session, document_id: str) -> None:
    """Backfill the keyword index for a document ingested before this
    feature existed (or whose FTS rows are otherwise incomplete) -- the same
    "lazily backfill on first use" pattern as `ensure_document_index` for
    TOC/PageText. A no-op for every document indexed going forward, since
    `index_chunks` already covers it at ingestion time.
    """
    if not is_available(db):
        return
    _ensure_table(db)
    from app.models import Chunk  # local import: avoids a module cycle with app.models -> ... -> app.db

    indexed = db.execute(text(f"SELECT COUNT(*) FROM {FTS_TABLE} WHERE document_id = :document_id"), {"document_id": document_id}).scalar_one()
    total = db.query(Chunk).filter(Chunk.document_id == document_id).count()
    if total == 0 or indexed >= total:
        return
    delete_document(db, document_id)
    chunks = db.query(Chunk).filter(Chunk.document_id == document_id).all()
    index_chunks(db, document_id, [c.id for c in chunks], [c.content for c in chunks], [c.page_number for c in chunks])
    db.commit()


def _match_expression(query_terms: list[str]) -> str | None:
    unique_terms = list(dict.fromkeys(query_terms))
    if not unique_terms:
        return None
    # Every FTS5 query term comes from `retrieval.terms()`, which only ever
    # emits [a-z0-9]+ tokens -- never a quote, space, or FTS5 operator -- so
    # wrapping each in double quotes is always a safe literal-token escape,
    # not string-built SQL (the MATCH argument itself is still bound as one
    # parameter).
    quoted = [f'"{term}"' for term in unique_terms]
    expression = " OR ".join(quoted)
    if len(unique_terms) >= 2:
        phrase = '"' + " ".join(unique_terms) + '"'
        expression = f"{phrase} OR {expression}"
    return expression


def search(db: Session, query_terms: list[str], document_ids: list[str] | None, limit: int) -> list[tuple[str, float]]:
    """Rank candidate chunks by SQLite's bm25(). Returns (chunk_id, score)
    pairs with higher = more relevant (bm25 itself is "lower is better", so
    it is negated here to match the vector-similarity convention used
    elsewhere in retrieval.py).
    """
    if not is_available(db):
        return []
    _ensure_table(db)
    match_expression = _match_expression(query_terms)
    if match_expression is None:
        return []

    sql = f"SELECT chunk_id, bm25({FTS_TABLE}) AS score FROM {FTS_TABLE} WHERE {FTS_TABLE} MATCH :match"
    params: dict[str, object] = {"match": match_expression, "limit": limit}
    statement = text(sql + (" AND document_id IN :document_ids" if document_ids else "") + " ORDER BY score LIMIT :limit")
    if document_ids:
        statement = statement.bindparams(bindparam("document_ids", expanding=True))
        params["document_ids"] = document_ids
    rows = db.execute(statement, params).all()
    return [(row.chunk_id, -row.score) for row in rows]
