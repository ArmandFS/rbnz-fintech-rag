from contextlib import contextmanager
from typing import Any, Iterator

import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import Json, RealDictCursor, execute_values

from config import get_settings


@contextmanager
def get_connection(*, register_vectors: bool = True) -> Iterator[psycopg2.extensions.connection]:
    settings = get_settings()
    conn = psycopg2.connect(settings.database_url)
    if register_vectors:
        register_vector(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(embedding_dimension: int) -> None:
    with get_connection(register_vectors=False) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id BIGSERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    source TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    checksum TEXT NOT NULL UNIQUE,
                    page_count INTEGER NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id BIGSERIAL PRIMARY KEY,
                    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    page_number INTEGER,
                    text TEXT NOT NULL,
                    embedding vector({embedding_dimension}) NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE(document_id, chunk_index)
                )
                """
            )
            cur.execute(
                """
                SELECT atttypmod
                FROM pg_attribute
                WHERE attrelid = 'chunks'::regclass
                  AND attname = 'embedding'
                """
            )
            vector_typmod = cur.fetchone()[0]
            existing_dimension = vector_typmod if vector_typmod and vector_typmod > 0 else None
            if existing_dimension and existing_dimension != embedding_dimension:
                raise ValueError(
                    "Existing chunks.embedding vector dimension is "
                    f"{existing_dimension}, but the configured embedding dimension is "
                    f"{embedding_dimension}. If you are changing embedding models, reset "
                    "the index and re-ingest the PDFs so all vectors use the same model."
                )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                ON chunks(document_id)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_collection
                ON documents(collection)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding
                ON chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
            )


def find_document_by_checksum(checksum: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM documents WHERE checksum = %s", (checksum,))
            row = cur.fetchone()
            return dict(row) if row else None


def insert_document(
    *,
    title: str,
    source: str,
    collection: str,
    file_path: str,
    checksum: str,
    page_count: int,
    metadata: dict[str, Any] | None = None,
) -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents
                    (title, source, collection, file_path, checksum, page_count, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    title,
                    source,
                    collection,
                    file_path,
                    checksum,
                    page_count,
                    Json(metadata or {}),
                ),
            )
            document_id = cur.fetchone()[0]
            return int(document_id)


def insert_chunks(document_id: int, chunks: list[dict[str, Any]]) -> None:
    rows = [
        (
            document_id,
            chunk["chunk_index"],
            chunk.get("page_number"),
            chunk["text"],
            chunk["embedding"],
            Json(chunk.get("metadata", {})),
        )
        for chunk in chunks
    ]
    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO chunks
                    (document_id, chunk_index, page_number, text, embedding, metadata)
                VALUES %s
                ON CONFLICT (document_id, chunk_index) DO NOTHING
                """,
                rows,
            )


def search_chunks(
    *,
    query_embedding: list[float],
    collection: str | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    filters = []
    params: list[Any] = [query_embedding]

    if collection:
        filters.append("d.collection = %s")
        params.append(collection)

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    params.extend([query_embedding, top_k])

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT
                    c.id AS chunk_id,
                    c.chunk_index,
                    c.page_number,
                    c.text,
                    c.metadata AS chunk_metadata,
                    d.id AS document_id,
                    d.title,
                    d.source,
                    d.collection,
                    d.file_path,
                    1 - (c.embedding <=> %s::vector) AS similarity
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                {where_clause}
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                params,
            )
            return [dict(row) for row in cur.fetchall()]


def get_index_summary() -> dict[str, Any]:
    with get_connection(register_vectors=False) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT COUNT(*) AS count FROM documents")
            document_count = cur.fetchone()["count"]

            cur.execute("SELECT COUNT(*) AS count FROM chunks")
            chunk_count = cur.fetchone()["count"]

            cur.execute(
                """
                SELECT d.collection, COUNT(DISTINCT d.id) AS documents, COUNT(c.id) AS chunks
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                GROUP BY d.collection
                ORDER BY d.collection
                """
            )
            collections = [dict(row) for row in cur.fetchall()]

    return {
        "document_count": document_count,
        "chunk_count": chunk_count,
        "collections": collections,
    }
