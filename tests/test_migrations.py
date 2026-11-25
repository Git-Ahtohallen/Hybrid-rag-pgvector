import psycopg
import pytest

from hybrid_rag.db import connect, migrate

pytestmark = pytest.mark.db


def test_migrate_is_idempotent(clean_db: str) -> None:
    assert migrate(clean_db) == ["001_init.sql", "002_embedding_model.sql"]
    assert migrate(clean_db) == []


def test_schema_has_tables_and_indexes(clean_db: str) -> None:
    migrate(clean_db)
    with psycopg.connect(clean_db) as conn:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = current_schema()"
            )
        }
        indexes = {
            r[0]
            for r in conn.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'chunks'")
        }
    assert {"documents", "chunks", "schema_migrations"} <= tables
    assert {"chunks_embedding_hnsw", "chunks_tsv_gin"} <= indexes


def test_tsv_column_is_generated_from_content(clean_db: str) -> None:
    migrate(clean_db)
    with connect(clean_db) as conn:
        doc_id = conn.execute(
            "INSERT INTO documents (external_id, title, doc_type) "
            "VALUES ('d1', 'Note', 'visit_note') RETURNING id"
        ).fetchone()[0]  # type: ignore[index]
        conn.execute(
            "INSERT INTO chunks (document_id, chunk_index, content, embedding, embedding_model) "
            "VALUES (%s, 0, 'Patient started metformin 500 mg', %s, 'test')",
            (doc_id, [0.0] * 383 + [1.0]),
        )
        hit = conn.execute(
            "SELECT count(*) FROM chunks WHERE tsv @@ plainto_tsquery('english', 'metformin')"
        ).fetchone()
    assert hit == (1,)


def test_deleting_document_cascades_to_chunks(clean_db: str) -> None:
    migrate(clean_db)
    with connect(clean_db) as conn:
        doc_id = conn.execute(
            "INSERT INTO documents (external_id, title, doc_type) "
            "VALUES ('d2', 'Note', 'visit_note') RETURNING id"
        ).fetchone()[0]  # type: ignore[index]
        conn.execute(
            "INSERT INTO chunks (document_id, chunk_index, content, embedding, embedding_model) "
            "VALUES (%s, 0, 'text', %s, 'test')",
            (doc_id, [1.0] + [0.0] * 383),
        )
        conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        remaining = conn.execute("SELECT count(*) FROM chunks").fetchone()
    assert remaining == (0,)


def test_embedding_model_is_required(clean_db: str) -> None:
    migrate(clean_db)
    with connect(clean_db) as conn:
        doc_id = conn.execute(
            "INSERT INTO documents (external_id, title, doc_type) "
            "VALUES ('d3', 'Note', 'visit_note') RETURNING id"
        ).fetchone()[0]  # type: ignore[index]
        with pytest.raises(psycopg.errors.NotNullViolation):
            conn.execute(
                "INSERT INTO chunks (document_id, chunk_index, content, embedding) "
                "VALUES (%s, 0, 'text', %s)",
                (doc_id, [1.0] + [0.0] * 383),
            )
