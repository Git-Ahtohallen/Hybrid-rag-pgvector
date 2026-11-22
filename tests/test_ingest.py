from pathlib import Path

import pytest

from hybrid_rag.cli import main
from hybrid_rag.db import connect, migrate
from hybrid_rag.documents import Document
from hybrid_rag.embedding import HashingEmbedder
from hybrid_rag.ingest import ingest_documents
from hybrid_rag.synthetic import generate_visit_notes

pytestmark = pytest.mark.db


def _counts(url: str) -> tuple[int, int]:
    with connect(url) as conn:
        docs = conn.execute("SELECT count(*) FROM documents").fetchone()
        chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()
    assert docs is not None and chunks is not None
    return docs[0], chunks[0]


def test_ingest_stores_documents_and_chunks(clean_db: str) -> None:
    migrate(clean_db)
    notes = generate_visit_notes(10)
    with connect(clean_db) as conn:
        stats = ingest_documents(conn, notes, HashingEmbedder(), max_chars=400, overlap_chars=80)
    assert stats.documents == 10
    assert stats.chunks > 10  # notes are longer than 400 chars
    assert _counts(clean_db) == (10, stats.chunks)

    with connect(clean_db) as conn:
        row = conn.execute(
            "SELECT d.metadata->>'patient_id', count(c.*) FROM documents d "
            "JOIN chunks c ON c.document_id = d.id WHERE d.external_id = 'visit-00001' "
            "GROUP BY 1"
        ).fetchone()
    assert row is not None
    assert row[0] == notes[0].metadata["patient_id"]


def test_reingest_replaces_instead_of_duplicating(clean_db: str) -> None:
    migrate(clean_db)
    notes = generate_visit_notes(5)
    with connect(clean_db) as conn:
        first = ingest_documents(conn, notes, HashingEmbedder())
        second = ingest_documents(conn, notes, HashingEmbedder())
    assert first == second
    assert _counts(clean_db) == (5, first.chunks)


def test_updated_document_gets_new_chunks(clean_db: str) -> None:
    migrate(clean_db)
    doc = Document("d1", "Note", "visit_note", "Old text about lisinopril.")
    with connect(clean_db) as conn:
        ingest_documents(conn, [doc], HashingEmbedder())
        ingest_documents(
            conn,
            [Document("d1", "Note v2", "visit_note", "New text: atorvastatin.")],
            HashingEmbedder(),
        )
        rows = conn.execute(
            "SELECT d.title, c.content FROM documents d JOIN chunks c ON c.document_id = d.id"
        ).fetchall()
    assert rows == [("Note v2", "New text: atorvastatin.")]


def test_keyword_search_finds_exact_codes(clean_db: str) -> None:
    migrate(clean_db)
    notes = generate_visit_notes(30)
    with connect(clean_db) as conn:
        ingest_documents(conn, notes, HashingEmbedder())
        hits = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT d.external_id FROM chunks c "
                "JOIN documents d ON d.id = c.document_id "
                "WHERE c.tsv @@ plainto_tsquery('english', 'metformin')"
            )
        }
    expected = {n.external_id for n in notes if "metformin" in n.metadata["medications"]}
    assert expected
    assert hits == expected


def test_wrong_embedding_dim_is_rejected(clean_db: str) -> None:
    migrate(clean_db)
    with connect(clean_db) as conn, pytest.raises(ValueError, match="dimensions"):
        ingest_documents(conn, generate_visit_notes(1), HashingEmbedder(dim=8))
    assert _counts(clean_db) == (0, 0)


def test_cli_generate_then_ingest(
    clean_db: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DATABASE_URL", clean_db)
    out = tmp_path / "notes.jsonl"
    assert main(["migrate"]) == 0
    assert main(["generate", str(out), "--count", "4"]) == 0
    assert main(["ingest", "--file", str(out)]) == 0
    assert "Ingested 4 documents" in capsys.readouterr().out
    assert _counts(clean_db)[0] == 4
