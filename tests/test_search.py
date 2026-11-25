import pytest

from hybrid_rag.cli import main
from hybrid_rag.db import connect, migrate
from hybrid_rag.documents import Document
from hybrid_rag.embedding import HashingEmbedder
from hybrid_rag.ingest import ingest_documents
from hybrid_rag.search import vector_search
from hybrid_rag.synthetic import generate_visit_notes

pytestmark = pytest.mark.db


@pytest.fixture()
def ingested(clean_db: str) -> str:
    migrate(clean_db)
    with connect(clean_db) as conn:
        ingest_documents(conn, generate_visit_notes(30), HashingEmbedder(), max_chars=400)
    return clean_db


def test_top_hits_are_closest_and_sorted(ingested: str) -> None:
    query = "metformin for type 2 diabetes mellitus"
    with connect(ingested) as conn:
        hits = vector_search(conn, query, HashingEmbedder(), k=5)
        contents = [r[0] for r in conn.execute("SELECT content FROM chunks")]
    assert len(hits) == 5
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)

    # Exact cosine similarity computed in Python agrees with the index-backed ranking.
    emb = HashingEmbedder()
    [q] = emb.embed([query])
    exact = sorted((float(q @ v) for v in emb.embed(contents)), reverse=True)[:5]
    assert scores == pytest.approx(exact, abs=1e-4)
    assert "diabetes" in hits[0].content.lower()


def test_hit_points_back_to_its_document(clean_db: str) -> None:
    migrate(clean_db)
    docs = [
        Document("a", "Asthma note", "visit_note", "Albuterol inhaler for wheezing."),
        Document("b", "Lipids note", "visit_note", "Started atorvastatin 20 mg nightly."),
    ]
    with connect(clean_db) as conn:
        ingest_documents(conn, docs, HashingEmbedder())
        [hit] = vector_search(conn, "atorvastatin 20 mg", HashingEmbedder(), k=1)
    assert (hit.external_id, hit.title, hit.chunk_index) == ("b", "Lipids note", 0)
    assert hit.content == "Started atorvastatin 20 mg nightly."
    assert 0.0 < hit.score <= 1.0


def test_k_above_hnsw_default_returns_k_rows(ingested: str) -> None:
    with connect(ingested) as conn:
        total = conn.execute("SELECT count(*) FROM chunks").fetchone()
        assert total is not None and total[0] > 60
        hits = vector_search(conn, "follow-up visit", HashingEmbedder(), k=60)
    assert len(hits) == 60


def test_empty_database_returns_no_hits(clean_db: str) -> None:
    migrate(clean_db)
    with connect(clean_db) as conn:
        assert vector_search(conn, "anything", HashingEmbedder()) == []


@pytest.mark.parametrize(("query", "k"), [("", 5), ("   ", 5), ("asthma", 0)])
def test_bad_arguments_are_rejected(ingested: str, query: str, k: int) -> None:
    with connect(ingested) as conn, pytest.raises(ValueError):
        vector_search(conn, query, HashingEmbedder(), k=k)


class RenamedEmbedder(HashingEmbedder):
    @property
    def name(self) -> str:
        return "some-other-model"


def test_query_model_must_match_stored_model(ingested: str) -> None:
    with connect(ingested) as conn, pytest.raises(ValueError, match="embedded with 'hashing-384'"):
        vector_search(conn, "asthma", RenamedEmbedder())


def test_ingest_refuses_to_mix_models(ingested: str) -> None:
    with connect(ingested) as conn:
        with pytest.raises(ValueError, match="before switching embedders"):
            ingest_documents(conn, generate_visit_notes(1), RenamedEmbedder())
        models = conn.execute("SELECT DISTINCT embedding_model FROM chunks").fetchall()
    assert models == [("hashing-384",)]


def test_cli_search(
    ingested: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DATABASE_URL", ingested)
    assert main(["search", "albuterol inhaler for asthma", "-k", "3"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 6  # a header line and a snippet line per hit
    assert lines[0].startswith(" 1. ")
    assert "visit-" in lines[0]
