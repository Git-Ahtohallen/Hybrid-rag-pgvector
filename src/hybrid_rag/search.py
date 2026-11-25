"""Vector search: cosine top-k over `chunks.embedding`.

The query is embedded with the same embedder that produced the stored vectors (search
checks the stored `embedding_model`), then ranked by pgvector's cosine distance operator
`<=>`, which the HNSW index on `chunks.embedding` serves.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg

from hybrid_rag.embedding import Embedder

# pgvector's default; HNSW scans return at most ef_search rows, so it is raised for larger k.
_DEFAULT_EF_SEARCH = 40


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_id: int
    external_id: str
    title: str
    chunk_index: int
    content: str
    score: float
    """Cosine similarity in [-1, 1]: 1 minus pgvector's cosine distance. Higher is closer."""


def vector_search(
    conn: psycopg.Connection, query: str, embedder: Embedder, *, k: int = 5
) -> list[SearchHit]:
    """The `k` chunks closest to `query`, best first. `conn` must come from `db.connect`."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not query.strip():
        raise ValueError("query is empty")
    with conn.transaction(), conn.cursor() as cur:
        stored = cur.execute("SELECT embedding_model FROM chunks LIMIT 1").fetchone()
        if stored is None:
            return []
        if stored[0] != embedder.name:
            raise ValueError(
                f"chunks were embedded with {stored[0]!r} but the query uses {embedder.name!r}"
            )
        [qvec] = embedder.embed([query])
        cur.execute(
            "SELECT set_config('hnsw.ef_search', %s, true)",
            (str(max(_DEFAULT_EF_SEARCH, k)),),
        )
        rows = cur.execute(
            """
            SELECT c.id, d.external_id, d.title, c.chunk_index, c.content,
                   1 - (c.embedding <=> %(q)s) AS score
              FROM chunks c
              JOIN documents d ON d.id = c.document_id
             ORDER BY c.embedding <=> %(q)s
             LIMIT %(k)s
            """,
            {"q": qvec, "k": k},
        ).fetchall()
    return [
        SearchHit(
            chunk_id=chunk_id,
            external_id=external_id,
            title=title,
            chunk_index=chunk_index,
            content=content,
            score=float(score),
        )
        for chunk_id, external_id, title, chunk_index, content, score in rows
    ]
