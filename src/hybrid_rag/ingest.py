"""Chunk, embed and store documents.

Ingestion is idempotent per document: a document whose `external_id` already exists is
updated in place and its chunks are replaced, so re-running ingestion never duplicates
rows. All documents in one call are written in a single transaction.

All chunks in the database must come from one embedder, because vectors from different
models are not comparable. Ingesting with a different embedder than the stored chunks is
refused; delete the existing documents first to switch models.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import psycopg
from psycopg.types.json import Jsonb

from hybrid_rag.chunking import chunk_text
from hybrid_rag.config import EMBEDDING_DIM
from hybrid_rag.documents import Document
from hybrid_rag.embedding import Embedder


@dataclass(frozen=True, slots=True)
class IngestStats:
    documents: int
    chunks: int


def ingest_documents(
    conn: psycopg.Connection,
    docs: Iterable[Document],
    embedder: Embedder,
    *,
    max_chars: int = 1000,
    overlap_chars: int = 150,
) -> IngestStats:
    """Store `docs` and their chunks. `conn` must come from `hybrid_rag.db.connect`."""
    if embedder.dim != EMBEDDING_DIM:
        raise ValueError(
            f"embedder produces {embedder.dim} dimensions but the schema expects {EMBEDDING_DIM}"
        )
    n_docs = n_chunks = 0
    with conn.transaction(), conn.cursor() as cur:
        other = cur.execute(
            "SELECT embedding_model FROM chunks WHERE embedding_model <> %s LIMIT 1",
            (embedder.name,),
        ).fetchone()
        if other is not None:
            raise ValueError(
                f"the database holds chunks embedded with {other[0]!r}, not {embedder.name!r}; "
                "delete the existing documents before switching embedders"
            )
        for doc in docs:
            chunks = chunk_text(doc.text, max_chars=max_chars, overlap_chars=overlap_chars)
            vectors = embedder.embed(chunks)
            cur.execute(
                """
                INSERT INTO documents (external_id, title, doc_type, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (external_id) DO UPDATE
                   SET title = EXCLUDED.title,
                       doc_type = EXCLUDED.doc_type,
                       metadata = EXCLUDED.metadata
                RETURNING id
                """,
                (doc.external_id, doc.title, doc.doc_type, Jsonb(doc.metadata)),
            )
            row = cur.fetchone()
            assert row is not None
            doc_id = row[0]
            cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
            cur.executemany(
                "INSERT INTO chunks"
                " (document_id, chunk_index, content, embedding, embedding_model)"
                " VALUES (%s, %s, %s, %s, %s)",
                [
                    (doc_id, i, text, vec, embedder.name)
                    for i, (text, vec) in enumerate(zip(chunks, vectors, strict=True))
                ],
            )
            n_docs += 1
            n_chunks += len(chunks)
    return IngestStats(documents=n_docs, chunks=n_chunks)
