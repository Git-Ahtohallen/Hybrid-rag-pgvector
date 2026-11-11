-- Documents and their searchable chunks, all inside one Postgres database.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id          BIGSERIAL PRIMARY KEY,
    external_id TEXT        NOT NULL UNIQUE,
    title       TEXT        NOT NULL,
    doc_type    TEXT        NOT NULL,
    metadata    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT      NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index INT         NOT NULL,
    content     TEXT        NOT NULL,
    -- Must match hybrid_rag.config.EMBEDDING_DIM.
    embedding   vector(384) NOT NULL,
    -- Kept in sync by Postgres; used for keyword search.
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (document_id, chunk_index)
);

-- Approximate nearest-neighbour search on cosine distance.
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);

-- Full-text search.
CREATE INDEX chunks_tsv_gin ON chunks USING gin (tsv);
