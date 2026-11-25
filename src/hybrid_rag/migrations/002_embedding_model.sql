-- Record which embedder produced each vector, so search never compares vectors from
-- different models. Rows written before this migration came from the hashing embedder.

ALTER TABLE chunks ADD COLUMN embedding_model TEXT NOT NULL DEFAULT 'hashing-384';
ALTER TABLE chunks ALTER COLUMN embedding_model DROP DEFAULT;
