# hybrid-rag-pgvector

Hybrid retrieval for RAG that keeps everything inside **one Postgres database**:

- **Vector search** with [pgvector](https://github.com/pgvector/pgvector) for meaning ("chest pain on exertion" ≈ "angina").
- **Full-text search** with Postgres `tsvector` for exact terms that embeddings often miss: drug names, dosages, billing codes such as `CPT 99214` or `ICD-10 E11.9`.
- **Hybrid ranking** (reciprocal rank fusion) that combines both result lists.
- **Retrieval evals** that measure recall@k for vector-only, keyword-only and hybrid search.

**Why one database?** When documents contain sensitive data (for example PHI in healthcare), adding a separate
vector database means another system to secure, audit and keep in sync. pgvector keeps embeddings next to the
source rows, under the same access controls and backups.

All sample data in this repo is **synthetic**. No real patient data is used.

> Status: early development. See the roadmap below.

## Quick start

1. Start Postgres with pgvector. Any of these works:
   - **Docker:** `docker compose up -d`
   - **Hosted:** a Neon or Supabase database with the `vector` extension enabled
2. Configure and install:

   ```bash
   cp .env.example .env          # then edit DATABASE_URL if not using Docker
   python -m venv .venv
   # Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. Create the schema and run the tests:

   ```bash
   hybrid-rag migrate
   pytest
   ```

Tests that need the database are marked `db` and are skipped when `DATABASE_URL` is not set.

## Synthetic data and ingestion

```bash
hybrid-rag generate data/notes.jsonl --count 50   # write synthetic visit notes to JSONL
hybrid-rag ingest --file data/notes.jsonl         # or just `hybrid-rag ingest --count 50`
```

- **Synthetic visit notes** (`hybrid_rag.synthetic`): outpatient notes for made-up patients
  (`SYN-00001`, ...) with real ICD-10-CM codes (`E11.9`, `I10`, `J45.909`, ...), CPT codes
  (`99214`, `83036`, ...) and drug names with doses. The same seed always gives the same notes.
- **Chunking** (`hybrid_rag.chunking`): packs paragraphs into chunks of up to `--max-chars`
  (default 1000), splitting long paragraphs at sentence boundaries, with `--overlap-chars`
  (default 150) carried into the next chunk.
- **Embeddings** (`hybrid_rag.embedding`): an `Embedder` interface and a dependency-free
  `HashingEmbedder` (hashed unigrams and bigrams, 384 dimensions) so ingestion runs offline and in
  CI. It measures word overlap, not meaning; a neural embedder plugs into the same interface later.
- **Ingestion** (`hybrid_rag.ingest`): one transaction per run, upserts documents by
  `external_id` and replaces their chunks, so re-running never duplicates rows.

## Roadmap

- [x] Project setup, Postgres + pgvector, schema migrations
- [x] Synthetic clinical documents and ingestion with chunking
- [ ] Vector search
- [ ] Full-text search
- [ ] Hybrid ranking and a FastAPI search endpoint
- [ ] Retrieval evals (recall@k) and results table

## License

[MIT](LICENSE)
