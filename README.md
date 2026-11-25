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
- **Embeddings** (`hybrid_rag.embedding`): an `Embedder` interface with two implementations, both
  384 dimensions:
  - `HashingEmbedder` (`--embedder hashing`, the default): hashed unigrams and bigrams, no
    dependencies, so ingestion runs offline and in CI. It measures word overlap, not meaning.
  - `SentenceTransformerEmbedder` (`--embedder minilm`): the neural
    [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model.
    Install it with `pip install -e ".[neural]"`. The model is downloaded on first use.
- **Ingestion** (`hybrid_rag.ingest`): one transaction per run, upserts documents by
  `external_id` and replaces their chunks, so re-running never duplicates rows.

## Vector search

```bash
pip install -e ".[neural]"
hybrid-rag ingest --embedder minilm
hybrid-rag search "chest pain when climbing stairs" -k 5 --embedder minilm
```

`hybrid_rag.search.vector_search(conn, query, embedder, k=5)` embeds the query and returns the `k`
closest chunks by cosine similarity, using pgvector's `<=>` operator and the HNSW index. Each hit
has the chunk text, its document's `external_id` and title, and a `score` (cosine similarity,
higher is closer).

Each chunk stores the name of the embedder that produced it (`chunks.embedding_model`). Vectors
from different models aren't comparable, so ingestion refuses to mix models, and search refuses a
query embedder that doesn't match the stored chunks. To switch models, delete the documents and
ingest again.

Tests that need the real model are marked `neural` and skipped when the extra isn't installed.

## Roadmap

- [x] Project setup, Postgres + pgvector, schema migrations
- [x] Synthetic clinical documents and ingestion with chunking
- [x] Vector search
- [ ] Full-text search
- [ ] Hybrid ranking and a FastAPI search endpoint
- [ ] Retrieval evals (recall@k) and results table

## License

[MIT](LICENSE)
