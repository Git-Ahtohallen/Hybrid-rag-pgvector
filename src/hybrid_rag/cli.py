"""Command-line entry point: `hybrid-rag <command>`."""

from __future__ import annotations

import argparse
import sys

from hybrid_rag.config import ConfigError
from hybrid_rag.db import connect, migrate
from hybrid_rag.documents import read_jsonl, write_jsonl
from hybrid_rag.embedding import EMBEDDER_CHOICES, make_embedder
from hybrid_rag.ingest import ingest_documents
from hybrid_rag.search import vector_search
from hybrid_rag.synthetic import generate_visit_notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hybrid-rag")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("migrate", help="create or update the database schema")

    gen = sub.add_parser("generate", help="write synthetic visit notes to a JSONL file")
    gen.add_argument("out", help="output .jsonl path")
    gen.add_argument("--count", type=int, default=50)
    gen.add_argument("--seed", type=int, default=7)

    ing = sub.add_parser("ingest", help="chunk, embed and store documents")
    src = ing.add_mutually_exclusive_group()
    src.add_argument("--file", help="JSONL file of documents (default: synthetic notes)")
    src.add_argument("--count", type=int, default=50, help="number of synthetic notes")
    ing.add_argument("--seed", type=int, default=7)
    ing.add_argument("--max-chars", type=int, default=1000)
    ing.add_argument("--overlap-chars", type=int, default=150)
    _add_embedder_arg(ing)

    srch = sub.add_parser("search", help="vector search over the ingested chunks")
    srch.add_argument("query")
    srch.add_argument("-k", type=int, default=5, help="number of results")
    _add_embedder_arg(srch)

    args = parser.parse_args(argv)

    try:
        if args.command == "migrate":
            applied = migrate()
            print(f"Applied: {', '.join(applied)}" if applied else "Schema is up to date.")
        elif args.command == "generate":
            n = write_jsonl(generate_visit_notes(args.count, args.seed), args.out)
            print(f"Wrote {n} synthetic notes to {args.out}")
        elif args.command == "ingest":
            docs = (
                read_jsonl(args.file) if args.file else generate_visit_notes(args.count, args.seed)
            )
            with connect() as conn:
                stats = ingest_documents(
                    conn,
                    docs,
                    make_embedder(args.embedder),
                    max_chars=args.max_chars,
                    overlap_chars=args.overlap_chars,
                )
            print(f"Ingested {stats.documents} documents as {stats.chunks} chunks.")
        elif args.command == "search":
            with connect() as conn:
                hits = vector_search(conn, args.query, make_embedder(args.embedder), k=args.k)
            if not hits:
                print("No chunks found. Run `hybrid-rag ingest` first.")
            for rank, hit in enumerate(hits, start=1):
                snippet = " ".join(hit.content.split())
                if len(snippet) > 160:
                    snippet = snippet[:157] + "..."
                print(
                    f"{rank:>2}. {hit.score:.3f}  {hit.external_id} #{hit.chunk_index}  {hit.title}"
                )
                print(f"    {snippet}")
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ImportError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _add_embedder_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embedder",
        choices=EMBEDDER_CHOICES,
        default="hashing",
        help="hashing: offline baseline (default); minilm: all-MiniLM-L6-v2 (neural extra)",
    )


if __name__ == "__main__":
    raise SystemExit(main())
