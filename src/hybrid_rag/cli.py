"""Command-line entry point: `hybrid-rag <command>`."""

from __future__ import annotations

import argparse
import sys

from hybrid_rag.config import ConfigError
from hybrid_rag.db import connect, migrate
from hybrid_rag.documents import read_jsonl, write_jsonl
from hybrid_rag.embedding import HashingEmbedder
from hybrid_rag.ingest import ingest_documents
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
                    HashingEmbedder(),
                    max_chars=args.max_chars,
                    overlap_chars=args.overlap_chars,
                )
            print(f"Ingested {stats.documents} documents as {stats.chunks} chunks.")
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
