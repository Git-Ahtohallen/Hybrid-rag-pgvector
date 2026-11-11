"""Command-line entry point: `hybrid-rag <command>`."""

from __future__ import annotations

import argparse
import sys

from hybrid_rag.config import ConfigError
from hybrid_rag.db import migrate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hybrid-rag")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("migrate", help="create or update the database schema")
    args = parser.parse_args(argv)

    try:
        if args.command == "migrate":
            applied = migrate()
            print(f"Applied: {', '.join(applied)}" if applied else "Schema is up to date.")
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
