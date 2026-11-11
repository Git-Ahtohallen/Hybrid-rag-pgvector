"""Database connection and schema migrations."""

from __future__ import annotations

from importlib import resources

import psycopg
from pgvector.psycopg import register_vector

from hybrid_rag.config import database_url

MIGRATIONS_PACKAGE = "hybrid_rag.migrations"


def connect(url: str | None = None) -> psycopg.Connection:
    """Open a connection with pgvector types registered.

    Run `migrate()` first on a fresh database: registering the vector type needs
    the extension to exist.
    """
    conn = psycopg.connect(url or database_url())
    register_vector(conn)
    return conn


def available_migrations() -> list[tuple[str, str]]:
    """(name, sql) for every bundled migration, in filename order."""
    files = [f for f in resources.files(MIGRATIONS_PACKAGE).iterdir() if f.name.endswith(".sql")]
    return [(f.name, f.read_text(encoding="utf-8")) for f in sorted(files, key=lambda f: f.name)]


def migrate(url: str | None = None) -> list[str]:
    """Apply pending migrations, each in its own transaction. Returns names applied."""
    applied_now: list[str] = []
    with psycopg.connect(url or database_url(), autocommit=True) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name       TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        done = {row[0] for row in conn.execute("SELECT name FROM schema_migrations")}
        for name, sql in available_migrations():
            if name in done:
                continue
            with conn.transaction():
                conn.execute(sql.encode())
                conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (name,))
            applied_now.append(name)
    return applied_now
