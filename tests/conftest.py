from __future__ import annotations

import os
from collections.abc import Iterator

import psycopg
import pytest

from hybrid_rag.config import load_dotenv


@pytest.fixture(scope="session")
def db_url() -> str:
    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set; skipping database tests")
    return url


@pytest.fixture()
def clean_db(db_url: str) -> Iterator[str]:
    """A database with no app tables, so every test starts from scratch."""

    def reset() -> None:
        with psycopg.connect(db_url, autocommit=True) as conn:
            conn.execute("DROP TABLE IF EXISTS chunks, documents, schema_migrations CASCADE")

    reset()
    yield db_url
    reset()
