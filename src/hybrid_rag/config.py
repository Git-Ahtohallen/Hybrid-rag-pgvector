"""Settings, read from environment variables (optionally via a local .env file)."""

from __future__ import annotations

import os
from pathlib import Path

# Must match the vector(...) column size in migrations/001_init.sql.
EMBEDDING_DIM = 384


class ConfigError(RuntimeError):
    pass


def load_dotenv(path: Path | str = ".env") -> None:
    """Load KEY=VALUE lines from a .env file without overriding real env vars."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def database_url() -> str:
    load_dotenv()
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise ConfigError(
            "DATABASE_URL is not set. Copy .env.example to .env, or export DATABASE_URL."
        )
    return url
