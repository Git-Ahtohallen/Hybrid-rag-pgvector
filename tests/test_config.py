from pathlib import Path

import pytest

from hybrid_rag import config
from hybrid_rag.db import available_migrations


def _unset_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # setenv first so monkeypatch records the original value and restores it after the
    # test; delenv alone does not track a variable that was already unset.
    monkeypatch.setenv("DATABASE_URL", "placeholder")
    monkeypatch.delenv("DATABASE_URL")


def test_database_url_missing_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # no .env here
    _unset_database_url(monkeypatch)
    with pytest.raises(config.ConfigError, match="DATABASE_URL"):
        config.database_url()


def test_dotenv_is_loaded_but_does_not_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text(
        "# comment\nDATABASE_URL='postgresql://from-file'\nOTHER=1\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    _unset_database_url(monkeypatch)
    assert config.database_url() == "postgresql://from-file"

    monkeypatch.setenv("DATABASE_URL", "postgresql://from-env")
    assert config.database_url() == "postgresql://from-env"


def test_migration_vector_size_matches_config() -> None:
    sql = dict(available_migrations())["001_init.sql"]
    assert f"vector({config.EMBEDDING_DIM})" in sql
