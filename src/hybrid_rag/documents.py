"""The document type that ingestion accepts, plus JSONL read/write helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    """A source document before chunking.

    `external_id` is the stable key: ingesting a document with the same id again
    replaces the stored copy instead of duplicating it.
    """

    external_id: str
    title: str
    doc_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("external_id", "title", "doc_type", "text"):
            if not getattr(self, name).strip():
                raise ValueError(f"Document.{name} must not be empty")


def write_jsonl(docs: Iterable[Document], path: Path | str) -> int:
    """Write one JSON object per line. Returns the number of documents written."""
    count = 0
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        for doc in docs:
            f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path | str) -> Iterator[Document]:
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield Document(**json.loads(line))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: invalid document: {exc}") from exc
