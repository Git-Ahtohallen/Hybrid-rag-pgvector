"""Turn text into vectors for the `chunks.embedding` column.

`HashingEmbedder` is a dependency-free baseline: it hashes word unigrams and bigrams
into a fixed number of buckets and L2-normalizes the result. It captures word overlap,
not meaning, so it is useful for tests, CI and as a lower bound in retrieval evals. A
neural embedder with the same `Embedder` interface can replace it without schema changes
as long as it produces `EMBEDDING_DIM` dimensions.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from hybrid_rag.config import EMBEDDING_DIM

Vector = npt.NDArray[np.float32]

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")


@runtime_checkable
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        """One unit-length vector per input text."""
        ...


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; keeps dotted codes such as `e11.9` together."""
    return _TOKEN.findall(text.lower())


class HashingEmbedder:
    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> Vector:
        vec = np.zeros(self._dim, dtype=np.float32)
        tokens = tokenize(text)
        features = tokens + [f"{a} {b}" for a, b in itertools.pairwise(tokens)]
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            h = int.from_bytes(digest, "little")
            sign = 1.0 if h & 1 else -1.0
            vec[(h >> 1) % self._dim] += sign
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            # Empty or symbol-only text: a fixed unit vector keeps cosine distance defined.
            vec[0] = 1.0
            return vec
        return vec / norm
