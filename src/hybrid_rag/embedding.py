"""Turn text into vectors for the `chunks.embedding` column.

Two embedders share the `Embedder` interface:

- `HashingEmbedder` is a dependency-free baseline: it hashes word unigrams and bigrams into
  a fixed number of buckets and L2-normalizes the result. It captures word overlap, not
  meaning, so it is used for tests and CI, and as a lower bound in retrieval evals.
- `SentenceTransformerEmbedder` wraps a sentence-transformers model, by default
  all-MiniLM-L6-v2 (384 dimensions, the same size as the schema). It needs the optional
  `neural` extra: `pip install -e ".[neural]"`.

Every embedder has a `name`. Ingestion stores it next to each vector, and search refuses to
compare a query vector against vectors from a different model.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from hybrid_rag.config import EMBEDDING_DIM

Vector = npt.NDArray[np.float32]

DEFAULT_NEURAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")


@runtime_checkable
class Embedder(Protocol):
    @property
    def name(self) -> str:
        """Identifies the model; vectors are only comparable within one name."""
        ...

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
    def name(self) -> str:
        return f"hashing-{self._dim}"

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


class SentenceEncoder(Protocol):
    """The part of `sentence_transformers.SentenceTransformer` this module uses."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> npt.NDArray[Any]: ...

    def get_sentence_embedding_dimension(self) -> int | None: ...


class SentenceTransformerEmbedder:
    """Neural embeddings from a sentence-transformers model.

    The model is downloaded from the Hugging Face Hub on first use and cached locally.
    Pass `model` to use an already-loaded model (or a test double) instead.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_NEURAL_MODEL,
        *,
        model: SentenceEncoder | None = None,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if model is None:
            model = _load_sentence_transformer(model_name, device)
        dim = model.get_sentence_embedding_dimension()
        if not dim:
            raise ValueError(f"model {model_name!r} does not report an embedding dimension")
        self._model = model
        self._name = model_name
        self._dim = dim
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        matrix = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return list(np.asarray(matrix, dtype=np.float32))


def _load_sentence_transformer(model_name: str, device: str | None) -> SentenceEncoder:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "SentenceTransformerEmbedder needs the optional 'neural' extra: "
            'pip install -e ".[neural]"'
        ) from exc
    model: SentenceEncoder = SentenceTransformer(model_name, device=device)
    return model


EMBEDDER_CHOICES = ("hashing", "minilm")


def make_embedder(kind: str) -> Embedder:
    """Build an embedder by its CLI name: `hashing` (offline) or `minilm` (neural)."""
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "minilm":
        return SentenceTransformerEmbedder()
    raise ValueError(f"unknown embedder {kind!r}; choose one of {', '.join(EMBEDDER_CHOICES)}")
