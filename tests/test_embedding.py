import sys
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from hybrid_rag.config import EMBEDDING_DIM
from hybrid_rag.embedding import (
    DEFAULT_NEURAL_MODEL,
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    make_embedder,
    tokenize,
)


def test_tokenize_keeps_codes_together() -> None:
    assert tokenize("ICD-10 E11.9, CPT 99214.") == ["icd", "10", "e11.9", "cpt", "99214"]


def test_vectors_are_unit_length_with_schema_dim() -> None:
    emb = HashingEmbedder()
    assert isinstance(emb, Embedder)
    vecs = emb.embed(["metformin 500 mg twice daily", "", "!!!"])
    for v in vecs:
        assert v.shape == (EMBEDDING_DIM,)
        assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-5)


def test_embedding_is_deterministic() -> None:
    a = HashingEmbedder().embed(["lisinopril 10 mg"])[0]
    b = HashingEmbedder().embed(["lisinopril 10 mg"])[0]
    assert np.array_equal(a, b)


def test_word_overlap_means_higher_similarity() -> None:
    emb = HashingEmbedder()
    query, near, far = emb.embed(
        [
            "metformin for type 2 diabetes",
            "patient takes metformin for diabetes",
            "inhaler technique for asthma",
        ]
    )
    assert float(query @ near) > float(query @ far)


def test_hashing_name_includes_dim() -> None:
    assert HashingEmbedder().name == "hashing-384"
    assert HashingEmbedder(dim=8).name == "hashing-8"


class FakeEncoder:
    """Stands in for a SentenceTransformer: returns unnormalized float64 rows."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim
        self.calls: list[dict[str, Any]] = []

    def get_sentence_embedding_dimension(self) -> int | None:
        return self.dim

    def encode(self, sentences: list[str], **kwargs: Any) -> npt.NDArray[np.float64]:
        self.calls.append({"sentences": sentences, **kwargs})
        rows = np.ones((len(sentences), self.dim))
        for i, s in enumerate(sentences):
            rows[i, 0] = len(s)
        return rows


def test_neural_embedder_wraps_model() -> None:
    fake = FakeEncoder()
    emb = SentenceTransformerEmbedder(model=fake, batch_size=8)
    assert isinstance(emb, Embedder)
    assert emb.name == DEFAULT_NEURAL_MODEL
    assert emb.dim == EMBEDDING_DIM

    vecs = emb.embed(("a", "bb"))
    assert len(vecs) == 2
    assert all(v.dtype == np.float32 and v.shape == (EMBEDDING_DIM,) for v in vecs)
    [call] = fake.calls
    assert call["sentences"] == ["a", "bb"]
    assert call["batch_size"] == 8
    assert call["normalize_embeddings"] is True
    assert call["show_progress_bar"] is False


def test_neural_embedder_skips_model_for_empty_input() -> None:
    fake = FakeEncoder()
    assert SentenceTransformerEmbedder(model=fake).embed([]) == []
    assert fake.calls == []


def test_neural_embedder_reports_model_dim() -> None:
    assert SentenceTransformerEmbedder("other", model=FakeEncoder(dim=768)).dim == 768


def test_neural_embedder_explains_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match=r"\[neural\]"):
        SentenceTransformerEmbedder()


def test_make_embedder() -> None:
    assert isinstance(make_embedder("hashing"), HashingEmbedder)
    with pytest.raises(ValueError, match="unknown embedder"):
        make_embedder("word2vec")
