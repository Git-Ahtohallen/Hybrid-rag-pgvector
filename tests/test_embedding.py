import numpy as np
import pytest

from hybrid_rag.config import EMBEDDING_DIM
from hybrid_rag.embedding import Embedder, HashingEmbedder, tokenize


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
