"""Checks against the real all-MiniLM-L6-v2 model. Skipped without the neural extra."""

import pytest

from hybrid_rag.config import EMBEDDING_DIM
from hybrid_rag.embedding import HashingEmbedder, SentenceTransformerEmbedder

pytest.importorskip("sentence_transformers")
pytestmark = pytest.mark.neural


@pytest.fixture(scope="module")
def minilm() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()


def test_minilm_matches_schema_dim(minilm: SentenceTransformerEmbedder) -> None:
    assert minilm.dim == EMBEDDING_DIM


def test_minilm_captures_meaning_without_word_overlap(
    minilm: SentenceTransformerEmbedder,
) -> None:
    query = "chest pain on exertion"
    texts = [
        query,
        "Patient reports angina when climbing stairs.",
        "Patient reports wheezing; reviewed inhaler technique.",
    ]
    q, angina, asthma = minilm.embed(texts)
    assert float(q @ angina) > float(q @ asthma)

    # The hashing baseline sees no shared words in either note, so it cannot tell them apart.
    hq, h_angina, h_asthma = HashingEmbedder().embed(texts)
    assert abs(float(hq @ h_angina) - float(hq @ h_asthma)) < 0.1
