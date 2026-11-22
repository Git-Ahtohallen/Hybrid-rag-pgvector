import itertools

import pytest

from hybrid_rag.chunking import chunk_text


def test_short_text_is_one_chunk() -> None:
    assert chunk_text("Patient started metformin 500 mg.") == ["Patient started metformin 500 mg."]


def test_empty_text_has_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("\n\n  \n") == []


def test_chunks_respect_max_chars() -> None:
    text = "\n\n".join(f"Paragraph {i}. " + "Some clinical detail here. " * 5 for i in range(30))
    chunks = chunk_text(text, max_chars=300, overlap_chars=60)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_paragraphs_are_not_split_when_they_fit() -> None:
    paras = [f"Section {i}: " + "x" * 80 for i in range(6)]
    chunks = chunk_text("\n\n".join(paras), max_chars=200, overlap_chars=0)
    for para in paras:
        assert sum(para in c for c in chunks) == 1


def test_consecutive_chunks_overlap() -> None:
    sentences = [f"Sentence number {i} about the plan." for i in range(40)]
    chunks = chunk_text(" ".join(sentences), max_chars=200, overlap_chars=80)
    for prev, nxt in itertools.pairwise(chunks):
        first_piece = nxt.split("\n")[0]
        assert first_piece in prev


def test_no_overlap_when_disabled() -> None:
    sentences = [f"Sentence {i}." for i in range(60)]
    chunks = chunk_text(" ".join(sentences), max_chars=100, overlap_chars=0)
    joined = "\n".join(chunks).split("\n")
    assert joined == sentences


def test_every_word_survives() -> None:
    text = "alpha beta " * 300 + "\n\n" + "ICD-10 E11.9 and CPT 99214. " * 40
    chunks = chunk_text(text, max_chars=120, overlap_chars=30)
    assert set(" ".join(chunks).split()) == set(text.split())
    assert all(len(c) <= 120 for c in chunks)


def test_huge_token_is_hard_cut() -> None:
    chunks = chunk_text("a" * 250, max_chars=100, overlap_chars=0)
    assert chunks == ["a" * 100, "a" * 100, "a" * 50]


@pytest.mark.parametrize(("max_chars", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_invalid_sizes(max_chars: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("text", max_chars=max_chars, overlap_chars=overlap)
