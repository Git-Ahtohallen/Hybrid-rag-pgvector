"""Split documents into overlapping chunks that respect paragraph and sentence breaks.

Chunks are packed greedily from paragraphs. A paragraph that is too long is split into
sentences, and a sentence that is still too long is split on whitespace. Each new
chunk starts with the last few pieces of the previous chunk (up to `overlap_chars`),
so a fact that straddles a boundary is still retrievable from one chunk.
"""

from __future__ import annotations

import re

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, max_chars: int = 1000, overlap_chars: int = 150) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars must be between 0 and max_chars - 1")

    pieces = _split_into_pieces(text, max_chars)
    chunks: list[str] = []
    current: list[str] = []
    size = 0  # length of "\n".join(current)

    for piece in pieces:
        added = len(piece) + (1 if current else 0)
        if current and size + added > max_chars:
            chunks.append("\n".join(current))
            current = _tail(current, overlap_chars, room=max_chars - len(piece) - 1)
            size = len("\n".join(current))
            added = len(piece) + (1 if current else 0)
        current.append(piece)
        size += added

    if current:
        chunks.append("\n".join(current))
    return chunks


def _split_into_pieces(text: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    for para in _PARAGRAPH_BREAK.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            pieces.append(para)
            continue
        for sentence in _SENTENCE_END.split(" ".join(para.split())):
            if len(sentence) <= max_chars:
                pieces.append(sentence)
            else:
                pieces.extend(_split_words(sentence, max_chars))
    return pieces


def _split_words(sentence: str, max_chars: int) -> list[str]:
    out: list[str] = []
    line = ""
    for word in sentence.split():
        while len(word) > max_chars:  # a single huge token: hard-cut it
            if line:
                out.append(line)
                line = ""
            out.append(word[:max_chars])
            word = word[max_chars:]
        candidate = f"{line} {word}" if line else word
        if len(candidate) > max_chars:
            out.append(line)
            line = word
        else:
            line = candidate
    if line:
        out.append(line)
    return out


def _tail(pieces: list[str], overlap_chars: int, room: int) -> list[str]:
    """The trailing pieces whose joined length fits in both the overlap and the room left."""
    budget = min(overlap_chars, room)
    tail: list[str] = []
    size = 0
    for piece in reversed(pieces):
        added = len(piece) + (1 if tail else 0)
        if size + added > budget:
            break
        tail.insert(0, piece)
        size += added
    return tail
