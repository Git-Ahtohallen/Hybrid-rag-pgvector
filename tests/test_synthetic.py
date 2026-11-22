import re
from pathlib import Path

import pytest

from hybrid_rag.documents import Document, read_jsonl, write_jsonl
from hybrid_rag.synthetic import CONDITIONS, generate_visit_notes


def test_same_seed_same_notes() -> None:
    assert generate_visit_notes(10, seed=1) == generate_visit_notes(10, seed=1)
    assert generate_visit_notes(10, seed=1) != generate_visit_notes(10, seed=2)


def test_notes_have_unique_ids_and_are_marked_synthetic() -> None:
    notes = generate_visit_notes(100)
    assert len({n.external_id for n in notes}) == 100
    for n in notes:
        assert n.metadata["synthetic"] is True
        assert re.fullmatch(r"SYN-\d{5}", n.metadata["patient_id"])
        assert n.text.startswith("Synthetic visit note.")


def test_codes_and_drugs_in_metadata_appear_in_text() -> None:
    for n in generate_visit_notes(30):
        for code in n.metadata["icd10"]:
            assert f"ICD-10 {code}" in n.text
        for code in n.metadata["cpt"]:
            assert f"CPT {code}" in n.text
        for drug in n.metadata["medications"]:
            assert drug in n.text


def test_every_condition_is_used() -> None:
    used = {key for n in generate_visit_notes(100) for key in n.metadata["conditions"]}
    assert used == {c.key for c in CONDITIONS}


def test_icd10_codes_are_well_formed() -> None:
    for c in CONDITIONS:
        assert re.fullmatch(r"[A-Z]\d{2}(\.\d{1,4})?", c.icd10), c.icd10
        for cpt, _ in c.orders:
            assert re.fullmatch(r"\d{5}", cpt), cpt


def test_jsonl_round_trip(tmp_path: Path) -> None:
    notes = generate_visit_notes(5)
    path = tmp_path / "notes.jsonl"
    assert write_jsonl(notes, path) == 5
    assert list(read_jsonl(path)) == notes


def test_document_rejects_empty_fields() -> None:
    with pytest.raises(ValueError, match="text"):
        Document(external_id="x", title="t", doc_type="note", text="  ")
