"""Deterministic synthetic outpatient visit notes.

Everything here is made up. Patients are identified only by synthetic IDs (`SYN-00001`),
and no note is based on a real person. The clinical vocabulary is realistic on purpose:
real ICD-10-CM diagnosis codes, CPT procedure codes and drug names with typical doses.
That is the kind of exact-term content where keyword search beats embeddings, which is
what the hybrid retrieval evals need to measure.

The same seed always produces the same notes, so evals and tests are reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta

from hybrid_rag.documents import Document

DOC_TYPE = "visit_note"


@dataclass(frozen=True, slots=True)
class Medication:
    name: str
    dose: str
    sig: str


@dataclass(frozen=True, slots=True)
class Condition:
    key: str
    name: str
    icd10: str
    complaints: tuple[str, ...]
    history: tuple[str, ...]
    exam: str
    medications: tuple[Medication, ...]
    orders: tuple[tuple[str, str], ...]  # (CPT code, description)
    plan: tuple[str, ...]


CONDITIONS: tuple[Condition, ...] = (
    Condition(
        key="t2dm",
        name="Type 2 diabetes mellitus without complications",
        icd10="E11.9",
        complaints=("follow-up of blood sugar control", "increased thirst and urination"),
        history=(
            "Reports home fasting glucose readings between 140 and 180 mg/dL.",
            "Has been walking 20 minutes most days and reducing sugary drinks.",
            "Denies hypoglycemic episodes, vision changes or foot numbness.",
        ),
        exam="Monofilament foot exam normal bilaterally. No skin breakdown.",
        medications=(
            Medication("metformin", "500 mg", "twice daily with meals"),
            Medication("empagliflozin", "10 mg", "once daily"),
        ),
        orders=(("83036", "Hemoglobin A1c"), ("82043", "Urine albumin, quantitative")),
        plan=(
            "Repeat HbA1c in 3 months.",
            "Reinforced diet and exercise counseling.",
            "Annual dilated eye exam referral placed.",
        ),
    ),
    Condition(
        key="htn",
        name="Essential (primary) hypertension",
        icd10="I10",
        complaints=("blood pressure recheck", "occasional morning headaches"),
        history=(
            "Home blood pressure log averages 148/92 over the past two weeks.",
            "Limits salt intake but eats out several times a week.",
            "Denies chest pain, shortness of breath or palpitations.",
        ),
        exam="Blood pressure 146/90 seated, repeat 142/88. Heart regular rate and rhythm.",
        medications=(
            Medication("lisinopril", "10 mg", "once daily"),
            Medication("amlodipine", "5 mg", "once daily"),
        ),
        orders=(("80048", "Basic metabolic panel"), ("93000", "Electrocardiogram, 12-lead")),
        plan=(
            "Continue home blood pressure monitoring twice daily.",
            "DASH diet handout provided.",
            "Recheck potassium and creatinine after dose change.",
        ),
    ),
    Condition(
        key="hld",
        name="Hyperlipidemia, unspecified",
        icd10="E78.5",
        complaints=("review of cholesterol results",),
        history=(
            "Recent LDL cholesterol 162 mg/dL. Family history of early heart disease.",
            "No muscle aches on current therapy.",
        ),
        exam="No xanthomas. Peripheral pulses intact.",
        medications=(Medication("atorvastatin", "20 mg", "once daily at bedtime"),),
        orders=(("80061", "Lipid panel"), ("80076", "Hepatic function panel")),
        plan=(
            "Repeat lipid panel in 6 to 8 weeks.",
            "Discussed 10-year cardiovascular risk estimate.",
        ),
    ),
    Condition(
        key="asthma",
        name="Unspecified asthma, uncomplicated",
        icd10="J45.909",
        complaints=("wheezing with exercise", "night-time cough"),
        history=(
            "Uses rescue inhaler about three times a week, more in cold weather.",
            "Woken by cough twice in the last month. No emergency visits this year.",
        ),
        exam="Scattered expiratory wheeze, good air movement. Oxygen saturation 98% on room air.",
        medications=(
            Medication("albuterol HFA", "90 mcg/actuation", "2 puffs every 4 to 6 hours as needed"),
            Medication("fluticasone HFA", "110 mcg/actuation", "2 puffs twice daily"),
        ),
        orders=(("94010", "Spirometry"), ("94664", "Inhaler technique demonstration")),
        plan=(
            "Written asthma action plan reviewed.",
            "Rinse mouth after inhaled steroid.",
            "Return sooner if rescue inhaler use exceeds twice a week.",
        ),
    ),
    Condition(
        key="gad",
        name="Generalized anxiety disorder",
        icd10="F41.1",
        complaints=("ongoing worry and poor sleep", "follow-up of anxiety medication"),
        history=(
            "GAD-7 score today is 12, down from 16 at the last visit.",
            "Sleeping about 6 hours a night. Denies thoughts of self-harm.",
        ),
        exam="Alert, cooperative, mildly anxious affect. Speech normal rate.",
        medications=(Medication("sertraline", "50 mg", "once daily"),),
        orders=(("96127", "Brief emotional/behavioral assessment"),),
        plan=(
            "Continue sertraline; consider increase to 100 mg if symptoms persist.",
            "Referral to cognitive behavioral therapy.",
            "Sleep hygiene counseling given.",
        ),
    ),
    Condition(
        key="gerd",
        name="Gastro-esophageal reflux disease without esophagitis",
        icd10="K21.9",
        complaints=("heartburn after meals", "sour taste at night"),
        history=(
            "Burning chest discomfort after large or spicy meals, worse when lying down.",
            "No trouble swallowing, weight loss or black stools.",
        ),
        exam="Abdomen soft, mild epigastric tenderness, no masses.",
        medications=(Medication("omeprazole", "20 mg", "once daily before breakfast"),),
        orders=(),
        plan=(
            "Raise the head of the bed and avoid meals within 3 hours of bedtime.",
            "Trial of omeprazole for 8 weeks, then reassess.",
        ),
    ),
    Condition(
        key="lbp",
        name="Low back pain, unspecified",
        icd10="M54.50",
        complaints=("low back pain after lifting", "stiff lower back for two weeks"),
        history=(
            "Pain started after moving furniture. Rated 5 out of 10, no radiation to the legs.",
            "No bowel or bladder changes, no fever, no numbness.",
        ),
        exam="Paraspinal muscle tenderness, straight leg raise negative bilaterally.",
        medications=(
            Medication("naproxen", "500 mg", "twice daily with food for up to 10 days"),
            Medication("cyclobenzaprine", "5 mg", "at bedtime as needed"),
        ),
        orders=(("97110", "Therapeutic exercise, physical therapy"),),
        plan=(
            "Stay active; avoid bed rest.",
            "Physical therapy referral for core strengthening.",
            "Return if red-flag symptoms develop.",
        ),
    ),
    Condition(
        key="uti",
        name="Urinary tract infection, site not specified",
        icd10="N39.0",
        complaints=("burning with urination", "urinary frequency for three days"),
        history=(
            "Dysuria and urgency without fever or flank pain.",
            "No vaginal discharge. Last similar episode over a year ago.",
        ),
        exam="No costovertebral angle tenderness. Mild suprapubic tenderness.",
        medications=(Medication("nitrofurantoin", "100 mg", "twice daily for 5 days"),),
        orders=(("81003", "Urinalysis, automated, without microscopy"), ("87086", "Urine culture")),
        plan=(
            "Increase fluid intake.",
            "Will adjust antibiotic if culture shows resistance.",
        ),
    ),
    Condition(
        key="migraine",
        name="Migraine, unspecified, not intractable, without status migrainosus",
        icd10="G43.909",
        complaints=("recurring headaches with light sensitivity",),
        history=(
            "Throbbing one-sided headaches about four times a month, lasting most of a day.",
            "Nausea and light sensitivity during attacks. Triggers include poor sleep.",
        ),
        exam="Neurological exam non-focal. Fundi normal.",
        medications=(
            Medication("sumatriptan", "50 mg", "at headache onset, may repeat once after 2 hours"),
            Medication("propranolol ER", "80 mg", "once daily for prevention"),
        ),
        orders=(),
        plan=(
            "Headache diary to track frequency and triggers.",
            "Limit acute medication to fewer than 10 days a month.",
        ),
    ),
)

_VISIT_CODES = (
    ("99213", "Office visit, established patient, low complexity"),
    ("99214", "Office visit, established patient, moderate complexity"),
)
_SEXES = ("female", "male")
_START = date(2025, 1, 6)


def generate_visit_notes(count: int = 50, seed: int = 7) -> list[Document]:
    """`count` synthetic visit notes, identical for the same seed."""
    if count < 0:
        raise ValueError("count must not be negative")
    rng = random.Random(seed)
    return [_note(i, rng) for i in range(1, count + 1)]


def _note(index: int, rng: random.Random) -> Document:
    patient_id = f"SYN-{rng.randint(1, 99_999):05d}"
    age = rng.randint(22, 84)
    sex = rng.choice(_SEXES)
    visit_date = _START + timedelta(days=rng.randint(0, 364))
    conditions = rng.sample(CONDITIONS, k=rng.choice((1, 2, 2, 3)))
    primary = conditions[0]
    visit_code = rng.choice(_VISIT_CODES) if len(conditions) == 1 else _VISIT_CODES[1]

    meds = [m for c in conditions for m in c.medications]
    orders = [visit_code, *(o for c in conditions for o in c.orders)]

    sections = [
        f"Synthetic visit note. Patient {patient_id}, {age}-year-old {sex}. "
        f"Visit date {visit_date.isoformat()}.",
        f"Chief complaint: {rng.choice(primary.complaints)}.",
        "History of present illness: "
        + " ".join(s for c in conditions for s in rng.sample(c.history, k=len(c.history))),
        "Exam: " + " ".join(c.exam for c in conditions),
        "Medications:\n" + "\n".join(f"- {m.name} {m.dose} {m.sig}" for m in meds),
        "Assessment:\n" + "\n".join(f"- {c.name} (ICD-10 {c.icd10})" for c in conditions),
        "Plan: " + " ".join(p for c in conditions for p in c.plan),
        "Billing:\n" + "\n".join(f"- CPT {code} {desc}" for code, desc in orders),
    ]
    return Document(
        external_id=f"visit-{index:05d}",
        title=f"Visit note: {primary.name}",
        doc_type=DOC_TYPE,
        text="\n\n".join(sections),
        metadata={
            "synthetic": True,
            "patient_id": patient_id,
            "visit_date": visit_date.isoformat(),
            "conditions": [c.key for c in conditions],
            "icd10": [c.icd10 for c in conditions],
            "cpt": [code for code, _ in orders],
            "medications": [m.name for m in meds],
        },
    )
