"""Clinical trial report ingestion (NLI4CT) and structured representation.

Each ClinicalTrials.gov record becomes a TrialRecord with four sections
(intervention, eligibility, results, adverse events) kept line by line. Line
indices are preserved because the NLI4CT gold annotations point at them, which
lets the evaluation suite measure evidence retrieval at sentence level.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

SECTION_KEYS = {
    "Intervention": "intervention",
    "Eligibility": "eligibility",
    "Results": "results",
    "Adverse Events": "adverse_events",
}

DRUG_LEXICON = [
    "trastuzumab", "pertuzumab", "lapatinib", "neratinib", "tucatinib", "trastuzumab emtansine", "t-dm1",
    "trastuzumab deruxtecan", "tamoxifen", "letrozole", "anastrozole", "exemestane", "fulvestrant",
    "goserelin", "palbociclib", "ribociclib", "abemaciclib", "everolimus", "alpelisib", "olaparib",
    "talazoparib", "capecitabine", "paclitaxel", "nab-paclitaxel", "docetaxel", "doxorubicin", "epirubicin",
    "cyclophosphamide", "carboplatin", "cisplatin", "gemcitabine", "vinorelbine", "eribulin", "ixabepilone",
    "bevacizumab", "pembrolizumab", "atezolizumab", "nivolumab", "durvalumab", "sacituzumab", "zoledronic acid",
    "denosumab", "fluorouracil", "methotrexate", "radiotherapy", "radiation", "surgery", "metformin",
    "celecoxib", "sunitinib", "sorafenib", "temsirolimus", "veliparib", "iniparib", "cixutumumab",
]

PHASE_RX = re.compile(r"\bphase\s*(i{1,3}|[123])(?:\s*/\s*(i{1,3}|[123]))?\b", re.I)


@dataclass
class TrialRecord:
    nct_id: str
    intervention: list[str] = field(default_factory=list)
    eligibility: list[str] = field(default_factory=list)
    results: list[str] = field(default_factory=list)
    adverse_events: list[str] = field(default_factory=list)
    title: str = ""
    agents: list[str] = field(default_factory=list)
    arms: list[str] = field(default_factory=list)
    setting: str = "unspecified"
    phase_hint: str = ""

    @property
    def url(self) -> str:
        return f"https://clinicaltrials.gov/study/{self.nct_id}"

    def section(self, name: str) -> list[str]:
        return getattr(self, name)

    def full_text(self) -> str:
        return " ".join(self.intervention + self.eligibility + self.results + self.adverse_events)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["url"] = self.url
        return d


def _clean(lines: list[str]) -> list[str]:
    """Strip trailing whitespace but keep every line, including blanks, so that
    positions stay aligned with the NLI4CT gold evidence indices."""
    return [ln.rstrip() for ln in lines]


def _arms(intervention: list[str]) -> list[str]:
    arms = []
    for i, ln in enumerate(intervention):
        if ln.strip().upper().startswith("INTERVENTION") and i + 1 < len(intervention):
            arms.append(intervention[i + 1].strip())
    return arms


def _agents(text: str) -> list[str]:
    low = text.lower()
    found = []
    for drug in DRUG_LEXICON:
        if re.search(rf"\b{re.escape(drug)}\b", low) and drug not in found:
            found.append(drug)
    return found


EARLY_RX = re.compile(
    r"operable|\bm0\b|adjuvant|non[- ]?metastatic|no evidence of (distant )?metasta|stage (0|i{1,3}|[123])[abc]?\b(?!\s?v)|"
    r"definitive (breast )?surgery|lumpectomy|mastectomy|completely resected|resected|early[- ]stage|post[- ]?operative",
    re.I,
)
ADVANCED_RX = re.compile(r"metastatic|stage iv|advanced|locally recurrent|unresectable|\bm1\b|measurable disease", re.I)
NEGATED_METASTASIS = re.compile(r"(no|without|absence of|not have)[^.;]{0,40}metasta", re.I)


def _setting(eligibility: list[str]) -> str:
    """Infer the trial population from the inclusion part of the criteria only,
    because exclusion lists mention metastatic disease in early stage trials."""
    inclusion = []
    for ln in eligibility:
        if re.match(r"^\s*exclusion criteria", ln, re.I):
            break
        inclusion.append(ln)
    text = " ".join(inclusion)
    low = text.lower()
    if re.search(r"neoadjuvant|preoperative|prior to (definitive )?surgery", low):
        return "neoadjuvant"
    early = len(EARLY_RX.findall(text))
    advanced = len(ADVANCED_RX.findall(NEGATED_METASTASIS.sub(" ", text)))
    if advanced > early:
        return "advanced or metastatic"
    if early:
        return "early or adjuvant"
    if re.search(r"prevention|high risk|healthy|no history of breast cancer|survivor", low):
        return "prevention or screening"
    if advanced:
        return "advanced or metastatic"
    return "unspecified"


def build_title(rec: TrialRecord) -> str:
    agents = [a for a in rec.agents if a not in {"surgery", "radiation"}][:3]
    setting = rec.setting if rec.setting != "unspecified" else "breast cancer"
    if agents:
        return f"{', '.join(a.title() for a in agents)} in {setting} breast cancer".replace("breast cancer breast cancer", "breast cancer")
    if rec.arms:
        return rec.arms[0][:110]
    return f"Clinical trial {rec.nct_id}"


def _iter_raw_trials(raw_dir: Path):
    folder = raw_dir / "nli4ct" / "Complete_dataset" / "CT json"
    for path in sorted(folder.glob("NCT*.json")):
        yield json.loads(path.read_text(encoding="utf-8"))
    live = raw_dir / "live_trials" / "live_trials.json"
    if live.exists():  # written by scripts/fetch_live_trials.py
        yield from json.loads(live.read_text(encoding="utf-8"))


def load_trials(raw_dir: Path) -> list[TrialRecord]:
    records: list[TrialRecord] = []
    seen: set[str] = set()
    for d in _iter_raw_trials(raw_dir):
        if not d.get("Clinical Trial ID") or d["Clinical Trial ID"] in seen:
            continue
        seen.add(d["Clinical Trial ID"])
        rec = TrialRecord(nct_id=d["Clinical Trial ID"])
        for key, attr in SECTION_KEYS.items():
            setattr(rec, attr, _clean(d.get(key, [])))
        joined = rec.full_text()
        rec.agents = _agents(" ".join(rec.intervention) or joined)
        rec.arms = _arms(rec.intervention)
        rec.setting = _setting(rec.eligibility)
        m = PHASE_RX.search(joined)
        rec.phase_hint = f"phase {m.group(0).split()[-1].upper()}" if m else ""
        rec.title = d.get("Title") or build_title(rec)
        records.append(rec)
    return records


@dataclass
class NLIStatement:
    uid: str
    kind: str
    section: str
    primary_id: str
    secondary_id: str | None
    statement: str
    label: str | None
    primary_evidence: list[int]
    secondary_evidence: list[int]
    split: str


def load_statements(raw_dir: Path) -> list[NLIStatement]:
    base = raw_dir / "nli4ct" / "Complete_dataset"
    out: list[NLIStatement] = []
    for split, fname in (("train", "train.json"), ("dev", "dev.json"), ("test", "Gold_test.json")):
        p = base / fname
        if not p.exists():
            continue
        for uid, v in json.loads(p.read_text()).items():
            out.append(
                NLIStatement(
                    uid=uid,
                    kind=v["Type"],
                    section=SECTION_KEYS.get(v["Section_id"], v["Section_id"]),
                    primary_id=v["Primary_id"],
                    secondary_id=v.get("Secondary_id"),
                    statement=v["Statement"],
                    label=v.get("Label"),
                    primary_evidence=list(v.get("Primary_evidence_index", [])),
                    secondary_evidence=list(v.get("Secondary_evidence_index", [])),
                    split=split,
                )
            )
    return out
