"""Parser and breast oncology filter for the PubMed 200k RCT corpus.

File format (one abstract per block):

    ###24491034
    BACKGROUND<TAB>The emergence of ...
    METHODS<TAB>We randomly assigned ...
    <blank line>

The sentence role labels (BACKGROUND, OBJECTIVE, METHODS, RESULTS,
CONCLUSIONS) are preserved because they drive section aware chunking and the
reranker's preference for results and conclusion sentences when a clinician
asks about efficacy.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

BREAST_CORE = re.compile(
    r"\b(breast|mammary|mastectom\w*|lumpectom\w*|ductal carcinoma in situ|dcis|mammogra\w*)\b",
    re.IGNORECASE,
)
BREAST_SPECIFIC_AGENTS = re.compile(
    r"\b(tamoxifen|letrozole|anastrozole|exemestane|fulvestrant|trastuzumab|pertuzumab|lapatinib|neratinib|"
    r"tucatinib|palbociclib|ribociclib|abemaciclib|alpelisib|olaparib|talazoparib|t-?dm1|trastuzumab emtansine|"
    r"trastuzumab deruxtecan|sacituzumab)\b",
    re.IGNORECASE,
)
CANCER_CONTEXT = re.compile(
    r"\b(cancer|carcinoma|tumou?rs?|neoplas\w*|oncolog\w*|malignan\w*|metasta\w*|mastectom\w*|lumpectom\w*|"
    r"dcis|chemotherap\w*|radiotherap\w*|adjuvant)\b",
    re.IGNORECASE,
)
LACTATION = re.compile(r"\b(breast[- ]?fe(e)?d\w*|breast ?milk|lactat\w*|breast[- ]?pump|infant formula|nursing mothers?)\b", re.IGNORECASE)
NON_BREAST_PRIMARY = re.compile(
    r"\b(prostate cancer|lung cancer|colorectal cancer|gastric cancer|ovarian cancer|pancreatic cancer)\b",
    re.IGNORECASE,
)

TOPIC_RULES: dict[str, re.Pattern[str]] = {
    "endocrine therapy": re.compile(r"tamoxifen|aromatase|letrozole|anastrozole|exemestane|fulvestrant|endocrine|ovarian (function )?suppression|goserelin", re.I),
    "her2 targeted therapy": re.compile(r"her2|her-2|erbb2|trastuzumab|pertuzumab|lapatinib|neratinib|tucatinib|t-?dm1|deruxtecan", re.I),
    "cdk4/6 inhibition": re.compile(r"palbociclib|ribociclib|abemaciclib|cdk ?4/6", re.I),
    "chemotherapy": re.compile(r"chemotherap|anthracycline|doxorubicin|epirubicin|taxane|paclitaxel|docetaxel|cyclophosphamide|capecitabine|carboplatin|gemcitabine|eribulin", re.I),
    "immunotherapy": re.compile(r"pembrolizumab|atezolizumab|nivolumab|durvalumab|immune checkpoint|pd-?l?1", re.I),
    "parp and targeted": re.compile(r"olaparib|talazoparib|parp|alpelisib|pik3ca|everolimus|mtor|sacituzumab", re.I),
    "radiotherapy": re.compile(r"radiotherap|radiation|irradiat|brachytherap|hypofraction|boost", re.I),
    "surgery": re.compile(r"mastectom|lumpectom|breast[- ]conserving|sentinel (lymph )?node|axillary (lymph node )?dissection|reconstruction|surgical", re.I),
    "neoadjuvant": re.compile(r"neoadjuvant|preoperative|pathologic(al)? complete response|\bpcr\b", re.I),
    "metastatic disease": re.compile(r"metasta|advanced breast|stage iv", re.I),
    "screening and imaging": re.compile(r"screen|mammogra|tomosynthesis|ultrasound|mri", re.I),
    "supportive care": re.compile(r"nausea|vomiting|fatigue|lymph(o)?edema|neuropath|hot flash|hot flush|pain|anxiety|depress|quality of life|exercise|insomnia|cognitive|alopecia|neutropeni", re.I),
    "survivorship and lifestyle": re.compile(r"survivor|diet|weight|physical activity|lifestyle|adherence", re.I),
    "bone health": re.compile(r"bisphosphonate|zoledronic|denosumab|bone (mineral )?density|osteopor|clodronate|skeletal", re.I),
    "prevention and risk": re.compile(r"prevention|chemoprevention|risk reduction|brca|high[- ]risk women|raloxifene", re.I),
    "triple negative": re.compile(r"triple[- ]negative|\btnbc\b|basal[- ]like", re.I),
}

DESIGN_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("phase 3", re.compile(r"phase (iii|3)\b", re.I)),
    ("phase 2", re.compile(r"phase (ii|2)\b", re.I)),
    ("phase 1", re.compile(r"phase (i|1)\b", re.I)),
    ("meta analysis", re.compile(r"meta[- ]analys", re.I)),
]


@dataclass
class Abstract:
    pmid: str
    sentences: list[tuple[str, str]] = field(default_factory=list)
    split: str = "train"

    @property
    def text(self) -> str:
        return " ".join(s for _, s in self.sentences)

    def section(self, label: str) -> str:
        return " ".join(s for lab, s in self.sentences if lab == label)

    @property
    def title_proxy(self) -> str:
        """Abstracts in this corpus have no title, so the OBJECTIVE sentence
        (or the first BACKGROUND sentence) acts as a display headline."""
        for wanted in ("OBJECTIVE", "BACKGROUND"):
            for lab, sent in self.sentences:
                if lab == wanted:
                    return sent
        return self.sentences[0][1] if self.sentences else ""


def iter_abstracts(path: Path, split: str) -> Iterator[Abstract]:
    current: Abstract | None = None
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("###"):
                if current and current.sentences:
                    yield current
                current = Abstract(pmid=line[3:].strip(), split=split)
            elif not line.strip():
                continue
            elif current is not None and "\t" in line:
                label, sentence = line.split("\t", 1)
                current.sentences.append((label.strip(), sentence.strip()))
    if current and current.sentences:
        yield current


def breast_relevance(abstract: Abstract) -> float:
    """Score in [0, 1]. A document qualifies when breast terms dominate."""
    text = abstract.text
    lowered = text.lower()
    if "breast" not in lowered and "mamm" not in lowered and not BREAST_SPECIFIC_AGENTS.search(text):
        return 0.0
    if LACTATION.search(text) or not CANCER_CONTEXT.search(text):
        return 0.0
    core = len(BREAST_CORE.findall(text))
    agents = len(BREAST_SPECIFIC_AGENTS.findall(text))
    competing = len(NON_BREAST_PRIMARY.findall(text))
    if core == 0 and agents < 2:
        return 0.0
    raw = core + 0.75 * agents - 1.5 * competing
    return max(0.0, min(1.0, raw / 4.0))


def tag_topics(text: str) -> list[str]:
    return [topic for topic, rx in TOPIC_RULES.items() if rx.search(text)]


def study_design(text: str) -> str:
    for label, rx in DESIGN_RULES:
        if rx.search(text):
            return label
    return "randomised controlled trial"


def load_breast_abstracts(raw_dir: Path, min_relevance: float = 0.5) -> list[dict]:
    """Return breast oncology abstracts as plain dictionaries ready for chunking."""
    records: list[dict] = []
    seen: set[str] = set()
    for split in ("train", "dev", "test"):
        path = raw_dir / "pubmed_rct" / f"{split}.txt"
        if not path.exists():
            continue
        for ab in iter_abstracts(path, split):
            if ab.pmid in seen:
                continue
            score = breast_relevance(ab)
            if score < min_relevance:
                continue
            seen.add(ab.pmid)
            text = ab.text
            records.append(
                {
                    "doc_id": f"PMID:{ab.pmid}",
                    "pmid": ab.pmid,
                    "source": "pubmed",
                    "title": ab.title_proxy,
                    "sentences": [{"label": lab, "text": s} for lab, s in ab.sentences],
                    "topics": tag_topics(text),
                    "design": study_design(text),
                    "relevance": round(score, 3),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{ab.pmid}/",
                    "split": split,
                }
            )
    return records
