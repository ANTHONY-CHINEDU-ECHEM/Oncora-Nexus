"""Query understanding: intent, clinical entities and retrieval priors.

The parsed query steers three later stages: which sections the reranker
prefers (results and conclusions for efficacy questions, adverse event tables
for toxicity questions), which entities must be covered, and how the answer is
framed.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from oncora.data.trials import DRUG_LEXICON
from oncora.rag.text import normalise, tokenize

INTENT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("toxicity", re.compile(r"toxicit|adverse|side effect|safety|tolerab|cardiotox|neuropath|nausea|neutropen|interstitial lung|pneumonitis|ejection fraction|arthralg|complication", re.I)),
    ("eligibility", re.compile(r"eligib|inclusion|exclusion|criteria|enrol|enroll|qualif|candidate for a trial", re.I)),
    ("prognosis", re.compile(r"prognos|survival estimate|risk of recurrence|recurrence risk|life expectancy|npi|nottingham|outlook|predict", re.I)),
    ("testing", re.compile(r"\btest(ing)?\b|assay|ihc|immunohisto|in situ hybrid|\bish\b|biomarker|genomic|oncotype|mammaprint|recurrence score|brca|germline|screening|mammogra|biopsy", re.I)),
    ("supportive", re.compile(r"quality of life|fatigue|exercise|lymph(o)?edema|anxiety|depress|hot fl|insomnia|bone health|psycholog|survivorship|fertility|pregnan", re.I)),
    ("efficacy", re.compile(r"efficac|benefit|improv|response|survival|reduce|effective|outcome|pcr|pathologic complete|recurrence|versus|vs\b|compared|superior|non ?inferior|works?\b", re.I)),
    ("recommendation", re.compile(r"recommend|should|standard of care|guideline|first line|second line|option|manage|treat|indicat|which|best|choice|sequenc", re.I)),
]

SECTION_PRIORS = {
    "efficacy": {"RESULTS": 1.0, "CONCLUSIONS": 1.0, "Results": 1.0},
    "toxicity": {"Adverse events": 1.0, "RESULTS": 0.6, "CONCLUSIONS": 0.4},
    "eligibility": {"Eligibility criteria": 1.0},
    "prognosis": {"RESULTS": 0.6, "CONCLUSIONS": 0.6},
    "testing": {"METHODS": 0.4, "RESULTS": 0.4},
    "supportive": {"RESULTS": 0.6, "CONCLUSIONS": 0.8},
    "recommendation": {"CONCLUSIONS": 0.8},
}

SUBTYPE_RULES = {
    "her2 positive": re.compile(r"her2 positive|her2 overexpress|her2 amplif|her2 enriched", re.I),
    "her2 low": re.compile(r"her2 low|her2 ultralow", re.I),
    "triple negative": re.compile(r"triple negative|basal", re.I),
    "hormone receptor positive": re.compile(r"hormone receptor positive|estrogen receptor positive|luminal|endocrine", re.I),
}

SETTING_RULES = {
    "metastatic": re.compile(r"metasta|advanced|stage iv|second line|first line", re.I),
    "neoadjuvant": re.compile(r"neoadjuvant|preoperative|before surgery", re.I),
    "adjuvant": re.compile(r"adjuvant|after surgery|postoperative|early breast cancer", re.I),
    "prevention": re.compile(r"prevent|risk reduc|high risk women|chemoprevention", re.I),
}

POPULATION_RULES = {
    "premenopausal": re.compile(r"premenopaus|young women|younger women", re.I),
    "postmenopausal": re.compile(r"postmenopaus", re.I),
    "older adults": re.compile(r"older|elderly|aged (65|70)|over (65|70)", re.I),
    "male": re.compile(r"\bmale\b|\bmen\b", re.I),
    "brca carriers": re.compile(r"brca", re.I),
}


@dataclass
class ParsedQuery:
    raw: str
    normalised: str
    tokens: list[str]
    intents: list[str] = field(default_factory=list)
    drugs: list[str] = field(default_factory=list)
    subtypes: list[str] = field(default_factory=list)
    settings: list[str] = field(default_factory=list)
    populations: list[str] = field(default_factory=list)

    @property
    def primary_intent(self) -> str:
        return self.intents[0] if self.intents else "general"

    def section_prior(self, section: str) -> float:
        best = 0.0
        for intent in self.intents[:2]:
            best = max(best, SECTION_PRIORS.get(intent, {}).get(section, 0.0))
        return best

    def to_dict(self) -> dict:
        d = asdict(self)
        d["primary_intent"] = self.primary_intent
        return d


def parse_query(text: str) -> ParsedQuery:
    norm = normalise(text)
    q = ParsedQuery(raw=text, normalised=norm, tokens=tokenize(text))
    q.intents = [name for name, rx in INTENT_RULES if rx.search(norm)]
    q.drugs = [d for d in DRUG_LEXICON if re.search(rf"(?<![a-z]){re.escape(d)}(?![a-z])", norm)]
    if "aromatase inhibitor" in norm:
        q.drugs.append("aromatase inhibitor")
    if "cdk4 inhibitor" in norm:
        q.drugs.append("cdk4 inhibitor")
    q.subtypes = [k for k, rx in SUBTYPE_RULES.items() if rx.search(norm)]
    q.settings = [k for k, rx in SETTING_RULES.items() if rx.search(norm)]
    q.populations = [k for k, rx in POPULATION_RULES.items() if rx.search(norm)]
    return q
