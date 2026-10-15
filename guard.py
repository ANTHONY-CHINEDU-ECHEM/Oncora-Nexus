"""Safety layer around the language model.

Input side
* PHI redaction: NHS numbers, hospital numbers, dates of birth, emails,
  phone numbers, UK postcodes and titled names are replaced with typed
  placeholders before anything is logged or sent to the model.
* Scope check: questions unrelated to breast oncology are declined politely.
* Crisis detection: expressions of self harm intent short circuit the pipeline
  and return signposting to urgent support.

Retrieval side
* Prompt injection scrubbing: retrieved text that tries to issue instructions
  to the model is neutralised before it enters the context window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

PHI_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("NHS_NUMBER", re.compile(r"\b\d{3}[ ]?\d{3}[ ]?\d{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+44\s?7\d{3}|\(?07\d{3}\)?|\(?0\d{3,4}\)?)\s?\d{3}\s?\d{3,4}(?!\d)")),
    ("DATE", re.compile(r"\b\d{1,2}[/.]\d{1,2}[/.](?:19|20)\d{2}\b")),
    ("DATE", re.compile(r"\b\d{1,2}(?:st|nd|rd|th)? (?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]* (?:19|20)\d{2}\b", re.I)),
    ("POSTCODE", re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b")),
    ("HOSPITAL_ID", re.compile(r"\b(?:MRN|hospital (?:no|number)|patient id)[:\s#]*[A-Z0-9-]{4,}\b", re.I)),
    ("NAME", re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?")),
]

ONCOLOGY_SCOPE = re.compile(
    r"breast|cancer|tumou?r|oncolog|carcinoma|mammo|her2|estrogen|oestrogen|progesterone|tamoxifen|letrozole|"
    r"anastrozole|chemo|radiother|mastectom|lumpectom|metasta|trastuzumab|pertuzumab|brca|lymph|axill|recurrence|"
    r"survival|trial|biopsy|screening|endocrine|aromatase|palbociclib|ribociclib|abemaciclib|olaparib|pembrolizumab|"
    r"capecitabine|docetaxel|paclitaxel|anthracycline|neoadjuvant|adjuvant|dcis|lobular|ductal|npi|grade|stage|"
    r"subtype|luminal|basal|ki67|oncotype|mammaprint|lvef|cardiac|neutropen|nausea|fatigue|lymph?oedema|lymphedema|"
    r"bone|bisphosphonate|denosumab|fertility|menopaus|hot fl|survivor|palliative|mdt|tumou?r board|ecog|eligib|"
    r"receptor|\ber\b|\bpr\b|ihc|immunohisto|pam50|hormone|ovar|nodes?\b|sentinel|mri\b|fish\b|margin|"
    r"radiotherapy|neoplasm|malignan|metastas",
    re.I,
)

CRISIS = re.compile(r"\b(kill myself|end my life|suicid\w*|self[- ]harm|want to die|overdose on purpose)\b", re.I)

INJECTION = re.compile(
    r"(ignore (all |any )?(previous|prior|above) (instructions|prompts)|disregard (the )?(system|previous)|"
    r"you are now|system prompt|<\s*/?\s*(system|assistant|instructions?)\s*>|act as (an?|the) )",
    re.I,
)

CRISIS_MESSAGE = (
    "It sounds like you may be going through something very difficult. If you are in immediate danger, call 999 "
    "or go to your nearest A and E. In the UK you can call Samaritans free on 116 123 at any time, or NHS 111. "
    "This system is a clinical decision support tool and cannot provide the support you deserve right now."
)

OUT_OF_SCOPE_MESSAGE = (
    "Oncora Nexus answers questions about breast cancer diagnosis, treatment, prognosis, trials and supportive care. "
    "This question appears to be outside that scope, so no answer was generated."
)


@dataclass
class GuardResult:
    text: str
    redactions: dict[str, int] = field(default_factory=dict)
    in_scope: bool = True
    crisis: bool = False

    @property
    def blocked(self) -> bool:
        return self.crisis or not self.in_scope

    @property
    def message(self) -> str | None:
        if self.crisis:
            return CRISIS_MESSAGE
        if not self.in_scope:
            return OUT_OF_SCOPE_MESSAGE
        return None


def redact_phi(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    out = text
    for label, rx in PHI_PATTERNS:
        out, n = rx.subn(f"[{label}]", out)
        if n:
            counts[label] = counts.get(label, 0) + n
    return out, counts


def check_input(text: str, case_context: str | None = None) -> GuardResult:
    from oncora.rag.text import normalise

    clean, counts = redact_phi(text)
    combined = normalise(f"{clean} {case_context or ''}")
    return GuardResult(
        text=clean,
        redactions=counts,
        in_scope=bool(ONCOLOGY_SCOPE.search(combined)),
        crisis=bool(CRISIS.search(text)),
    )


def scrub_injection(text: str) -> str:
    return INJECTION.sub("[removed instruction like text]", text)
