"""Clinical text normalisation shared by every retriever.

The same tokenizer is used for indexing and querying, and it is ported line
for line to the browser engine (frontend/src/engine/text.ts) so that the
static demo ranks documents the same way as the Python service.

Normalisation steps:
1. British to American spelling (oestrogen to estrogen, tumour to tumor) so a
   UK clinician's query matches US authored abstracts.
2. Clinical synonym and abbreviation folding (T-DM1 to trastuzumab emtansine).
3. Lower casing, tokenisation that keeps tokens such as her2, cdk4 and 5fu.
4. Stop word removal and a conservative suffix stemmer.
"""

from __future__ import annotations

import re
from functools import lru_cache

UK_TO_US = {
    "oestrogen": "estrogen",
    "oestradiol": "estradiol",
    "tumour": "tumor",
    "tumours": "tumors",
    "oedema": "edema",
    "lymphoedema": "lymphedema",
    "haematological": "hematological",
    "haemoglobin": "hemoglobin",
    "anaemia": "anemia",
    "randomised": "randomized",
    "randomisation": "randomization",
    "centre": "center",
    "centres": "centers",
    "hospitalisation": "hospitalization",
    "paediatric": "pediatric",
    "foetal": "fetal",
    "oesophageal": "esophageal",
    "behaviour": "behavior",
    "favourable": "favorable",
    "unfavourable": "unfavorable",
    "programme": "program",
    "labelled": "labeled",
    "optimise": "optimize",
    "organisation": "organization",
    "characterised": "characterized",
    "localised": "localized",
    "metastasised": "metastasized",
    "hot flushes": "hot flashes",
    "hot flush": "hot flash",
}

# Phrase level canonicalisation. Keys are matched as whole phrases on the
# lower cased text before tokenisation.
SYNONYMS = {
    "t-dm1": "trastuzumab emtansine",
    "tdm1": "trastuzumab emtansine",
    "kadcyla": "trastuzumab emtansine",
    "t-dxd": "trastuzumab deruxtecan",
    "tdxd": "trastuzumab deruxtecan",
    "enhertu": "trastuzumab deruxtecan",
    "herceptin": "trastuzumab",
    "perjeta": "pertuzumab",
    "tykerb": "lapatinib",
    "ibrance": "palbociclib",
    "kisqali": "ribociclib",
    "verzenio": "abemaciclib",
    "verzenios": "abemaciclib",
    "arimidex": "anastrozole",
    "femara": "letrozole",
    "aromasin": "exemestane",
    "faslodex": "fulvestrant",
    "keytruda": "pembrolizumab",
    "lynparza": "olaparib",
    "trodelvy": "sacituzumab govitecan",
    "xeloda": "capecitabine",
    "taxol": "paclitaxel",
    "taxotere": "docetaxel",
    "zometa": "zoledronic acid",
    "tnbc": "triple negative breast cancer",
    "triple-negative": "triple negative",
    "her-2": "her2",
    "her 2": "her2",
    "erbb2": "her2",
    "her2-positive": "her2 positive",
    "her2-negative": "her2 negative",
    "her2+": "her2 positive",
    "her2-": "her2 negative",
    "er+": "estrogen receptor positive",
    "er-": "estrogen receptor negative",
    "hr+": "hormone receptor positive",
    "hr-positive": "hormone receptor positive",
    "er-positive": "estrogen receptor positive",
    "er-negative": "estrogen receptor negative",
    "pr-positive": "progesterone receptor positive",
    "aromatase inhibitors": "aromatase inhibitor",
    "aromatase-inhibitor": "aromatase inhibitor",
    "cdk4/6": "cdk4 inhibitor",
    "cdk 4/6": "cdk4 inhibitor",
    "pcr": "pathologic complete response",
    "pathological complete response": "pathologic complete response",
    "dfs": "disease free survival",
    "idfs": "invasive disease free survival",
    "pfs": "progression free survival",
    "efs": "event free survival",
    "rfs": "recurrence free survival",
    "mbc": "metastatic breast cancer",
    "ebc": "early breast cancer",
    "slnb": "sentinel lymph node biopsy",
    "alnd": "axillary lymph node dissection",
    "bcs": "breast conserving surgery",
    "pmrt": "postmastectomy radiotherapy",
    "lvef": "left ventricular ejection fraction",
    "ecog": "ecog performance status",
    "ovarian function suppression": "ovarian suppression",
    "ofs": "ovarian suppression",
    "radiation therapy": "radiotherapy",
    "breast-conserving": "breast conserving",
    "dose-dense": "dose dense",
    "side effects": "adverse events toxicity",
    "side effect": "adverse event toxicity",
    # Lay and colloquial clinical language mapped to indexed terminology.
    "echo": "echocardiogram",
    "echocardiography": "echocardiogram",
    "heart scan": "echocardiogram",
    "heart problems": "cardiac toxicity",
    "lumpectomy": "breast conserving surgery",
    "wide local excision": "breast conserving surgery",
    "arm swelling": "lymphedema",
    "swollen arm": "lymphedema",
    "tiredness": "fatigue",
    "peau d orange": "inflammatory breast cancer",
    "red swollen breast": "inflammatory breast cancer",
    "low neutrophils": "neutropenia",
    "neutropenic fever": "neutropenic sepsis",
    "fever": "febrile",
    "feeling sick": "nausea",
    "sickness": "nausea vomiting",
    "hair loss": "alopecia",
    "have children": "fertility",
    "children in future": "fertility",
    "get pregnant": "pregnancy fertility",
    "family history": "familial risk",
    "preventive medication": "chemoprevention",
    "preventive medicine": "chemoprevention",
    "rebiopsy": "biopsy",
    "re-biopsy": "biopsy",
    "weight training": "resistance exercise",
    "physical activity": "exercise",
    "antidepressants": "ssri antidepressant",
    "chemo": "chemotherapy",
    "radiation": "radiotherapy",
    "ec chemotherapy": "epirubicin cyclophosphamide chemotherapy",
    "leg weakness": "spinal cord compression",
    "bone pain": "bone metastases",
    "night sweats": "hot flashes",
    "brain spread": "brain metastases",
}

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have having he
    her here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not
    now of off on once only or other our ours ourselves out over own same she should so some such than that the
    their theirs them themselves then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours yourself yourselves patients patient
    study studies trial trials group groups using used use may might among within versus vs compared whether
    however therefore thus via per respectively total participants subjects""".split()
)

KEEP_SHORT = frozenset({"er", "pr", "os", "ai", "ct", "rt", "gy", "iv", "ki", "pd", "mri", "her2"})

_TOKEN_RX = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_NUMBER_RX = re.compile(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*(%|percent|mg|gy|months?|years?|weeks?|days?)?")


@lru_cache(maxsize=1)
def _phrase_patterns() -> list[tuple[re.Pattern[str], str]]:
    pairs = list(UK_TO_US.items()) + list(SYNONYMS.items())
    pairs.sort(key=lambda kv: -len(kv[0]))
    return [(re.compile(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])"), v) for k, v in pairs]


def normalise(text: str) -> str:
    low = f" {text.lower()} "
    for rx, repl in _phrase_patterns():
        low = rx.sub(repl, low)
    return low.strip()


def stem(token: str) -> str:
    """Conservative suffix stripping. It deliberately avoids aggressive Porter
    rules that collapse clinically distinct words (for example, therapy and
    therapist)."""
    if len(token) <= 4 or token.isdigit():
        return token
    if token.endswith(("shes", "ches", "xes", "zes")):
        return token[:-2]
    for suffix, repl in (
        ("ational", "ate"),
        ("ization", "ize"),
        ("ies", "y"),
        ("sses", "ss"),
        ("ness", ""),
        ("ments", "ment"),
        ("ings", ""),
        ("ing", ""),
        ("edly", ""),
        ("ed", ""),
        ("es", "e"),
        ("s", ""),
    ):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            if suffix == "s" and token.endswith(("ss", "us", "is")):
                return token
            return token[: len(token) - len(suffix)] + repl
    return token


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN_RX.findall(normalise(text)):
        if not keep_stopwords and tok in STOPWORDS:
            continue
        if len(tok) < 3 and tok not in KEEP_SHORT and not tok.isdigit():
            continue
        out.append(stem(tok))
    return out


def bigrams(tokens: list[str]) -> list[str]:
    return [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]


def numbers(text: str) -> set[str]:
    """Extract numeric facts (value plus optional unit) for faithfulness checks."""
    found = set()
    # Rejoin numbers split by the spaced punctuation of the PubMed RCT release ("10 .9", "18 %").
    text = re.sub(r"(\d)\s+\.(\d)", r"\1.\2", text)
    text = re.sub(r"(\d)\s+%", r"\1%", text)
    for value, unit in _NUMBER_RX.findall(text.lower()):
        v = value.rstrip("0").rstrip(".") if "." in value else value
        found.add(v)
        if unit:
            canonical = "%" if unit in ("%", "percent") else unit[0]
            found.add(f"{v}{canonical}")
    return found


SENT_SPLIT_RX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENT_SPLIT_RX.split(text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])
