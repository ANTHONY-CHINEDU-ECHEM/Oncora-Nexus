"""Extractive evidence composer.

Used when no local language model is reachable, as the deterministic engine of
the hosted static demo, and as a baseline in the evaluation. It never
generates new wording for clinical facts: every statement is a verbatim
sentence from a cited passage, chosen for relevance to the question, so its
faithfulness is high by construction and its weakness is fluency.
"""

from __future__ import annotations

import re

from oncora.rag.query import ParsedQuery
from oncora.rag.retriever import Hit
from oncora.rag.text import split_sentences, tokenize

LABEL_RX = re.compile(r"^(BACKGROUND|OBJECTIVE|METHODS|RESULTS|CONCLUSIONS):\s*")
PREFERRED = {
    "efficacy": ("CONCLUSIONS", "RESULTS"),
    "toxicity": ("RESULTS", "CONCLUSIONS"),
    "supportive": ("CONCLUSIONS", "RESULTS"),
    "recommendation": ("CONCLUSIONS",),
}


MIN_SENTENCE_SCORE = 0.3


def _trim(sentence: str) -> str:
    """Tidy the spaced punctuation of the PubMed RCT release and drop the final stop."""
    s = re.sub(r"\s+([,.;:)\]%])", r"\1", sentence)
    s = re.sub(r"([(\[])\s+", r"\1", s)
    return s.strip().rstrip(".")


def _sentences(hit: Hit) -> list[tuple[str, str]]:
    """Return (label, sentence) pairs for a passage."""
    p = hit.passage
    if p.source == "trial":
        return [("", s.strip()) for line in p.text.split(" | ") for s in split_sentences(line) if len(s.split()) >= 4]
    out, label = [], ""
    for sent in split_sentences(p.text):
        m = LABEL_RX.match(sent)
        if m:
            label = m.group(1)
            sent = sent[m.end():]
        out.append((label, sent.strip()))
    return out


def _score(sent: str, label: str, q: ParsedQuery) -> float:
    toks = set(tokenize(sent))
    if not toks:
        return 0.0
    overlap = len(toks & set(q.tokens)) / max(len(set(q.tokens)), 1)
    prior = 0.25 if label in PREFERRED.get(q.primary_intent, ("CONCLUSIONS", "RESULTS")) else 0.0
    numeric = 0.1 if re.search(r"\d", sent) and q.primary_intent in {"efficacy", "toxicity"} else 0.0
    length_pen = 0.15 if len(sent.split()) > 70 else 0.0
    background_pen = 0.2 if label in {"BACKGROUND", "METHODS"} else 0.0
    return overlap + prior + numeric - length_pen - background_pen


def best_sentences(hit: Hit, q: ParsedQuery, n: int = 1) -> list[str]:
    scored = sorted(_sentences(hit), key=lambda ls: -_score(ls[1], ls[0], q))
    return [s for _, s in scored[:n]]


def compose(question: str, q: ParsedQuery, hits: list[Hit]) -> str:
    if not hits:
        return "BOTTOM LINE: The retrieved evidence does not answer this question."
    lead_idx = next((i for i, h in enumerate(hits) if h.passage.source == "guideline"), 0)
    lead = hits[lead_idx]
    lead_sents = best_sentences(lead, q, n=2)
    bottom = " ".join(f"{_trim(s)} [S{lead_idx + 1}]." for s in lead_sents)
    bullets = []
    for i, h in enumerate(hits, start=1):
        if i == lead_idx + 1:
            continue
        if h.passage.section == "Eligibility criteria" and q.primary_intent != "eligibility":
            continue
        scored = sorted(_sentences(h), key=lambda ls: -_score(ls[1], ls[0], q))
        if not scored or _score(scored[0][1], scored[0][0], q) < MIN_SENTENCE_SCORE:
            continue
        bullets.append(f"* {_trim(scored[0][1])} [S{i}].")
    caveats = []
    sources = {h.passage.source for h in hits}
    if "guideline" not in sources:
        caveats.append("* No curated guideline digest matched this question closely, so the answer rests on individual trials.")
    if "trial" in sources:
        caveats.append("* Trial registry reports summarise posted results and have not necessarily been peer reviewed.")
    if q.subtypes or q.settings:
        caveats.append("* Check that each cited population matches the patient's subtype and disease setting before applying a finding.")
    caveats.append("* Extractive mode: statements are verbatim evidence sentences selected by relevance; a local language model can produce a fuller synthesis.")
    return "BOTTOM LINE: " + bottom + "\nKEY EVIDENCE:\n" + "\n".join(bullets[:6]) + "\nCAVEATS:\n" + "\n".join(caveats)
