"""Post hoc grounding verification for generated answers.

Every sentence of an answer is checked against the passages it cites:

* lexical support: share of the sentence's content tokens found in the best
  cited passage (after clinical normalisation and stemming);
* numeric fidelity: every number in the sentence must occur in a cited
  passage, because invented percentages and hazard ratios are the most
  dangerous hallucination in oncology;
* citation validity: cited passage numbers must exist.

The answer receives a faithfulness score (share of supported claim sentences)
and each sentence a status that the interface renders inline, so a clinician
can see exactly which statements are backed by which evidence.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from oncora.rag.text import numbers, split_sentences, tokenize

CITE_RX = re.compile(r"\[S(\d+)\]")
HEADER_RX = re.compile(r"^(BOTTOM LINE|KEY EVIDENCE|CAVEATS)\s*:?\s*", re.I)
NON_CLAIM_RX = re.compile(r"^(the retrieved evidence does not answer|no (guideline|evidence)|this (is|was) |caveat)", re.I)

SUPPORTED = 0.5
PARTIAL = 0.3


@dataclass
class SentenceCheck:
    text: str
    citations: list[int]
    support: float
    numbers_ok: bool
    status: str
    best_source: int | None = None


@dataclass
class VerificationReport:
    sentences: list[SentenceCheck] = field(default_factory=list)
    faithfulness: float = 0.0
    citation_coverage: float = 0.0
    invalid_citations: list[int] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["faithfulness"] = round(self.faithfulness, 3)
        d["citation_coverage"] = round(self.citation_coverage, 3)
        return d


def _claims(answer: str) -> list[tuple[str, str]]:
    """Split an answer into (section, sentence) pairs."""
    out: list[tuple[str, str]] = []
    section = "BOTTOM LINE"
    for line in answer.splitlines():
        header = HEADER_RX.match(line.strip())
        if header:
            section = header.group(1).upper()
        line = HEADER_RX.sub("", line.strip()).lstrip("*•").strip()
        if not line:
            continue
        for sent in split_sentences(line):
            lead = re.match(r"^((?:\[S\d+\]\s*)+)", sent)
            if lead and out:
                # A citation placed after the full stop belongs to the previous sentence.
                out[-1] = (out[-1][0], out[-1][1] + " " + lead.group(1).strip())
                sent = sent[lead.end():].strip()
            if len(tokenize(CITE_RX.sub("", sent))) >= 3:
                out.append((section, sent))
    return out


def verify_answer(answer: str, evidence_texts: list[str]) -> VerificationReport:
    ev_tokens = [set(tokenize(t)) for t in evidence_texts]
    ev_numbers = [numbers(t) for t in evidence_texts]
    report = VerificationReport()
    cited_any = 0
    for section, sent in _claims(answer):
        cites = [int(c) for c in CITE_RX.findall(sent)]
        clean = CITE_RX.sub("", sent)
        toks = set(tokenize(clean))
        valid = [c for c in cites if 1 <= c <= len(evidence_texts)]
        report.invalid_citations += [c for c in cites if c not in valid]
        best, best_src = 0.0, None
        for c in valid:
            s = len(toks & ev_tokens[c - 1]) / max(len(toks), 1)
            if s > best:
                best, best_src = s, c
        sent_nums = {n for n in numbers(clean) if not re.fullmatch(r"\d", n)}
        pool = set().union(*(ev_numbers[c - 1] for c in valid)) if valid else set()
        missing = sorted(n for n in sent_nums if n not in pool)
        nums_ok = not missing
        if missing and valid:
            report.unsupported_numbers += missing
        if NON_CLAIM_RX.search(clean.strip()) or (section == "CAVEATS" and not valid):
            status = "meta"
        elif not valid:
            status = "uncited"
        elif best >= SUPPORTED and nums_ok:
            status = "supported"
        elif best >= PARTIAL and nums_ok:
            status = "partial"
        else:
            status = "unsupported"
        if valid and status != "meta":
            cited_any += 1
        report.sentences.append(SentenceCheck(sent, valid, round(best, 3), nums_ok, status, best_src))
    claims = [s for s in report.sentences if s.status != "meta"]
    if claims:
        report.faithfulness = sum(1.0 if s.status == "supported" else 0.5 if s.status == "partial" else 0.0 for s in claims) / len(claims)
        report.citation_coverage = cited_any / len(claims)
    report.invalid_citations = sorted(set(report.invalid_citations))
    report.unsupported_numbers = sorted(set(report.unsupported_numbers))
    return report
