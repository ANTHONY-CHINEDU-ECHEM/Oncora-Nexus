"""Unified passage model and source specific chunking strategies.

Three source families are chunked differently because their structure differs:

* PubMed abstracts: section aware sentence windows (about 180 words) with one
  sentence of overlap. The abstract headline is prepended to the indexed text
  (contextual chunking) so that a results chunk still knows which trial it
  belongs to.
* ClinicalTrials.gov reports: one stream per section (eligibility,
  intervention, results, adverse events), grouped into windows of lines with
  their original line numbers retained for evidence level evaluation.
* Knowledge cards: one passage per card; they are already concise.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from oncora.data.trials import TrialRecord

EVIDENCE_LEVELS = {
    "guideline": (1, "Curated guideline digest"),
    "meta analysis": (2, "Meta analysis of randomised trials"),
    "phase 3": (2, "Randomised controlled trial, phase 3"),
    "randomised controlled trial": (3, "Randomised controlled trial"),
    "phase 2": (3, "Randomised controlled trial, phase 2"),
    "phase 1": (4, "Early phase randomised study"),
    "trial": (4, "Trial registry report with posted results"),
}

SECTION_DISPLAY = {
    "intervention": "Interventions",
    "eligibility": "Eligibility criteria",
    "results": "Results",
    "adverse_events": "Adverse events",
}


@dataclass
class Passage:
    chunk_id: str
    doc_id: str
    source: str
    title: str
    url: str
    section: str
    text: str
    index_text: str
    evidence_rank: int
    evidence_label: str
    topics: list[str] = field(default_factory=list)
    line_start: int | None = None
    line_end: int | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Passage:
        return cls(**d)


def _words(text: str) -> int:
    return len(text.split())


def chunk_pubmed(record: dict, target_words: int = 180, overlap: int = 1) -> list[Passage]:
    sents = record["sentences"]
    rank, label = EVIDENCE_LEVELS.get(record.get("design", ""), EVIDENCE_LEVELS["randomised controlled trial"])
    windows: list[list[int]] = []
    start = 0
    while start < len(sents):
        end, words = start, 0
        while end < len(sents) and (words < target_words or end == start):
            words += _words(sents[end]["text"])
            end += 1
        windows.append(list(range(start, end)))
        if end >= len(sents):
            break
        start = max(end - overlap, start + 1)
    passages = []
    for i, idx in enumerate(windows):
        labels = [sents[j]["label"] for j in idx]
        dominant = max(set(labels), key=labels.count)
        display = " ".join(f"{sents[j]['label']}: {sents[j]['text']}" if (j == idx[0] or sents[j]['label'] != sents[j - 1]['label']) else sents[j]["text"] for j in idx)
        plain = " ".join(sents[j]["text"] for j in idx)
        passages.append(
            Passage(
                chunk_id=f"{record['doc_id']}#{i}",
                doc_id=record["doc_id"],
                source="pubmed",
                title=record["title"],
                url=record["url"],
                section=dominant,
                text=display,
                index_text=f"{record['title']} {plain}",
                evidence_rank=rank,
                evidence_label=label,
                topics=record.get("topics", []),
                line_start=idx[0],
                line_end=idx[-1],
                meta={"pmid": record["pmid"], "design": record.get("design"), "sections": sorted(set(labels))},
            )
        )
    return passages


def chunk_trial(rec: TrialRecord, target_words: int = 130) -> list[Passage]:
    rank, label = EVIDENCE_LEVELS["trial"]
    passages: list[Passage] = []
    for section in ("eligibility", "intervention", "results", "adverse_events"):
        lines = rec.section(section)
        if not any(ln.strip() for ln in lines):
            continue
        k, start = 0, 0
        while start < len(lines):
            end, words = start, 0
            while end < len(lines) and (words < target_words or end == start):
                words += _words(lines[end])
                end += 1
            block = [ln.strip() for ln in lines[start:end] if ln.strip()]
            if block:
                text = " | ".join(block)
                passages.append(
                    Passage(
                        chunk_id=f"{rec.nct_id}:{section}:{k}",
                        doc_id=rec.nct_id,
                        source="trial",
                        title=rec.title,
                        url=rec.url,
                        section=SECTION_DISPLAY[section],
                        text=text,
                        index_text=f"{rec.title} {SECTION_DISPLAY[section]} {text}",
                        evidence_rank=rank,
                        evidence_label=label,
                        topics=[rec.setting] + rec.agents[:6],
                        line_start=start,
                        line_end=end - 1,
                        meta={"nct_id": rec.nct_id, "section_key": section, "setting": rec.setting},
                    )
                )
                k += 1
            start = end
    return passages


def chunk_cards(cards_path: Path) -> list[Passage]:
    data = json.loads(cards_path.read_text(encoding="utf-8"))
    rank, label = EVIDENCE_LEVELS["guideline"]
    out = []
    for card in data["cards"]:
        refs = "; ".join(card.get("references", []))
        out.append(
            Passage(
                chunk_id=f"{card['id']}#0",
                doc_id=card["id"],
                source="guideline",
                title=card["title"],
                url="",
                section=card["category"],
                text=card["body"],
                index_text=f"{card['title']} {card['category']} {' '.join(card.get('tags', []))} {card['body']}",
                evidence_rank=rank,
                evidence_label=label,
                topics=card.get("tags", []),
                meta={"references": card.get("references", []), "references_text": refs, "reviewed": data["meta"]["reviewed"]},
            )
        )
    return out
