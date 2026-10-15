"""Scale the literature corpus with the NCBI Entrez utilities.

The bundled corpus uses the breast oncology subset of PubMed 200k RCT (about
3,000 abstracts). This script pulls any PubMed query, by default every
randomised trial, meta analysis and practice guideline indexed under the MeSH
heading Breast Neoplasms (well over 100,000 records), and writes them in the
same structure as oncora/data/pubmed_rct.py. The next run of
scripts/build_index.py picks the file up automatically and indexes it
alongside the bundled data, removing duplicate PMIDs.

    NCBI_API_KEY=... python scripts/fetch_pubmed.py --max 50000

Respect NCBI usage policy: at most 3 requests per second without a key and 10
with one, and include a contact email.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oncora.config import settings  # noqa: E402
from oncora.data.pubmed_rct import Abstract, breast_relevance, study_design, tag_topics  # noqa: E402

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_QUERY = '"breast neoplasms"[MeSH Terms] AND (randomized controlled trial[pt] OR meta-analysis[pt] OR practice guideline[pt]) AND english[la] AND hasabstract'
LABEL_MAP = {"BACKGROUND": "BACKGROUND", "OBJECTIVE": "OBJECTIVE", "OBJECTIVES": "OBJECTIVE", "AIMS": "OBJECTIVE", "METHODS": "METHODS", "RESULTS": "RESULTS", "CONCLUSIONS": "CONCLUSIONS", "CONCLUSION": "CONCLUSIONS"}


def _get(url: str, params: dict) -> bytes:
    key = os.environ.get("NCBI_API_KEY")
    if key:
        params["api_key"] = key
    params["tool"] = "oncora_nexus"
    params["email"] = os.environ.get("NCBI_EMAIL", "")
    with urllib.request.urlopen(f"{url}?{urllib.parse.urlencode(params)}", timeout=120) as r:
        return r.read()


def search(query: str, max_records: int) -> tuple[str, str, int]:
    res = json.loads(_get(f"{EUTILS}/esearch.fcgi", {"db": "pubmed", "term": query, "usehistory": "y", "retmode": "json", "retmax": 0}))
    r = res["esearchresult"]
    return r["webenv"], r["querykey"], min(int(r["count"]), max_records)


def fetch_batch(webenv: str, key: str, start: int, size: int) -> list[dict]:
    xml = _get(f"{EUTILS}/efetch.fcgi", {"db": "pubmed", "query_key": key, "WebEnv": webenv, "retstart": start, "retmax": size, "retmode": "xml"})
    out = []
    for art in ET.fromstring(xml).iter("PubmedArticle"):
        pmid = art.findtext(".//PMID")
        title = (art.findtext(".//ArticleTitle") or "").strip()
        year = art.findtext(".//PubDate/Year") or ""
        sents = []
        for node in art.iter("AbstractText"):
            label = LABEL_MAP.get((node.get("NlmCategory") or node.get("Label") or "").upper(), "RESULTS" if sents else "BACKGROUND")
            text = "".join(node.itertext()).strip()
            if text:
                sents.append((label, text))
        if not pmid or not sents:
            continue
        ab = Abstract(pmid=pmid, sentences=sents, split="eutils")
        text = ab.text
        out.append(
            {
                "doc_id": f"PMID:{pmid}",
                "pmid": pmid,
                "source": "pubmed",
                "title": title or ab.title_proxy,
                "year": year,
                "sentences": [{"label": l, "text": s} for l, s in sents],
                "topics": tag_topics(text),
                "design": study_design(text + " " + title),
                "relevance": round(breast_relevance(ab), 3),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "split": "eutils",
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--max", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=200)
    args = ap.parse_args()
    webenv, key, total = search(args.query, args.max)
    print(f"{total} records to fetch")
    records: list[dict] = []
    delay = 0.11 if os.environ.get("NCBI_API_KEY") else 0.35
    for start in range(0, total, args.batch):
        records += fetch_batch(webenv, key, start, args.batch)
        print(f"  {len(records)} abstracts", end="\r")
        time.sleep(delay)
    out = settings.paths.processed / "pubmed_eutils.json"
    out.write_text(json.dumps(records))
    print(f"\nSaved {len(records)} abstracts to {out}")


if __name__ == "__main__":
    main()
