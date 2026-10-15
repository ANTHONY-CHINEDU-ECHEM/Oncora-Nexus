"""Fetch currently recruiting breast cancer trials from the ClinicalTrials.gov API (v2).

The demonstration corpus contains completed trials with posted results. In a
deployment, run this script on a schedule (for example nightly) to screen
patients against studies that are open now. Records are converted to the same
TrialRecord structure, so parsing, matching and the interface work unchanged.

    python scripts/fetch_live_trials.py --max 2000 --country "United Kingdom"

Output: data/raw/live_trials/live_trials.json
API reference: https://clinicaltrials.gov/data-api/api
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oncora.config import settings  # noqa: E402

API = "https://clinicaltrials.gov/api/v2/studies"


def fetch(max_records: int, country: str | None, status: str = "RECRUITING") -> list[dict]:
    params = {
        "query.cond": "breast cancer",
        "filter.overallStatus": status,
        "pageSize": "100",
        "format": "json",
    }
    if country:
        params["query.locn"] = country
    out: list[dict] = []
    token = None
    while len(out) < max_records:
        if token:
            params["pageToken"] = token
        url = f"{API}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            payload = json.loads(resp.read())
        out.extend(payload.get("studies", []))
        token = payload.get("nextPageToken")
        if not token:
            break
        time.sleep(0.3)
    return out[:max_records]


def to_record(study: dict) -> dict:
    ps = study.get("protocolSection", {})
    ident = ps.get("identificationModule", {})
    elig = ps.get("eligibilityModule", {})
    arms = ps.get("armsInterventionsModule", {})
    design = ps.get("designModule", {})
    locations = ps.get("contactsLocationsModule", {}).get("locations", [])
    criteria = [ln.rstrip() for ln in (elig.get("eligibilityCriteria") or "").splitlines()]
    min_age, max_age = elig.get("minimumAge"), elig.get("maximumAge")
    header = []
    if min_age or max_age:
        header.append(f"  Age {min_age or 'any'} to {max_age or 'any'}")
    if elig.get("sex") == "FEMALE":
        header.append("  Female participants only")
    intervention = []
    for i, it in enumerate(arms.get("interventions", []), start=1):
        intervention += [f"INTERVENTION {i}: ", f"  {it.get('name', '')}", f"  {it.get('description', '')}"]
    return {
        "Clinical Trial ID": ident.get("nctId"),
        "Title": ident.get("briefTitle"),
        "Phases": design.get("phases", []),
        "Status": ps.get("statusModule", {}).get("overallStatus"),
        "Locations": sorted({f"{l.get('city', '')}, {l.get('country', '')}" for l in locations})[:50],
        "Intervention": intervention,
        "Eligibility": ["Inclusion Criteria:"] + header + criteria if header else criteria,
        "Results": [],
        "Adverse Events": [],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max", type=int, default=1000)
    ap.add_argument("--country", default=None)
    args = ap.parse_args()
    studies = fetch(args.max, args.country)
    records = [to_record(s) for s in studies if s.get("protocolSection")]
    out = settings.paths.raw / "live_trials"
    out.mkdir(parents=True, exist_ok=True)
    (out / "live_trials.json").write_text(json.dumps(records, indent=1))
    print(f"Saved {len(records)} recruiting trials to {out / 'live_trials.json'}")


if __name__ == "__main__":
    main()
