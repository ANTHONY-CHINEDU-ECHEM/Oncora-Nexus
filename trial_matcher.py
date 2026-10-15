"""Criterion level clinical trial matching.

Free text eligibility criteria are parsed into typed constraints, each linked to
the source line it came from. A patient profile is then evaluated against every
constraint and each one receives a verdict:

    met        the patient satisfies the criterion
    not_met    the patient violates it, which excludes the trial
    unknown    the criterion is understood but the profile lacks the field
    manual     recognised as needing human review (laboratory values,
               investigator judgement, consent, contraception)

Trials are ranked into three tiers (likely eligible, needs review, excluded)
and, within a tier, by the share of criteria confirmed and by the lexical
relevance of the trial to the patient summary. Every verdict is shown with its
source line so the research nurse can audit the reasoning in seconds.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np

from oncora.data.trials import TrialRecord
from oncora.models.patient import PatientProfile
from oncora.rag.bm25 import BM25Index
from oncora.rag.text import tokenize

INCLUSION_HEADER = re.compile(r"^\s*(inclusion criteria|disease characteristics|patient characteristics)\s*:?\s*$", re.I)
EXCLUSION_HEADER = re.compile(r"^\s*(exclusion criteria)\s*:?\s*$", re.I)
PRIOR_HEADER = re.compile(r"^\s*(prior concurrent therapy)\s*:?\s*$", re.I)
NEGATED = re.compile(r"^\s*(no|not|without|must not|patients? (who|with) (have|has) (not )?(received|had)? ?no|absence of|none)\b", re.I)
INELIGIBLE = re.compile(r"\b(ineligible|not eligible|are excluded|will be excluded|not allowed|not permitted|is excluded)\b", re.I)
LIMIT_OR_WASHOUT = re.compile(
    r"\b(no more than|not more than|up to \d|maximum of \d|at least \d+ (?:weeks?|days?|months?|years?) (?:since|after|from)|"
    r"since (?:prior|previous|last)|within (?:the )?(?:last |past )?\d+ (?:weeks?|days?|months?|years?)|\d+ or fewer|\d+ or more prior)\b",
    re.I,
)

THERAPY_TERMS = {
    "trastuzumab": r"trastuzumab|herceptin",
    "pertuzumab": r"pertuzumab",
    "taxane": r"taxane|paclitaxel|docetaxel",
    "anthracycline": r"anthracycline|doxorubicin|epirubicin",
    "cdk4_6_inhibitor": r"cdk ?4/6|palbociclib|ribociclib|abemaciclib",
    "aromatase_inhibitor": r"aromatase inhibitor|letrozole|anastrozole|exemestane",
    "endocrine_therapy": r"hormonal therapy|endocrine therapy|hormone therapy|tamoxifen",
    "chemotherapy": r"chemotherapy",
    "immunotherapy": r"immunotherapy|checkpoint|pd-?1|pd-?l1|pembrolizumab|atezolizumab",
    "parp_inhibitor": r"parp|olaparib|talazoparib",
    "capecitabine": r"capecitabine",
    "platinum": r"platinum|carboplatin|cisplatin",
    "radiotherapy": r"radiotherapy|radiation therapy",
}

EXPANDS = {
    "chemotherapy": {"chemotherapy", "taxane", "anthracycline", "capecitabine", "platinum"},
    "endocrine_therapy": {"endocrine_therapy", "aromatase_inhibitor", "tamoxifen", "letrozole", "anastrozole", "exemestane", "fulvestrant"},
    "aromatase_inhibitor": {"aromatase_inhibitor", "letrozole", "anastrozole", "exemestane"},
    "cdk4_6_inhibitor": {"cdk4_6_inhibitor", "palbociclib", "ribociclib", "abemaciclib"},
    "trastuzumab": {"trastuzumab", "t_dm1", "trastuzumab_emtansine"},
}

MANUAL_RX = re.compile(
    r"neutrophil|platelet|hemoglobin|haemoglobin|bilirubin|creatinine|ast\b|alt\b|aminotransferase|clearance|"
    r"informed consent|contracepti|life expectancy|measurable|investigator|willing|able to|compliance|"
    r"psychiatric|infection|hepatitis|hiv|cardiac disease|myocardial|arrhythmia|neuropathy|allerg|hypersensitiv",
    re.I,
)


@dataclass
class Criterion:
    line: int
    text: str
    kind: str
    ctype: str
    requirement: dict
    verdict: str = "unknown"
    explanation: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParsedTrial:
    nct_id: str
    criteria: list[Criterion] = field(default_factory=list)
    manual_lines: int = 0
    unparsed_lines: int = 0

    def to_dict(self) -> dict:
        return {
            "nct_id": self.nct_id,
            "criteria": [c.to_dict() for c in self.criteria],
            "manual_lines": self.manual_lines,
            "unparsed_lines": self.unparsed_lines,
        }


def _age(text: str) -> dict | None:
    t = text.lower()
    if not re.search(r"\bage|\byears?\b|\byrs\b", t):
        return None
    m = re.search(r"(\d{2})\s*(?:to|-|and)\s*(\d{2})\s*(?:years|yrs)", t) or re.search(r"(?:aged?|between)\s*(\d{2})\s*(?:to|-|and)\s*(\d{2})", t)
    if m:
        return {"min": int(m.group(1)), "max": int(m.group(2))}
    m = re.search(r"(?:age\s*)?(?:>=|≥|=>|greater than or equal to|at least|of at least|minimum of)\s*(\d{2})", t) or re.search(
        r"(\d{2})\s*(?:years|yrs)?\s*(?:of age )?(?:or|and) (?:older|over|above)", t
    )
    if m:
        return {"min": int(m.group(1))}
    m = re.search(r"(?:over|older than|>)\s*(\d{2})\s*(?:years|yrs)", t) or re.search(r"age\s*>\s*(\d{2})", t)
    if m:
        return {"min": int(m.group(1)) + 1}
    m = re.search(r"(?:age\s*)?(?:<=|≤|=<|no older than|up to|under|less than or equal to)\s*(\d{2})\s*(?:years|yrs)?", t)
    if m and "age" in t:
        return {"max": int(m.group(1))}
    return None


def _ecog(text: str) -> dict | None:
    t = text.lower()
    if "ecog" in t or "performance status" in t or "zubrod" in t or "who performance" in t:
        m = re.search(r"(?:0|zero)\s*(?:-|to|or|,)\s*([12])", t) or re.search(r"(?:<=|≤|=<|of|less than or equal to)\s*([0-3])\b", t)
        if m:
            return {"max": int(m.group(1))}
    m = re.search(r"karnofsky[^0-9]{0,40}(\d{2,3})\s*%?", t)
    if m:
        k = int(m.group(1))
        return {"max": 1 if k >= 80 else 2 if k >= 60 else 3}
    return None


AMBIGUOUS_RECEPTOR = re.compile(r"positive or negative|negative or positive|either|regardless|irrespective|any (er|her2|hormone)|\(1\).*\(2\)|following subtypes", re.I)


def _her2(text: str) -> str | None:
    t = text.lower()
    if AMBIGUOUS_RECEPTOR.search(t):
        return None
    if not re.search(r"her-?2|erbb2|neu\b", t):
        return None
    if re.search(r"her-?2[- ]?(negative|non[- ]?(amplified|overexpress))|her-?2 (0|1\+)|not (over)?express\w* her-?2|her-?2[- ]?normal|her2/neu[- ]negative", t):
        return "negative"
    if re.search(r"her-?2[- ]?low", t):
        return "low"
    if re.search(r"(positive|overexpress\w*|amplif\w*|3\+|fish[- ]positive)", t):
        return "positive"
    return None


def _hormone(text: str) -> str | None:
    t = text.lower()
    if AMBIGUOUS_RECEPTOR.search(t):
        return None
    if re.search(r"triple[- ]negative", t):
        return "triple_negative"
    if not re.search(r"estrogen|oestrogen|progesterone|hormone receptor|\ber\b|\bpgr?\b|\ber[+-]|\bhr[+-]", t):
        return None
    if re.search(r"(estrogen|oestrogen|hormone)[- ]receptor[- ]?(\(er\)|\(hr\))?[- ]?(and|or|and/or)?[- ]?(progesterone receptor[- ]?(\(pr\)|\(pgr\))?)?[- ]?negative|\ber[- ]?negative|\ber-\b|er and pr negative|both er and pr negative", t):
        return "negative"
    if re.search(r"positive|\ber\+|\bhr\+|expressing", t):
        return "positive"
    return None


def _setting(text: str) -> str | None:
    t = text.lower()
    if re.search(r"no (evidence of )?(distant )?metasta|non[- ]?metastatic|without (distant )?metasta|\bm0\b|stage (0|i|ii|iii|1|2|3)[abc]?\b(?! ?v)|stage 0|operable|resectable(?! )|early[- ]stage|adjuvant", t) and not re.search(r"stage iv|unresectable", t):
        return "early"
    if re.search(r"metastatic|stage iv|\bm1\b|advanced|locally recurrent|unresectable|recurrent", t):
        return "metastatic"
    return None


def _menopause(text: str) -> str | None:
    t = text.lower()
    if "postmenopausal" in t or "post-menopausal" in t:
        return "postmenopausal"
    if "premenopausal" in t or "pre-menopausal" in t:
        return "premenopausal"
    return None


def _therapies(text: str) -> list[str]:
    t = text.lower()
    return [k for k, rx in THERAPY_TERMS.items() if re.search(rx, t)]


def parse_eligibility(rec: TrialRecord) -> ParsedTrial:
    parsed = ParsedTrial(nct_id=rec.nct_id)
    section = "inclusion"
    for i, raw in enumerate(rec.eligibility):
        line = raw.strip()
        if not line:
            continue
        if INCLUSION_HEADER.match(line):
            section = "inclusion"
            continue
        if EXCLUSION_HEADER.match(line):
            section = "exclusion"
            continue
        if PRIOR_HEADER.match(line):
            section = "prior"
            continue
        if LIMIT_OR_WASHOUT.search(line):
            # Line limits or times prior treatment rather than defining the population: needs human review.
            parsed.manual_lines += 1
            continue
        negated = bool(NEGATED.match(line)) or bool(INELIGIBLE.search(line))
        effective = "exclusion" if (section == "exclusion" or negated) else "inclusion"
        found = False

        if (a := _age(line)) and effective == "inclusion":
            parsed.criteria.append(Criterion(i, line, "inclusion", "age", a))
            found = True
        if (e := _ecog(line)) and effective == "inclusion":
            parsed.criteria.append(Criterion(i, line, "inclusion", "ecog", e))
            found = True
        if h := _her2(line):
            parsed.criteria.append(Criterion(i, line, effective, "her2", {"status": h}))
            found = True
        if hr := _hormone(line):
            parsed.criteria.append(Criterion(i, line, effective, "hormone", {"status": hr}))
            found = True
        if (s := _setting(line)) and section != "prior":
            parsed.criteria.append(Criterion(i, line, effective, "setting", {"setting": s}))
            found = True
        if m := _menopause(line):
            parsed.criteria.append(Criterion(i, line, effective, "menopause", {"status": m}))
            found = True
        if re.search(r"brca ?1|brca ?2|brca mutation|deleterious brca|gbrca", line, re.I):
            parsed.criteria.append(Criterion(i, line, effective, "brca", {"carrier": True}))
            found = True
        if re.search(r"brain metasta|cns metasta|central nervous system metasta|leptomeningeal", line, re.I):
            parsed.criteria.append(Criterion(i, line, "exclusion" if effective == "exclusion" or section != "inclusion" else "inclusion", "brain_mets", {"allowed_if_treated": bool(re.search(r"treated|stable|controlled", line, re.I))}))
            found = True
        if re.search(r"pregnan|breast[- ]?feeding|lactating|nursing", line, re.I) and effective == "exclusion":
            parsed.criteria.append(Criterion(i, line, "exclusion", "pregnancy", {}))
            found = True
        if m := re.search(r"(?:lvef|ejection fraction)[^0-9]{0,40}(\d{2})\s*%", line, re.I):
            parsed.criteria.append(Criterion(i, line, "inclusion", "lvef", {"min": int(m.group(1))}))
            found = True
        therapies = _therapies(line)
        if therapies and (section == "prior" or effective == "exclusion" or re.search(r"prior|previous|received|progress|refractory|failed", line, re.I)):
            prior_kind = "exclusion" if (negated or section == "exclusion") else "inclusion"
            if re.search(r"progress|refractory|failed|resistan|must have (received|had)|previously treated with|after", line, re.I) and not negated:
                prior_kind = "inclusion"
            if section == "prior" and not negated and not re.search(r"must|required|progress|refractory|failed|at least", line, re.I):
                prior_kind = "allowed"
            if prior_kind != "allowed":
                parsed.criteria.append(Criterion(i, line, prior_kind, "prior_therapy", {"therapies": therapies}))
                found = True
        if not found:
            if MANUAL_RX.search(line):
                parsed.manual_lines += 1
            else:
                parsed.unparsed_lines += 1
    trial_setting = {
        "advanced or metastatic": "metastatic",
        "early or adjuvant": "early",
        "neoadjuvant": "early",
        "prevention or screening": "early",
    }.get(rec.setting)
    if trial_setting and not any(c.ctype == "setting" and c.kind == "inclusion" for c in parsed.criteria):
        parsed.criteria.append(
            Criterion(-1, f"Trial population inferred from the record as {rec.setting} disease", "inclusion", "setting", {"setting": trial_setting})
        )
    return parsed


def _patient_therapies(profile: PatientProfile) -> set[str]:
    base = {p.lower() for p in profile.prior_therapies}
    out = set(base)
    for group, members in EXPANDS.items():
        if base & members:
            out.add(group)
    if base & {"tamoxifen", "letrozole", "anastrozole", "exemestane", "fulvestrant", "aromatase_inhibitor"}:
        out.add("endocrine_therapy")
    return out


def evaluate(criterion: Criterion, profile: PatientProfile, derived: dict) -> Criterion:
    c, r = criterion, criterion.requirement
    excl = c.kind == "exclusion"

    def verdict(holds: bool | None, why: str) -> None:
        if holds is None:
            c.verdict, c.explanation = "unknown", why
        else:
            ok = holds if not excl else not holds
            c.verdict, c.explanation = ("met" if ok else "not_met"), why

    if c.ctype == "age":
        if profile.age is None:
            verdict(None, "Age not recorded")
        else:
            ok = profile.age >= r.get("min", 0) and profile.age <= r.get("max", 200)
            verdict(ok, f"Age {profile.age:g} against range {r.get('min', 'any')} to {r.get('max', 'any')}")
    elif c.ctype == "ecog":
        verdict(None if profile.ecog is None else profile.ecog <= r["max"], "ECOG not recorded" if profile.ecog is None else f"ECOG {profile.ecog} against maximum {r['max']}")
    elif c.ctype == "her2":
        if profile.her2_status == "unknown":
            verdict(None, "HER2 status not recorded")
        else:
            want = r["status"]
            have = profile.her2_status
            holds = have == want or (want == "negative" and have == "low")
            verdict(holds, f"HER2 {have}; criterion refers to HER2 {want}")
    elif c.ctype == "hormone":
        want = r["status"]
        if profile.er_status == "unknown":
            verdict(None, "Receptor status not recorded")
        elif want == "triple_negative":
            verdict(derived["triple_negative"], "Triple negative status from ER, PR and HER2")
        else:
            verdict(derived["hr_positive"] == (want == "positive"), f"Hormone receptor {'positive' if derived['hr_positive'] else 'negative'}; criterion refers to {want}")
    elif c.ctype == "setting":
        have = "metastatic" if profile.setting == "metastatic" else "early"
        verdict(have == r["setting"], f"Disease setting {profile.setting.replace('_', ' ')}; criterion refers to {r['setting']} disease")
    elif c.ctype == "menopause":
        if profile.menopausal_status == "unknown":
            verdict(None, "Menopausal status not recorded")
        else:
            verdict(profile.menopausal_status == r["status"], f"Patient is {profile.menopausal_status}")
    elif c.ctype == "brca":
        if profile.brca == "unknown":
            verdict(None, "Germline BRCA status unknown")
        else:
            verdict(derived["brca_carrier"], f"BRCA status {profile.brca}")
    elif c.ctype == "brain_mets":
        if c.kind == "exclusion":
            if profile.brain_metastases and r.get("allowed_if_treated"):
                c.verdict, c.explanation = "unknown", "Brain metastases present; trial allows treated or stable disease, verify"
            else:
                c.verdict, c.explanation = ("not_met", "Brain metastases present") if profile.brain_metastases else ("met", "No brain metastases recorded")
        else:
            verdict(profile.brain_metastases, "Trial requires brain metastases")
    elif c.ctype == "pregnancy":
        c.verdict, c.explanation = ("not_met", "Patient is pregnant") if profile.pregnant else ("met", "Not pregnant")
    elif c.ctype == "lvef":
        verdict(None if profile.lvef is None else profile.lvef >= r["min"], "LVEF not recorded" if profile.lvef is None else f"LVEF {profile.lvef:g} percent against minimum {r['min']}")
    elif c.ctype == "prior_therapy":
        received = _patient_therapies(profile)
        hits = [t for t in r["therapies"] if t in received]
        if not profile.prior_therapies and profile.setting != "metastatic":
            verdict(False, "No prior therapy recorded")
        else:
            verdict(bool(hits), ("Received " + ", ".join(hits)) if hits else "No matching prior therapy recorded")
    return c


TIER_ORDER = {"likely eligible": 0, "needs review": 1, "excluded": 2}


class TrialMatcher:
    def __init__(self, trials: list[TrialRecord]) -> None:
        self.trials = {t.nct_id: t for t in trials}
        self.parsed = {t.nct_id: parse_eligibility(t) for t in trials}
        self.ids = list(self.trials)
        self.bm25 = BM25Index().fit([tokenize(t.title + " " + " ".join(t.eligibility) + " " + " ".join(t.intervention)) for t in trials])

    def parsing_stats(self) -> dict:
        crit = [c for p in self.parsed.values() for c in p.criteria]
        by_type: dict[str, int] = {}
        for c in crit:
            by_type[c.ctype] = by_type.get(c.ctype, 0) + 1
        return {
            "trials": len(self.parsed),
            "structured_criteria": len(crit),
            "criteria_per_trial": round(len(crit) / max(len(self.parsed), 1), 2),
            "by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
            "manual_review_lines": sum(p.manual_lines for p in self.parsed.values()),
            "unparsed_lines": sum(p.unparsed_lines for p in self.parsed.values()),
        }

    def match(self, profile: PatientProfile, top_k: int = 25, include_excluded: bool = False, query: str | None = None) -> dict:
        derived = profile.derived()
        rel = self.bm25.score(tokenize((query or "") + " " + profile.summary()))
        rel = rel / (rel.max() or 1.0)
        rows = []
        for j, nct in enumerate(self.ids):
            p = self.parsed[nct]
            evaluated = [evaluate(Criterion(**{k: v for k, v in asdict(c).items()}), profile, derived) for c in p.criteria]
            met = sum(c.verdict == "met" for c in evaluated)
            not_met = [c for c in evaluated if c.verdict == "not_met"]
            unknown = sum(c.verdict == "unknown" for c in evaluated)
            tier = "excluded" if not_met else ("likely eligible" if unknown == 0 and met >= 2 else "needs review")
            confirmed = met / max(len(evaluated), 1)
            score = 0.45 * confirmed + 0.25 * min(met, 8) / 8 + 0.3 * float(rel[j])
            rows.append((tier, score, nct, evaluated, met, len(not_met), unknown, p.manual_lines))
        rows.sort(key=lambda r: (TIER_ORDER[r[0]], -r[1]))
        tiers = {t: sum(1 for r in rows if r[0] == t) for t in TIER_ORDER}
        out = []
        for tier, score, nct, evaluated, met, n_not, unk, manual in rows:
            if tier == "excluded" and not include_excluded:
                continue
            t = self.trials[nct]
            out.append(
                {
                    "nct_id": nct,
                    "title": t.title,
                    "url": t.url,
                    "setting": t.setting,
                    "agents": t.agents[:8],
                    "arms": t.arms[:4],
                    "tier": tier,
                    "score": round(score, 4),
                    "criteria_met": met,
                    "criteria_not_met": n_not,
                    "criteria_unknown": unk,
                    "manual_review_items": manual,
                    "criteria": [c.to_dict() for c in evaluated],
                }
            )
            if len(out) >= top_k:
                break
        return {"summary": profile.summary(), "tiers": tiers, "trials": out}

    def export_parsed(self) -> list[dict]:
        out = []
        for nct, p in self.parsed.items():
            t = self.trials[nct]
            out.append(
                {
                    "nct_id": nct,
                    "title": t.title,
                    "setting": t.setting,
                    "agents": t.agents[:8],
                    "arms": t.arms[:4],
                    "manual_lines": p.manual_lines,
                    "criteria": [{"line": c.line, "text": c.text[:400], "kind": c.kind, "ctype": c.ctype, "requirement": c.requirement} for c in p.criteria],
                    "index_text": " ".join(tokenize(t.title + " " + " ".join(t.eligibility[:30]))),
                }
            )
        return out


def bm25_relevance(matcher: TrialMatcher, text: str) -> np.ndarray:
    return matcher.bm25.score(tokenize(text))
