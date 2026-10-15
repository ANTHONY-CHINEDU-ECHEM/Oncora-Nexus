"""Patient profile schema, derived fields and the treatment pathway rule engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

RULES_PATH = Path(__file__).with_name("therapy_rules.json")


class PatientProfile(BaseModel):
    """Deidentified clinical summary. Every field is optional so partial
    information still produces useful, clearly caveated output."""

    age: float | None = Field(None, ge=15, le=110)
    sex: Literal["female", "male"] = "female"
    menopausal_status: Literal["premenopausal", "postmenopausal", "unknown"] = "unknown"
    ecog: int | None = Field(None, ge=0, le=4)
    setting: Literal["early", "locally_advanced", "metastatic"] = "early"
    tumour_size_mm: float | None = Field(None, ge=0, le=300)
    grade: int | None = Field(None, ge=1, le=3)
    positive_nodes: int | None = Field(None, ge=0, le=80)
    er_status: Literal["positive", "low", "negative", "unknown"] = "unknown"
    pr_status: Literal["positive", "negative", "unknown"] = "unknown"
    her2_status: Literal["positive", "low", "negative", "unknown"] = "unknown"
    ki67_percent: float | None = Field(None, ge=0, le=100)
    brca: Literal["brca1", "brca2", "negative", "unknown"] = "unknown"
    pdl1_cps: float | None = Field(None, ge=0, le=100)
    pik3ca_mutation: bool | None = None
    esr1_mutation: bool | None = None
    residual_disease: bool | None = None
    prior_therapies: list[str] = Field(default_factory=list)
    brain_metastases: bool = False
    visceral_crisis: bool = False
    lvef: float | None = Field(None, ge=10, le=90)
    recurrence_score: float | None = Field(None, ge=0, le=100)
    pregnant: bool = False
    hormone_therapy_planned: bool | None = None
    notes: str | None = Field(None, max_length=2000)

    def derived(self) -> dict[str, Any]:
        d: dict[str, Any] = self.model_dump()
        d["hr_positive"] = self.er_status in ("positive", "low") or self.pr_status == "positive"
        d["er_low"] = self.er_status == "low"
        d["her2_positive"] = self.her2_status == "positive"
        d["her2_low"] = self.her2_status == "low"
        d["triple_negative"] = self.er_status == "negative" and self.pr_status != "positive" and self.her2_status in ("negative", "low")
        nodes = self.positive_nodes
        d["node_positive"] = bool(nodes and nodes >= 1)
        d["high_risk_nodes"] = bool(
            nodes is not None and (nodes >= 4 or (nodes >= 1 and ((self.grade == 3) or (self.tumour_size_mm or 0) >= 50)))
        )
        d["brca_carrier"] = self.brca in ("brca1", "brca2")
        pt = {p.lower() for p in self.prior_therapies}
        d["prior_cdk46"] = bool(pt & {"cdk4_6_inhibitor", "palbociclib", "ribociclib", "abemaciclib"})
        d["prior_her2_therapy"] = bool(pt & {"trastuzumab", "pertuzumab", "t_dm1", "trastuzumab_emtansine"})
        d["prior_therapies"] = sorted(pt)
        return d

    def surrogate_features(self) -> dict[str, float | None]:
        nodes = self.positive_nodes
        npi = None
        if self.tumour_size_mm is not None and self.grade is not None and nodes is not None:
            node_stage = 1 if nodes == 0 else 2 if nodes <= 3 else 3
            npi = round(0.2 * self.tumour_size_mm / 10 + self.grade + node_stage, 2)
        return {
            "er_positive": None if self.er_status == "unknown" else float(self.er_status in ("positive", "low")),
            "pr_positive": None if self.pr_status == "unknown" else float(self.pr_status == "positive"),
            "her2_positive": None if self.her2_status == "unknown" else float(self.her2_status == "positive"),
            "grade": None if self.grade is None else float(self.grade),
            "age": self.age,
            "npi": npi,
        }

    def survival_features(self) -> dict[str, float | None]:
        hormone = self.hormone_therapy_planned
        if hormone is None and self.er_status != "unknown":
            hormone = self.er_status in ("positive", "low")
        meno = None if self.menopausal_status == "unknown" else float(self.menopausal_status == "postmenopausal")
        return {
            "age": self.age,
            "tumour_size_mm": self.tumour_size_mm,
            "grade": self.grade,
            "positive_nodes": self.positive_nodes,
            "er_positive": None if self.er_status == "unknown" else float(self.er_status in ("positive", "low")),
            "pr_positive": None if self.pr_status == "unknown" else float(self.pr_status == "positive"),
            "postmenopausal": meno,
            "hormone_therapy": None if hormone is None else float(hormone),
        }

    def npi(self) -> dict | None:
        f = self.surrogate_features()
        if f["npi"] is None:
            return None
        v = f["npi"]
        group = "good" if v <= 3.4 else "moderate" if v <= 5.4 else "poor"
        return {"value": v, "group": group}

    def summary(self) -> str:
        parts = []
        if self.age:
            parts.append(f"{int(self.age)} year old {self.sex}")
        if self.menopausal_status != "unknown":
            parts.append(self.menopausal_status)
        parts.append(self.setting.replace("_", " ") + " breast cancer")
        if self.tumour_size_mm is not None:
            parts.append(f"{self.tumour_size_mm:g} mm")
        if self.grade:
            parts.append(f"grade {self.grade}")
        if self.positive_nodes is not None:
            parts.append(f"{self.positive_nodes} positive node{'' if self.positive_nodes == 1 else 's'}")
        parts.append(f"ER {self.er_status}, PR {self.pr_status}, HER2 {self.her2_status}")
        if self.brca in ("brca1", "brca2"):
            parts.append(f"germline {self.brca.upper()} carrier")
        if self.residual_disease:
            parts.append("residual disease after neoadjuvant therapy")
        if self.prior_therapies:
            parts.append("prior " + ", ".join(self.prior_therapies))
        return ", ".join(parts)


def _check(cond: dict, d: dict) -> bool:
    op, field = cond["op"], cond["field"]
    if op == "any":
        return any(_check(c, d) for c in cond["value"])
    if op == "all":
        return all(_check(c, d) for c in cond["value"])
    v = d.get(field)
    target = cond.get("value")
    if op == "true":
        return v is True
    if op == "false":
        return v is False or v is None
    if op == "missing":
        return v is None
    if v is None:
        return op == "in" and None in (target or [])
    if op == "eq":
        return v == target
    if op == "in":
        return v in target
    if op == "gt":
        return v > target
    if op == "gte":
        return v >= target
    if op == "lt":
        return v < target
    if op == "lte":
        return v <= target
    if op == "includes_any":
        return bool(set(v) & set(target))
    raise ValueError(f"Unknown operator {op}")


def load_rules() -> dict:
    return json.loads(RULES_PATH.read_text())


def evaluate_pathway(profile: PatientProfile, rules: dict | None = None) -> list[dict]:
    rules = rules or load_rules()
    d = profile.derived()
    fired = []
    order = {"standard": 0, "consider": 1, "caution": 2}
    for rule in rules["rules"]:
        if all(_check(c, d) for c in rule["when"]):
            fired.append({k: rule[k] for k in ("id", "category", "strength", "title", "text", "cards")})
    return sorted(fired, key=lambda r: (order.get(r["strength"], 3), r["id"]))
