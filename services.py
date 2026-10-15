"""Service container: loads every artefact once and exposes typed operations."""

from __future__ import annotations

import json
import logging
import time
from functools import cached_property
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from oncora.config import settings
from oncora.data.trials import load_trials
from oncora.models.patient import PatientProfile, evaluate_pathway, load_rules
from oncora.models.subtype import SUBTYPE_DESCRIPTIONS, predict_surrogate
from oncora.models.survival import cox_predict
from oncora.models.trial_matcher import TrialMatcher
from oncora.rag.index import load_index
from oncora.rag.pipeline import AnswerEngine
from oncora.rag.retriever import HybridRetriever

log = logging.getLogger(__name__)


def _json(path: Path) -> dict | list | None:
    return json.loads(path.read_text()) if path.exists() else None


class Services:
    def __init__(self) -> None:
        self.paths = settings.paths
        self.started = time.time()
        t0 = time.time()
        self.index = load_index()
        self.retriever = HybridRetriever(self.index)
        self.engine = AnswerEngine(self.retriever)
        self.rules = load_rules()
        self.cox_core = _json(self.paths.models / "cox_core.json")
        self.cox_extended = _json(self.paths.models / "cox_extended.json")
        self.surrogate = _json(self.paths.models / "subtype_surrogate.json")
        self.cohort = _json(self.paths.models / "cohort_analytics.json")
        self.registry = _json(self.paths.models / "registry.json") or {}
        cards = json.loads(self.paths.knowledge_cards.read_text())
        self.cards = {c["id"]: c for c in cards["cards"]}
        self.cards_meta = cards["meta"]
        _ = self.matcher  # parse every trial protocol at start up rather than on the first request
        log.info("Services ready in %.1fs (%s passages)", time.time() - t0, self.index.size)

    @cached_property
    def matcher(self) -> TrialMatcher:
        return TrialMatcher(load_trials(self.paths.raw))

    @cached_property
    def subtype_models(self) -> dict | None:
        p = self.paths.models / "subtype_models.joblib"
        return joblib.load(p) if p.exists() else None

    def reports(self) -> dict:
        out = {}
        for name in ("retrieval_eval", "survival_eval", "subtype_eval", "nli4ct_eval", "answer_eval_default", "answer_eval_llm", "trial_parsing"):
            data = _json(self.paths.reports / f"{name}.json")
            if data is not None:
                out[name] = data
        return out

    # Prognosis
    def prognosis(self, patient: PatientProfile, model: str = "core", extended_inputs: dict | None = None) -> dict:
        spec = self.cox_core if model == "core" else self.cox_extended
        if spec is None:
            raise RuntimeError("Survival model not trained. Run scripts/train_models.py")
        feats = patient.survival_features()
        if model == "extended":
            feats.update(
                {"her2_positive": None if patient.her2_status == "unknown" else float(patient.her2_status == "positive")}
            )
            feats.update(extended_inputs or {})
        res = cox_predict(spec, feats, horizons=(36, 60, 120))
        res["model"] = spec["model"] + ("_extended" if model == "extended" else "_core")
        res["endpoint"] = spec["endpoint"]
        res["imputed_fields"] = [c["feature"] for c in res["contributions"] if c["imputed"]]
        res["npi"] = patient.npi()
        res["training_n"] = spec["training_n"]
        res["caveats"] = [
            "Estimates reflect METABRIC patients diagnosed between 1977 and 2005 under the treatments they received; contemporary therapy, including HER2 targeted agents, will usually improve outcomes.",
            "External validation on GBSG2 showed good discrimination but optimistic absolute estimates (observed survival lower than predicted), so absolute values should be reconciled with PREDICT Breast.",
            "This is a prognostic model. It does not estimate what would happen under a different treatment choice.",
        ]
        if res["imputed_fields"]:
            res["caveats"].insert(0, "Missing inputs were set to the cohort median: " + ", ".join(res["imputed_fields"]) + ".")
        return res

    # Subtype
    def subtype(self, patient: PatientProfile | None, expression: dict | None) -> dict:
        out: dict = {"descriptions": SUBTYPE_DESCRIPTIONS}
        if patient is not None and self.surrogate:
            out["surrogate"] = predict_surrogate(self.surrogate, patient.surrogate_features())
            out["surrogate"]["model_note"] = "Pathology surrogate: cross validated macro F1 about 0.42 against PAM50. Treat as a prior, not a call."
        if expression and self.subtype_models:
            model = self.subtype_models["expression"]
            genes = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
            if genes is None:
                genes = sorted(expression)
            row = pd.DataFrame([[expression.get(g, np.nan) for g in genes]], columns=genes)
            missing = [g for g in genes if g not in expression]
            row = row.fillna(row.mean(axis=1).iloc[0] if not row.isna().all(axis=1).iloc[0] else 0.0)
            probs = model.predict_proba(row)[0]
            classes = list(model.classes_)
            p = {c: round(float(v), 4) for c, v in zip(classes, probs, strict=False)}
            top = max(p, key=p.get)
            out["expression"] = {"predicted": top, "probabilities": p, "missing_genes": missing, "description": SUBTYPE_DESCRIPTIONS.get(top, "")}
        return out

    # Pathway
    def pathway(self, patient: PatientProfile) -> list[dict]:
        fired = evaluate_pathway(patient, self.rules)
        for r in fired:
            r["card_titles"] = {cid: self.cards[cid]["title"] for cid in r["cards"] if cid in self.cards}
        return fired

    # Case brief
    def case_brief(self, patient: PatientProfile, question: str | None, mode: str) -> dict:
        t0 = time.time()
        pathway = self.pathway(patient)
        prognosis = self.prognosis(patient) if patient.setting != "metastatic" else None
        subtype = self.subtype(patient, None)
        trials = self.matcher.match(patient, top_k=8)
        summary = patient.summary()
        q = question or f"What are the evidence based treatment options for {summary}?"
        evidence = self.engine.answer(q, case_context=summary, mode=mode)
        return {
            "summary": summary,
            "npi": patient.npi(),
            "pathway": pathway,
            "prognosis": prognosis,
            "subtype": subtype,
            "trials": {"tiers": trials["tiers"], "top": trials["trials"]},
            "evidence": evidence,
            "latency_ms": int((time.time() - t0) * 1000),
            "disclaimer": "Decision support for qualified clinicians. Options are presented for multidisciplinary discussion and do not constitute a treatment decision.",
        }

    def health(self) -> dict:
        return {
            "status": "ok",
            "uptime_seconds": int(time.time() - self.started),
            "index_passages": self.index.size,
            "llm_available": self.engine.llm_available(),
            "llm_model": self.engine.llm.model,
            "reranker_trained": self.retriever.reranker.trained,
            "models_loaded": {
                "cox_core": self.cox_core is not None,
                "cox_extended": self.cox_extended is not None,
                "subtype_surrogate": self.surrogate is not None,
            },
        }
