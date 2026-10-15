"""Prognostic modelling of relapse free survival.

Development cohort: METABRIC complete cases on the core clinical schema.
External validation: GBSG2 (node positive patients from a German randomised
trial with a different era, assay and population). The comparison baseline is
the Nottingham Prognostic Index, which every UK MDT already uses, so the
report answers the practical question: does the model add anything to NPI?

Three learners are compared under identical five fold cross validation:
Cox proportional hazards (ridge penalised), random survival forest and
gradient boosted survival trees. The Cox model is the deployed model because
it is calibrated, exportable to the browser as a handful of coefficients and
fully explainable per patient; the tree ensembles are reported to show what
non linear models would add.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv

from oncora.data.cohorts import CORE_FEATURES, EXTENDED_FEATURES, FEATURE_LABELS, kaplan_meier, logrank_test

log = logging.getLogger(__name__)

# Model inputs are transformed versions of the clinical features. The same
# transform is reproduced in the browser engine, so keep it simple and explicit.
TRANSFORMS = {
    "age": ("scale", 0.1),  # per decade
    "tumour_size_mm": ("cap_scale", (100.0, 0.1)),  # centimetres, capped at 10 cm
    "grade": ("identity", None),
    "positive_nodes": ("log1p", None),
    "er_positive": ("identity", None),
    "pr_positive": ("identity", None),
    "postmenopausal": ("identity", None),
    "hormone_therapy": ("identity", None),
    "her2_positive": ("identity", None),
    "chemotherapy": ("identity", None),
    "radiotherapy": ("identity", None),
    "mastectomy": ("identity", None),
    "lobular_histology": ("identity", None),
}


def transform(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for f in features:
        kind, arg = TRANSFORMS[f]
        x = df[f].astype(float)
        if kind == "scale":
            x = x * arg
        elif kind == "cap_scale":
            x = x.clip(upper=arg[0]) * arg[1]
        elif kind == "log1p":
            x = np.log1p(x.clip(lower=0))
        out[f] = x
    return out


def npi(df: pd.DataFrame) -> pd.Series:
    node_stage = np.where(df["positive_nodes"] == 0, 1, np.where(df["positive_nodes"] <= 3, 2, 3))
    return 0.2 * (df["tumour_size_mm"] / 10.0) + df["grade"] + node_stage


def make_learners(seed: int) -> dict:
    return {
        "cox_ph": Pipeline([("scale", StandardScaler()), ("model", CoxPHSurvivalAnalysis(alpha=0.5, ties="efron"))]),
        "random_survival_forest": RandomSurvivalForest(
            n_estimators=300, min_samples_leaf=20, max_features="sqrt", n_jobs=-1, random_state=seed
        ),
        "gradient_boosted_survival": GradientBoostingSurvivalAnalysis(
            n_estimators=250, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=seed
        ),
    }


@dataclass
class SurvivalData:
    X: pd.DataFrame
    y: np.ndarray
    raw: pd.DataFrame


def prepare(df: pd.DataFrame, features: list[str], time_col: str = "rfs_months", event_col: str = "rfs_event") -> SurvivalData:
    keep = df.dropna(subset=features + [time_col, event_col]).copy()
    keep = keep[keep[time_col] > 0]
    X = transform(keep, features)
    y = Surv.from_arrays(event=keep[event_col].astype(bool).values, time=keep[time_col].astype(float).values)
    return SurvivalData(X=X, y=y, raw=keep)


def _risk(model, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float)


def _surv_matrix(model, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    fns = model.predict_survival_function(X)
    return np.vstack([fn(times) for fn in fns])


def evaluate(model, train_y, X_test, y_test, horizons: list[float], ibs_grid: np.ndarray, tau: float) -> dict:
    risk = _risk(model, X_test)
    c_h = concordance_index_censored(y_test["event"], y_test["time"], risk)[0]
    c_u = concordance_index_ipcw(train_y, y_test, risk, tau=tau)[0]
    max_t = min(train_y["time"].max(), y_test["time"].max()) - 1e-6
    hz = [h for h in horizons if h < max_t]
    aucs, mean_auc = cumulative_dynamic_auc(train_y, y_test, risk, hz)
    grid = ibs_grid[(ibs_grid > y_test["time"].min()) & (ibs_grid < max_t)]
    surv = _surv_matrix(model, X_test, grid)
    ibs = integrated_brier_score(train_y, y_test, surv, grid)
    return {
        "c_index_harrell": round(float(c_h), 4),
        "c_index_uno": round(float(c_u), 4),
        "time_dependent_auc": {f"{int(h)}m": round(float(a), 4) for h, a in zip(hz, aucs, strict=False)},
        "mean_auc": round(float(mean_auc), 4),
        "integrated_brier_score": round(float(ibs), 4),
    }


def calibration_table(model, X, y, horizon: float, bins: int = 5) -> list[dict]:
    """Predicted versus Kaplan Meier observed survival at a horizon, by risk quintile."""
    surv = _surv_matrix(model, X, np.array([horizon]))[:, 0]
    order = pd.qcut(surv, bins, labels=False, duplicates="drop")
    rows = []
    for b in sorted(set(order)):
        m = order == b
        km = kaplan_meier(y["time"][m], y["event"][m].astype(int), np.array([horizon]))
        rows.append(
            {
                "bin": int(b) + 1,
                "n": int(m.sum()),
                "predicted": round(float(surv[m].mean()), 4),
                "observed": km["survival"][0],
                "observed_lower": km["lower"][0],
                "observed_upper": km["upper"][0],
            }
        )
    return rows


def cross_validate(data: SurvivalData, seed: int, folds: int = 5) -> dict:
    horizons = [36.0, 60.0, 120.0]
    grid = np.arange(12.0, 181.0, 6.0)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    results: dict[str, list[dict]] = {}
    npi_scores = []
    for fold, (tr, te) in enumerate(skf.split(data.X, data.y["event"])):
        Xtr, Xte = data.X.iloc[tr], data.X.iloc[te]
        ytr, yte = data.y[tr], data.y[te]
        for name, learner in make_learners(seed + fold).items():
            learner.fit(Xtr, ytr)
            results.setdefault(name, []).append(evaluate(learner, ytr, Xte, yte, horizons, grid, tau=120.0))
        if {"tumour_size_mm", "grade", "positive_nodes"} <= set(data.raw.columns):
            n = npi(data.raw.iloc[te]).values
            npi_scores.append(concordance_index_censored(yte["event"], yte["time"], n)[0])
    summary = {}
    for name, folds_res in results.items():
        summary[name] = {
            "c_index_harrell_mean": round(float(np.mean([r["c_index_harrell"] for r in folds_res])), 4),
            "c_index_harrell_sd": round(float(np.std([r["c_index_harrell"] for r in folds_res])), 4),
            "c_index_uno_mean": round(float(np.mean([r["c_index_uno"] for r in folds_res])), 4),
            "mean_auc_mean": round(float(np.mean([r["mean_auc"] for r in folds_res])), 4),
            "integrated_brier_score_mean": round(float(np.mean([r["integrated_brier_score"] for r in folds_res])), 4),
            "folds": folds_res,
        }
    if npi_scores:
        summary["npi_baseline"] = {"c_index_harrell_mean": round(float(np.mean(npi_scores)), 4), "c_index_harrell_sd": round(float(np.std(npi_scores)), 4)}
    return summary


def export_cox(model: Pipeline, data: SurvivalData, features: list[str], grid_max: int = 180) -> dict:
    scaler: StandardScaler = model.named_steps["scale"]
    cox: CoxPHSurvivalAnalysis = model.named_steps["model"]
    grid = np.arange(0, grid_max + 1, 1.0)
    baseline = cox.baseline_survival_
    s0 = np.clip(baseline(np.clip(grid, baseline.x.min(), baseline.x.max())), 1e-6, 1.0)
    s0[grid < baseline.x.min()] = 1.0
    lp = _risk(model, data.X)
    cuts = np.quantile(lp, [1 / 3, 2 / 3])
    coef = cox.coef_
    return {
        "model": "cox_ph_ridge",
        "endpoint": "relapse free survival",
        "time_unit": "months",
        "features": features,
        "feature_labels": {f: FEATURE_LABELS[f] for f in features},
        "transforms": {f: {"kind": TRANSFORMS[f][0], "arg": TRANSFORMS[f][1]} for f in features},
        "mean": scaler.mean_.round(6).tolist(),
        "scale": scaler.scale_.round(6).tolist(),
        "coef": coef.round(6).tolist(),
        "hazard_ratio_per_sd": np.exp(coef).round(4).tolist(),
        "hazard_ratio_per_unit": np.exp(coef / scaler.scale_).round(4).tolist(),
        "baseline_survival": {"time": grid.tolist(), "survival": s0.round(6).tolist()},
        "risk_group_cuts": cuts.round(6).tolist(),
        "training_n": int(len(data.X)),
        "training_events": int(data.y["event"].sum()),
        "cohort_reference": {f: round(float(data.raw[f].median()), 3) for f in features},
    }


def cox_predict(spec: dict, patient: dict, horizons: tuple[int, ...] = (36, 60, 120)) -> dict:
    """Score one patient with an exported Cox specification (mirrors the browser engine)."""
    z, contributions = [], []
    for i, f in enumerate(spec["features"]):
        val = patient.get(f)
        if val is None:
            val = spec["cohort_reference"][f]
        kind, arg = spec["transforms"][f]["kind"], spec["transforms"][f]["arg"]
        x = float(val)
        if kind == "scale":
            x *= arg
        elif kind == "cap_scale":
            x = min(x, arg[0]) * arg[1]
        elif kind == "log1p":
            x = float(np.log1p(max(x, 0.0)))
        zi = (x - spec["mean"][i]) / spec["scale"][i]
        z.append(zi)
        contributions.append({"feature": f, "label": spec["feature_labels"][f], "value": val, "log_hazard": round(spec["coef"][i] * zi, 4), "imputed": patient.get(f) is None})
    lp = float(np.dot(spec["coef"], z))
    s0 = np.array(spec["baseline_survival"]["survival"])
    curve = np.power(s0, np.exp(lp))
    t = spec["baseline_survival"]["time"]
    group = "low" if lp < spec["risk_group_cuts"][0] else "intermediate" if lp < spec["risk_group_cuts"][1] else "high"
    return {
        "linear_predictor": round(lp, 4),
        "relative_hazard": round(float(np.exp(lp)), 4),
        "risk_group": group,
        "survival_at": {f"{h}m": round(float(curve[min(h, len(curve) - 1)]), 4) for h in horizons},
        "curve": {"time": t, "survival": curve.round(4).tolist(), "baseline": s0.round(4).tolist()},
        "contributions": sorted(contributions, key=lambda c: -abs(c["log_hazard"])),
    }


def train_and_report(metabric: pd.DataFrame, gbsg: pd.DataFrame, out_dir: Path, seed: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"endpoint": "relapse free survival (months)"}

    core = prepare(metabric, CORE_FEATURES)
    report["development_cohort"] = {"name": "METABRIC", "n": int(len(core.X)), "events": int(core.y["event"].sum()), "median_follow_up_months": round(float(np.median(core.y["time"])), 1)}
    log.info("Core model cross validation on %s patients", len(core.X))
    report["core_cross_validation"] = cross_validate(core, seed)

    ext = prepare(metabric, EXTENDED_FEATURES)
    log.info("Extended model cross validation on %s patients", len(ext.X))
    report["extended_cross_validation"] = cross_validate(ext, seed)
    report["extended_cohort_n"] = int(len(ext.X))

    # Final models on the full development cohort.
    learners = make_learners(seed)
    for learner in learners.values():
        learner.fit(core.X, core.y)
    cox = learners["cox_ph"]
    spec = export_cox(cox, core, CORE_FEATURES)
    (out_dir / "cox_core.json").write_text(json.dumps(spec))
    joblib.dump(learners, out_dir / "survival_core_models.joblib", compress=3)

    ext_cox = make_learners(seed)["cox_ph"].fit(ext.X, ext.y)
    ext_spec = export_cox(ext_cox, ext, EXTENDED_FEATURES)
    (out_dir / "cox_extended.json").write_text(json.dumps(ext_spec))

    # External validation on GBSG2.
    val = prepare(gbsg, CORE_FEATURES)
    report["external_cohort"] = {"name": "GBSG2", "n": int(len(val.X)), "events": int(val.y["event"].sum()), "note": "All GBSG2 patients are node positive; follow up is shorter (maximum about 7 years), so horizons are 36 and 60 months."}
    ext_res = {}
    grid = np.arange(6.0, 73.0, 3.0)
    for name, learner in learners.items():
        ext_res[name] = evaluate(learner, core.y, val.X, val.y, [36.0, 60.0], grid, tau=60.0)
    gbsg_npi = npi(val.raw).values
    ext_res["npi_baseline"] = {"c_index_harrell": round(float(concordance_index_censored(val.y["event"], val.y["time"], gbsg_npi)[0]), 4)}
    report["external_validation"] = ext_res
    report["external_calibration_60m"] = calibration_table(cox, val.X, val.y, 60.0)
    report["internal_calibration_120m"] = calibration_table(cox, core.X, core.y, 120.0)
    report["internal_calibration_60m"] = calibration_table(cox, core.X, core.y, 60.0)

    # Same comparison on the node positive METABRIC subset for a like for like reference.
    npos = core.raw["positive_nodes"] > 0
    sub_X, sub_y = core.X[npos.values], core.y[npos.values]
    report["metabric_node_positive_apparent"] = {
        "n": int(npos.sum()),
        "c_index_cox": round(float(concordance_index_censored(sub_y["event"], sub_y["time"], _risk(cox, sub_X))[0]), 4),
        "c_index_npi": round(float(concordance_index_censored(sub_y["event"], sub_y["time"], npi(core.raw[npos.values]).values)[0]), 4),
    }

    # Risk groups and Kaplan Meier curves for the interface.
    km_grid = np.arange(0, 241, 6.0)
    lp = _risk(cox, core.X)
    groups = np.where(lp < spec["risk_group_cuts"][0], "low", np.where(lp < spec["risk_group_cuts"][1], "intermediate", "high"))
    report["risk_groups_metabric"] = {
        g: kaplan_meier(core.y["time"][groups == g], core.y["event"][groups == g].astype(int), km_grid) for g in ("low", "intermediate", "high")
    }
    report["risk_groups_metabric_logrank"] = logrank_test(core.y["time"], core.y["event"].astype(int), groups)
    lpv = _risk(cox, val.X)
    gv = np.where(lpv < spec["risk_group_cuts"][0], "low", np.where(lpv < spec["risk_group_cuts"][1], "intermediate", "high"))
    vgrid = np.arange(0, 85, 3.0)
    report["risk_groups_gbsg2"] = {g: kaplan_meier(val.y["time"][gv == g], val.y["event"][gv == g].astype(int), vgrid) for g in ("low", "intermediate", "high") if (gv == g).sum() > 0}
    report["risk_groups_gbsg2_logrank"] = logrank_test(val.y["time"], val.y["event"].astype(int), gv)
    report["cox_hazard_ratios"] = [
        {"feature": f, "label": FEATURE_LABELS[f], "hr_per_sd": spec["hazard_ratio_per_sd"][i], "hr_per_unit": spec["hazard_ratio_per_unit"][i], "coef": spec["coef"][i]}
        for i, f in enumerate(CORE_FEATURES)
    ]
    report["cox_extended_hazard_ratios"] = [
        {"feature": f, "label": FEATURE_LABELS[f], "hr_per_sd": ext_spec["hazard_ratio_per_sd"][i]} for i, f in enumerate(EXTENDED_FEATURES)
    ]
    # Permutation importance of the forest on the development cohort (C index drop).
    rsf = learners["random_survival_forest"]
    base_c = concordance_index_censored(core.y["event"], core.y["time"], _risk(rsf, core.X))[0]
    rng = np.random.default_rng(seed)
    imp = []
    for f in CORE_FEATURES:
        Xp = core.X.copy()
        Xp[f] = rng.permutation(Xp[f].values)
        c = concordance_index_censored(core.y["event"], core.y["time"], _risk(rsf, Xp))[0]
        imp.append({"feature": f, "label": FEATURE_LABELS[f], "c_index_drop": round(float(base_c - c), 4)})
    report["rsf_permutation_importance"] = sorted(imp, key=lambda d: -d["c_index_drop"])
    return report
