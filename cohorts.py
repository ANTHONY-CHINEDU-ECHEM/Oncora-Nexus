"""Loading and harmonisation of the patient cohorts.

Three cohorts are harmonised onto one clinical schema so that a model trained
on one can be validated on another:

* METABRIC (n = 2,509): development cohort, relapse free survival.
* GBSG2 (n = 686): German Breast Cancer Study Group trial, external validation.
* TCGA BRCA PanCancer Atlas (n = 1,084): cohort analytics and subtype mix.

The core schema intentionally contains only variables a multidisciplinary
team has at the first post surgical meeting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CORE_FEATURES = [
    "age",
    "tumour_size_mm",
    "grade",
    "positive_nodes",
    "er_positive",
    "pr_positive",
    "postmenopausal",
    "hormone_therapy",
]

EXTENDED_FEATURES = CORE_FEATURES + [
    "her2_positive",
    "chemotherapy",
    "radiotherapy",
    "mastectomy",
    "lobular_histology",
]

FEATURE_LABELS = {
    "age": "Age at diagnosis (years)",
    "tumour_size_mm": "Tumour size (mm)",
    "grade": "Histological grade (1 to 3)",
    "positive_nodes": "Positive lymph nodes",
    "er_positive": "Oestrogen receptor positive",
    "pr_positive": "Progesterone receptor positive",
    "postmenopausal": "Postmenopausal",
    "hormone_therapy": "Received endocrine therapy",
    "her2_positive": "HER2 positive",
    "chemotherapy": "Received chemotherapy",
    "radiotherapy": "Received radiotherapy",
    "mastectomy": "Mastectomy (vs breast conserving)",
    "lobular_histology": "Lobular or mixed lobular histology",
}

SUBTYPE_ORDER = ["LumA", "LumB", "Her2", "Basal", "claudin-low", "Normal"]


def _read_cbio(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", comment="#", low_memory=False)


def _yes(series: pd.Series) -> pd.Series:
    return series.map({"YES": 1.0, "NO": 0.0, "Yes": 1.0, "No": 0.0})


def load_metabric(raw_dir: Path) -> pd.DataFrame:
    pat = _read_cbio(raw_dir / "metabric" / "data_clinical_patient.txt")
    smp = _read_cbio(raw_dir / "metabric" / "data_clinical_sample.txt")
    df = pat.merge(smp, on="PATIENT_ID", how="left")
    out = pd.DataFrame({"patient_id": df["PATIENT_ID"]})
    out["age"] = df["AGE_AT_DIAGNOSIS"]
    out["tumour_size_mm"] = df["TUMOR_SIZE"]
    out["grade"] = df["GRADE"]
    out["positive_nodes"] = df["LYMPH_NODES_EXAMINED_POSITIVE"]
    out["er_positive"] = df["ER_STATUS"].map({"Positive": 1.0, "Negative": 0.0})
    out["pr_positive"] = df["PR_STATUS"].map({"Positive": 1.0, "Negative": 0.0})
    out["postmenopausal"] = df["INFERRED_MENOPAUSAL_STATE"].map({"Post": 1.0, "Pre": 0.0})
    out["hormone_therapy"] = _yes(df["HORMONE_THERAPY"])
    out["her2_positive"] = df["HER2_STATUS"].map({"Positive": 1.0, "Negative": 0.0})
    out["chemotherapy"] = _yes(df["CHEMOTHERAPY"])
    out["radiotherapy"] = _yes(df["RADIO_THERAPY"])
    out["mastectomy"] = df["BREAST_SURGERY"].map({"MASTECTOMY": 1.0, "BREAST CONSERVING": 0.0})
    hist = df["CANCER_TYPE_DETAILED"].fillna("")
    out["lobular_histology"] = hist.str.contains("Lobular").astype(float)
    out["npi"] = df["NPI"]
    out["stage"] = df["TUMOR_STAGE"]
    out["histology"] = hist
    out["subtype"] = df["CLAUDIN_SUBTYPE"].where(df["CLAUDIN_SUBTYPE"] != "NC")
    out["integrative_cluster"] = df["INTCLUST"]
    out["rfs_months"] = df["RFS_MONTHS"]
    out["rfs_event"] = df["RFS_STATUS"].map({"1:Recurred": 1, "0:Not Recurred": 0})
    out["os_months"] = df["OS_MONTHS"]
    out["os_event"] = df["OS_STATUS"].map({"1:DECEASED": 1, "0:LIVING": 0})
    out["dss_event"] = df["VITAL_STATUS"].map({"Died of Disease": 1, "Died of Other Causes": 0, "Living": 0})
    out["cohort"] = "METABRIC"
    return out


def load_gbsg2() -> pd.DataFrame:
    """GBSG2 ships with scikit-survival, so it needs no network access."""
    from sksurv.datasets import load_gbsg2 as _load

    X, y = _load()
    out = pd.DataFrame(index=X.index)
    out["patient_id"] = [f"GBSG-{i:04d}" for i in range(len(X))]
    out["age"] = X["age"].astype(float)
    out["tumour_size_mm"] = X["tsize"].astype(float)
    out["grade"] = X["tgrade"].astype(str).map({"I": 1.0, "II": 2.0, "III": 3.0})
    out["positive_nodes"] = X["pnodes"].astype(float)
    # Conventional 10 fmol/mg cytosol cut point for receptor positivity.
    out["er_positive"] = (X["estrec"].astype(float) >= 10).astype(float)
    out["pr_positive"] = (X["progrec"].astype(float) >= 10).astype(float)
    out["postmenopausal"] = (X["menostat"].astype(str) == "Post").astype(float)
    out["hormone_therapy"] = (X["horTh"].astype(str) == "yes").astype(float)
    out["rfs_months"] = y["time"] / 30.4375
    out["rfs_event"] = y["cens"].astype(int)
    out["cohort"] = "GBSG2"
    return out.reset_index(drop=True)


TCGA_SUBTYPE = {
    "BRCA_LumA": "LumA",
    "BRCA_LumB": "LumB",
    "BRCA_Her2": "Her2",
    "BRCA_Basal": "Basal",
    "BRCA_Normal": "Normal",
}


def _stage_number(value: str | float) -> float:
    if not isinstance(value, str):
        return np.nan
    v = value.upper().replace("STAGE", "").strip()
    for roman, num in (("IV", 4.0), ("III", 3.0), ("II", 2.0), ("I", 1.0)):
        if v.startswith(roman):
            return num
    return np.nan


def load_tcga(raw_dir: Path) -> pd.DataFrame:
    pat = _read_cbio(raw_dir / "tcga_brca" / "data_clinical_patient.txt")
    out = pd.DataFrame({"patient_id": pat["PATIENT_ID"]})
    out["age"] = pd.to_numeric(pat["AGE"], errors="coerce")
    out["stage"] = pat["AJCC_PATHOLOGIC_TUMOR_STAGE"].map(_stage_number)
    out["subtype"] = pat["SUBTYPE"].map(TCGA_SUBTYPE)
    out["os_months"] = pd.to_numeric(pat["OS_MONTHS"], errors="coerce")
    out["os_event"] = pat["OS_STATUS"].map({"1:DECEASED": 1, "0:LIVING": 0})
    out["pfs_months"] = pd.to_numeric(pat["PFS_MONTHS"], errors="coerce")
    out["pfs_event"] = pat["PFS_STATUS"].map({"1:PROGRESSION": 1, "0:CENSORED": 0})
    out["cohort"] = "TCGA BRCA"
    return out


def load_pam50_expression(raw_dir: Path) -> pd.DataFrame:
    """Samples by genes matrix (log2 intensity) for the PAM50 panel."""
    path = raw_dir / "metabric" / "expression_pam50.tsv"
    expr = pd.read_csv(path, sep="\t")
    expr = expr.drop(columns=[c for c in ("Entrez_Gene_Id",) if c in expr.columns])
    expr = expr.groupby("Hugo_Symbol").mean(numeric_only=True)
    expr.index = expr.index.str.replace("ORC6L", "ORC6")
    return expr.T


def kaplan_meier(time: np.ndarray, event: np.ndarray, grid: np.ndarray) -> dict:
    """Product limit estimator with Greenwood confidence bands on a fixed grid."""
    order = np.argsort(time)
    t, e = time[order], event[order]
    uniq = np.unique(t[e == 1])
    surv, var_acc = 1.0, 0.0
    steps_t, steps_s, steps_lo, steps_hi = [0.0], [1.0], [1.0], [1.0]
    for ut in uniq:
        at_risk = np.sum(t >= ut)
        d = np.sum((t == ut) & (e == 1))
        if at_risk == 0:
            continue
        surv *= 1 - d / at_risk
        if at_risk > d:
            var_acc += d / (at_risk * (at_risk - d))
        se = surv * np.sqrt(var_acc)
        steps_t.append(float(ut))
        steps_s.append(surv)
        steps_lo.append(max(0.0, surv - 1.96 * se))
        steps_hi.append(min(1.0, surv + 1.96 * se))
    idx = np.searchsorted(np.array(steps_t), grid, side="right") - 1
    at_risk_grid = [int(np.sum(time >= g)) for g in grid]
    return {
        "time": grid.tolist(),
        "survival": [round(float(steps_s[i]), 4) for i in idx],
        "lower": [round(float(steps_lo[i]), 4) for i in idx],
        "upper": [round(float(steps_hi[i]), 4) for i in idx],
        "at_risk": at_risk_grid,
        "n": int(len(time)),
        "events": int(event.sum()),
    }


def logrank_test(time: np.ndarray, event: np.ndarray, group: np.ndarray) -> dict:
    """Multi group log rank test (chi square with k minus 1 degrees of freedom)."""
    from scipy.stats import chi2

    groups = np.unique(group)
    k = len(groups)
    if k < 2:
        return {"chi2": 0.0, "df": 0, "p_value": 1.0}
    times = np.unique(time[event == 1])
    O = np.zeros(k)
    E = np.zeros(k)
    V = np.zeros((k, k))
    for t in times:
        at_risk = np.array([np.sum((time >= t) & (group == g)) for g in groups], dtype=float)
        deaths = np.array([np.sum((time == t) & (event == 1) & (group == g)) for g in groups], dtype=float)
        n, d = at_risk.sum(), deaths.sum()
        if n <= 1:
            continue
        O += deaths
        E += d * at_risk / n
        factor = d * (n - d) / (n * n * (n - 1))
        V += factor * (np.diag(at_risk * n) - np.outer(at_risk, at_risk))
    diff = (O - E)[:-1]
    Vr = V[:-1, :-1]
    stat = float(diff @ np.linalg.pinv(Vr) @ diff)
    return {"chi2": round(stat, 3), "df": k - 1, "p_value": float(chi2.sf(stat, k - 1))}
