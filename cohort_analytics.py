"""Population analytics for the Cohort Explorer: survival by intrinsic subtype,
receptor mix, age distribution and treatment patterns across METABRIC and TCGA."""

from __future__ import annotations

import numpy as np
import pandas as pd

from oncora.data.cohorts import SUBTYPE_ORDER, kaplan_meier, logrank_test


def _km_by(df: pd.DataFrame, group: str, time: str, event: str, grid: np.ndarray, min_n: int = 30) -> dict:
    d = df.dropna(subset=[group, time, event])
    d = d[d[time] > 0]
    curves = {}
    for g in [s for s in SUBTYPE_ORDER if s in set(d[group])] + sorted(set(d[group]) - set(SUBTYPE_ORDER)):
        sub = d[d[group] == g]
        if len(sub) >= min_n:
            curves[str(g)] = kaplan_meier(sub[time].values.astype(float), sub[event].values.astype(int), grid)
    kept = d[d[group].isin(list(curves))]
    test = logrank_test(kept[time].values.astype(float), kept[event].values.astype(int), kept[group].astype(str).values)
    return {"curves": curves, "logrank": test}


def _hist(values: pd.Series, bins: list[float]) -> dict:
    counts, edges = np.histogram(values.dropna(), bins=bins)
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}


def build(metabric: pd.DataFrame, tcga: pd.DataFrame) -> dict:
    grid_long = np.arange(0, 241, 6.0)
    grid_tcga = np.arange(0, 181, 6.0)
    m = metabric.copy()
    receptor = np.select(
        [m["her2_positive"] == 1, (m["er_positive"] == 1) | (m["pr_positive"] == 1), m["er_positive"] == 0],
        ["HER2 positive", "HR positive, HER2 negative", "Triple negative"],
        default="Unknown",
    )
    m["receptor_group"] = receptor
    out = {
        "metabric": {
            "n": int(len(m)),
            "os_by_subtype": _km_by(m, "subtype", "os_months", "os_event", grid_long),
            "rfs_by_subtype": _km_by(m, "subtype", "rfs_months", "rfs_event", grid_long),
            "rfs_by_receptor_group": _km_by(m[m["receptor_group"] != "Unknown"], "receptor_group", "rfs_months", "rfs_event", grid_long),
            "subtype_counts": {k: int(v) for k, v in m["subtype"].value_counts().items()},
            "receptor_counts": {k: int(v) for k, v in pd.Series(receptor).value_counts().items()},
            "age_histogram": _hist(m["age"], list(range(20, 101, 5))),
            "grade_counts": {str(int(k)): int(v) for k, v in m["grade"].value_counts().sort_index().items()},
            "treatment_rates": {
                "chemotherapy": round(float(m["chemotherapy"].mean()), 3),
                "hormone_therapy": round(float(m["hormone_therapy"].mean()), 3),
                "radiotherapy": round(float(m["radiotherapy"].mean()), 3),
                "mastectomy": round(float(m["mastectomy"].mean()), 3),
            },
            "median_age": round(float(m["age"].median()), 1),
            "median_follow_up_months": round(float(m["os_months"].median()), 1),
        },
        "tcga": {
            "n": int(len(tcga)),
            "os_by_subtype": _km_by(tcga, "subtype", "os_months", "os_event", grid_tcga, min_n=25),
            "subtype_counts": {k: int(v) for k, v in tcga["subtype"].value_counts().items()},
            "stage_counts": {f"Stage {int(k)}": int(v) for k, v in tcga["stage"].value_counts().sort_index().items()},
            "age_histogram": _hist(tcga["age"], list(range(20, 101, 5))),
            "median_age": round(float(tcga["age"].median()), 1),
            "median_follow_up_months": round(float(tcga["os_months"].median()), 1),
        },
    }
    return out
