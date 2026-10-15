"""Intrinsic molecular subtype classification.

Two complementary classifiers are trained on METABRIC:

1. Expression model: multinomial logistic regression on the 50 PAM50 genes
   (Illumina HT 12, log2) predicting the PAM50 plus Claudin low label. It is
   benchmarked against a nearest centroid classifier, which is how PAM50 is
   applied in practice. Caveat stated in the model card: METABRIC labels were
   themselves derived from expression, so this measures how well a compact,
   portable model reproduces the reference call, not agreement with outcome.
2. Surrogate model: gradient boosting on routinely available pathology (ER,
   PR, HER2, grade, age, NPI). It quantifies how far the immunohistochemical
   surrogate diverges from the molecular truth, which is clinically useful:
   it tells the MDT when a genomic assay is most likely to change the label.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from oncora.data.cohorts import SUBTYPE_ORDER

log = logging.getLogger(__name__)

SURROGATE_FEATURES = ["er_positive", "pr_positive", "her2_positive", "grade", "age", "npi"]

SUBTYPE_DESCRIPTIONS = {
    "LumA": "Luminal A: hormone receptor rich, low proliferation, best prognosis, endocrine sensitive.",
    "LumB": "Luminal B: hormone receptor positive with high proliferation; benefits more often from chemotherapy.",
    "Her2": "HER2 enriched: driven by the ERBB2 amplicon; candidate for HER2 targeted therapy.",
    "Basal": "Basal like: mostly triple negative, high grade, chemotherapy sensitive, early relapse risk.",
    "claudin-low": "Claudin low: low cell adhesion genes, immune and stem cell features, usually triple negative.",
    "Normal": "Normal like: resembles normal breast tissue; often reflects low tumour content in the sample.",
}


def _cv_report(y_true, y_pred, labels) -> dict:
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", labels=labels)), 4),
        "cohen_kappa": round(float(cohen_kappa_score(y_true, y_pred)), 4),
        "per_class": {k: {m: round(float(v[m]), 4) for m in ("precision", "recall", "f1-score", "support")} for k, v in rep.items() if k in labels},
        "confusion_matrix": {"labels": labels, "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()},
    }


def train_expression_model(expr: pd.DataFrame, metabric: pd.DataFrame, seed: int) -> tuple[Pipeline, dict]:
    labels = metabric.set_index("patient_id")["subtype"].dropna()
    common = expr.index.intersection(labels.index)
    X = expr.loc[common].astype(float)
    X = X.fillna(X.median())
    y = labels.loc[common].values
    classes = [c for c in SUBTYPE_ORDER if c in set(y)]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    lr = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced"))])
    centroid = Pipeline([("scale", StandardScaler()), ("model", NearestCentroid())])
    pred_lr = cross_val_predict(lr, X, y, cv=skf)
    pred_nc = cross_val_predict(centroid, X, y, cv=skf)
    lr.fit(X, y)
    report = {
        "n": int(len(y)),
        "genes": list(X.columns),
        "class_counts": {c: int((y == c).sum()) for c in classes},
        "logistic_regression_cv": _cv_report(y, pred_lr, classes),
        "nearest_centroid_cv": _cv_report(y, pred_nc, classes),
    }
    coef = lr.named_steps["model"].coef_
    top = {}
    for ci, cls in enumerate(lr.named_steps["model"].classes_):
        order = np.argsort(-coef[ci])
        top[cls] = [{"gene": X.columns[j], "weight": round(float(coef[ci, j]), 3)} for j in order[:6]]
    report["top_genes_per_subtype"] = top
    return lr, report


def train_surrogate_model(metabric: pd.DataFrame, seed: int) -> tuple[object, dict, dict]:
    df = metabric.dropna(subset=["subtype", "er_positive", "her2_positive", "grade"]).copy()
    X = df[SURROGATE_FEATURES].astype(float)
    y = df["subtype"].values
    classes = [c for c in SUBTYPE_ORDER if c in set(y)]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    gbm = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=250, class_weight="balanced", random_state=seed)
    pred = cross_val_predict(gbm, X, y, cv=skf)
    gbm.fit(X, y)
    # Interpretable, exportable variant for the browser engine (multinomial logistic regression).
    Xi = X.fillna(X.median())
    lr = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced"))])
    pred_lr = cross_val_predict(lr, Xi, y, cv=skf)
    lr.fit(Xi, y)
    # Rule based St Gallen style surrogate as a clinical baseline.
    rule = np.where(
        df["her2_positive"] == 1,
        "Her2",
        np.where(df["er_positive"] == 0, "Basal", np.where((df["grade"] == 3) | (df["pr_positive"] == 0), "LumB", "LumA")),
    )
    report = {
        "n": int(len(y)),
        "features": SURROGATE_FEATURES,
        "gradient_boosting_cv": _cv_report(y, pred, classes),
        "logistic_regression_cv": _cv_report(y, pred_lr, classes),
        "st_gallen_rule_baseline": _cv_report(y, rule, classes),
    }
    scaler, model = lr.named_steps["scale"], lr.named_steps["model"]
    spec = {
        "features": SURROGATE_FEATURES,
        "medians": X.median().round(4).to_dict(),
        "mean": scaler.mean_.round(6).tolist(),
        "scale": scaler.scale_.round(6).tolist(),
        "classes": list(model.classes_),
        "coef": model.coef_.round(6).tolist(),
        "intercept": model.intercept_.round(6).tolist(),
        "descriptions": SUBTYPE_DESCRIPTIONS,
    }
    return {"gbm": gbm, "lr": lr}, report, spec


def predict_surrogate(spec: dict, patient: dict) -> dict:
    z = []
    for i, f in enumerate(spec["features"]):
        v = patient.get(f)
        v = spec["medians"][f] if v is None else float(v)
        z.append((v - spec["mean"][i]) / spec["scale"][i])
    logits = np.array(spec["coef"]) @ np.array(z) + np.array(spec["intercept"])
    p = np.exp(logits - logits.max())
    p /= p.sum()
    probs = {c: round(float(pi), 4) for c, pi in zip(spec["classes"], p, strict=False)}
    top = max(probs, key=probs.get)
    return {"predicted": top, "probabilities": probs, "description": spec["descriptions"].get(top, "")}


def train_all(expr: pd.DataFrame, metabric: pd.DataFrame, out_dir: Path, seed: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    expr_model, expr_report = train_expression_model(expr, metabric, seed)
    sur_models, sur_report, sur_spec = train_surrogate_model(metabric, seed)
    joblib.dump({"expression": expr_model, "surrogate": sur_models}, out_dir / "subtype_models.joblib", compress=3)
    (out_dir / "subtype_surrogate.json").write_text(json.dumps(sur_spec))
    return {"expression_pam50": expr_report, "surrogate_pathology": sur_report, "descriptions": SUBTYPE_DESCRIPTIONS}
