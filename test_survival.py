import numpy as np

from oncora.data.cohorts import kaplan_meier, logrank_test
from oncora.models.survival import cox_predict

SPEC = {
    "features": ["age", "positive_nodes"],
    "feature_labels": {"age": "Age", "positive_nodes": "Nodes"},
    "transforms": {"age": {"kind": "scale", "arg": 0.1}, "positive_nodes": {"kind": "log1p", "arg": None}},
    "mean": [6.0, 0.5],
    "scale": [1.0, 0.5],
    "coef": [0.1, 0.5],
    "baseline_survival": {"time": list(range(0, 121)), "survival": list(np.linspace(1.0, 0.6, 121))},
    "risk_group_cuts": [-0.2, 0.2],
    "cohort_reference": {"age": 60, "positive_nodes": 0},
}


def test_more_nodes_means_lower_survival():
    low = cox_predict(SPEC, {"age": 60, "positive_nodes": 0})
    high = cox_predict(SPEC, {"age": 60, "positive_nodes": 10})
    assert high["survival_at"]["60m"] < low["survival_at"]["60m"]
    assert high["risk_group"] == "high"


def test_missing_inputs_are_imputed_and_flagged():
    res = cox_predict(SPEC, {"age": 60})
    assert any(c["imputed"] for c in res["contributions"])


def test_kaplan_meier_matches_hand_calculation():
    t = np.array([1, 2, 3, 4, 5], dtype=float)
    e = np.array([1, 0, 1, 0, 1])
    km = kaplan_meier(t, e, np.array([0, 1, 3, 5]))
    assert km["survival"][0] == 1.0
    assert abs(km["survival"][1] - 0.8) < 1e-6
    assert abs(km["survival"][2] - 0.8 * (2 / 3)) < 1e-4


def test_logrank_detects_separation():
    rng = np.random.default_rng(0)
    t = np.concatenate([rng.exponential(10, 200), rng.exponential(30, 200)])
    e = np.ones(400, dtype=int)
    g = np.array(["a"] * 200 + ["b"] * 200)
    assert logrank_test(t, e, g)["p_value"] < 1e-6
