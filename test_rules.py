from oncora.models.patient import PatientProfile, evaluate_pathway


def ids(**kw):
    return {r["id"] for r in evaluate_pathway(PatientProfile(**kw))}


def test_postmenopausal_hr_positive_high_risk():
    fired = ids(age=63, menopausal_status="postmenopausal", tumour_size_mm=32, grade=3, positive_nodes=2,
                er_status="positive", pr_status="positive", her2_status="negative")
    assert {"E03", "E05", "E18", "T01"} <= fired
    assert "E01" not in fired and "E11" not in fired


def test_her2_positive_residual_disease_triggers_tdm1_and_cardiac_monitoring():
    fired = ids(age=52, menopausal_status="postmenopausal", tumour_size_mm=35, grade=3, positive_nodes=1,
                er_status="negative", pr_status="negative", her2_status="positive", residual_disease=True, lvef=45)
    assert {"E11", "E12", "E13", "E14"} <= fired


def test_triple_negative_young_patient():
    fired = ids(age=38, menopausal_status="premenopausal", tumour_size_mm=28, grade=3, positive_nodes=0,
                er_status="negative", pr_status="negative", her2_status="negative")
    assert {"E15", "G01", "S01"} <= fired


def test_metastatic_after_cdk46_with_pik3ca():
    fired = ids(age=58, setting="metastatic", menopausal_status="postmenopausal", er_status="positive",
                pr_status="positive", her2_status="low", prior_therapies=["palbociclib", "letrozole"], pik3ca_mutation=True)
    assert {"M03", "M04", "M09", "M13"} <= fired
    assert "M01" not in fired


def test_derived_fields():
    d = PatientProfile(er_status="low", pr_status="negative", her2_status="negative", positive_nodes=4).derived()
    assert d["hr_positive"] and d["er_low"] and d["high_risk_nodes"] and not d["triple_negative"]


def test_npi_calculation():
    npi = PatientProfile(tumour_size_mm=25, grade=3, positive_nodes=2).npi()
    assert npi == {"value": 5.5, "group": "poor"}
