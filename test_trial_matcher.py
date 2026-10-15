from oncora.data.trials import TrialRecord
from oncora.models.patient import PatientProfile
from oncora.models.trial_matcher import Criterion, TrialMatcher, evaluate, parse_eligibility

TRIAL = TrialRecord(
    nct_id="NCT99999999",
    eligibility=[
        "Inclusion Criteria:",
        "  Postmenopausal women aged 18 years or older",
        "  Histologically confirmed metastatic breast cancer",
        "  Estrogen receptor positive disease",
        "  HER2-negative disease",
        "  ECOG performance status 0-1",
        "Exclusion Criteria:",
        "  Prior treatment with a CDK4/6 inhibitor",
        "  Active brain metastases",
        "  Pregnant or breastfeeding",
    ],
    intervention=["INTERVENTION 1: ", "  Letrozole plus palbociclib"],
    agents=["letrozole", "palbociclib"],
    title="Letrozole, Palbociclib in advanced or metastatic breast cancer",
    setting="advanced or metastatic",
)


def test_parser_extracts_typed_constraints():
    parsed = parse_eligibility(TRIAL)
    types = {(c.ctype, c.kind) for c in parsed.criteria}
    assert ("age", "inclusion") in types
    assert ("menopause", "inclusion") in types
    assert ("hormone", "inclusion") in types
    assert ("her2", "inclusion") in types
    assert ("ecog", "inclusion") in types
    assert ("prior_therapy", "exclusion") in types
    assert ("brain_mets", "exclusion") in types


def test_eligible_patient_is_likely_eligible():
    m = TrialMatcher([TRIAL])
    p = PatientProfile(age=60, setting="metastatic", menopausal_status="postmenopausal", ecog=1,
                       er_status="positive", pr_status="positive", her2_status="negative")
    res = m.match(p)
    assert res["trials"][0]["tier"] == "likely eligible"


def test_prior_cdk46_excludes():
    m = TrialMatcher([TRIAL])
    p = PatientProfile(age=60, setting="metastatic", menopausal_status="postmenopausal", ecog=1,
                       er_status="positive", pr_status="positive", her2_status="negative", prior_therapies=["palbociclib"])
    res = m.match(p, include_excluded=True)
    assert res["trials"][0]["tier"] == "excluded"


def test_missing_fields_need_review():
    m = TrialMatcher([TRIAL])
    res = m.match(PatientProfile(setting="metastatic", er_status="positive", her2_status="negative"))
    assert res["trials"][0]["tier"] == "needs review"


def test_her2_low_counts_as_her2_negative():
    c = Criterion(0, "HER2-negative", "inclusion", "her2", {"status": "negative"})
    p = PatientProfile(her2_status="low")
    assert evaluate(c, p, p.derived()).verdict == "met"
