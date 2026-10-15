from oncora.rag.verify import verify_answer

EVIDENCE = [
    "Letrozole reduced the risk of recurrence by 18 percent compared with tamoxifen, hazard ratio 0.82.",
    "Supervised exercise improved fatigue scores in women receiving chemotherapy.",
]


def test_supported_sentence_with_matching_numbers():
    r = verify_answer("BOTTOM LINE: Letrozole reduced recurrence by 18% versus tamoxifen [S1].", EVIDENCE)
    assert r.sentences[0].status == "supported"
    assert r.faithfulness == 1.0


def test_invented_number_is_flagged():
    r = verify_answer("BOTTOM LINE: Letrozole reduced recurrence by 35% versus tamoxifen [S1].", EVIDENCE)
    assert r.sentences[0].status == "unsupported"
    assert "35%" in r.unsupported_numbers


def test_wrong_citation_is_unsupported_and_invalid_citation_reported():
    r = verify_answer("BOTTOM LINE: Exercise improved fatigue during chemotherapy [S1]. Trastuzumab doubles survival [S9].", EVIDENCE)
    assert r.sentences[0].status == "unsupported"
    assert 9 in r.invalid_citations


def test_uncited_claims_and_caveats_are_treated_differently():
    r = verify_answer("KEY EVIDENCE:\n* Exercise improved fatigue in women receiving chemotherapy.\nCAVEATS:\n* Evidence limited to small trials.", EVIDENCE)
    statuses = [s.status for s in r.sentences]
    assert statuses == ["uncited", "meta"]
