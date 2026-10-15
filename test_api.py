"""API integration tests. Skipped automatically when the index has not been built."""

import pytest

from oncora.config import settings

pytestmark = pytest.mark.skipif(not (settings.paths.index / "manifest.json").exists(), reason="index not built")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from oncora.api.main import app

    with TestClient(app) as c:
        yield c


PATIENT = {"age": 63, "menopausal_status": "postmenopausal", "tumour_size_mm": 32, "grade": 3, "positive_nodes": 2,
           "er_status": "positive", "pr_status": "positive", "her2_status": "negative"}


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["index_passages"] > 1000


def test_ask_returns_verified_cited_answer(client):
    r = client.post("/api/ask", json={"question": "What is the recommended whole breast radiotherapy fractionation after breast conserving surgery?", "mode": "extractive"})
    body = r.json()
    assert r.status_code == 200
    assert body["evidence"] and body["verification"]["faithfulness"] >= 0.5
    assert any(e["doc_id"] == "KC-025" for e in body["evidence"])


def test_ask_blocks_out_of_scope(client):
    assert client.post("/api/ask", json={"question": "best pizza in town?"}).json()["mode"] == "blocked"


def test_prognosis(client):
    body = client.post("/api/prognosis", json={"patient": PATIENT}).json()
    assert 0 < body["survival_at"]["60m"] < 1 and body["npi"]["group"] in {"good", "moderate", "poor"}


def test_trials_match(client):
    body = client.post("/api/trials/match", json={"patient": PATIENT, "top_k": 5}).json()
    assert body["trials"] and "criteria" in body["trials"][0]


def test_pathway_and_subtype(client):
    assert client.post("/api/pathway", json={"patient": PATIENT}).json()["recommendations"]
    sub = client.post("/api/subtype", json={"patient": PATIENT}).json()
    assert "surrogate" in sub


def test_validation_error(client):
    assert client.post("/api/prognosis", json={"patient": {"age": 5}}).status_code == 422
