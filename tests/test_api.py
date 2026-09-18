from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from financial_rag_agent.api.routers.ingestion import get_ingest_service
from financial_rag_agent.api.routers.retrieval import get_query_service
from financial_rag_agent.main import app
from financial_rag_agent.retrieval.vector_retriever import RetrievedChunk


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ingest_rejects_malformed_cik_before_calling_service(client):
    calls = []
    app.dependency_overrides[get_ingest_service] = lambda: calls.append

    resp = client.post("/api/v1/filings/ingest", json={"cik": "not-a-cik"})

    assert resp.status_code == 400
    assert "Invalid CIK" in resp.json()["detail"]
    assert calls == []  # never reached the (fake) ingestion service


def test_ingest_maps_value_error_from_service_to_400(client):
    def fake_ingest(cik: str):
        raise ValueError(f"No 10-K filing found for CIK {cik}")

    app.dependency_overrides[get_ingest_service] = lambda: fake_ingest

    resp = client.post("/api/v1/filings/ingest", json={"cik": "1045810"})

    assert resp.status_code == 400
    assert "No 10-K filing found" in resp.json()["detail"]


def test_query_uses_injected_service_not_real_retrieval(client):
    fake_chunk = RetrievedChunk(
        chunk_id=uuid4(),
        score=0.42,
        text="fake text",
        item_label="Item 1",
        item_heading="Item 1. Business",
        filing_accession_number="0000000000-00-000000",
        modality="text",
        table_data=None,
        citation_sentences=[],
    )
    app.dependency_overrides[get_query_service] = lambda: (
        lambda q, k, filing_id=None, modality=None: [fake_chunk]
    )

    resp = client.get("/api/v1/query", params={"q": "anything", "k": 1})

    assert resp.status_code == 200
    body = resp.json()
    assert body["results"][0]["text"] == "fake text"
    assert body["results"][0]["score"] == 0.42


def test_runtime_error_maps_to_500_with_detail(client):
    def fake_query(q, k, filing_id=None, modality=None):
        raise RuntimeError("configured embedding_dimension=768 but model returned a 384-dim vector")

    app.dependency_overrides[get_query_service] = lambda: fake_query

    resp = client.get("/api/v1/query", params={"q": "anything", "k": 1})

    assert resp.status_code == 500
    assert "embedding_dimension" in resp.json()["detail"]
