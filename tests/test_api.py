"""
Tests des endpoints FastAPI.

Lancer avec :
    pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app.api.main import app
    return TestClient(app)


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "service" in data
        assert data["service"] == "french-labor-law-rag"
        assert "qdrant_connected" in data


class TestAskEndpoint:

    def test_ask_requires_question(self, client):
        response = client.post("/ask", json={})
        assert response.status_code == 422

    def test_ask_rejects_empty_question(self, client):
        response = client.post("/ask", json={"question": ""})
        assert response.status_code == 422

    def test_ask_rejects_short_question(self, client):
        response = client.post("/ask", json={"question": "ab"})
        assert response.status_code == 422

    def test_ask_response_structure(self, client):
        """Nécessite Qdrant et LM Studio actifs."""
        response = client.post(
            "/ask",
            json={"question": "Quelle est la durée légale du travail en France ?"},
        )
        if response.status_code == 200:
            data = response.json()
            assert "question" in data
            assert "answer" in data
            assert "sources" in data
            assert "intent" in data
            assert "latency_ms" in data
