"""Tests for the product API endpoints and schema contract."""

import pytest
from fastapi.testclient import TestClient

from ai_act_qa.api import create_app
from ai_act_qa.config import load_config
from ai_act_qa.search import Hit
from ai_act_qa.service import Service


class DummyRetriever:
    """Fast dummy retriever for API contract tests without re-indexing."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def search(self, question: str, k: int | None = None) -> list[Hit]:
        k = k or 5
        return [
            Hit(
                chunk_id="art3.60",
                label="Article 3(60)",
                text=(
                    "Article 3 - Definitions\n(60) 'deep fake' means AI-generated "
                    "image, audio or video."
                ),
                spans=[(0, 50)],
                score=0.95,
            ),
            Hit(
                chunk_id="art50.4",
                label="Article 50(4)",
                text=(
                    "Article 50 - Transparency obligations\n4. Deployers of an AI system "
                    "that generates deep fakes..."
                ),
                spans=[(50, 100)],
                score=0.85,
            ),
        ][:k]


@pytest.fixture(scope="module")
def mock_service():
    cfg = load_config()

    def mock_chat(msgs: list[dict]) -> dict:
        user_content = msgs[1]["content"]
        if "unanswerable" in user_content.lower() or "quantum" in user_content.lower():
            text = "The AI Act passages provided do not answer this question."
        else:
            text = "A deep fake is AI-generated image, audio or video [1]."
        return {
            "text": text,
            "finish_reason": "stop",
            "prompt_tokens": 80,
            "completion_tokens": 18,
            "seconds": 0.05,
        }

    retriever = DummyRetriever(cfg["retrieval"])
    return Service(cfg=cfg, retriever=retriever, chat_fn=mock_chat)


@pytest.fixture(scope="module")
def client(mock_service):
    app = create_app(service=mock_service)
    return TestClient(app)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "ai-act-qa"}


def test_info(client):
    r = client.get("/info")
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "ai-act-qa"
    assert data["model"] == "gemma-4-e2b"
    assert "celex" in data["document"]
    assert "prompt" in data
    assert "retrieval" in data


def test_passages_endpoint(client):
    r = client.get("/passages", params={"question": "What is a deep fake?", "k": 2})
    assert r.status_code == 200
    passages = r.json()
    assert len(passages) == 2
    assert passages[0]["label"] == "Article 3(60)"
    assert passages[0]["number"] == 1
    assert "deep fake" in passages[0]["text"]


def test_ask_valid_answer_with_citations(client):
    r = client.post("/ask", json={"question": "What is a deep fake?"})
    assert r.status_code == 200
    data = r.json()
    assert data["question"] == "What is a deep fake?"
    assert not data["declined"]
    assert len(data["citations"]) == 1
    assert data["citations"][0]["number"] == 1
    assert data["citations"][0]["label"] == "Article 3(60)"
    assert data["citations"][0]["chapter"] == "Chapter I - General provisions"
    assert len(data["passages"]) == 2
    assert data["latency"]["total_ms"] >= 0
    assert data["provenance"]["model"] == "gemma-4-e2b"


def test_ask_decline_response(client):
    r = client.post("/ask", json={"question": "Does the Act regulate quantum teleportation?"})
    assert r.status_code == 200
    data = r.json()
    assert data["declined"] is True
    assert data["exact_decline"] is True
    assert len(data["citations"]) == 0


def test_ask_input_validation(client):
    # Empty question
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422

    # Oversized question (> 2000 chars)
    r = client.post("/ask", json={"question": "x" * 2001})
    assert r.status_code == 422


def test_openapi_schema(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert schema["info"]["title"] == "EU AI Act Question Answering API"
    paths = schema["paths"]
    assert "/healthz" in paths
    assert "/info" in paths
    assert "/ready" in paths
    assert "/passages" in paths
    assert "/ask" in paths
