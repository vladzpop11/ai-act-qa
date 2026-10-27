"""Load and concurrency tests: evaluates API throughput, parallelism, and queue stability."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from ai_act_qa.api import create_app
from ai_act_qa.config import load_config
from ai_act_qa.search import Hit
from ai_act_qa.service import Service


@pytest.fixture(scope="module")
def concurrent_client():
    cfg = load_config()

    def mock_chat(msgs: list[dict]) -> dict:
        q = msgs[1]["content"]
        # Extract query id from message
        return {
            "text": f"Response to {q[-10:]} [1].",
            "finish_reason": "stop",
            "prompt_tokens": 50,
            "completion_tokens": 12,
            "seconds": 0.01,
        }

    class ConcurrentRetriever:
        def search(self, q: str, k: int | None = None) -> list[Hit]:
            return [Hit("art5.1.a", "Article 5(1)(a)", "Prohibited text", [(0, 10)], 0.9)]

    svc = Service(cfg=cfg, retriever=ConcurrentRetriever(), chat_fn=mock_chat)
    return TestClient(create_app(service=svc))


def test_concurrent_requests_remain_isolated_and_stable(concurrent_client):
    """Fire 20 concurrent requests across multiple threads to ensure no thread races or crashes."""
    concurrency = 20
    queries = [f"What is prohibited under section {i:03d}?" for i in range(concurrency)]

    def send(q: str):
        return concurrent_client.post("/ask", json={"question": q})

    with ThreadPoolExecutor(max_workers=5) as pool:
        responses = list(pool.map(send, queries))

    assert len(responses) == concurrency
    for q, res in zip(queries, responses, strict=True):
        assert res.status_code == 200
        data = res.json()
        assert data["question"] == q
        assert len(data["citations"]) == 1
        assert not data["declined"]


def test_passages_endpoint_concurrency(concurrent_client):
    """Test concurrent passage search."""
    queries = [f"Article search {i}" for i in range(15)]

    def send(q: str):
        return concurrent_client.get("/passages", params={"question": q, "k": 1})

    with ThreadPoolExecutor(max_workers=5) as pool:
        responses = list(pool.map(send, queries))

    for res in responses:
        assert res.status_code == 200
        assert len(res.json()) == 1
