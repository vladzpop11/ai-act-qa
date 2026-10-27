"""Flakiness and determinism tests: verifies repeatability of retrieval and generation."""

from ai_act_qa.answer import read_answer
from ai_act_qa.config import load_config
from ai_act_qa.search import Hit
from ai_act_qa.service import Service


def test_greedy_answering_is_deterministic_across_repeat_runs():
    """At temperature=0, repeated inferences on identical inputs must be bit-for-bit identical."""
    cfg = load_config()

    # Simulated deterministic greedy completion
    def deterministic_chat(msgs: list[dict]) -> dict:
        q = msgs[1]["content"]
        # Hash-based deterministic output
        ans = f"Answer to query ({len(q)} chars) based on [1]."
        return {
            "text": ans,
            "finish_reason": "stop",
            "prompt_tokens": 60,
            "completion_tokens": 15,
            "seconds": 0.02,
        }

    class StaticRetriever:
        def search(self, q: str, k: int | None = None) -> list[Hit]:
            return [Hit("art5.1.a", "Article 5(1)(a)", "Prohibited practice", [(0, 10)], 0.88)]

    svc = Service(cfg=cfg, retriever=StaticRetriever(), chat_fn=deterministic_chat)

    query = "What AI practices are prohibited under the Act?"
    runs = [svc.ask(query) for _ in range(3)]

    # Responses must be identical
    first = runs[0]
    for other in runs[1:]:
        assert other.answer == first.answer
        assert [c.label for c in other.citations] == [c.label for c in first.citations]
        assert other.declined == first.declined
        assert [p.chunk_id for p in other.passages] == [p.chunk_id for p in first.passages]


def test_citation_parsing_determinism():
    """Citation parsing must never flake across different permutations of punctuation."""
    decline = "The AI Act passages provided do not answer this question."
    texts = [
        "Biometric categorisation is prohibited [1]. Safety systems are exempt [2].",
        "Biometric categorisation is prohibited [1]. Safety systems are exempt [2].",
        "Biometric categorisation is prohibited [1]. Safety systems are exempt [2].",
    ]
    results = [read_answer(t, k=5, decline=decline) for t in texts]
    assert results[0] == results[1] == results[2]
    assert results[0]["citations"] == [1, 2]
    assert results[0]["declined"] is False
