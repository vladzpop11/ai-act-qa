"""Product QA service: coordinates retrieval, LLM generation, citation resolution and provenance.

Exposes the core application logic used by the API and product tests.
"""

import os
import time
from collections.abc import Callable

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, Field

from ai_act_qa.answer import Prompt, load_prompt, messages, read_answer
from ai_act_qa.chunk import build
from ai_act_qa.config import load_config
from ai_act_qa.llm import chat
from ai_act_qa.parse import Unit, load_units
from ai_act_qa.search import Hit, Retriever


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=2000, description="Question about the EU AI Act"
    )
    k: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Optional override for the number of passages to retrieve",
    )


class Citation(BaseModel):
    number: int = Field(..., description="Passage number [1..k] cited by the model")
    chunk_id: str
    label: str = Field(..., description="Legal citation label, e.g. Article 5(1)(a)")
    chapter: str = Field(..., description="Chapter title")
    text: str = Field(..., description="Full text of the cited passage")


class Passage(BaseModel):
    number: int
    chunk_id: str
    label: str
    score: float
    text: str


class LatencyBreakdown(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class Provenance(BaseModel):
    model: str
    prompt_file: str
    prompt_sha256: str
    document_sha256: str
    act_title: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    declined: bool
    exact_decline: bool
    citations: list[Citation]
    invalid_citations: list[int]
    passages: list[Passage]
    latency: LatencyBreakdown
    provenance: Provenance


class Service:
    """The unified QA service."""

    def __init__(
        self,
        cfg: dict | None = None,
        llm_url: str | None = None,
        retriever: Retriever | None = None,
        chat_fn: Callable | None = None,
    ) -> None:
        self.cfg = cfg or load_config()
        self.llm_url = llm_url or os.environ.get("LLM_URL", "http://127.0.0.1:8090")
        self.canonical, unit_list = load_units()
        self.units: dict[str, Unit] = {u.id: u for u in unit_list}
        self.prompt: Prompt = load_prompt(self.cfg["answers"]["prompt"])
        self.model_name: str = self.cfg.get("selection", {}).get("selected") or "gemma-4-e2b"
        self.decline: str = self.cfg["answers"]["decline"]

        if retriever:
            self.retriever = retriever
        else:
            self.retriever = Retriever(self.cfg["retrieval"])
            chunks = build(self.canonical, unit_list, self.cfg["chunking"])
            self.retriever.index(chunks)

        self._chat_fn = chat_fn

    def chat_call(self, msgs: list[dict]) -> dict:
        if self._chat_fn:
            return self._chat_fn(msgs)
        return chat(
            self.llm_url,
            msgs,
            max_tokens=self.cfg["answers"]["max_tokens"],
            temperature=self.cfg["answers"]["temperature"],
            seed=self.cfg["answers"]["seed"],
            thinking=self.cfg["answers"].get("thinking", False),
        )

    def retrieve_passages(self, question: str, k: int | None = None) -> list[Hit]:
        """Search relevant passages for a question."""
        return self.retriever.search(question, k=k)

    def ask(self, question: str, k: int | None = None) -> AnswerResponse:
        """End-to-end question answering pipeline."""
        t_start = time.perf_counter()
        hits = self.retrieve_passages(question, k=k)
        t_retrieved = time.perf_counter()

        sys_msg, user_msg = messages(self.prompt, question, hits, self.decline)

        try:
            chat_result = self.chat_call([sys_msg, user_msg])
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            msg = f"LLM backend at {self.llm_url} is unreachable. Ensure llama-server is running."
            raise HTTPException(status_code=503, detail=msg) from exc
        except httpx.HTTPStatusError as exc:
            msg = f"LLM backend returned error: {exc.response.status_code} {exc.response.text}"
            raise HTTPException(status_code=502, detail=msg) from exc

        t_generated = time.perf_counter()

        answer_text = chat_result.get("text", "")
        parsed = read_answer(answer_text, len(hits), self.decline)

        citations = []
        for n in parsed["citations"]:
            hit = hits[n - 1]
            unit = self.units.get(hit.chunk_id)
            chapter = unit.chapter if unit else ""
            citations.append(
                Citation(
                    number=n,
                    chunk_id=hit.chunk_id,
                    label=hit.label,
                    chapter=chapter,
                    text=hit.text,
                )
            )

        passages = [
            Passage(
                number=i,
                chunk_id=h.chunk_id,
                label=h.label,
                score=round(h.score, 4),
                text=h.text,
            )
            for i, h in enumerate(hits, 1)
        ]

        retrieval_ms = (t_retrieved - t_start) * 1000
        generation_ms = (t_generated - t_retrieved) * 1000
        total_ms = (t_generated - t_start) * 1000

        provenance = Provenance(
            model=self.model_name,
            prompt_file=self.prompt.path,
            prompt_sha256=self.prompt.sha256,
            document_sha256=self.cfg["source"]["sha256"],
            act_title=self.cfg["source"]["title"],
        )

        return AnswerResponse(
            question=question,
            answer=answer_text,
            declined=parsed["declined"],
            exact_decline=parsed["exact_decline"],
            citations=citations,
            invalid_citations=parsed["invalid"],
            passages=passages,
            latency=LatencyBreakdown(
                retrieval_ms=round(retrieval_ms, 2),
                generation_ms=round(generation_ms, 2),
                total_ms=round(total_ms, 2),
            ),
            provenance=provenance,
        )
