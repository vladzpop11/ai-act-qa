"""FastAPI web service for AI Act question answering."""

import argparse
import os
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from ai_act_qa.config import load_config
from ai_act_qa.service import AnswerResponse, AskRequest, Passage, Service

_service: Service | None = None


def get_service() -> Service:
    global _service
    if _service is None:
        _service = Service()
    return _service


def set_service(service: Service) -> None:
    global _service
    _service = service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly initialize retriever on startup
    get_service()
    yield


ServiceDep = Annotated[Service, Depends(get_service)]


def create_app(service: Service | None = None) -> FastAPI:
    if service is not None:
        set_service(service)

    app = FastAPI(
        title="EU AI Act Question Answering API",
        description=(
            "Answers legal questions about the EU AI Act (Regulation (EU) 2024/1689 "
            "consolidated text of 27 July 2026, including the Digital Omnibus 2026/1744) "
            "with open-weight models, citing exact statutory passages."
        ),
        version="0.1.0",
        lifespan=lifespan if service is None else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz", summary="Liveness check", tags=["System"])
    def healthz():
        return {"status": "ok", "service": "ai-act-qa"}

    @app.get("/ready", summary="Readiness check (retriever & LLM)", tags=["System"])
    def ready(svc: ServiceDep):
        llm_ready = False
        try:
            r = httpx.get(f"{svc.llm_url}/health", timeout=3.0)
            llm_ready = r.status_code == 200
        except httpx.HTTPError:
            llm_ready = False

        return {
            "status": "ready" if llm_ready else "degraded",
            "retriever": "ready",
            "llm": "ready" if llm_ready else "unreachable",
            "llm_url": svc.llm_url,
            "model": svc.model_name,
        }

    @app.get("/info", summary="Service metadata and provenance", tags=["System"])
    def info(svc: ServiceDep):
        cfg = svc.cfg
        return {
            "name": "ai-act-qa",
            "version": "0.1.0",
            "document": {
                "celex": cfg["source"]["celex"],
                "title": cfg["source"]["title"],
                "sha256": cfg["source"]["sha256"],
            },
            "model": svc.model_name,
            "prompt": {
                "file": svc.prompt.path,
                "sha256": svc.prompt.sha256,
            },
            "retrieval": {
                "mode": cfg["retrieval"]["mode"],
                "embedding": cfg["retrieval"]["embedding"],
                "reranker": cfg["retrieval"].get("reranker"),
                "k": cfg["retrieval"]["k"],
            },
        }

    @app.get(
        "/passages",
        response_model=list[Passage],
        summary="Search passages only",
        tags=["Q&A"],
    )
    def passages(
        question: str = Query(..., min_length=1, max_length=2000, description="Query text"),
        k: int | None = Query(None, ge=1, le=10, description="Number of passages"),
        svc: ServiceDep = None,
    ):
        hits = svc.retrieve_passages(question, k=k)
        return [
            Passage(
                number=i,
                chunk_id=h.chunk_id,
                label=h.label,
                score=round(h.score, 4),
                text=h.text,
            )
            for i, h in enumerate(hits, 1)
        ]

    @app.post(
        "/ask",
        response_model=AnswerResponse,
        summary="Answer question with citations",
        tags=["Q&A"],
    )
    def ask(request: AskRequest, svc: ServiceDep = None):
        return svc.ask(request.question, k=request.k)

    return app


app = create_app()


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Start the AI Act Q&A API server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--llm-url",
        default=None,
        help="URL of running llama-server (default: from config/env)",
    )
    args = parser.parse_args()

    cfg = load_config()
    if args.llm_url:
        cfg["llm_url"] = args.llm_url
        os.environ["LLM_URL"] = args.llm_url

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
