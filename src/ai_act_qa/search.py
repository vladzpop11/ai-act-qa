"""Retrieval: dense, keyword (BM25) or hybrid search in Qdrant, optionally reranked.

- dense: a sentence embedding (fastembed, ONNX on the CPU) and cosine similarity.
- bm25: Qdrant's sparse BM25 vectors, with IDF computed by Qdrant over the collection.
- hybrid: both, fused by reciprocal rank (RRF) in one Qdrant query.
- rerank: a cross-encoder rescores the first `candidates` results and the best `k` are kept.

The same code runs against an in-memory Qdrant (evaluation, tests, CI) and the Qdrant server in
docker compose (the API): only `location` differs.
"""

import hashlib
from dataclasses import dataclass

import numpy as np
from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models

from ai_act_qa.chunk import Chunk
from ai_act_qa.config import project_path

COLLECTION = "ai_act"
BM25 = "Qdrant/bm25"


@dataclass
class Hit:
    chunk_id: str
    label: str
    text: str
    spans: list[tuple[int, int]]
    score: float


def cached_embeddings(model: TextEmbedding, name: str, texts: list[str]) -> list[np.ndarray]:
    """Dense embeddings, cached on disk by model and text (embedding the Act takes minutes on
    a 2-core CPU, and every chunking and retrieval setting reuses most chunks)."""
    path = project_path(f"data/cache/{name.replace('/', '__')}.npz")
    cache: dict[str, np.ndarray] = {}
    if path.exists():
        stored = np.load(path)
        cache = dict(zip(stored["keys"].tolist(), stored["vectors"], strict=True))
    keys = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
    missing = [(k, t) for k, t in dict(zip(keys, texts, strict=True)).items() if k not in cache]
    if missing:
        vectors = model.embed([t for _, t in missing], batch_size=32)
        cache.update({k: v for (k, _), v in zip(missing, vectors, strict=True)})
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, keys=np.array(list(cache)), vectors=np.stack(list(cache.values())))
    return [cache[k] for k in keys]


class Retriever:
    def __init__(self, cfg: dict, location: str = ":memory:") -> None:
        """`cfg`: the `retrieval` section of the config (models, mode, k, rerank)."""
        self.cfg = cfg
        self.client = QdrantClient(location=location) if location == ":memory:" else QdrantClient(
            url=location)  # fmt: skip
        self.dense = TextEmbedding(cfg["embedding"])
        self.sparse = SparseTextEmbedding(BM25)
        self.reranker = TextCrossEncoder(cfg["reranker"]) if cfg.get("rerank") else None

    def index(self, chunks: list[Chunk]) -> None:
        texts = [c.text for c in chunks]
        dense = cached_embeddings(self.dense, self.cfg["embedding"], texts)
        sparse = list(self.sparse.embed(texts, batch_size=32))
        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)
        self.client.create_collection(
            COLLECTION,
            vectors_config={"dense": models.VectorParams(size=len(dense[0]),
                                                         distance=models.Distance.COSINE)},
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )  # fmt: skip
        points = [
            models.PointStruct(
                id=i,
                vector={
                    "dense": d.tolist(),
                    "bm25": models.SparseVector(
                        indices=s.indices.tolist(), values=s.values.tolist()
                    ),
                },
                payload={
                    "chunk_id": c.id,
                    "label": c.label,
                    "text": c.text,
                    "spans": [list(x) for x in c.spans],
                },
            )  # fmt: skip
            for i, (c, d, s) in enumerate(zip(chunks, dense, sparse, strict=True))
        ]
        self.client.upload_points(COLLECTION, points, batch_size=256, wait=True)

    def search(self, question: str, k: int | None = None) -> list[Hit]:
        k = k or self.cfg["k"]
        mode = self.cfg["mode"]
        limit = self.cfg["candidates"] if self.reranker else k
        query = self.cfg.get("query_prefix", "") + question
        if mode == "dense":
            result = self.client.query_points(COLLECTION, query=self.dense_vector(query),
                                              using="dense", limit=limit)  # fmt: skip
        elif mode == "bm25":
            result = self.client.query_points(COLLECTION, query=self.sparse_vector(question),
                                              using="bm25", limit=limit)  # fmt: skip
        elif mode == "hybrid":
            result = self.client.query_points(
                COLLECTION,
                prefetch=[
                    models.Prefetch(query=self.dense_vector(query), using="dense",
                                    limit=self.cfg["candidates"]),
                    models.Prefetch(query=self.sparse_vector(question), using="bm25",
                                    limit=self.cfg["candidates"]),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
            )  # fmt: skip
        else:
            raise ValueError(f"unknown retrieval mode {mode!r}")
        hits = [
            Hit(p.payload["chunk_id"], p.payload["label"], p.payload["text"],
                [tuple(x) for x in p.payload["spans"]], p.score)
            for p in result.points
        ]  # fmt: skip
        if self.reranker and hits:
            scores = list(self.reranker.rerank(question, [h.text for h in hits]))
            for hit, score in zip(hits, scores, strict=True):
                hit.score = float(score)
            hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def dense_vector(self, text: str) -> list[float]:
        return next(iter(self.dense.query_embed(text))).tolist()

    def sparse_vector(self, text: str) -> models.SparseVector:
        s = next(iter(self.sparse.query_embed(text)))
        return models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
