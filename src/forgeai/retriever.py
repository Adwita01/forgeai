"""Hybrid retriever: BM25 (lexical) + dense vectors, fused by a tunable alpha.

Dense vectors come from the configured embedding model (e.g. Ollama nomic-embed-text).
If it is unreachable, a TF-IDF cosine is used instead so the factory never stops for lack
of an embedding model - the fallback is recorded in `self.dense_backend`.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from .llm import LLM
from .manifest import STOPWORDS
from .parsing import Chunk, LoadedDoc, chunk_docs
from .schemas import RetrievalConfig, RetrievedChunk

_TOK = re.compile(r"[a-z0-9][a-z0-9\-\.]*")


def tokenize(text: str) -> list[str]:
    toks = []
    for t in _TOK.findall(text.lower()):
        t = t.strip(".-")
        if not t or t in STOPWORDS:
            continue
        toks.append(t)
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            toks.append(t[:-1])  # cheap plural folding
    return toks


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf))
        for i, tf in enumerate(self.tf):
            s = 0.0
            for q in query:
                f = tf.get(q)
                if not f:
                    continue
                s += self.idf.get(q, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1)))
            out[i] = s
        return out


class TfidfDense:
    def __init__(self, docs: list[list[str]]):
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        self.vocab = {t: i for i, t in enumerate(df)}
        n = len(docs)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in self.vocab])
        self.mat = np.vstack([self._vec(d) for d in docs]) if docs else np.zeros((0, len(self.vocab)))

    def _vec(self, toks: list[str]) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for t, c in Counter(toks).items():
            j = self.vocab.get(t)
            if j is not None:
                v[j] = (1 + math.log(c)) * self.idf[j]
        nrm = np.linalg.norm(v)
        return v / nrm if nrm else v

    def scores(self, query: list[str]) -> np.ndarray:
        return self.mat @ self._vec(query) if len(self.mat) else np.zeros(0)


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


class HybridRetriever:
    def __init__(self, docs: list[LoadedDoc], cfg: RetrievalConfig, llm: Optional[LLM] = None, cache_dir: Optional[Path] = None):
        self.cfg = cfg
        self.llm = llm
        self.chunks: list[Chunk] = chunk_docs(docs, cfg.chunk_size, cfg.chunk_overlap)
        self.toks = [tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25(self.toks)
        self.cache_dir = cache_dir
        self.dense_backend = "tfidf"
        self._emb: Optional[np.ndarray] = None
        self._tfidf: Optional[TfidfDense] = None
        self._qcache: dict[str, np.ndarray] = {}
        self._init_dense()

    # ------------------------------------------------------------------ dense
    def _init_dense(self) -> None:
        if self.llm is not None and self.llm.s.embed_model and self.chunks:
            texts = [c.text for c in self.chunks]
            key = hashlib.sha256(("\x1e".join(texts) + self.llm.s.embed_model).encode()).hexdigest()[:20]
            path = (self.cache_dir / f"emb_{key}.npy") if self.cache_dir else None
            if path is not None and path.exists():
                self._emb = np.load(path)
            else:
                vecs = self.llm.embed(texts)
                if vecs is not None and len(vecs) == len(texts):
                    self._emb = np.array(vecs, dtype=np.float32)
                    if path is not None:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        np.save(path, self._emb)
            if self._emb is not None:
                nrm = np.linalg.norm(self._emb, axis=1, keepdims=True)
                self._emb = self._emb / np.where(nrm == 0, 1, nrm)
                self.dense_backend = f"embed:{self.llm.s.embed_model}"
                return
        self._tfidf = TfidfDense(self.toks)

    def _dense_scores(self, query: str) -> np.ndarray:
        if self._emb is not None and self.llm is not None:
            q = self._qcache.get(query)
            if q is None:
                v = self.llm.embed([query])
                if v is None:
                    if self._tfidf is None:
                        self._tfidf = TfidfDense(self.toks)
                    return self._tfidf.scores(tokenize(query))
                q = np.array(v[0], dtype=np.float32)
                q = q / (np.linalg.norm(q) or 1)
                self._qcache[query] = q
            return self._emb @ q
        assert self._tfidf is not None
        return self._tfidf.scores(tokenize(query))

    # ------------------------------------------------------------------ search
    def search(self, query: str, top_k: Optional[int] = None, alpha: Optional[float] = None) -> list[RetrievedChunk]:
        if not self.chunks:
            return []
        k = top_k or self.cfg.top_k
        a = self.cfg.hybrid_alpha if alpha is None else alpha
        q = tokenize(query)
        bm = _minmax(self.bm25.scores(q))
        de = _minmax(self._dense_scores(query))
        fused = a * de + (1 - a) * bm
        order = np.argsort(-fused)
        out: list[RetrievedChunk] = []
        for i in order[: max(k * 3, k)]:
            if fused[i] < self.cfg.min_score and out:
                break
            c = self.chunks[i]
            out.append(RetrievedChunk(chunk_id=c.chunk_id, filename=c.filename, locator=c.locator, text=c.text, score=round(float(fused[i]), 4)))
            if len(out) >= k:
                break
        return out

    def raw_bm25_max(self, query: str) -> float:
        s = self.bm25.scores(tokenize(query))
        return float(s.max()) if s.size else 0.0

    def stats(self) -> dict:
        return {"n_chunks": len(self.chunks), "dense_backend": self.dense_backend, **json.loads(self.cfg.model_dump_json())}
