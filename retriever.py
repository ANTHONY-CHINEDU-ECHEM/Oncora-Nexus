"""Hybrid retrieval: BM25 and dense candidates, reciprocal rank fusion, then a
learned feature reranker.

Stage 1  BM25 top N and dense top N (N = candidate_pool, default 60).
Stage 2  Reciprocal rank fusion, k = 60 (Cormack et al., SIGIR 2009).
Stage 3  Logistic regression reranker over eleven interpretable features.
         When no trained reranker is available a hand weighted linear
         combination of the same features is used, so the system degrades
         gracefully rather than failing.
Stage 4  Maximal marginal relevance diversification so that the final
         evidence set does not contain five chunks of the same abstract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from oncora.config import settings
from oncora.rag.documents import Passage
from oncora.rag.index import KnowledgeIndex
from oncora.rag.query import ParsedQuery, parse_query
from oncora.rag.text import bigrams, normalise

FEATURE_NAMES = [
    "bm25_norm",
    "dense_cosine",
    "rrf",
    "query_coverage",
    "bigram_overlap",
    "drug_match",
    "context_match",
    "section_prior",
    "is_guideline",
    "is_trial",
    "evidence_strength",
]

DEFAULT_WEIGHTS = np.array([1.6, 1.4, 22.0, 2.2, 1.0, 1.2, 0.8, 0.6, 0.5, 0.0, 0.3])


@dataclass
class Hit:
    passage: Passage
    score: float
    rank: int = 0
    features: dict[str, float] = field(default_factory=dict)

    def to_dict(self, include_text: bool = True) -> dict:
        p = self.passage
        d = {
            "chunk_id": p.chunk_id,
            "doc_id": p.doc_id,
            "source": p.source,
            "title": p.title,
            "url": p.url,
            "section": p.section,
            "evidence_label": p.evidence_label,
            "evidence_rank": p.evidence_rank,
            "topics": p.topics[:6],
            "score": round(self.score, 4),
            "rank": self.rank,
            "features": {k: round(v, 4) for k, v in self.features.items()},
            "meta": p.meta,
            "line_start": p.line_start,
            "line_end": p.line_end,
        }
        if include_text:
            d["text"] = p.text
        return d


class Reranker:
    """Pointwise logistic regression reranker on interpretable features."""

    def __init__(self, model=None) -> None:
        self.model = model

    @classmethod
    def load(cls, path: Path) -> Reranker:
        if path.exists():
            return cls(joblib.load(path))
        return cls(None)

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return X @ DEFAULT_WEIGHTS
        return self.model.decision_function(X)

    @property
    def trained(self) -> bool:
        return self.model is not None


class HybridRetriever:
    def __init__(self, index: KnowledgeIndex, reranker: Reranker | None = None) -> None:
        self.index = index
        self.reranker = reranker or Reranker.load(settings.paths.models / "reranker.joblib")
        self.cfg = settings.retrieval

    # Stage 1 and 2
    def candidates(self, q: ParsedQuery, sources: list[str] | None, pool: int) -> tuple[np.ndarray, dict]:
        mask = self.index.mask_for(sources)
        bm = self.index.bm25.score(q.tokens, mask)
        qv = self.index.encoder.encode([q.normalised])[0]
        dn = self.index.dense @ qv
        if mask is not None:
            dn = np.where(mask, dn, -1.0)
        bm_top = np.argsort(-bm)[:pool]
        bm_top = bm_top[bm[bm_top] > 0]
        dn_top = np.argsort(-dn)[:pool]
        rrf: dict[int, float] = {}
        for rank, i in enumerate(bm_top):
            rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (self.cfg.rrf_k + rank + 1)
        for rank, i in enumerate(dn_top):
            rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (self.cfg.rrf_k + rank + 1)
        cand = np.array(sorted(rrf, key=lambda i: -rrf[i]), dtype=int)
        bm_max = float(bm[bm_top[0]]) if len(bm_top) else 1.0
        return cand, {"bm25": bm, "dense": dn, "rrf": rrf, "bm_max": max(bm_max, 1e-9)}

    def _norm_text(self, i: int) -> str:
        cache = self.__dict__.setdefault("_norm_cache", {})
        if i not in cache:
            cache[i] = normalise(self.index.passages[i].index_text)
        return cache[i]

    def features(self, q: ParsedQuery, cand: np.ndarray, stats: dict) -> np.ndarray:
        qset = set(q.tokens)
        qbi = set(bigrams(q.tokens))
        context_terms = q.subtypes + q.settings + q.populations
        rows = []
        for i in cand:
            p = self.index.passages[i]
            toks = self.index.token_sets[i]
            text_norm = self._norm_text(int(i))
            coverage = len(qset & toks) / max(len(qset), 1)
            bi = 0.0
            if qbi:
                bi = sum(1 for b in qbi if b.replace("_", " ") in text_norm) / len(qbi)
            drug = 0.0
            if q.drugs:
                drug = sum(1 for d in q.drugs if d in text_norm) / len(q.drugs)
            ctx = 0.0
            if context_terms:
                ctx = sum(1 for c in context_terms if c.split()[0] in text_norm) / len(context_terms)
            rows.append(
                [
                    stats["bm25"][i] / stats["bm_max"],
                    max(float(stats["dense"][i]), 0.0),
                    stats["rrf"].get(int(i), 0.0),
                    coverage,
                    bi,
                    drug,
                    ctx,
                    q.section_prior(p.section),
                    1.0 if p.source == "guideline" else 0.0,
                    1.0 if p.source == "trial" else 0.0,
                    (5 - p.evidence_rank) / 4.0,
                ]
            )
        return np.array(rows, dtype=np.float64)

    def _mmr(self, order: list[int], scores: np.ndarray, k: int, lam: float = 0.75, per_doc: int = 2) -> list[int]:
        chosen: list[int] = []
        doc_counts: dict[str, int] = {}
        pool = order[: max(k * 4, 30)]
        if not pool:
            return []
        s = (scores[pool] - scores[pool].min()) / (np.ptp(scores[pool]) + 1e-9)
        rel = dict(zip(pool, s, strict=False))
        while pool and len(chosen) < k:
            best, best_val = None, -1e9
            for i in pool:
                doc = self.index.passages[i].doc_id
                if doc_counts.get(doc, 0) >= per_doc:
                    continue
                sim = max((float(self.index.dense[i] @ self.index.dense[j]) for j in chosen), default=0.0)
                val = lam * rel[i] - (1 - lam) * sim
                if val > best_val:
                    best, best_val = i, val
            if best is None:
                break
            chosen.append(best)
            pool.remove(best)
            doc = self.index.passages[best].doc_id
            doc_counts[doc] = doc_counts.get(doc, 0) + 1
        return chosen

    def search(
        self,
        query: str | ParsedQuery,
        k: int | None = None,
        sources: list[str] | None = None,
        diversify: bool = True,
        rerank: bool = True,
        per_doc: int = 2,
    ) -> list[Hit]:
        q = query if isinstance(query, ParsedQuery) else parse_query(query)
        k = k or self.cfg.final_k
        cand, stats = self.candidates(q, sources, self.cfg.candidate_pool)
        if len(cand) == 0:
            return []
        X = self.features(q, cand, stats)
        if rerank:
            scores_c = self.reranker.score(X)
        else:
            scores_c = np.array([stats["rrf"][int(i)] for i in cand])
        full = np.full(self.index.size, -1e9)
        full[cand] = scores_c
        order = [int(cand[j]) for j in np.argsort(-scores_c)]
        picked = self._mmr(order, full, k, per_doc=per_doc) if diversify else order[:k]
        feat_lookup = {int(c): X[j] for j, c in enumerate(cand)}
        hits = []
        for r, i in enumerate(picked, start=1):
            hits.append(
                Hit(
                    passage=self.index.passages[i],
                    score=float(full[i]),
                    rank=r,
                    features=dict(zip(FEATURE_NAMES, map(float, feat_lookup[i]), strict=False)),
                )
            )
        return hits

    # Variants used by the ablation study
    def search_bm25(self, query: str, k: int, sources: list[str] | None = None) -> list[int]:
        q = parse_query(query)
        s = self.index.bm25.score(q.tokens, self.index.mask_for(sources))
        return [int(i) for i in np.argsort(-s)[:k]]

    def search_dense(self, query: str, k: int, sources: list[str] | None = None) -> list[int]:
        q = parse_query(query)
        v = self.index.encoder.encode([q.normalised])[0]
        s = self.index.dense @ v
        mask = self.index.mask_for(sources)
        if mask is not None:
            s = np.where(mask, s, -1.0)
        return [int(i) for i in np.argsort(-s)[:k]]

    def search_ids(self, query: str, k: int, sources: list[str] | None = None, rerank: bool = True) -> list[int]:
        hits = self.search(query, k=k, sources=sources, diversify=False, rerank=rerank, per_doc=99)
        by = self.index.by_chunk() if not hasattr(self, "_by") else self._by
        self._by = by
        return [by[h.passage.chunk_id] for h in hits]
