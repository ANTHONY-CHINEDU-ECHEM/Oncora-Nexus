"""Okapi BM25 over a sparse matrix.

Term saturation depends only on term frequency and document length, so the
full BM25 weight of every (document, term) pair is computed once at index time
and stored in a CSR matrix. Scoring a query is then a column slice and a row
sum, which keeps latency in single milliseconds for tens of thousands of
passages.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import sparse


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.weights: sparse.csc_matrix | None = None
        self.n_docs = 0
        self.avgdl = 0.0

    def fit(self, tokenised_docs: list[list[str]]) -> BM25Index:
        rows, cols, vals = [], [], []
        lengths = np.zeros(len(tokenised_docs), dtype=np.float64)
        for i, toks in enumerate(tokenised_docs):
            lengths[i] = len(toks)
            for term, tf in Counter(toks).items():
                j = self.vocab.setdefault(term, len(self.vocab))
                rows.append(i)
                cols.append(j)
                vals.append(tf)
        self.n_docs = len(tokenised_docs)
        self.avgdl = float(lengths.mean()) if self.n_docs else 0.0
        tf = sparse.csr_matrix((np.array(vals, dtype=np.float64), (rows, cols)), shape=(self.n_docs, len(self.vocab)))
        df = np.bincount(tf.indices, minlength=len(self.vocab)).astype(np.float64)
        self.idf = np.log1p((self.n_docs - df + 0.5) / (df + 0.5))
        norm = self.k1 * (1 - self.b + self.b * lengths / max(self.avgdl, 1e-9))
        tf = tf.tocoo()
        w = tf.data * (self.k1 + 1) / (tf.data + norm[tf.row]) * self.idf[tf.col]
        self.weights = sparse.csc_matrix((w, (tf.row, tf.col)), shape=tf.shape)
        return self

    def query_ids(self, tokens: list[str]) -> list[int]:
        return [self.vocab[t] for t in tokens if t in self.vocab]

    def score(self, tokens: list[str], mask: np.ndarray | None = None) -> np.ndarray:
        ids = self.query_ids(tokens)
        if not ids:
            return np.zeros(self.n_docs)
        counts = Counter(ids)
        cols = list(counts)
        q = np.array([counts[c] for c in cols], dtype=np.float64)
        scores = np.asarray(self.weights[:, cols] @ q).ravel()
        if mask is not None:
            scores = np.where(mask, scores, 0.0)
        return scores

    def state(self) -> dict:
        w = self.weights.tocsc()
        return {
            "k1": self.k1,
            "b": self.b,
            "vocab": self.vocab,
            "idf": self.idf,
            "data": w.data,
            "indices": w.indices,
            "indptr": w.indptr,
            "shape": w.shape,
            "n_docs": self.n_docs,
            "avgdl": self.avgdl,
        }

    @classmethod
    def from_state(cls, st: dict) -> BM25Index:
        obj = cls(st["k1"], st["b"])
        obj.vocab = st["vocab"]
        obj.idf = st["idf"]
        obj.weights = sparse.csc_matrix((st["data"], st["indices"], st["indptr"]), shape=tuple(st["shape"]))
        obj.n_docs = st["n_docs"]
        obj.avgdl = st["avgdl"]
        return obj
