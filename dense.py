"""Dense semantic encoders.

Two interchangeable backends implement the same interface:

* LSAEncoder (default, fully offline): TF IDF over unigrams and clinical
  bigrams followed by truncated SVD to 256 dimensions. It captures term
  co occurrence (for example, linking "cardiotoxicity" with "ejection
  fraction") without any downloaded weights, so the whole platform runs on an
  air gapped hospital server.
* SentenceTransformerEncoder (optional): any local sentence transformer such
  as BAAI/bge-small-en-v1.5 or ncbi/MedCPT. Enabled with
  ONCORA_EMBEDDINGS=sentence_transformers once the weights are cached.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from oncora.rag.text import tokenize


def _analyzer(text: str) -> list[str]:
    toks = tokenize(text)
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]


class LSAEncoder:
    name = "lsa"

    def __init__(self, dimensions: int = 256, seed: int = 0) -> None:
        self.dimensions = dimensions
        self.seed = seed
        self.vectorizer = TfidfVectorizer(analyzer=_analyzer, min_df=2, max_df=0.5, sublinear_tf=True, max_features=60_000, dtype=np.float32)
        self.svd = TruncatedSVD(n_components=dimensions, random_state=seed, algorithm="randomized", n_iter=7)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        tfidf = self.vectorizer.fit_transform(texts)
        dims = min(self.dimensions, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        self.svd.set_params(n_components=dims)
        emb = self.svd.fit_transform(tfidf)
        self.compact()
        return normalize(emb).astype(np.float32)

    def compact(self) -> None:
        """Drop fitting artefacts that are not needed at query time. The pruned
        term list kept by scikit learn is large and has no runtime use."""
        if hasattr(self.vectorizer, "stop_words_"):
            self.vectorizer.stop_words_ = None
        self.svd.components_ = self.svd.components_.astype(np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        return normalize(self.svd.transform(self.vectorizer.transform(texts))).astype(np.float32)

    @property
    def explained_variance(self) -> float:
        return float(self.svd.explained_variance_ratio_.sum())


class SentenceTransformerEncoder:  # pragma: no cover, requires model weights
    name = "sentence_transformers"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def cosine_top(matrix: np.ndarray, query: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    scores = matrix @ query.ravel()
    if mask is not None:
        scores = np.where(mask, scores, -1.0)
    return scores
