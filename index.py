"""Build, persist and load the hybrid knowledge index."""

from __future__ import annotations

import gzip
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from oncora.config import settings
from oncora.data.pubmed_rct import load_breast_abstracts
from oncora.data.trials import load_trials
from oncora.rag.bm25 import BM25Index
from oncora.rag.dense import LSAEncoder
from oncora.rag.documents import Passage, chunk_cards, chunk_pubmed, chunk_trial
from oncora.rag.text import tokenize

log = logging.getLogger(__name__)


def collect_passages(raw_dir: Path, processed_dir: Path, cards_path: Path) -> list[Passage]:
    cache = processed_dir / "pubmed_breast.json"
    if cache.exists():
        abstracts = json.loads(cache.read_text())
    else:
        abstracts = load_breast_abstracts(raw_dir)
        cache.write_text(json.dumps(abstracts))
    extra = processed_dir / "pubmed_eutils.json"
    if extra.exists():
        seen = {a["pmid"] for a in abstracts}
        added = [a for a in json.loads(extra.read_text()) if a["pmid"] not in seen and a.get("relevance", 1) >= 0.25]
        log.info("Adding %s abstracts fetched through the NCBI Entrez utilities", len(added))
        abstracts = abstracts + added
    passages: list[Passage] = []
    passages += chunk_cards(cards_path)
    for rec in abstracts:
        passages += chunk_pubmed(rec, target_words=settings.retrieval.chunk_target_tokens)
    trials = load_trials(raw_dir)
    (processed_dir / "trials.json").write_text(json.dumps([t.to_dict() for t in trials]))
    for t in trials:
        passages += chunk_trial(t)
    return passages


@dataclass
class KnowledgeIndex:
    passages: list[Passage]
    bm25: BM25Index
    dense: np.ndarray
    encoder: LSAEncoder
    token_sets: list[frozenset[str]]
    source_array: np.ndarray
    manifest: dict

    @property
    def size(self) -> int:
        return len(self.passages)

    def mask_for(self, sources: list[str] | None) -> np.ndarray | None:
        if not sources:
            return None
        return np.isin(self.source_array, sources)

    def by_chunk(self) -> dict[str, int]:
        return {p.chunk_id: i for i, p in enumerate(self.passages)}


def build_index(out_dir: Path | None = None) -> KnowledgeIndex:
    paths = settings.paths.ensure()
    out_dir = out_dir or paths.index
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    passages = collect_passages(paths.raw, paths.processed, paths.knowledge_cards)
    log.info("Collected %s passages in %.1fs", len(passages), time.time() - t0)
    tokens = [tokenize(p.index_text) for p in passages]
    bm25 = BM25Index(settings.retrieval.bm25_k1, settings.retrieval.bm25_b).fit(tokens)
    encoder = LSAEncoder(settings.retrieval.dense_dimensions, seed=settings.random_seed)
    dense = encoder.fit_transform([p.index_text for p in passages])
    by_source = Counter(p.source for p in passages)
    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "passages": len(passages),
        "documents": len({p.doc_id for p in passages}),
        "by_source": dict(by_source),
        "documents_by_source": dict(Counter(p.source for p in {p.doc_id: p for p in passages}.values())),
        "vocabulary": len(bm25.vocab),
        "dense_backend": encoder.name,
        "dense_dimensions": int(dense.shape[1]),
        "lsa_explained_variance": round(encoder.explained_variance, 4),
        "avg_passage_tokens": round(float(np.mean([len(t) for t in tokens])), 1),
        "build_seconds": round(time.time() - t0, 1),
    }
    with gzip.open(out_dir / "passages.jsonl.gz", "wt", encoding="utf-8") as fh:
        for p in passages:
            fh.write(json.dumps(p.to_dict()) + "\n")
    joblib.dump(bm25.state(), out_dir / "bm25.joblib", compress=3)
    np.save(out_dir / "dense.npy", dense)
    joblib.dump(encoder, out_dir / "encoder.joblib", compress=3)
    joblib.dump(tokens, out_dir / "tokens.joblib", compress=3)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Index built: %s", manifest)
    return _assemble(passages, bm25, dense, encoder, manifest, tokens)


def _assemble(passages, bm25, dense, encoder, manifest, tokens=None) -> KnowledgeIndex:
    tokens = tokens or [tokenize(p.index_text) for p in passages]
    return KnowledgeIndex(
        passages=passages,
        bm25=bm25,
        dense=dense,
        encoder=encoder,
        token_sets=[frozenset(t) for t in tokens],
        source_array=np.array([p.source for p in passages]),
        manifest=manifest,
    )


def load_index(index_dir: Path | None = None) -> KnowledgeIndex:
    index_dir = index_dir or settings.paths.index
    if not (index_dir / "manifest.json").exists():
        raise FileNotFoundError(f"No index at {index_dir}. Run: python scripts/build_index.py")
    with gzip.open(index_dir / "passages.jsonl.gz", "rt", encoding="utf-8") as fh:
        passages = [Passage.from_dict(json.loads(line)) for line in fh]
    bm25 = BM25Index.from_state(joblib.load(index_dir / "bm25.joblib"))
    dense = np.load(index_dir / "dense.npy")
    encoder = joblib.load(index_dir / "encoder.joblib")
    manifest = json.loads((index_dir / "manifest.json").read_text())
    tok_path = index_dir / "tokens.joblib"
    tokens = joblib.load(tok_path) if tok_path.exists() else None
    return _assemble(passages, bm25, dense, encoder, manifest, tokens)
