"""Retrieval benchmark and learning to rank.

Three query sets, each judged at document level:

1. PubMed known item queries. For a held out abstract, the OBJECTIVE
   sentence with 50 percent of its words removed (seeded) is the query and the
   abstract is the single relevant document. This is the classic known item
   protocol for corpora without human judgements.
2. Card topical queries (training only): random subsets of a card's title,
   tags and body words, so the reranker sees queries for which a guideline
   digest is the right answer.
3. Trial registry queries (training only): heavily thinned results,
   intervention and adverse event text, so the reranker learns when trial
   reports are the right evidence.
4. Clinical questions, development set: 50 clinician style questions with
   relevant knowledge cards. Never used for training, but error analysis on
   this set informed the lay language lexicon, so its scores are optimistic.
5. Clinical questions, blind set: 20 further questions written after the
   lexicon was frozen and scored once. This is the headline figure.

The reranker is trained on sets 1, 2 and 3 (train partitions) and evaluated on
the held out partition of set 1 and on sets 4 and 5, which it never sees.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from oncora.config import settings
from oncora.rag.index import KnowledgeIndex
from oncora.rag.query import parse_query
from oncora.rag.retriever import FEATURE_NAMES, HybridRetriever, Reranker

log = logging.getLogger(__name__)

QUESTIONS_PATH = Path(__file__).with_name("clinical_questions.json")


def _dropout(text: str, rng: random.Random, keep: float = 0.5) -> str:
    words = [w for w in text.split() if any(c.isalnum() for c in w)]
    kept = [w for w in words if rng.random() < keep]
    if len(kept) < 3:
        kept = words[: max(3, len(words) // 2)]
    return " ".join(kept)


def build_query_sets(index: KnowledgeIndex, n_train: int = 1200, n_test: int = 500, seed: int = 7) -> dict:
    rng = random.Random(seed)
    pubmed_docs: dict[str, str] = {}
    for p in index.passages:
        if p.source == "pubmed" and p.doc_id not in pubmed_docs:
            pubmed_docs[p.doc_id] = p.title
    doc_ids = sorted(pubmed_docs)
    rng.shuffle(doc_ids)
    train_docs, test_docs = doc_ids[:n_train], doc_ids[n_train : n_train + n_test]
    pub_train = [{"query": _dropout(pubmed_docs[d], rng), "relevant": [d]} for d in train_docs]
    pub_test = [{"query": _dropout(pubmed_docs[d], rng), "relevant": [d]} for d in test_docs]
    card_train = []
    for p in index.passages:
        if p.source != "guideline":
            continue
        vocab = (p.title + " " + " ".join(p.topics) + " " + p.text).split()
        for _ in range(8):
            n = rng.randint(4, 8)
            card_train.append({"query": " ".join(rng.sample(vocab, min(n, len(vocab)))), "relevant": [p.doc_id]})
    trial_chunks = [p for p in index.passages if p.source == "trial" and p.section in ("Results", "Adverse events", "Interventions")]
    rng.shuffle(trial_chunks)
    trial_train = [{"query": _dropout(p.text.replace(" | ", " "), rng, keep=0.25), "relevant": [p.doc_id]} for p in trial_chunks[:400]]
    data = json.loads(QUESTIONS_PATH.read_text())
    clinical_q = [{"query": q["question"], "relevant": q["relevant"], "id": q["id"]} for q in data["questions"]]
    blind_q = [{"query": q["question"], "relevant": q["relevant"], "id": q["id"]} for q in data.get("blind_questions", [])]
    return {
        "pubmed_train": pub_train,
        "pubmed_test": pub_test,
        "card_train": card_train,
        "trial_train": trial_train,
        "clinical": clinical_q,
        "clinical_blind": blind_q,
    }


def train_reranker(retriever: HybridRetriever, train_queries: list[dict], out_path: Path) -> dict:
    X_all, y_all, groups = [], [], 0
    pool = retriever.cfg.candidate_pool
    for item in train_queries:
        q = parse_query(item["query"])
        cand, stats = retriever.candidates(q, None, pool)
        if len(cand) == 0:
            continue
        labels = np.array([1 if retriever.index.passages[i].doc_id in item["relevant"] else 0 for i in cand])
        if labels.sum() == 0:
            continue  # nothing to learn when the target is outside the candidate pool
        X_all.append(retriever.features(q, cand, stats))
        y_all.append(labels)
        groups += 1
    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    model.fit(X, y)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    coefs = model[-1].coef_[0]
    return {
        "training_queries": groups,
        "training_pairs": int(len(y)),
        "positive_pairs": int(y.sum()),
        "coefficients_standardised": {n: round(float(c), 4) for n, c in zip(FEATURE_NAMES, coefs, strict=False)},
    }


def _metrics(ranked_docs: list[list[str]], relevant: list[list[str]], k: int = 10) -> dict:
    r1 = r5 = r10 = mrr = ndcg = 0.0
    for docs, rel in zip(ranked_docs, relevant, strict=False):
        seen, uniq = set(), []
        for d in docs:
            if d not in seen:
                seen.add(d)
                uniq.append(d)
        ranks = [i for i, d in enumerate(uniq[:k]) if d in rel]
        if ranks:
            first = ranks[0]
            r1 += first < 1
            r5 += first < 5
            r10 += first < 10
            mrr += 1.0 / (first + 1)
        dcg = sum(1.0 / math.log2(i + 2) for i in ranks)
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
        ndcg += dcg / ideal if ideal else 0.0
    n = max(len(relevant), 1)
    return {
        "recall_at_1": round(r1 / n, 4),
        "recall_at_5": round(r5 / n, 4),
        "recall_at_10": round(r10 / n, 4),
        "mrr_at_10": round(mrr / n, 4),
        "ndcg_at_10": round(ndcg / n, 4),
        "queries": len(relevant),
    }


def evaluate_variants(retriever: HybridRetriever, queries: list[dict], k: int = 10) -> dict:
    idx = retriever.index
    variants = {
        "bm25": lambda q: retriever.search_bm25(q, k * 3),
        "dense_lsa": lambda q: retriever.search_dense(q, k * 3),
        "hybrid_rrf": lambda q: retriever.search_ids(q, k * 3, rerank=False),
        "hybrid_rrf_reranked": lambda q: retriever.search_ids(q, k * 3, rerank=True),
    }
    out = {}
    for name, fn in variants.items():
        t0 = time.time()
        ranked = [[idx.passages[i].doc_id for i in fn(item["query"])] for item in queries]
        m = _metrics(ranked, [item["relevant"] for item in queries], k)
        m["mean_latency_ms"] = round((time.time() - t0) * 1000 / max(len(queries), 1), 1)
        out[name] = m
        log.info("%s %s", name, m)
    return out


def run(retriever: HybridRetriever, report_path: Path | None = None) -> dict:
    sets = build_query_sets(retriever.index)
    reranker_path = settings.paths.models / "reranker.joblib"
    training = train_reranker(retriever, sets["pubmed_train"] + sets["card_train"] + sets["trial_train"], reranker_path)
    retriever.reranker = Reranker.load(reranker_path)
    results = {
        "protocol": __doc__.strip(),
        "index": retriever.index.manifest,
        "reranker_training": training,
        "pubmed_known_item_test": evaluate_variants(retriever, sets["pubmed_test"]),
        "clinical_questions_development": evaluate_variants(retriever, sets["clinical"]),
    }
    if sets["clinical_blind"]:
        results["clinical_questions_blind"] = evaluate_variants(retriever, sets["clinical_blind"])
    per_question = []
    for item in sets["clinical"] + sets["clinical_blind"]:
        hits = retriever.search(item["query"], k=5)
        per_question.append(
            {
                "id": item["id"],
                "question": item["query"],
                "relevant": item["relevant"],
                "top5": [h.passage.doc_id for h in hits],
                "hit_at_5": any(h.passage.doc_id in item["relevant"] for h in hits),
            }
        )
    results["clinical_questions_detail"] = per_question
    if report_path:
        report_path.write_text(json.dumps(results, indent=2))
    return results
