"""Export the static bundle that powers the in browser demo engine.

The hosted demo runs without a server: retrieval, reranking, answer
composition, verification, prognosis, subtype, pathway rules and trial
screening all execute in the visitor's browser from the files written here.

Retrieval in the browser is lexical (BM25 plus a learned reranker over
lexical features) because shipping the dense encoder would add tens of
megabytes. A dedicated lexical reranker is therefore trained here, and its
quality is measured on the same held out question sets and reported next to
the full server pipeline so the gap is visible rather than hidden.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from oncora.config import settings
from oncora.data.trials import DRUG_LEXICON
from oncora.eval.retrieval_eval import _metrics, build_query_sets
from oncora.rag import query as query_mod
from oncora.rag import text as text_mod
from oncora.rag.index import KnowledgeIndex
from oncora.rag.query import ParsedQuery, parse_query
from oncora.rag.text import tokenize

log = logging.getLogger(__name__)

LEXICAL_FEATURES = [
    "bm25_norm",
    "bm25_reciprocal_rank",
    "query_coverage",
    "bigram_overlap",
    "drug_match",
    "context_match",
    "section_prior",
    "is_guideline",
    "is_trial",
    "evidence_strength",
]

SOURCE_CODE = {"guideline": "g", "pubmed": "p", "trial": "t"}


def _drug_tokens(drug: str) -> list[str]:
    return tokenize(drug)


def lexical_features(index: KnowledgeIndex, q: ParsedQuery, cand: np.ndarray, bm: np.ndarray) -> np.ndarray:
    qset = set(q.tokens)
    qbi = {f"{a} {b}" for a, b in zip(q.tokens, q.tokens[1:], strict=False)}
    drugs = [_drug_tokens(d) for d in q.drugs]
    ctx = [tokenize(c)[:1] for c in q.subtypes + q.settings + q.populations]
    ctx = [c for c in ctx if c]
    bm_max = max(float(bm[cand[0]]), 1e-9) if len(cand) else 1.0
    rows = []
    for rank, i in enumerate(cand):
        p = index.passages[i]
        toks = index.token_sets[i]
        rows.append(
            [
                bm[i] / bm_max,
                1.0 / (rank + 1),
                len(qset & toks) / max(len(qset), 1),
                (sum(1 for b in qbi if all(t in toks for t in b.split())) / len(qbi)) if qbi else 0.0,
                (sum(1 for d in drugs if d and all(t in toks for t in d)) / len(drugs)) if drugs else 0.0,
                (sum(1 for c in ctx if c[0] in toks) / len(ctx)) if ctx else 0.0,
                q.section_prior(p.section),
                1.0 if p.source == "guideline" else 0.0,
                1.0 if p.source == "trial" else 0.0,
                (5 - p.evidence_rank) / 4.0,
            ]
        )
    return np.array(rows)


def _bm25_candidates(index: KnowledgeIndex, q: ParsedQuery, pool: int = 60) -> tuple[np.ndarray, np.ndarray]:
    bm = index.bm25.score(q.tokens)
    cand = np.argsort(-bm)[:pool]
    return cand[bm[cand] > 0], bm


def train_lexical_reranker(index: KnowledgeIndex) -> tuple[dict, dict]:
    sets = build_query_sets(index)
    X_all, y_all = [], []
    for item in sets["pubmed_train"] + sets["card_train"] + sets["trial_train"]:
        q = parse_query(item["query"])
        cand, bm = _bm25_candidates(index, q)
        if len(cand) == 0:
            continue
        y = np.array([1 if index.passages[i].doc_id in item["relevant"] else 0 for i in cand])
        if y.sum() == 0:
            continue
        X_all.append(lexical_features(index, q, cand, bm))
        y_all.append(y)
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000)).fit(np.vstack(X_all), np.concatenate(y_all))
    scaler, lr = model[0], model[-1]
    spec = {
        "features": LEXICAL_FEATURES,
        "mean": scaler.mean_.round(6).tolist(),
        "scale": scaler.scale_.round(6).tolist(),
        "coef": lr.coef_[0].round(6).tolist(),
        "intercept": float(lr.intercept_[0]),
    }

    def rank(query: str, rerank: bool) -> list[str]:
        q = parse_query(query)
        cand, bm = _bm25_candidates(index, q)
        if len(cand) == 0:
            return []
        if rerank:
            s = model.decision_function(lexical_features(index, q, cand, bm))
            cand = cand[np.argsort(-s)]
        return [index.passages[i].doc_id for i in cand[:30]]

    evaluation = {}
    for name in ("clinical", "clinical_blind", "pubmed_test"):
        qs = sets[name]
        evaluation[name] = {
            "bm25": _metrics([rank(x["query"], False) for x in qs], [x["relevant"] for x in qs]),
            "bm25_lexical_reranker": _metrics([rank(x["query"], True) for x in qs], [x["relevant"] for x in qs]),
        }
    return spec, evaluation


def _gz_json(obj, path: Path) -> int:
    raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    with gzip.open(path, "wb", compresslevel=9) as fh:
        fh.write(raw)
    return path.stat().st_size


def _js_pattern(rx: re.Pattern[str]) -> str:
    return rx.pattern


def export(index: KnowledgeIndex, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = settings.paths
    sizes: dict[str, int] = {}

    labels = sorted({p.evidence_label for p in index.passages})
    label_ix = {l: i for i, l in enumerate(labels)}
    corpus = {
        "evidence_labels": labels,
        "passages": [
            [
                p.chunk_id,
                p.doc_id,
                SOURCE_CODE[p.source],
                p.title,
                p.url,
                p.section,
                p.text,
                p.evidence_rank,
                label_ix[p.evidence_label],
                " ".join(tokenize(p.index_text)),
                p.line_start if p.line_start is not None else -1,
            ]
            for p in index.passages
        ],
        "columns": ["chunk_id", "doc_id", "source", "title", "url", "section", "text", "evidence_rank", "evidence_label", "tokens", "line_start"],
    }
    sizes["corpus.json.gz"] = _gz_json(corpus, out_dir / "corpus.json.gz")

    lexicon = {
        "uk_to_us": text_mod.UK_TO_US,
        "synonyms": text_mod.SYNONYMS,
        "stopwords": sorted(text_mod.STOPWORDS),
        "keep_short": sorted(text_mod.KEEP_SHORT),
        "intent_rules": [[n, _js_pattern(rx)] for n, rx in query_mod.INTENT_RULES],
        "section_priors": query_mod.SECTION_PRIORS,
        "subtype_rules": {k: _js_pattern(v) for k, v in query_mod.SUBTYPE_RULES.items()},
        "setting_rules": {k: _js_pattern(v) for k, v in query_mod.SETTING_RULES.items()},
        "population_rules": {k: _js_pattern(v) for k, v in query_mod.POPULATION_RULES.items()},
        "drugs": DRUG_LEXICON,
    }
    (out_dir / "lexicon.json").write_text(json.dumps(lexicon))

    spec, browser_eval = train_lexical_reranker(index)
    (out_dir / "reranker_lexical.json").write_text(json.dumps(spec))
    (paths.reports / "browser_engine_eval.json").write_text(json.dumps(browser_eval, indent=2))

    for name in ("cox_core.json", "cox_extended.json", "subtype_surrogate.json", "cohort_analytics.json", "registry.json"):
        src = paths.models / name
        if src.exists():
            (out_dir / name).write_text(src.read_text())
    sizes["trials_parsed.json.gz"] = _gz_json(json.loads((paths.models / "trials_parsed.json").read_text()), out_dir / "trials_parsed.json.gz")
    (out_dir / "therapy_rules.json").write_text((paths.root / "oncora" / "models" / "therapy_rules.json").read_text())
    (out_dir / "cards.json").write_text(paths.knowledge_cards.read_text())

    reports = {}
    for name in ("retrieval_eval", "survival_eval", "subtype_eval", "nli4ct_eval", "answer_eval_default", "trial_parsing", "browser_engine_eval"):
        p = paths.reports / f"{name}.json"
        if p.exists():
            data = json.loads(p.read_text())
            if name == "retrieval_eval":
                data.pop("protocol", None)
            if name == "answer_eval_default":
                data.pop("per_question", None)
            reports[name] = data
    sizes["reports.json.gz"] = _gz_json(reports, out_dir / "reports.json.gz")

    meta = {
        "index": index.manifest,
        "browser_engine": {
            "retrieval": "BM25 with learned lexical reranker",
            "generation": "extractive composer with sentence level verification",
            "evaluation": browser_eval,
        },
        "sizes_bytes": sizes,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta))
    total = sum(f.stat().st_size for f in out_dir.iterdir())
    log.info("Demo bundle written to %s (%.1f MB)", out_dir, total / 1e6)
    return meta


__all__ = ["export", "LEXICAL_FEATURES"]
