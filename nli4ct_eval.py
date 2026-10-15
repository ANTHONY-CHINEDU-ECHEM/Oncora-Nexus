"""Evaluation of evidence selection and claim verification on NLI4CT.

NLI4CT (SemEval 2023 Task 7) provides expert written statements about breast
cancer trial reports, each labelled Entailment or Contradiction, with gold
evidence line indices. It is a direct test of two capabilities Oncora Nexus
relies on when it answers questions about a specific trial:

Task A, evidence selection: given a statement and the relevant section of a
trial report, which lines support or refute it? This is the same operation the
answer verifier performs when it checks a generated sentence against a cited
passage. Measured with precision, recall and F1 against the gold lines.

Task B, claim verification: is the statement entailed or contradicted? A
lightweight, fully local classifier built on the verifier's own signals
(lexical support, numeric agreement, negation and quantifier mismatch) is
trained on the official training split and tested on the gold test split.
Reported honestly against a majority baseline: this shows how far simple
grounding checks go and where a dedicated NLI model would be needed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from oncora.data.trials import NLIStatement, TrialRecord
from oncora.rag.bm25 import BM25Index
from oncora.rag.text import numbers, tokenize

NEGATION = re.compile(r"\b(no|not|none|never|without|neither|nor|fail\w*|did not|does not|cannot)\b", re.I)
# Claim features use a focused evidence window rather than the set level optimum,
# which degenerates to selecting every line on this dataset.
CLAIM_EVIDENCE_THRESHOLD = 0.2

QUANT_ALL = re.compile(r"\b(all|every|each|both|entire)\b", re.I)
QUANT_SOME = re.compile(r"\b(some|at least|most|majority|more than|less than|fewer|higher|lower|greater)\b", re.I)


def _line_scores(statement: str, lines: list[str]) -> np.ndarray:
    toks = [tokenize(l) for l in lines]
    if not any(toks):
        return np.zeros(len(lines))
    idx = BM25Index(k1=1.2, b=0.5).fit([t if t else ["_"] for t in toks])
    s = idx.score(tokenize(statement))
    q = set(tokenize(statement))
    cov = np.array([len(q & set(t)) / max(len(q), 1) for t in toks])
    s = s / (s.max() or 1.0)
    return 0.6 * s + 0.4 * cov


def select_evidence(statement: str, lines: list[str], threshold: float, min_lines: int = 1) -> list[int]:
    if not lines:
        return []
    sc = _line_scores(statement, lines)
    chosen = [i for i, v in enumerate(sc) if v >= threshold and lines[i].strip()]
    if len(chosen) < min_lines:
        chosen = [int(i) for i in np.argsort(-sc)[:min_lines]]
    return sorted(chosen)


def _prf(pred: list[int], gold: list[int]) -> tuple[float, float, float]:
    p_set, g_set = set(pred), set(gold)
    tp = len(p_set & g_set)
    prec = tp / len(p_set) if p_set else 0.0
    rec = tp / len(g_set) if g_set else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def _section(trials: dict[str, TrialRecord], nct: str, section: str) -> list[str]:
    t = trials.get(nct)
    return t.section(section) if t else []


def _average_precision(order: list[int], gold: set[int]) -> float:
    hits, total = 0, 0.0
    for rank, i in enumerate(order, start=1):
        if i in gold:
            hits += 1
            total += hits / rank
    return total / len(gold) if gold else 0.0


def evidence_ranking(statements: list[NLIStatement], trials: dict[str, TrialRecord], seed: int = 0) -> dict:
    """Ranking quality of the line scorer versus a random ordering of the same lines."""
    rng = np.random.default_rng(seed)
    ap, ap_rand, p3, p3_rand = [], [], [], []
    for st in statements:
        for nct, gold in ((st.primary_id, st.primary_evidence), (st.secondary_id, st.secondary_evidence)):
            lines = _section(trials, nct, st.section) if nct else []
            valid = [i for i, l in enumerate(lines) if l.strip()]
            g = set(gold) & set(valid)
            if not g or len(valid) < 4:
                continue
            sc = _line_scores(st.statement, lines)
            order = sorted(valid, key=lambda i: -sc[i])
            rand = list(rng.permutation(valid))
            ap.append(_average_precision(order, g))
            ap_rand.append(_average_precision(rand, g))
            p3.append(len(set(order[:3]) & g) / 3)
            p3_rand.append(len(set(rand[:3]) & g) / 3)
    return {
        "pairs": len(ap),
        "mean_average_precision": round(float(np.mean(ap)), 4),
        "mean_average_precision_random": round(float(np.mean(ap_rand)), 4),
        "precision_at_3": round(float(np.mean(p3)), 4),
        "precision_at_3_random": round(float(np.mean(p3_rand)), 4),
    }


def evidence_selection(statements: list[NLIStatement], trials: dict[str, TrialRecord], threshold: float) -> dict:
    ps, rs, fs = [], [], []
    all_p, all_r, all_f = [], [], []
    for st in statements:
        for nct, gold in ((st.primary_id, st.primary_evidence), (st.secondary_id, st.secondary_evidence)):
            if not nct:
                continue
            lines = _section(trials, nct, st.section)
            pred = select_evidence(st.statement, lines, threshold)
            p, r, f = _prf(pred, gold)
            ps.append(p)
            rs.append(r)
            fs.append(f)
            ap, ar, af = _prf(list(range(len(lines))), gold)
            all_p.append(ap)
            all_r.append(ar)
            all_f.append(af)
    return {
        "pairs": len(fs),
        "precision": round(float(np.mean(ps)), 4),
        "recall": round(float(np.mean(rs)), 4),
        "f1": round(float(np.mean(fs)), 4),
        "baseline_select_all": {"precision": round(float(np.mean(all_p)), 4), "recall": round(float(np.mean(all_r)), 4), "f1": round(float(np.mean(all_f)), 4)},
    }


def tune_threshold(statements: list[NLIStatement], trials: dict[str, TrialRecord]) -> tuple[float, list[dict]]:
    curve = []
    best, best_f = 0.5, -1.0
    for th in np.arange(0.0, 0.85, 0.05):
        res = evidence_selection(statements, trials, float(th))
        curve.append({"threshold": round(float(th), 2), "f1": res["f1"], "precision": res["precision"], "recall": res["recall"]})
        if res["f1"] > best_f:
            best, best_f = float(th), res["f1"]
    return round(best, 2), curve


def claim_features(st: NLIStatement, trials: dict[str, TrialRecord], threshold: float) -> list[float]:
    feats = []
    s_tokens = set(tokenize(st.statement))
    s_nums = {n for n in numbers(st.statement) if len(n) > 1}
    neg_s = len(NEGATION.findall(st.statement))
    for nct in (st.primary_id, st.secondary_id):
        lines = _section(trials, nct, st.section) if nct else []
        sel = select_evidence(st.statement, lines, threshold) if lines else []
        ev = " ".join(lines[i] for i in sel)
        e_tokens = set(tokenize(ev))
        e_nums = numbers(ev)
        coverage = len(s_tokens & e_tokens) / max(len(s_tokens), 1)
        num_match = (len(s_nums & e_nums) / len(s_nums)) if s_nums else 1.0
        neg_e = len(NEGATION.findall(ev))
        feats += [coverage, num_match, float(len(s_nums)), float(neg_e > 0), float(len(sel)), float(bool(nct))]
    feats += [
        float(neg_s),
        float(bool(QUANT_ALL.search(st.statement))),
        float(bool(QUANT_SOME.search(st.statement))),
        float(st.kind == "Comparison"),
        float(len(st.statement.split())),
    ]
    feats += [float(st.section == s) for s in ("eligibility", "intervention", "results", "adverse_events")]
    return feats


def run(trials_list: list[TrialRecord], statements: list[NLIStatement], seed: int, out_path: Path | None = None) -> dict:
    trials = {t.nct_id: t for t in trials_list}
    train = [s for s in statements if s.split == "train"]
    dev = [s for s in statements if s.split == "dev"]
    test = [s for s in statements if s.split == "test"]
    threshold, curve = tune_threshold(train + dev, trials)
    ev_test = evidence_selection(test, trials, threshold)
    Xtr = np.array([claim_features(s, trials, CLAIM_EVIDENCE_THRESHOLD) for s in train + dev])
    ytr = np.array([1 if s.label == "Entailment" else 0 for s in train + dev])
    Xte = np.array([claim_features(s, trials, CLAIM_EVIDENCE_THRESHOLD) for s in test])
    yte = np.array([1 if s.label == "Entailment" else 0 for s in test])
    clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=seed).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    majority = int(round(ytr.mean()))
    results = {
        "protocol": __doc__.strip(),
        "splits": {"train": len(train), "dev": len(dev), "test": len(test)},
        "evidence_selection": {"tuned_threshold": threshold, "threshold_curve_train_dev": curve, "test": ev_test},
        "evidence_ranking_test": evidence_ranking(test, trials, seed),
        "claim_verification_test": {
            "accuracy": round(float(accuracy_score(yte, pred)), 4),
            "macro_f1": round(float(f1_score(yte, pred, average="macro")), 4),
            "entailment_precision": round(float(precision_score(yte, pred)), 4),
            "entailment_recall": round(float(recall_score(yte, pred)), 4),
            "majority_baseline_accuracy": round(float(accuracy_score(yte, np.full_like(yte, majority))), 4),
        },
        "interpretation": (
            "NLI4CT gold evidence often spans most of a section, so selecting every line is a strong set based baseline; "
            "the line ranker is better judged by ranking quality, where it places gold lines well above a random ordering. "
            "Claim verification from lexical and numeric signals alone is only modestly above chance, because "
            "NLI4CT statements hinge on arithmetic, comparison and paraphrase. Oncora Nexus therefore treats its "
            "verifier as a guard against unsupported and numerically invented sentences, not as proof of entailment."
        ),
    }
    if out_path:
        out_path.write_text(json.dumps(results, indent=2))
    return results
