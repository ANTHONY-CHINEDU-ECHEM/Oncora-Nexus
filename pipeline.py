"""End to end question answering: guard, understand, retrieve, generate, verify."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator

from oncora.config import settings
from oncora.rag.composer import compose
from oncora.rag.guard import check_input, redact_phi, scrub_injection
from oncora.rag.llm import LLMUnavailable, OllamaClient
from oncora.rag.prompts import CASE_TEMPLATE, PROMPT_VERSION, SYSTEM_PROMPT, USER_TEMPLATE, format_evidence
from oncora.rag.query import parse_query
from oncora.rag.retriever import Hit, HybridRetriever
from oncora.rag.verify import verify_answer

log = logging.getLogger(__name__)


def confidence(hits: list[Hit], faithfulness: float) -> dict:
    strong = sum(1 for h in hits if h.features.get("query_coverage", 0) >= 0.5)
    high_level = any(h.passage.evidence_rank <= 2 for h in hits[:4])
    score = 0.4 * min(1.0, strong / 3) + 0.2 * (1.0 if high_level else 0.0) + 0.4 * faithfulness
    level = "high" if score >= 0.75 else "moderate" if score >= 0.5 else "low"
    return {
        "score": round(score, 3),
        "level": level,
        "strong_passages": strong,
        "high_level_evidence": high_level,
        "faithfulness": round(faithfulness, 3),
    }


class AnswerEngine:
    def __init__(self, retriever: HybridRetriever, llm: OllamaClient | None = None) -> None:
        self.retriever = retriever
        self.llm = llm or OllamaClient()
        self._llm_ok: tuple[float, bool] | None = None

    def llm_available(self) -> bool:
        now = time.time()
        if self._llm_ok and now - self._llm_ok[0] < 30:
            return self._llm_ok[1]
        ok = settings.llm.provider == "ollama" and self.llm.available()
        self._llm_ok = (now, ok)
        return ok

    def _prepare(self, question: str, case_context: str | None, sources: list[str] | None, k: int):
        guard = check_input(question, case_context)
        case_clean = redact_phi(case_context)[0] if case_context else None
        if guard.blocked:
            return guard, None, [], case_clean
        q = parse_query(guard.text + (" " + case_clean if case_clean else ""))
        q_question = parse_query(guard.text)
        # Retrieval uses the question plus case context; intent comes from the question alone.
        q.intents = q_question.intents or q.intents
        hits = self.retriever.search(q, k=k, sources=sources)
        for h in hits:
            h.passage.text = scrub_injection(h.passage.text)
        return guard, q, hits, case_clean

    def _messages(self, question: str, case_clean: str | None, hits: list[Hit]) -> list[dict]:
        case_block = CASE_TEMPLATE.format(case=case_clean) if case_clean else ""
        user = USER_TEMPLATE.format(question=question, case_block=case_block, evidence=format_evidence(hits))
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

    def _package(self, request_id, question, guard, q, hits, answer, mode, notice, started) -> dict:
        report = verify_answer(answer, [h.passage.text for h in hits])
        return {
            "request_id": request_id,
            "question": guard.text,
            "mode": mode,
            "notice": notice,
            "answer": answer,
            "query": q.to_dict(),
            "evidence": [h.to_dict() for h in hits],
            "verification": report.to_dict(),
            "confidence": confidence(hits, report.faithfulness),
            "redactions": guard.redactions,
            "model": self.llm.model if mode == "llm" else "extractive composer",
            "prompt_version": PROMPT_VERSION,
            "latency_ms": int((time.time() - started) * 1000),
        }

    def answer(
        self,
        question: str,
        case_context: str | None = None,
        sources: list[str] | None = None,
        mode: str = "auto",
        k: int | None = None,
    ) -> dict:
        started = time.time()
        request_id = uuid.uuid4().hex[:12]
        k = k or settings.retrieval.final_k
        guard, q, hits, case_clean = self._prepare(question, case_context, sources, k)
        if guard.blocked:
            return {
                "request_id": request_id,
                "question": guard.text,
                "mode": "blocked",
                "answer": guard.message,
                "evidence": [],
                "verification": None,
                "confidence": {"level": "none", "score": 0.0},
                "redactions": guard.redactions,
                "latency_ms": int((time.time() - started) * 1000),
            }
        notice = None
        use_llm = mode == "llm" or (mode == "auto" and self.llm_available())
        if use_llm and hits:
            try:
                answer = self.llm.chat(self._messages(guard.text, case_clean, hits))
                return self._package(request_id, question, guard, q, hits, answer, "llm", None, started)
            except LLMUnavailable as exc:
                log.warning("LLM unavailable, falling back to extractive mode: %s", exc)
                notice = "The local language model could not be reached, so an extractive evidence brief is shown."
        answer = compose(guard.text, q, hits)
        return self._package(request_id, question, guard, q, hits, answer, "extractive", notice, started)

    def stream(self, question: str, case_context: str | None = None, sources: list[str] | None = None, k: int | None = None) -> Iterator[dict]:
        """Server sent event stream: evidence first, then tokens, then the verified result."""
        started = time.time()
        request_id = uuid.uuid4().hex[:12]
        k = k or settings.retrieval.final_k
        guard, q, hits, case_clean = self._prepare(question, case_context, sources, k)
        if guard.blocked:
            yield {"type": "final", "data": {"request_id": request_id, "mode": "blocked", "answer": guard.message, "evidence": []}}
            return
        yield {"type": "evidence", "data": [h.to_dict() for h in hits]}
        mode, notice, answer = "extractive", None, ""
        if self.llm_available() and hits:
            try:
                for piece in self.llm.stream(self._messages(guard.text, case_clean, hits)):
                    answer += piece
                    yield {"type": "token", "data": piece}
                mode = "llm"
            except LLMUnavailable:
                notice = "The local language model stopped responding, so an extractive evidence brief is shown."
                answer = ""
        if mode == "extractive":
            answer = compose(guard.text, q, hits)
            yield {"type": "token", "data": answer}
        yield {"type": "final", "data": self._package(request_id, question, guard, q, hits, answer, mode, notice, started)}
