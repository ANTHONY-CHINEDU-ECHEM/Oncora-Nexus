"""The LLM path, exercised with a stub client so it runs without Ollama."""

import pytest

from oncora.config import settings

pytestmark = pytest.mark.skipif(not (settings.paths.index / "manifest.json").exists(), reason="index not built")


class StubLLM:
    model = "stub-model"

    def __init__(self, answer):
        self.answer = answer
        self.messages = None

    def available(self):
        return True

    def chat(self, messages):
        self.messages = messages
        return self.answer


@pytest.fixture(scope="module")
def retriever():
    from oncora.rag.index import load_index
    from oncora.rag.retriever import HybridRetriever

    return HybridRetriever(load_index())


def test_llm_answer_is_verified_and_prompt_contains_evidence(retriever):
    from oncora.rag.pipeline import AnswerEngine

    stub = StubLLM("BOTTOM LINE: Hypofractionated whole breast radiotherapy of 26 Gy in five fractions is recommended [S1].\nKEY EVIDENCE:\n* It reduced recurrence by 99% [S1].\nCAVEATS:\n* Evidence applies to whole breast treatment.")
    engine = AnswerEngine(retriever, llm=stub)
    res = engine.answer("What whole breast radiotherapy fractionation is recommended after breast conserving surgery?", mode="llm")
    assert res["mode"] == "llm" and res["model"] == "stub-model"
    user_prompt = stub.messages[1]["content"]
    assert "[S1]" in user_prompt and "Evidence passages" in user_prompt
    statuses = [s["status"] for s in res["verification"]["sentences"]]
    assert "supported" in statuses
    assert "99%" in res["verification"]["unsupported_numbers"]


def test_llm_failure_falls_back_to_extractive(retriever):
    from oncora.rag.llm import LLMUnavailable
    from oncora.rag.pipeline import AnswerEngine

    class Broken(StubLLM):
        def chat(self, messages):
            raise LLMUnavailable("connection refused")

    engine = AnswerEngine(retriever, llm=Broken(""))
    res = engine.answer("Does exercise reduce fatigue during chemotherapy?", mode="llm")
    assert res["mode"] == "extractive" and res["notice"]
