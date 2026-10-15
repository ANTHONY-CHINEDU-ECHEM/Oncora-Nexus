"""Oncora Nexus HTTP API.

Run locally:
    uvicorn oncora.api.main:app --host 0.0.0.0 --port 8000

Interactive documentation is served at /docs (Swagger UI) and /redoc.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from oncora.api.audit import AuditLog
from oncora.api.schemas import AskRequest, CaseBriefRequest, PrognosisRequest, SearchRequest, SubtypeRequest, TrialMatchRequest
from oncora.api.services import Services
from oncora.config import settings
from oncora.rag.guard import redact_phi
from oncora.rag.prompts import PROMPT_VERSION

log = logging.getLogger("oncora.api")
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    STATE["services"] = Services()
    STATE["audit"] = AuditLog(settings.paths.audit_db)
    yield
    STATE.clear()


app = FastAPI(
    title="Oncora Nexus API",
    version=settings.app_version,
    description=(
        "Evidence grounded decision support for breast cancer multidisciplinary teams: hybrid retrieval over "
        "trials, literature and curated guidance; local LLM synthesis with sentence level verification; "
        "prognostic and subtype models; and criterion level clinical trial screening."
    ),
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["*"], allow_headers=["*"])


def svc() -> Services:
    s = STATE.get("services")
    if s is None:
        raise HTTPException(503, "Service is starting")
    return s


def audit() -> AuditLog:
    return STATE["audit"]


@app.middleware("http")
async def timing_header(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    response.headers["X-Response-Time-ms"] = str(int((time.time() - t0) * 1000))
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/api/health", tags=["system"])
def health() -> dict:
    return svc().health()


@app.get("/api/meta", tags=["system"])
def meta() -> dict:
    s = svc()
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "prompt_version": PROMPT_VERSION,
        "index": s.index.manifest,
        "registry": s.registry,
        "knowledge_cards": {"count": len(s.cards), **s.cards_meta},
        "llm": {"provider": settings.llm.provider, "model": settings.llm.model, "available": s.engine.llm_available()},
    }


@app.post("/api/ask", tags=["evidence"])
def ask(req: AskRequest) -> dict:
    s = svc()
    res = s.engine.answer(req.question, req.case_context, req.sources, req.mode, req.k)
    audit().record(
        "ask", res.get("question", ""), res.get("request_id"), res.get("mode"), res.get("model"), res.get("prompt_version"),
        res.get("latency_ms"), {"confidence": res.get("confidence", {}).get("level"), "faithfulness": (res.get("verification") or {}).get("faithfulness")},
    )
    return res


@app.post("/api/ask/stream", tags=["evidence"])
def ask_stream(req: AskRequest) -> StreamingResponse:
    s = svc()

    def events():
        for ev in s.engine.stream(req.question, req.case_context, req.sources, req.k):
            if ev["type"] == "final":
                d = ev["data"]
                audit().record("ask_stream", d.get("question", ""), d.get("request_id"), d.get("mode"), d.get("model"), d.get("prompt_version"), d.get("latency_ms"), {"confidence": (d.get("confidence") or {}).get("level")})
            yield f"event: {ev['type']}\ndata: {json.dumps(ev['data'])}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/search", tags=["evidence"])
def search(req: SearchRequest) -> dict:
    s = svc()
    t0 = time.time()
    text, _ = redact_phi(req.query)
    hits = s.retriever.search(text, k=req.k, sources=req.sources)
    return {"query": text, "hits": [h.to_dict() for h in hits], "latency_ms": int((time.time() - t0) * 1000)}


@app.get("/api/documents/{doc_id}", tags=["evidence"])
def document(doc_id: str) -> dict:
    s = svc()
    passages = [p.to_dict() for p in s.index.passages if p.doc_id == doc_id]
    if not passages:
        raise HTTPException(404, f"Unknown document {doc_id}")
    if doc_id in s.cards:
        return {"doc_id": doc_id, "source": "guideline", "card": s.cards[doc_id], "passages": passages}
    if doc_id.startswith("NCT") and doc_id in s.matcher.trials:
        t = s.matcher.trials[doc_id]
        return {"doc_id": doc_id, "source": "trial", "trial": t.to_dict(), "criteria": s.matcher.parsed[doc_id].to_dict(), "passages": passages}
    return {"doc_id": doc_id, "source": passages[0]["source"], "passages": passages}


@app.get("/api/cards", tags=["evidence"])
def cards() -> dict:
    s = svc()
    return {"meta": s.cards_meta, "cards": list(s.cards.values())}


@app.post("/api/prognosis", tags=["models"])
def prognosis(req: PrognosisRequest) -> dict:
    s = svc()
    t0 = time.time()
    try:
        res = s.prognosis(req.patient, req.model, req.extended_inputs)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    audit().record("prognosis", req.patient.summary(), latency_ms=int((time.time() - t0) * 1000), model=res["model"], outcome={"risk_group": res["risk_group"]})
    return res


@app.post("/api/subtype", tags=["models"])
def subtype(req: SubtypeRequest) -> dict:
    if req.patient is None and not req.expression:
        raise HTTPException(422, "Provide a patient profile, expression values or both")
    return svc().subtype(req.patient, req.expression)


@app.post("/api/pathway", tags=["models"])
def pathway(req: PrognosisRequest) -> dict:
    return {"summary": req.patient.summary(), "rules_version": svc().rules["version"], "recommendations": svc().pathway(req.patient)}


@app.post("/api/trials/match", tags=["trials"])
def trials_match(req: TrialMatchRequest) -> dict:
    s = svc()
    t0 = time.time()
    res = s.matcher.match(req.patient, top_k=req.top_k, include_excluded=req.include_excluded, query=req.query)
    res["latency_ms"] = int((time.time() - t0) * 1000)
    res["data_note"] = "Demonstration corpus: 999 breast cancer trials with posted results (NLI4CT). Run scripts/fetch_live_trials.py to screen against currently recruiting studies."
    audit().record("trials_match", req.patient.summary(), latency_ms=res["latency_ms"], outcome=res["tiers"])
    return res


@app.get("/api/trials/{nct_id}", tags=["trials"])
def trial(nct_id: str) -> dict:
    s = svc()
    if nct_id not in s.matcher.trials:
        raise HTTPException(404, f"Unknown trial {nct_id}")
    return {"trial": s.matcher.trials[nct_id].to_dict(), "criteria": s.matcher.parsed[nct_id].to_dict()}


@app.post("/api/case/brief", tags=["case"])
def case_brief(req: CaseBriefRequest) -> dict:
    s = svc()
    res = s.case_brief(req.patient, req.question, req.mode)
    audit().record("case_brief", res["summary"], res["evidence"].get("request_id"), res["evidence"].get("mode"), res["evidence"].get("model"), PROMPT_VERSION, res["latency_ms"], {"rules_fired": len(res["pathway"])})
    return res


@app.get("/api/cohorts", tags=["analytics"])
def cohorts() -> dict:
    data = svc().cohort
    if data is None:
        raise HTTPException(503, "Cohort analytics not built. Run scripts/train_models.py")
    return data


@app.get("/api/evaluation", tags=["governance"])
def evaluation() -> dict:
    return svc().reports()


@app.get("/api/audit", tags=["governance"])
def audit_log(limit: int = Query(50, ge=1, le=500)) -> dict:
    return {"stats": audit().stats(), "entries": audit().recent(limit)}


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


# Serve the built interface when present (single container deployment).
WEB_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")
    if (WEB_DIST / "demo").exists():
        app.mount("/demo", StaticFiles(directory=WEB_DIST / "demo"), name="demo")

    @app.get("/", include_in_schema=False)
    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str = ""):
        if path.startswith("api/"):
            raise HTTPException(404)
        candidate = WEB_DIST / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(WEB_DIST / "index.html")
