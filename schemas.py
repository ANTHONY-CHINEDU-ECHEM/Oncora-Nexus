"""Request and response schemas for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from oncora.models.patient import PatientProfile

Source = Literal["guideline", "pubmed", "trial"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, examples=["Does T-DM1 help residual HER2 positive disease?"])
    case_context: str | None = Field(None, max_length=4000)
    sources: list[Source] | None = None
    mode: Literal["auto", "llm", "extractive"] = "auto"
    k: int = Field(8, ge=3, le=15)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    sources: list[Source] | None = None
    k: int = Field(10, ge=1, le=50)


class PrognosisRequest(BaseModel):
    patient: PatientProfile
    model: Literal["core", "extended"] = "core"
    extended_inputs: dict[str, float] | None = Field(
        None, description="Optional treatment and histology inputs for the extended model: her2_positive, chemotherapy, radiotherapy, mastectomy, lobular_histology"
    )


class SubtypeRequest(BaseModel):
    patient: PatientProfile | None = None
    expression: dict[str, float] | None = Field(None, description="PAM50 gene log2 expression values keyed by gene symbol")


class TrialMatchRequest(BaseModel):
    patient: PatientProfile
    query: str | None = Field(None, max_length=500, description="Optional free text such as an agent of interest")
    top_k: int = Field(20, ge=1, le=100)
    include_excluded: bool = False


class CaseBriefRequest(BaseModel):
    patient: PatientProfile
    question: str | None = Field(None, max_length=1000)
    mode: Literal["auto", "llm", "extractive"] = "auto"
