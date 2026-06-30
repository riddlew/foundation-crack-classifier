from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class ClassifierResult(BaseModel):
    severity_level: str
    urgency: str
    final_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    raw_probabilities: dict[str, float]
    why_this_result: str
    customer_summary: str
    disclaimer: str
    recommended_action: str

    model_config = ConfigDict(extra="forbid")


class FileClassificationResponse(BaseModel):
    filename: str
    ok: bool
    result: ClassifierResult | None
    error: str | None

    model_config = ConfigDict(extra="forbid")


class ClassifyResponse(BaseModel):
    results: list[FileClassificationResponse]

    model_config = ConfigDict(extra="forbid")
