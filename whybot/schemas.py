"""Structured contracts between agents. Every LLM output is validated against one of these."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

HypothesisKind = Literal["metric", "weather", "holiday", "day_of_week", "fare_mix", "zone_mix", "data_quality", "other"]


class Hypothesis(BaseModel):
    id: str = Field(description="H0 is reserved for quantifying the metric change itself; then H1, H2, ...")
    kind: HypothesisKind
    statement: str
    support_if: str = Field(description="What numbers would support it")
    refute_if: str = Field(description="What numbers would refute it")


class QuerySpec(BaseModel):
    hypothesis_id: str
    purpose: str = Field(description="Exactly what the query must return, incl. grouping, filters and windows")
    tables: list[str] = Field(description="Fully qualified tables from the catalog, e.g. NY_TAXI.yellow_tripdata")


class Plan(BaseModel):
    metric: str
    target_window: str
    baseline_window: str
    hypotheses: list[Hypothesis]
    queries: list[QuerySpec]
    rationale: str = ""


class SQLDraft(BaseModel):
    sql: str
    tables_used: list[str]
    note: str = ""


class CitedValue(BaseModel):
    qid: str
    column: str
    value: float | int | str | None = Field(description="the exact cell value; null only when the cell itself is NULL")
    row_hint: str = Field(default="", description="column=value identifying the row, e.g. trip_date=2024-02-13")


class DerivedValue(BaseModel):
    name: str = Field(description="e.g. pct_change_vs_baseline")
    value: float
    formula: str = Field(description="Arithmetic over cited values v1..vn (1-based order of cited_values), + - * / ( ), avg(), sum(), min(), max(), abs(). e.g. (v1 - avg(v2,v3,v4)) / avg(v2,v3,v4) * 100")


class Claim(BaseModel):
    cid: str
    hypothesis_id: str
    statement: str
    direction: Literal["supports", "refutes"]
    cited_values: list[CitedValue]
    derived: list[DerivedValue] = Field(default=[], description="Every percentage/ratio/average used in the statement, with its formula")


class ClaimSet(BaseModel):
    claims: list[Claim]


class ClaimVerdict(BaseModel):
    cid: str
    verdict: Literal["supported", "weak", "unsupported"]
    strength: float = Field(ge=0, le=1)
    reason: str


class HypothesisStatus(BaseModel):
    hypothesis_id: str
    status: Literal["supported", "refuted", "inconclusive", "untested"]
    reason: str


class FollowUp(BaseModel):
    hypothesis_id: str
    purpose: str
    tables: list[str]


class CriticReport(BaseModel):
    verdicts: list[ClaimVerdict]
    hypothesis_status: list[HypothesisStatus]
    decision: Literal["accept", "retry", "insufficient_evidence"]
    feedback: str
    follow_up_queries: list[FollowUp] = []


class Citation(BaseModel):
    qid: str
    sql_hash: str
    table: str


class FinalClaim(BaseModel):
    text: str
    citations: list[Citation]


class FinalAnswer(BaseModel):
    headline: str
    answer: str
    recommendation: str
    claims: list[FinalClaim]
    caveats: list[str] = []
    confidence: float = Field(ge=0, le=1)
    lessons: list[str] = Field(default=[], description="0-3 reusable lessons for future investigations (schema quirks, joins, filters)")
