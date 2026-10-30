"""Pydantic models shared by retrieval, the model loop, verification and the API."""

from pydantic import BaseModel, Field


class FigureResult(BaseModel):
    """One row of the figures table, optionally with its search distance."""

    figure_id: str
    country_code: str
    country_name: str
    indicator_code: str
    indicator_name: str
    year: int
    value: float
    unit: str | None
    source_url: str
    distance: float | None = None  # cosine distance, only set by search_figures


class Claim(BaseModel):
    statement: str  # the sentence or phrase containing the number
    value: float  # the number exactly as stated in the answer
    figure_id: str  # id of the figure it came from


class Answer(BaseModel):
    text: str
    claims: list[Claim]
    insufficient_data: bool


class VerifiedClaim(Claim):
    passed: bool
    actual_value: float | None
    source_url: str | None
    reason: str | None  # why it failed, e.g. "figure not found", "value mismatch"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class AskResponse(BaseModel):
    answer: str
    insufficient_data: bool
    claims: list[VerifiedClaim]  # each carries the source_url of its figure
    citation_pass_rate: float  # passed / total; 1.0 when there are no claims
