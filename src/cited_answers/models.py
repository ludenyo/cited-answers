"""Pydantic models shared by retrieval, the model loop, verification and the API."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

ClaimKind = Literal["figure", "derived"]
Operation = Literal["difference", "percent_change", "ratio"]


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


class IndicatorInfo(BaseModel):
    indicator_code: str
    indicator_name: str
    unit: str | None


class Claim(BaseModel):
    """A number in the answer and where it came from.

    A figure claim cites one stored figure. A derived claim is computed from two
    stored figures, listed as [from, to]:
      difference      = to - from
      percent_change  = (to - from) / from * 100
      ratio           = from / to
    """

    # This docstring is sent to the model as the schema description, so it speaks
    # to the model. Developer notes live in comments like this one.
    # Every field is required (null or empty when unused) because the strict
    # submit_answer schema needs all properties present. extra="forbid" makes the
    # generated schema carry additionalProperties: false, which strict also needs.
    model_config = ConfigDict(extra="forbid")

    statement: str  # the sentence or phrase containing the number
    value: float  # the number exactly as stated in the answer
    kind: ClaimKind = Field(
        description="figure: read straight from a tool result. derived: computed."
    )
    figure_id: str | None = Field(
        description="Figure claims: the figure's id. Derived claims: null."
    )
    source_figure_ids: list[str] = Field(
        description="Derived claims: exactly two figure ids, [from, to]. Figure claims: []."
    )
    operation: Operation | None = Field(
        description="Derived claims: difference, percent_change or ratio. Figure claims: null."
    )

    @model_validator(mode="after")
    def _fields_must_match_kind(self) -> Self:
        if self.kind == "figure":
            if not self.figure_id:
                raise ValueError("a figure claim needs figure_id")
            if self.source_figure_ids or self.operation is not None:
                raise ValueError("a figure claim must have source_figure_ids [] and operation null")
        else:
            if self.figure_id is not None:
                raise ValueError("a derived claim must have figure_id null")
            if len(self.source_figure_ids) != 2:
                raise ValueError("a derived claim needs exactly two source_figure_ids")
            if self.operation is None:
                raise ValueError("a derived claim needs an operation")
        return self


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    claims: list[Claim]
    insufficient_data: bool


class VerifiedClaim(Claim):
    passed: bool
    actual_value: float | None  # the stored value, or the recomputed value for derived claims
    source_url: str | None  # figure claims only; derived claims use source_urls
    source_urls: list[str]  # every figure behind the claim, for figure and derived claims
    reason: str | None  # why it failed, e.g. "figure not found", "value mismatch"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class AskResponse(BaseModel):
    answer: str
    insufficient_data: bool
    claims: list[VerifiedClaim]  # each carries the source_url of its figure
    citation_pass_rate: float  # passed / total; 1.0 when there are no claims
    model: str
    prompt_version: str  # first 8 characters of the SHA-256 of the prompt file
    latency_ms: int
    input_tokens: int  # summed over every model call made for this question
    output_tokens: int
    # Guardrails. A refused question was never sent to the answering model.
    refused: bool = False
    refusal_reason: Literal["out_of_scope", "suspicious"] | None = None
    # True when claims that failed verification, even after one retry, were dropped.
    # Clients must show partial_answer_explanation whenever this is true.
    partial_answer: bool = False
    partial_answer_explanation: str | None = None
    guardrails: list[str] = Field(default_factory=list)  # every guardrail that fired
