"""Structured-output schemas for the LLM decision steps.

Used with Groq's strict json_schema mode (constrained decoding on gpt-oss), so
labels can't come back as 'Yes.', 'in-scope', or a verdict buried mid-text.
Field order matters: reasoning-like fields come before the decision field so
the model writes them first."""

from typing import Literal
from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    label: Literal["in_scope", "out_of_scope"]


class GradeDecision(BaseModel):
    relevant: bool = Field(
        description="True if the document helps answer the question."
    )


class ValidationResult(BaseModel):
    unsupported_claims: list[str] = Field(
        description="Each factual claim in the answer that the sources do not "
        "support, quoted or closely paraphrased. Empty if all are supported."
    )
    grounded: bool = Field(
        description="True only if unsupported_claims is empty."
    )
