from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=1, description="The user prompt.")
    system_instruction: str | None = Field(
        default=None,
        max_length=4000,
        description="Optional system instruction sent to Gemini.",
    )
    model: str | None = Field(
        default=None,
        description="Gemini model name. Defaults to GEMINI_DEFAULT_MODEL.",
    )
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1, le=2048)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1, le=100)
    response_mime_type: Literal["text/plain", "application/json"] = "text/plain"

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("prompt cannot be empty or whitespace only")
        return cleaned

    @field_validator("system_instruction")
    @classmethod
    def validate_system_instruction(cls, value: str | None) -> str | None:
        if value is None:
            return value
        cleaned = value.strip()
        return cleaned or None


class GenerateResponse(BaseModel):
    request_id: str
    model: str
    output_text: str
    provider_latency_ms: float
    total_latency_ms: float


class ErrorResponse(BaseModel):
    detail: str
