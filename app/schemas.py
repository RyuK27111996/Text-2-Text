from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class GenerateRequest(BaseModel):
    """Request body for text generation across supported providers."""

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
    response_json_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional Gemini JSON Schema for strict structured outputs. Requires response_mime_type=application/json.",
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """Strip and reject empty prompt values."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("prompt cannot be empty or whitespace only")
        return cleaned

    @field_validator("system_instruction")
    @classmethod
    def validate_system_instruction(cls, value: str | None) -> str | None:
        """Normalize blank system instructions to `None`."""
        if value is None:
            return value
        cleaned = value.strip()
        return cleaned or None

    @field_validator("response_json_schema")
    @classmethod
    def validate_response_json_schema(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        """Reject empty schema objects when strict JSON mode is requested."""
        if value is None:
            return value
        if not value:
            raise ValueError("response_json_schema cannot be empty")
        return value

    @model_validator(mode="after")
    def validate_json_mode_requirements(self):
        """Ensure schema-based output validation is only used with JSON responses."""
        if self.response_json_schema is not None and self.response_mime_type != "application/json":
            raise ValueError("response_json_schema requires response_mime_type='application/json'")
        return self


class GenerateResponse(BaseModel):
    """Normalized API response returned by the service."""

    request_id: str
    model: str
    output_text: str
    provider_latency_ms: float
    total_latency_ms: float


class ErrorResponse(BaseModel):
    """Standard error response payload."""

    detail: str
