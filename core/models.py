import re
from pydantic import BaseModel, field_validator
from typing import Optional


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    rate: Optional[str] = "+0%"

    @field_validator("text", mode='before')
    @classmethod
    def clean_text(cls, v: str) -> str:
        """Clean text by removing newlines, tabs, carriage returns, and normalizing whitespace.
        
        - Replaces \\n, \\r, \\t with spaces
        - Normalizes multiple spaces to single space
        - Trims leading/trailing whitespace
        """
        if not isinstance(v, str):
            return v
        
        # Replace newlines, carriage returns, and tabs with spaces
        cleaned = v.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        # Normalize multiple spaces to single space
        cleaned = re.sub(r'\s+', ' ', cleaned)
        # Trim leading/trailing whitespace
        return cleaned.strip()

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v

    @field_validator("text")
    @classmethod
    def text_length_ok(cls, v: str) -> str:
        if len(v) > 1000:
            raise ValueError("Text too long (max 1000 chars)")
        return v


class STTResponse(BaseModel):
    text: str
    language: str = "yue"
    duration_seconds: float
    processing_time_ms: int
    confidence: Optional[float] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None