"""Pydantic response models for OpenAPI schema documentation."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Chat completion models (OpenAI-compatible)
# ---------------------------------------------------------------------------


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletion(BaseModel):
    """Non-streaming chat completion response (OpenAI-compatible + engine metadata)."""

    id: str
    object: str
    created: int
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo
    engine: str
    prompt: str
    elapsed_ms: int


# ---------------------------------------------------------------------------
# Model list models (OpenAI-compatible)
# ---------------------------------------------------------------------------


class ModelEntry(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str
    capabilities: Optional[dict[str, bool]] = None


class ModelList(BaseModel):
    object: str
    data: list[ModelEntry]


class LegacyModelEntry(ModelEntry):
    """Extended model entry with legacy fields for /models endpoint."""

    name: str
    limits: Optional[dict[str, Any]] = None
    supported_models: Optional[list[str]] = None


class LegacyModelList(BaseModel):
    object: str
    data: list[LegacyModelEntry]


# ---------------------------------------------------------------------------
# Misc response models
# ---------------------------------------------------------------------------


class PingResponse(BaseModel):
    status: str
    service: str
