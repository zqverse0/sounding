"""会话类型：字段名与 inspect_scout `TranscriptInfo` / `Transcript` 相同（扫描提示模板按这些名字取值）。

只有消息，没有事件与时间线（输入是 OpenAI chat 消息）。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, JsonValue

from .._ai.model import ChatMessage


class Transcript(BaseModel):
    transcript_id: str
    source_type: str | None = Field(default=None)
    source_id: str | None = Field(default=None)
    source_uri: str | None = Field(default=None)
    date: str | None = Field(default=None)
    task_set: str | None = Field(default=None)
    task_id: str | None = Field(default=None)
    task_repeat: int | None = Field(default=None)
    agent: str | None = Field(default=None)
    agent_args: dict[str, Any] | None = Field(default=None)
    model: str | None = Field(default=None)
    model_options: dict[str, Any] | None = Field(default=None)
    score: JsonValue | None = Field(default=None)
    score_explanation: str | None = Field(default=None)
    success: bool | None = Field(default=None)
    message_count: int | None = Field(default=None)
    total_time: float | None = Field(default=None)
    total_tokens: int | None = Field(default=None)
    error: str | None = Field(default=None)
    limit: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)
    messages: list[ChatMessage] = Field(default_factory=list)
