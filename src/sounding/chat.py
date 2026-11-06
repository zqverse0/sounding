"""输入格式：OpenAI chat 消息（user / 带 `tool_calls` 的 assistant / tool）→ 消息对象。

各 Agent 的原始会话文件怎么转成这个格式，由调用方负责（如 Berth 的 `session show --messages`）。
"""

from __future__ import annotations

import json
from typing import Any

from ._ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    ToolCall,
)


def chat_messages(openai_messages: list[dict[str, Any]]) -> list[ChatMessage]:
    """OpenAI chat 消息 → 消息对象；消息 id 为 m1、m2…（结果里的引用指回第几条）。"""
    messages: list[ChatMessage] = []
    names: dict[str, str] = {}
    for message in openai_messages:
        content = message.get("content") or ""
        if message["role"] == "user":
            messages.append(ChatMessageUser(content=content))
        elif message["role"] == "assistant":
            calls = [
                ToolCall(
                    id=call.get("id") or f"call-{len(messages)}-{index}",
                    function=call["function"]["name"],
                    arguments=_arguments(call["function"]["arguments"]),
                )
                for index, call in enumerate(message.get("tool_calls") or [])
            ]
            names.update((call.id, call.function) for call in calls)
            messages.append(
                ChatMessageAssistant(content=content, tool_calls=calls or None)
            )
        elif message["role"] == "tool":
            call_id = message.get("tool_call_id")
            messages.append(
                ChatMessageTool(
                    content=content,
                    tool_call_id=call_id,
                    function=names.get(call_id or ""),
                )
            )
    for index, chat in enumerate(messages, start=1):
        chat.id = f"m{index}"
    return messages


def _arguments(arguments: str) -> dict[str, Any]:
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {"input": arguments}
    return parsed if isinstance(parsed, dict) else {"input": parsed}
