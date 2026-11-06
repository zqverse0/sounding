"""inspect_ai 工具层的最小替身：structured_generate 用到的工具定义、参数解析与执行。

错误文本与 inspect_ai 0.3.287 `model/_call_tools.py` 一致（会回给模型，属于请求内容）。
"""

from __future__ import annotations

import json
import string
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Literal,
    Protocol,
    Sequence,
    TypeAlias,
    cast,
    runtime_checkable,
)

from ._tool_params import ToolParams as ToolParams
from .model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ModelOutput,
    ToolCall,
    ToolCallError,
)

Tool: TypeAlias = Callable[..., Awaitable[Any]]
ToolChoice: TypeAlias = Literal["auto", "any", "none"] | Any


@runtime_checkable
class ToolSource(Protocol):
    async def tools(self) -> list[Any]: ...


@dataclass
class ToolInfo:
    name: str | None
    description: str | None
    parameters: ToolParams = field(default_factory=ToolParams)
    options: dict[str, Any] | None = None


class ToolDef:
    """同 inspect_ai ToolDef：name / description 缺省时取自函数。"""

    def __init__(
        self,
        tool: Tool,
        name: str | None = None,
        description: str | None = None,
        parameters: ToolParams | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.tool = tool
        self.name: str = name or tool.__name__
        self.description: str = description or (tool.__doc__ or "")
        self.parameters = parameters or ToolParams()
        self.options = options


def tool_infos(tools: Sequence[Any]) -> list[ToolInfo]:
    return [
        ToolInfo(
            name=t.name,
            description=t.description,
            parameters=t.parameters,
            options=t.options,
        )
        for t in tools
        if isinstance(t, (ToolDef, ToolInfo))
    ]


def parse_tool_call(id: str, function: str, arguments: str | None) -> ToolCall:
    """同 inspect_ai parse_tool_call 的 JSON 分支。

    ponytail: 非 `{` 开头的参数 inspect_ai 用 YAML 解析并映射到首个参数；这里按解析失败处理（模型极少这样回）。
    嵌套深度上限（100）也不检查。
    """
    error: str | None = None
    arguments_dict: dict[str, Any] = {}
    text = (arguments or "").strip()
    try:
        parsed = json.loads(text) if text.startswith("{") else None
        if parsed is None and text:
            raise json.JSONDecodeError("Expecting value", text, 0)
        arguments_dict = parsed or {}
    except json.JSONDecodeError as ex:
        recovered = _object_with_trailing_quotes(text)
        if recovered is not None:
            arguments_dict = recovered
        else:
            error = tool_parse_error_message(arguments, ex)
    return ToolCall(
        id=id, function=function, arguments=arguments_dict, parse_error=error
    )


def tool_parse_error_message(arguments: str | None, ex: Exception) -> str:
    raw = arguments or ""
    limit = 16 * 1024
    encoded = raw.encode()
    if len(encoded) > limit:
        half = limit // 2
        shown = (
            encoded[:half].decode(errors="ignore")
            + "\n...\n"
            + encoded[-half:].decode(errors="ignore")
            + f"\n\n(arguments middle-truncated from {len(encoded)} bytes)"
        )
    else:
        shown = raw
    return f"Error parsing the following tool call arguments:\n\n{shown}\n\nError details: {ex}"


def _object_with_trailing_quotes(arguments: str) -> dict[str, Any] | None:
    # 同 inspect_ai：完整 JSON 对象后面只跟着多余的双引号时恢复。
    try:
        value, index = json.JSONDecoder().raw_decode(arguments)
    except (json.JSONDecodeError, RecursionError):
        return None
    if not isinstance(value, dict):
        return None
    if set(arguments[index:]) - {'"'} - set(string.whitespace):
        return None
    return cast(dict[str, Any], value)


def validate_tool_input(input: dict[str, Any], parameters: ToolParams) -> str | None:
    # 同 inspect_ai validate_tool_input（jsonschema Draft7）。
    from jsonschema import Draft7Validator

    errors = list(
        Draft7Validator(parameters.model_dump(exclude_none=True)).iter_errors(input)
    )
    if not errors:
        return None
    return "\n".join(
        [f"Found {len(errors)} validation errors parsing tool input arguments:"]
        + [f"- {error.message}" for error in errors]
    )


async def execute_tools(
    messages: list[ChatMessage], tools: Sequence[Any], max_output: int | None = None
) -> tuple[list[ChatMessage], ModelOutput | None]:
    """执行最后一条 assistant 消息里的全部工具调用，每个调用配一条 tool 结果。"""
    message = messages[-1]
    if not isinstance(message, ChatMessageAssistant) or not message.tool_calls:
        return [], None
    defs = [t for t in tools if isinstance(t, ToolDef)]
    results: list[ChatMessage] = []
    for call in message.tool_calls:
        tool_def = next((t for t in defs if t.name == call.function), None)
        error: str | None
        content = ""
        if call.parse_error:
            error = call.parse_error
        elif tool_def is None:
            error = f"Tool {call.function} not found"
        else:
            error = validate_tool_input(call.arguments, tool_def.parameters)
            if error is None:
                content = str(await tool_def.tool(**call.arguments))
        results.append(
            ChatMessageTool(
                content=content,
                tool_call_id=call.id,
                function=call.function,
                error=ToolCallError(type="parsing", message=error) if error else None,
            )
        )
    return results, None
