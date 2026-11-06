"""inspect_ai 模型层的最小替身：scout 单会话扫描用到的消息、输出、生成与 token 计数。

只支持 OpenAI 兼容的 Chat Completions（`openai-api/<服务>/<模型>`，读 `<服务>_BASE_URL` /
`<服务>_API_KEY`，与 inspect_ai 同名）。请求体与 inspect_ai 的 `OpenAICompatibleAPI` 逐字段一致，
由 `tests/golden` 对照验证；规则出处见各函数注释（inspect_ai 0.3.287 `model/_openai.py` 等）。
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence, TypeAlias, Union, cast

from pydantic import BaseModel, Field, JsonValue

# ---- 内容与消息（字段名与 inspect_ai 相同，只保留 scout 渲染与序列化用到的）----


class ContentText(BaseModel):
    type: Literal["text"] = "text"
    text: str
    refusal: bool | None = None
    internal: JsonValue | None = None


class ContentReasoning(BaseModel):
    type: Literal["reasoning"] = "reasoning"
    reasoning: str
    summary: str | None = None
    signature: str | None = None
    redacted: bool = False
    internal: JsonValue | None = None


Content: TypeAlias = Union[ContentText, ContentReasoning]


def _message_uuid() -> str:
    return uuid.uuid4().hex


class ChatMessageBase(BaseModel):
    id: str | None = Field(default_factory=_message_uuid)
    content: str | list[Content]
    source: Literal["input", "generate"] | None = None
    metadata: dict[str, Any] | None = None
    role: str

    @property
    def text(self) -> str:
        # 同 inspect_ai ChatMessageBase.text：text 部分以换行连接。
        if isinstance(self.content, str):
            return self.content
        return "\n".join(c.text for c in self.content if c.type == "text")


class ChatMessageSystem(ChatMessageBase):
    role: Literal["system"] = "system"


class ChatMessageUser(ChatMessageBase):
    role: Literal["user"] = "user"


class ToolCall(BaseModel):
    id: str
    function: str
    arguments: dict[str, Any]
    parse_error: str | None = None
    type: Literal["function", "custom"] = "function"


class ChatMessageAssistant(ChatMessageBase):
    role: Literal["assistant"] = "assistant"
    tool_calls: list[ToolCall] | None = None
    model: str | None = None


class ToolCallError(BaseModel):
    type: str = "parsing"
    message: str


class ChatMessageTool(ChatMessageBase):
    role: Literal["tool"] = "tool"
    tool_call_id: str | None = None
    function: str | None = None
    error: ToolCallError | None = None


ChatMessage: TypeAlias = Union[
    ChatMessageSystem, ChatMessageUser, ChatMessageAssistant, ChatMessageTool
]

StopReason: TypeAlias = str


class ModelUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    message: ChatMessageAssistant
    stop_reason: StopReason = "unknown"


class ModelOutput(BaseModel):
    model: str = ""
    choices: list[ChatCompletionChoice] = Field(default_factory=list)
    usage: ModelUsage | None = None
    completion_override: str | None = Field(default=None, exclude=True)

    @property
    def message(self) -> ChatMessageAssistant:
        return self.choices[0].message

    @property
    def stop_reason(self) -> StopReason:
        return self.choices[0].stop_reason

    @property
    def completion(self) -> str:
        # 同 inspect_ai：首个 choice 的 text；structured_generate 会改写它。
        if self.completion_override is not None:
            return self.completion_override
        return self.choices[0].message.text if self.choices else ""

    @completion.setter
    def completion(self, value: str) -> None:
        self.completion_override = value


@dataclass
class GenerateConfig:
    cache_prompt: bool | None = None
    cache: Any = None
    parallel_tool_calls: bool | None = None
    max_tokens: int | None = None
    temperature: float | None = None

    def merge(self, other: "GenerateConfig") -> "GenerateConfig":
        merged = GenerateConfig(**self.__dict__)
        for key, value in other.__dict__.items():
            if value is not None:
                setattr(merged, key, value)
        return merged


CachePolicy: TypeAlias = Any

# ---- 生成 ----

_TIMEOUT = 600
_RETRIES = 5


@dataclass
class _Api:
    model_name: str


@dataclass
class Model:
    """OpenAI 兼容服务上的一个模型。"""

    name: str
    base_url: str
    api_key: str
    service: str = "openai-api"
    api: _Api = field(init=False)

    def __post_init__(self) -> None:
        self.api = _Api(model_name=self.name)

    def __str__(self) -> str:
        return f"{self.service}/{self.name}"

    async def generate(
        self,
        input: str | list[ChatMessage],
        tools: Sequence[Any] | Any = (),
        tool_choice: Any = None,
        config: GenerateConfig | None = None,
    ) -> ModelOutput:
        from .tool import tool_infos

        config = config or GenerateConfig()
        messages = [ChatMessageUser(content=input)] if isinstance(input, str) else input
        infos = tool_infos(tools)
        body: dict[str, Any] = {
            "model": self.name,
            "messages": [_openai_message(m) for m in messages],
        }
        # 同 openai_completion_params：只发显式设置的参数；parallel_tool_calls 仅在有工具时发。
        if config.max_tokens is not None:
            body["max_tokens"] = config.max_tokens
        if config.temperature is not None:
            body["temperature"] = config.temperature
        if infos:
            if config.parallel_tool_calls is not None:
                body["parallel_tool_calls"] = config.parallel_tool_calls
            body["tools"] = [_openai_tool(info) for info in infos]
            body["tool_choice"] = _openai_tool_choice(tool_choice)
        response = await asyncio.to_thread(self._post, body)
        return _model_output(response)

    async def count_tokens(
        self, input: str | list[ChatMessage], config: Any = None
    ) -> int:
        text = input if isinstance(input, str) else "\n".join(m.text for m in input)
        return count_text_tokens(text)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        # ponytail: 只重试 429 / 5xx / 网络错误，指数退避；inspect_ai 的重试策略按服务商更细。
        for attempt in range(_RETRIES):
            try:
                with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
                    return cast(dict[str, Any], json.loads(response.read()))
            except urllib.error.HTTPError as error:
                retryable = error.code == 429 or error.code >= 500
                if not retryable or attempt == _RETRIES - 1:
                    raise RuntimeError(
                        f"{error.code} {error.read().decode(errors='replace')}"
                    ) from error
            except (urllib.error.URLError, TimeoutError):
                if attempt == _RETRIES - 1:
                    raise
            time.sleep(min(60, 2**attempt) + random.random())
        raise AssertionError("unreachable")


def _openai_message(message: ChatMessage) -> dict[str, Any]:
    # 同 inspect_ai openai_chat_message（system_role="system"，无缓存断点）。
    if message.role in ("system", "user"):
        if isinstance(message.content, str):
            return {"role": message.role, "content": message.content}
        return {
            "role": message.role,
            "content": [
                {"type": "text", "text": c.text}
                for c in message.content
                if c.type == "text"
            ],
        }
    if isinstance(message, ChatMessageAssistant):
        content, extra = _assistant_content(message)
        param: dict[str, Any] = {"role": "assistant", "content": content}
        if message.tool_calls:
            param["tool_calls"] = [
                {
                    "id": call.id,
                    "function": {
                        "name": call.function,
                        "arguments": json.dumps(call.arguments),
                    },
                    "type": "function",
                }
                for call in message.tool_calls
            ]
        return param | extra
    assert isinstance(message, ChatMessageTool)
    return {
        "role": "tool",
        "content": f"Error: {message.error.message}" if message.error else message.text,
        "tool_call_id": str(message.tool_call_id),
    }


def _assistant_content(message: ChatMessageAssistant) -> tuple[str, dict[str, Any]]:
    # 同 inspect_ai openai_assistant_content + default_reasoning_handler：
    # str 原样；列表时每段前加换行，来自 reasoning_content / reasoning 字段的推理放回同名字段。
    if isinstance(message.content, str):
        return message.content, {}
    content = ""
    extra: dict[str, Any] = {}
    for c in message.content:
        if c.type == "reasoning":
            if str(c.internal) in (
                "reasoning",
                "reasoning_content",
                "reasoning_details",
            ):
                extra = {**extra, str(c.internal): c.reasoning}
            else:
                content = f"{content}\n<think>\n{c.reasoning}\n</think>\n"
        elif c.type == "text":
            content = f"{content}\n{c.text}"
    return content, extra


def _openai_tool(info: Any) -> dict[str, Any]:
    from ._json import JSON_SCHEMA_EXTENDED_FIELDS, json_schema_dump

    # 同 OpenAICompatibleAPI.tools_to_openai：排除扩展校验字段，strict=True。
    return {
        "type": "function",
        "function": {
            "name": info.name,
            "description": info.description,
            "parameters": json_schema_dump(
                info.parameters, exclude=JSON_SCHEMA_EXTENDED_FIELDS
            ),
            "strict": True,
        },
    }


def _openai_tool_choice(tool_choice: Any) -> Any:
    if tool_choice in (None, "auto"):
        return "auto"
    if tool_choice == "any":
        return "required"
    if tool_choice == "none":
        return "none"
    return {"type": "function", "function": {"name": tool_choice.name}}


def _model_output(response: dict[str, Any]) -> ModelOutput:
    from .tool import parse_tool_call

    choices = []
    for choice in response.get("choices", []):
        raw = choice.get("message") or {}
        text = str(raw.get("refusal") or raw.get("content") or "")
        # 同 chat_message_assistant_from_openai / parse_reasoning_content。
        # ponytail: 不解析正文里的 <think> 标签与 OpenRouter reasoning_details 列表；deepseek 用 reasoning_content。
        reasoning = next(
            (
                (source, raw[source])
                for source in ("reasoning_details", "reasoning_content", "reasoning")
                if raw.get(source)
            ),
            None,
        )
        content: str | list[Content]
        if reasoning is not None:
            content = [
                ContentReasoning(reasoning=str(reasoning[1]), internal=reasoning[0]),
                ContentText(text=text, refusal=True if raw.get("refusal") else None),
            ]
        elif raw.get("refusal") is not None:
            content = [ContentText(text=text, refusal=True)]
        else:
            content = text
        calls = [
            parse_tool_call(
                call["id"], call["function"]["name"], call["function"].get("arguments")
            )
            for call in raw.get("tool_calls") or []
        ]
        message = ChatMessageAssistant(
            content=content,
            model=response.get("model"),
            source="generate",
            tool_calls=calls or None,
        )
        choices.append(
            ChatCompletionChoice(
                message=message, stop_reason=as_stop_reason(choice.get("finish_reason"))
            )
        )
    usage = response.get("usage") or {}
    return ModelOutput(
        model=response.get("model") or "",
        choices=choices,
        usage=ModelUsage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        ),
    )


def as_stop_reason(reason: str | None) -> StopReason:
    # 同 inspect_ai model/_model_output.py as_stop_reason。
    match reason:
        case "stop" | "eos":
            return "stop"
        case "length":
            return "max_tokens"
        case "tool_calls" | "function_call":
            return "tool_calls"
        case "content_filter" | "model_length" | "max_tokens":
            return reason
        case _:
            return "unknown"


# ---- 模型解析 ----

_DEFAULT_MODEL_ENV = "INSPECT_EVAL_MODEL"


def get_model(model: str | Model | None = None, *, role: str | None = None) -> Model:
    """`openai-api/<服务>/<模型>` → Model；None 取 `INSPECT_EVAL_MODEL`（与 inspect_ai 相同）。"""
    if isinstance(model, Model):
        return model
    model = model or os.environ.get(_DEFAULT_MODEL_ENV)
    if not model:
        raise ValueError(
            "No model specified (and no model environment variable defined)"
        )
    provider, _, rest = model.partition("/")
    if provider != "openai-api" or "/" not in rest:
        raise ValueError(
            f"sounding only supports openai-api/<service>/<model>, got {model!r}"
        )
    service, _, name = rest.partition("/")
    prefix = service.upper().replace("-", "_")
    base_url = os.environ.get(f"{prefix}_BASE_URL")
    api_key = os.environ.get(f"{prefix}_API_KEY")
    if not base_url or not api_key:
        raise ValueError(
            f"{prefix}_BASE_URL and {prefix}_API_KEY must be set for {model!r}"
        )
    return Model(
        name=name, base_url=base_url, api_key=api_key, service=f"openai-api/{service}"
    )


@dataclass
class ModelInfo:
    input_tokens: int | None = None


def get_model_info(model: str | Model) -> ModelInfo | None:
    """同 inspect_ai get_model_info：Model 按规范名（服务里的模型名）查资料库，字符串直接查。"""
    from ._model_info import _get_model_info_db, _lookup_in_db

    name = model.name if isinstance(model, Model) else model
    input_tokens = _lookup_in_db(name, _get_model_info_db())
    return None if input_tokens is None else ModelInfo(input_tokens=input_tokens)


def count_text_tokens(text: str) -> int:
    # 同 inspect_ai model/_tokens.py：o200k_base 计数加 10% 余量。
    # tiktoken 首次使用时下载词表，默认缓存在 /tmp（重启即丢）；改到持久目录，只下载一次。
    os.environ.setdefault(
        "TIKTOKEN_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "sounding", "tiktoken"),
    )
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")
    return max(1, int(len(enc.encode(text, disallowed_special=())) * 1.1))


# inspect_ai 从 model 导出 execute_tools；放在末尾，避免与 tool 循环导入。
from .tool import execute_tools as execute_tools  # noqa: E402
