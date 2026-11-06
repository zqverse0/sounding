"""配置 → 扫描器：整体问题（llm / grep）用 inspect_scout 的扫描器，逐条判断（jev）见 `jev.py`。

这里只做「配置 → 扫描器」的翻译，整体问题的分析全部是摘取的 scout 代码。配置（JSON）：

    {
      "model": "openai-api/openai/deepseek-flash",
      "questions": [
        {"id": "fix", "type": "llm", "question": "用户是否纠正了 Agent？", "answer": "boolean"},
        {"id": "outcome", "type": "llm", "question": "结果？",
         "answer": {"labels": ["已解决", "未解决"], "multiple": false, "allowNone": false}},
        {"id": "issues", "type": "llm", "question": "列出问题",
         "answer": {"fields": [{"name": "title", "type": "string", "description": "一句话"}], "multiple": true},
         "exclude": {"toolUsage": true}},
        {"id": "errors", "type": "grep", "patterns": ["error", "失败"], "regex": false},
        {"id": "corrected", "type": "jev", "unit": "user", "question": "是否在纠正 Agent？",
         "criteria": {"true": "指出 Agent 做错并要求改", "false": "新要求、提问或确认"}},
        {"id": "how", "type": "llm", "question": "用户纠正了 Agent 什么？", "focus": "corrected",
         "answer": {"fields": [{"name": "wrong", "type": "string", "description": "错在哪"}], "multiple": true}}
      ]
    }

`answer` 取 "boolean" / "numeric" / "string"、选项（labels）或结构化字段（fields，multiple 时为多条结果）。
对话模型写作 `openai-api/<服务>/<模型>`，读 `<服务>_BASE_URL` / `<服务>_API_KEY`（如 OPENAI_*）。

`jev` 问题是逐条判断（见 `jev.py`）：先于 llm 问题执行；llm 问题的 `focus` 指向一个 jev 问题时，
把它命中的位置附在问题后面交给 scout 重点核对。Jev 读 TYPESAFE_API_KEY / TYPESAFE_BASE_URL（TypeSafe 官方 SDK 的变量名）。
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from typing import Any

from pydantic import Field, create_model

from . import jev
from ._ai.model import get_model
from ._grep_scanner._grep_scanner import grep_scanner
from ._llm_scanner._llm_scanner import llm_scanner
from ._llm_scanner.types import AnswerMultiLabel, AnswerStructured
from ._scanner.extract import MessagesPreprocessor
from ._transcript.types import Transcript
from .chat import chat_messages

_FIELD_TYPES: dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
}


def _answer(spec: Any) -> Any:
    if isinstance(spec, str):
        return spec
    if "labels" in spec:
        if spec.get("multiple"):
            return AnswerMultiLabel(
                labels=spec["labels"], allow_none=bool(spec.get("allowNone"))
            )
        return list(spec["labels"])
    fields = {
        f["name"]: (
            _FIELD_TYPES[f.get("type", "string")],
            Field(description=f["description"]),
        )
        for f in spec["fields"]
    }
    model = create_model(spec.get("name") or "Answer", **fields)  # type: ignore[call-overload]
    return AnswerStructured(type=list[model] if spec.get("multiple") else model)  # type: ignore[valid-type]


def _scanner(question: dict[str, Any], model: Any, nonce: str, note: str = "") -> Any:
    if question.get("type", "llm") == "grep":
        return grep_scanner(
            question["patterns"],
            regex=bool(question.get("regex")),
            ignore_case=question.get("ignoreCase", True),
            word_boundary=bool(question.get("wordBoundary")),
        )
    exclude = question.get("exclude") or {}
    return llm_scanner(
        question=question["question"] + note,
        answer=_answer(question["answer"]),
        preprocessor=MessagesPreprocessor(
            exclude_system=exclude.get("system", True),
            exclude_reasoning=bool(exclude.get("reasoning")),
            exclude_tool_usage=bool(exclude.get("toolUsage")),
        ),
        model=model,
        nonce=nonce,
    )


def _dump(result: Any) -> Any:
    if isinstance(result, list):
        return [_dump(r) for r in result]
    return json.loads(result.model_dump_json())


async def analyze(
    session: Transcript | list[dict[str, Any]],
    config: dict[str, Any],
    model_name: str | None = None,
) -> dict[str, Any]:
    """按问题配置分析一个会话。`session` 是 Transcript，或 OpenAI chat 格式的消息列表。

    配置写法见模块说明；对话模型读 `<服务>_BASE_URL` / `<服务>_API_KEY`（`openai-api/openai/<模型>` 即
    OPENAI_BASE_URL / OPENAI_API_KEY），逐条判断读 TYPESAFE_API_KEY / TYPESAFE_BASE_URL（见 `jev.py`）。
    """
    transcript = (
        session
        if isinstance(session, Transcript)
        else Transcript(transcript_id="session", messages=chat_messages(session))
    )
    questions = config.get("questions") or []
    needs_model = any(q.get("type", "llm") == "llm" for q in questions)
    model = None
    if needs_model:
        model_name = model_name or config.get("model")
        model = get_model(model_name)
        # 与 scout 的 scan(model=...) 相同：不带 model 的调用（多段文本答案的 LLM 合并）用它。
        os.environ.setdefault("INSPECT_EVAL_MODEL", model_name or "")
    # scout 建议每次构建扫描器时随机生成，会话里的内容无法预知、也就无法伪造消息头。
    nonce = secrets.token_hex(4)

    def failed(question: dict[str, Any], error: Exception) -> dict[str, Any]:
        return {
            "id": question.get("id"),
            "result": None,
            "error": f"{type(error).__name__}: {error}",
        }

    async def judge(question: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await asyncio.to_thread(
                jev.judge, transcript.messages, question, config.get("jevModel")
            )
            return {"id": question.get("id"), "result": result, "error": None}
        except Exception as error:  # 一个问题失败不影响其他问题
            return failed(question, error)

    # 逐条判断先跑：它的命中要交给 focus 指向它的 llm 问题
    judged = {
        r["id"]: r
        for r in await asyncio.gather(
            *(judge(q) for q in questions if q.get("type") == "jev")
        )
    }

    async def one(question: dict[str, Any]) -> dict[str, Any]:
        if question.get("type") == "jev":
            return judged[question.get("id")]
        focus = judged.get(question.get("focus")) if question.get("focus") else None
        note = jev.focus_note(focus["result"]) if focus and focus["result"] else ""
        try:
            result = await _scanner(question, model, nonce, note)(transcript)
            return {"id": question.get("id"), "result": _dump(result), "error": None}
        except Exception as error:  # 一个问题失败不影响其他问题
            return failed(question, error)

    results = await asyncio.gather(*(one(q) for q in questions))
    return {
        "session": transcript.transcript_id,
        "agent": transcript.agent,
        "model": str(model) if model else None,
        "messages": len(transcript.messages),
        "results": results,
    }
