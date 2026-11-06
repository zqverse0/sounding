"""逐条判断：用 Jev（System One 决策模型）对会话里的每条用户消息或每个工具调用做局部判断。

scout 擅长通读整条轨迹、推理后作答，但慢、每次找到的不完全一样；Jev 一次只看一小段，快、便宜、同样输入
给出几乎相同的概率。分工是「Jev 先筛，scout 再深查」：判断结果可以作为 llm 问题的 `focus`，
把概率高的位置交给 scout 重点核对。

连接用 TypeSafe 官方 SDK 的变量名：TYPESAFE_API_KEY（必需）、TYPESAFE_BASE_URL（API 根地址，默认
https://api.typesafe.ai，不带 `/v1`）、TYPESAFE_DEFAULT_MODEL（默认 jev-latest；配置的 `jevModel` 优先）。
协议：POST `<根地址>/v1/systemone`，`{"model", "state", "questions"}` → `answers.<id>.noul`（「是」的概率）。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ._ai.model import ChatMessage

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"
# 每段原文保留的字符数；Jev 单次输入上限约 32k token，局部判断远用不到。
CONTEXT_CHARS = 1500
WORKERS = 6
ATTEMPTS = 3
# llm 问题的 focus 最多列出这么多位置。
FOCUS_LIMIT = 15

UNITS = ("user", "action")


def items(messages: list[ChatMessage], unit: str) -> list[dict[str, Any]]:
    """要判断的对象，各带一小段上下文（不用模型）。

    - user：每条有正文的用户消息（会话第一条之前没有 Agent 回复，跳过），带 Agent 上一条回复；
    - action：每个工具调用，带 Agent 动手前说的话与上一个工具结果。
    """
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
    out: list[dict[str, Any]] = []
    reply = result = ""
    for message in messages:
        text = message.text
        if message.role == "assistant":
            if text.strip():
                reply = text
            for call in getattr(message, "tool_calls", None) or []:
                if unit == "action":
                    out.append(
                        {
                            "id": message.id,
                            "text": f"{call.function} {json.dumps(call.arguments, ensure_ascii=False)}",
                            "state": {
                                "action": {
                                    "tool": call.function,
                                    "arguments": json.dumps(
                                        call.arguments, ensure_ascii=False
                                    )[:CONTEXT_CHARS],
                                },
                                "agent_said": reply[-CONTEXT_CHARS:],
                                "previous_result": result[-CONTEXT_CHARS:],
                            },
                        }
                    )
        elif message.role == "tool":
            result = text
        # `<…>` 开头的是 Agent 注入的提醒 / 命令回显，不是用户说的话
        elif (
            message.role == "user"
            and text.strip()
            and not text.lstrip().startswith("<")
        ):
            if unit == "user" and reply:
                out.append(
                    {
                        "id": message.id,
                        "text": text,
                        "state": {
                            "agent_reply_before": reply[-CONTEXT_CHARS:],
                            "user_message": text[:CONTEXT_CHARS],
                        },
                    }
                )
            reply = ""  # 新的一问：之前的回复不再算动手理由
    return out


def _instructions(question: dict[str, Any]) -> str:
    target = (
        "user_message（agent_reply_before 是 Agent 上一条回复）"
        if question["unit"] == "user"
        else "action（agent_said 是 Agent 动手前说的话，previous_result 是上一个工具结果）"
    )
    return f"针对 {target}：{question['question']}"


def _call(url: str, token: str, body: dict[str, Any]) -> float:
    request = urllib.request.Request(
        url,
        json.dumps(body).encode(),
        {"Authorization": f"Bearer {token}", "content-type": "application/json"},
    )
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return float(json.load(response)["answers"]["c"]["noul"])
        except urllib.error.HTTPError as error:
            # 4xx 是题目或输入不合要求，重试无用
            if error.code < 500 and error.code != 429:
                detail = error.read().decode(errors="replace")[:300]
                raise RuntimeError(f"Jev HTTP {error.code}: {detail}") from error
            if attempt == ATTEMPTS - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == ATTEMPTS - 1:
                raise
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def judge(
    messages: list[ChatMessage], question: dict[str, Any], model: str | None = None
) -> dict[str, Any]:
    """逐条判断一个问题。返回会话结论 `value`（有一条概率 ≥ threshold）与命中的位置（概率从高到低）。"""
    base = os.environ.get("TYPESAFE_BASE_URL") or DEFAULT_BASE_URL
    token = os.environ.get("TYPESAFE_API_KEY")
    if not token:
        raise RuntimeError(
            "逐条判断需要 Jev：设置 TYPESAFE_API_KEY（可选 TYPESAFE_BASE_URL）"
        )
    url = base.rstrip("/") + "/v1/systemone"
    criteria = question["criteria"]
    if (
        not str(criteria.get("true", "")).strip()
        or not str(criteria.get("false", "")).strip()
    ):
        # Jev 在标准没写清时给 ~0.5，没有区分度
        raise ValueError("criteria 需要同时写清 true 与 false")
    threshold = float(question.get("threshold", 0.5))
    questions = {
        "c": {
            "type": "noul",
            "instructions": _instructions(question),
            "criteria": criteria,
        }
    }
    targets = items(messages, question["unit"])
    with ThreadPoolExecutor(WORKERS) as pool:
        scores = list(
            pool.map(
                lambda item: _call(
                    url,
                    token,
                    {
                        "model": model
                        or os.environ.get("TYPESAFE_DEFAULT_MODEL")
                        or DEFAULT_MODEL,
                        "state": item["state"],
                        "questions": questions,
                    },
                ),
                targets,
            )
        )
    hits = sorted(
        (
            {"id": item["id"], "p": round(p, 4), "text": item["text"][:200]}
            for item, p in zip(targets, scores, strict=True)
            if p >= threshold
        ),
        key=lambda hit: -hit["p"],
    )
    return {
        "value": bool(hits),
        "max": round(max(scores), 4) if scores else None,
        "checked": len(targets),
        "threshold": threshold,
        "hits": hits,
    }


def focus_note(result: dict[str, Any]) -> str:
    """把逐条判断的命中写成 llm 问题的补充说明：先核对这些位置，再通读全程补上遗漏。"""
    hits = result["hits"][:FOCUS_LIMIT]
    if not hits:
        return ""
    lines = "\n".join(
        f"- {hit['p']:.2f}「{' '.join(hit['text'].split())[:80]}」" for hit in hits
    )
    return (
        "\n\n预先逐条判断筛出的位置（概率与开头）：\n"
        f"{lines}\n先逐个核对这些位置；然后通读全程，补上没列出来的。只输出符合问题的结果，核对后不符合的不要列出。"
    )
