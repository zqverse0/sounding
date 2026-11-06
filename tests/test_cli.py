"""sounding scan：配置 → 扫描器的翻译、每题独立出错、输出形状。"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from mock_openai import MockOpenAI, text_reply, tool_reply  # noqa: E402
from scenarios import SESSION  # noqa: E402

from sounding import Transcript  # noqa: E402
from sounding.analyze import analyze  # noqa: E402
from sounding.chat import chat_messages  # noqa: E402

CONFIG = {
    "model": "openai-api/berth/deepseek-flash",
    "questions": [
        {"id": "fix", "question": "用户是否纠正了 Agent？", "answer": "boolean"},
        {
            "id": "count",
            "question": "几次工具调用？",
            "answer": "numeric",
            "exclude": {"toolUsage": True},
        },
        {
            "id": "outcome",
            "question": "结果？",
            "answer": {"labels": ["已解决", "未解决"]},
        },
        {
            "id": "kinds",
            "question": "问题类型？",
            "answer": {"labels": ["幻觉", "掩盖"], "multiple": True, "allowNone": True},
        },
        {
            "id": "issues",
            "question": "列出问题",
            "answer": {
                "multiple": True,
                "fields": [
                    {"name": "title", "type": "string", "description": "一句话"},
                    {"name": "severity", "type": "integer", "description": "1-3"},
                ],
            },
        },
        {
            "id": "bad",
            "question": "x",
            "answer": {
                "fields": [{"name": "a", "type": "nosuchtype", "description": "d"}]
            },
        },
        {"id": "errors", "type": "grep", "patterns": ["TypeError"]},
    ],
}


def reply(body: dict) -> dict:
    prompt = "".join(p["text"] for p in body["messages"][0]["content"])
    if body.get("tools"):
        return tool_reply(
            "answer",
            {"results": [{"title": "掩盖", "severity": 3, "explanation": "[M4]"}]},
        )
    if "几次工具调用" in prompt:
        assert (
            "pnpm test login" not in prompt
        )  # exclude.toolUsage 生效：工具调用不进提示
        return text_reply("ANSWER: 2")
    if "yes or no" in prompt:
        return text_reply("[M7]\nANSWER: yes")
    if "comma-separated" in prompt:
        return text_reply("ANSWER: B")
    return text_reply("ANSWER: B")


def test_scan_translates_every_question_kind(monkeypatch) -> None:
    mock = MockOpenAI(reply)
    monkeypatch.setenv("BERTH_BASE_URL", mock.base_url)
    monkeypatch.setenv("BERTH_API_KEY", "k")
    try:
        transcript = Transcript(
            transcript_id="s1", agent="claude", messages=chat_messages(SESSION)
        )
        output = asyncio.run(analyze(transcript, CONFIG, None))
    finally:
        mock.close()
    by_id = {r["id"]: r for r in output["results"]}
    assert output["model"] == "openai-api/berth/deepseek-flash" and output[
        "messages"
    ] == len(SESSION)
    assert by_id["fix"]["result"]["value"] is True
    assert by_id["fix"]["result"]["references"][0]["id"] == "m7"
    assert by_id["count"]["result"]["value"] == 2.0
    assert by_id["outcome"]["result"]["answer"] == "未解决"
    assert by_id["kinds"]["result"]["value"] == ["掩盖"]
    issues = by_id["issues"]["result"]
    assert issues["type"] == "resultset" and issues["value"][0]["value"] == {
        "title": "掩盖",
        "severity": 3,
    }
    assert by_id["bad"]["result"] is None and "KeyError" in by_id["bad"]["error"]
    assert by_id["errors"]["result"]["value"] == 1
    json.dumps(output, ensure_ascii=False)
