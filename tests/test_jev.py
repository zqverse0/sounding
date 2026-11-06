"""逐条判断：取对象与上下文、调 Jev、会话结论与命中、focus 交给 llm 问题。"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from mock_openai import MockOpenAI, text_reply  # noqa: E402
from scenarios import SESSION  # noqa: E402

from sounding import Transcript, jev  # noqa: E402
from sounding.analyze import analyze  # noqa: E402
from sounding.chat import chat_messages  # noqa: E402

CRITERIA = {"true": "指出 Agent 做错并要求改", "false": "新要求、提问或确认"}


class MockJev:
    """假 systemone 端点：用户消息里有「不对」判 0.9，其余 0.1；记录请求。"""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                mock.requests.append(
                    {
                        "path": self.path,
                        "auth": self.headers["Authorization"],
                        "body": body,
                    }
                )
                p = (
                    0.9
                    if "不对" in json.dumps(body["state"], ensure_ascii=False)
                    else 0.1
                )
                data = json.dumps(
                    {"model": "jev-test", "answers": {"c": {"type": "noul", "noul": p}}}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()


def test_items_carry_local_context() -> None:
    messages = chat_messages(SESSION)
    users = jev.items(messages, "user")
    # 第一条用户消息之前没有 Agent 回复，不判
    assert [u["id"] for u in users] == ["m7"]
    assert users[0]["state"]["agent_reply_before"] == "改好了：给 token 加了可选链。"
    actions = jev.items(messages, "action")
    assert [a["id"] for a in actions] == ["m2", "m4"]
    assert actions[0]["state"]["agent_said"] == "先看一下报错日志。"
    assert "TypeError" in actions[1]["state"]["previous_result"]


def test_scan_runs_jev_first_and_hands_hits_to_focus(monkeypatch) -> None:
    jev_mock = MockJev()
    llm_prompts: list[str] = []

    def reply(body: dict) -> dict:
        llm_prompts.append("".join(p["text"] for p in body["messages"][0]["content"]))
        return text_reply("[M7]\nANSWER: yes")

    llm_mock = MockOpenAI(reply)
    monkeypatch.setenv("TYPESAFE_BASE_URL", jev_mock.url)
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    monkeypatch.setenv("BERTH_BASE_URL", llm_mock.base_url)
    monkeypatch.setenv("BERTH_API_KEY", "k")
    config = {
        "model": "openai-api/berth/deepseek-flash",
        "questions": [
            {
                "id": "fix",
                "question": "用户纠正了 Agent 吗？",
                "answer": "boolean",
                "focus": "corrected",
            },
            {
                "id": "corrected",
                "type": "jev",
                "unit": "user",
                "question": "是否在纠正 Agent？",
                "criteria": CRITERIA,
            },
            {
                "id": "bad",
                "type": "jev",
                "unit": "user",
                "question": "x",
                "criteria": {"true": "", "false": "y"},
            },
        ],
    }
    try:
        transcript = Transcript(
            transcript_id="s1", agent="claude", messages=chat_messages(SESSION)
        )
        output = asyncio.run(analyze(transcript, config, None))
    finally:
        jev_mock.close()
        llm_mock.close()
    by_id = {r["id"]: r for r in output["results"]}
    judged = by_id["corrected"]["result"]
    assert (
        judged["value"] is True
        and judged["checked"] == 1
        and judged["hits"][0]["id"] == "m7"
    )
    request = jev_mock.requests[0]
    assert request["path"] == "/v1/systemone" and request["auth"] == "Bearer ts-test"
    assert request["body"]["model"] == "jev-latest"
    assert request["body"]["questions"]["c"]["criteria"] == CRITERIA
    # focus：命中的位置附在 llm 问题后面
    assert "预先逐条判断筛出的位置" in llm_prompts[0] and "0.90「不对" in llm_prompts[0]
    assert by_id["fix"]["result"]["value"] is True
    assert by_id["bad"]["result"] is None and "criteria" in by_id["bad"]["error"]


def test_jev_without_api_key_fails_clearly(monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    try:
        jev.judge(
            chat_messages(SESSION),
            {"unit": "user", "question": "x", "criteria": CRITERIA},
        )
    except RuntimeError as error:
        assert "TYPESAFE_API_KEY" in str(error)
    else:
        raise AssertionError("expected RuntimeError")


def test_cli_runs_standalone_from_stdin(monkeypatch, capsys) -> None:
    """不经 Berth：消息从 stdin 进，模型与 Jev 都只靠各自官方 SDK 的环境变量。"""
    import io

    from sounding.cli import main

    jev_mock = MockJev()
    llm_mock = MockOpenAI(lambda body: text_reply("[M7]\nANSWER: yes"))
    for name in ("BERTH_CLI", "BERTH_PROVIDER_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", llm_mock.base_url)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("TYPESAFE_BASE_URL", jev_mock.url)
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    config = pathlib.Path(__file__).parent / "_standalone_config.json"
    config.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "id": "corrected",
                        "type": "jev",
                        "unit": "user",
                        "question": "纠正？",
                        "criteria": CRITERIA,
                    },
                    {
                        "id": "fix",
                        "question": "纠正了吗？",
                        "answer": "boolean",
                        "focus": "corrected",
                    },
                ]
            },
            ensure_ascii=False,
        )
    )
    monkeypatch.setattr(
        "sys.stdin", io.StringIO(json.dumps(SESSION, ensure_ascii=False))
    )
    try:
        main(
            [
                "scan",
                "--file",
                "-",
                "--config",
                str(config),
                "--model",
                "openai-api/openai/m",
            ]
        )
    finally:
        config.unlink()
        jev_mock.close()
        llm_mock.close()
    output = json.loads(capsys.readouterr().out)
    by_id = {r["id"]: r for r in output["results"]}
    assert output["session"] == "stdin"
    assert by_id["corrected"]["result"]["hits"][0]["id"] == "m7"
    assert by_id["fix"]["result"]["value"] is True
