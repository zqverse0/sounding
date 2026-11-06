"""本地假 OpenAI 兼容服务：记录每个请求体，按顺序回放预设回复。对照测试用。"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable


class MockOpenAI:
    def __init__(
        self,
        replies: list[dict[str, Any]] | Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        # 列表按顺序回放；函数按请求体决定回复（并发请求到达顺序不定时用）。
        self.replies = replies if callable(replies) else list(replies)
        self.requests: list[dict[str, Any]] = []
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                mock.requests.append({"path": self.path, "body": body})
                if callable(mock.replies):
                    reply = mock.replies(body)
                else:
                    reply = mock.replies.pop(0) if mock.replies else _text("ANSWER: no")
                data = json.dumps(
                    {
                        "id": "chatcmpl-mock",
                        "object": "chat.completion",
                        "created": 0,
                        "model": body.get("model"),
                        "choices": [{"index": 0, **reply}],
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                        },
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()


def _text(content: str, finish: str = "stop") -> dict[str, Any]:
    return {
        "message": {"role": "assistant", "content": content},
        "finish_reason": finish,
    }


def text_reply(content: str, finish: str = "stop") -> dict[str, Any]:
    return _text(content, finish)


def tool_reply(
    name: str, arguments: dict[str, Any], call_id: str = "call_1"
) -> dict[str, Any]:
    return {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        },
        "finish_reason": "tool_calls",
    }
