"""`sounding scan`：按问题配置分析一个会话，结果 JSON 打到 stdout。

会话来源二选一：
- `--file <路径|->`：OpenAI chat 格式的消息，`[...]` 或 `{"messages": [...], "session_id"?, "agent"?}`；
- `<会话 id>`：从 Berth 读（可选来源，见 `berth.py`）。

配置写法见 `analyze.py`。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from importlib.metadata import version
from typing import Any

from ._transcript.types import Transcript
from .analyze import analyze
from .chat import chat_messages


def _read(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sounding")
    parser.add_argument(
        "--version", action="version", version=f"sounding {version('sounding')}"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    scan_parser = sub.add_parser("scan", help="分析一个会话，结果 JSON 输出到 stdout")
    scan_parser.add_argument(
        "session", nargs="?", help="Berth 会话 id（`berth project <p> session ls`）"
    )
    scan_parser.add_argument(
        "--file", help="OpenAI chat 格式的会话消息 JSON 文件；`-` 从 stdin 读"
    )
    scan_parser.add_argument(
        "--config", required=True, help="问题配置 JSON 文件；`-` 从 stdin 读"
    )
    scan_parser.add_argument(
        "--project", default=".", help="（Berth 来源）项目名称、id 或路径"
    )
    scan_parser.add_argument("--agent", help="（Berth 来源）只在这个工具 id 的会话里找")
    scan_parser.add_argument("--model", help="覆盖配置里的对话模型")
    args = parser.parse_args(argv)

    if bool(args.session) == bool(args.file):
        parser.error("give exactly one of: a Berth session id, or --file")
    if args.file == "-" and args.config == "-":
        parser.error("--file and --config cannot both read stdin")
    config = _read(args.config)
    if args.file:
        shown = _read(args.file)
        if isinstance(shown, list):
            shown = {"messages": shown}
        transcript = Transcript(
            transcript_id=shown.get("session_id")
            or ("stdin" if args.file == "-" else os.path.basename(args.file)),
            agent=shown.get("agent"),
            messages=chat_messages(shown["messages"]),
        )
    else:
        from .berth import load_session

        transcript = load_session(args.session, project=args.project, agent=args.agent)
    output = asyncio.run(analyze(transcript, config, args.model))
    json.dump(output, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
