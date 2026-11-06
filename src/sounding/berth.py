"""可选来源：从 Berth 读一个会话（`berth project <p> session show <id> --messages`）→ Transcript。

只有 `sounding scan <会话 id>` 用到；`BERTH_CLI` 可覆盖 berth 可执行文件。不用 Berth 时改用 `--file`。
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from ._transcript.types import Transcript
from .chat import chat_messages


def load_session(
    session_id: str, *, project: str = ".", agent: str | None = None
) -> Transcript:
    args = ["session", "show", session_id, "--messages"] + (
        ["--agent", agent] if agent else []
    )
    shown = _berth(project, args)
    return Transcript(
        transcript_id=session_id,
        source_type="berth",
        source_id=session_id,
        source_uri=shown.get("path"),
        task_set=shown.get("cwd"),
        agent=shown.get("agent"),
        messages=chat_messages(shown["messages"]),
        message_count=len(shown["messages"]),
    )


def _berth(project: str, args: list[str]) -> Any:
    binary = os.environ.get("BERTH_CLI", "berth")
    result = subprocess.run(
        [binary, "project", project, *args], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout.strip() or result.stderr.strip())
    return json.loads(result.stdout)
