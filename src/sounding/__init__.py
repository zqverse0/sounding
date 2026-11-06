"""sounding：Agent 会话分析——inspect_scout 单会话扫描的精简版，加 Jev 逐条判断（`jev.py`）。

入口：`await analyze(messages, config)`（messages 为 OpenAI chat 格式），或命令行 `sounding scan --file`。

扫描器的提示、渲染、解析、分段与合并摘自 inspect_scout（见 UPSTREAM.md），对同一会话发出的请求与
得到的结果与 inspect_scout 一致（tests/golden 对照）。去掉了会话库、批量扫描、结果存储与界面。
"""

from . import jev
from ._ai.model import get_model
from ._grep_scanner._grep_scanner import grep_scanner
from ._llm_scanner._llm_scanner import llm_scanner
from ._llm_scanner._reducer import ResultReducer
from ._llm_scanner.generate import generate_answer
from ._llm_scanner.types import AnswerMultiLabel, AnswerStructured
from ._scanner.extract import MessagesPreprocessor, message_numbering, messages_as_str
from ._scanner.result import Reference, Result
from ._transcript.types import Transcript
from .analyze import analyze
from .chat import chat_messages

__all__ = [
    "AnswerMultiLabel",
    "AnswerStructured",
    "MessagesPreprocessor",
    "Reference",
    "Result",
    "ResultReducer",
    "Transcript",
    "analyze",
    "chat_messages",
    "generate_answer",
    "get_model",
    "grep_scanner",
    "jev",
    "llm_scanner",
    "message_numbering",
    "messages_as_str",
]
