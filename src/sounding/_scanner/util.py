# 摘自 inspect_scout 7521262 src/inspect_scout/_scanner/util.py（MIT）：只保留 _message_id。
from .._ai.model import ChatMessage
from .._ai.util import auto_id


def _message_id(message: ChatMessage) -> str:
    return message.id or auto_id("message", message.text)
