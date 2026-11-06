"""把 inspect_scout 与 sounding 包成同名接口，供场景调用。"""

from __future__ import annotations

import types
from typing import Any


def scout() -> Any:
    import inspect_scout
    from inspect_ai.model import get_model
    from inspect_scout import AnswerMultiLabel, AnswerStructured, MessagesPreprocessor
    from inspect_scout.sources._berth.transcripts import _chat_messages

    return types.SimpleNamespace(
        name="inspect_scout",
        llm_scanner=inspect_scout.llm_scanner,
        grep_scanner=inspect_scout.grep_scanner,
        Transcript=inspect_scout.Transcript,
        AnswerMultiLabel=AnswerMultiLabel,
        AnswerStructured=AnswerStructured,
        MessagesPreprocessor=MessagesPreprocessor,
        get_model=get_model,
        chat_messages=_chat_messages,
    )


def lite() -> Any:
    import sounding
    from sounding.chat import chat_messages

    return types.SimpleNamespace(
        name="sounding",
        llm_scanner=sounding.llm_scanner,
        grep_scanner=sounding.grep_scanner,
        Transcript=sounding.Transcript,
        AnswerMultiLabel=sounding.AnswerMultiLabel,
        AnswerStructured=sounding.AnswerStructured,
        MessagesPreprocessor=sounding.MessagesPreprocessor,
        get_model=sounding.get_model,
        chat_messages=chat_messages,
    )
