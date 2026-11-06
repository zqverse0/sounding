"""inspect_ai 零散工具的替身（行为与原实现相同，出处见各注释）。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from logging import Logger
from typing import Any, Awaitable, Callable, Literal, TypeAlias

from pydantic import BaseModel, Field
from pydantic_core import to_jsonable_python

from ._json import JSONSchema as JSONSchema  # inspect_ai.util 导出 JSONSchema

ValueToFloat: TypeAlias = Callable[[Any], float]


class Reference(BaseModel):
    """同 inspect_ai scorer Reference。"""

    type: Literal["message", "event"]
    cite: str | None = Field(default=None)
    id: str


class PrerequisiteError(RuntimeError):
    pass


def jsonable_python(x: Any) -> Any:
    # 同 inspect_ai _util/json.py。
    return to_jsonable_python(x, exclude_none=True, fallback=lambda _x: None)


def to_json_str_safe(x: Any) -> str:
    return json.dumps(jsonable_python(x))


_warned: set[str] = set()


def warn_once(logger: Logger, message: str) -> None:
    if message not in _warned:
        _warned.add(message)
        logger.warning(message)


async def tg_collect(funcs: list[Callable[[], Awaitable[Any]]]) -> list[Any]:
    return list(await asyncio.gather(*(f() for f in funcs)))


_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def auto_id(base: str, index: str) -> str:
    # 同 inspect_ai analysis auto_id：md5 → UUID → shortuuid.encode（int_to_string，补足 22 位）。
    number = uuid.UUID(bytes=hashlib.md5(f"{base}_{index}".encode()).digest()).int
    output = ""
    while number:
        number, digit = divmod(number, len(_ALPHABET))
        output += _ALPHABET[digit]
    output += _ALPHABET[0] * max(22 - len(output), 0)
    return output[::-1]
