# 摘自 inspect_scout 7521262 src/inspect_scout/_scanner/result.py（MIT）：只保留 Result 与 as_resultset。
from typing import Any, Generic

from pydantic import BaseModel, Field, JsonValue
from pydantic.json_schema import SkipJsonSchema
from typing_extensions import TypeVar

from .._ai.util import Reference, jsonable_python

__all__ = ["ParsedT", "Reference", "Result", "as_resultset"]


def _uuid() -> str:
    import uuid

    return uuid.uuid4().hex


ParsedT = TypeVar("ParsedT", default=Any)
"""Type of `Result.parsed` (defaults to `Any` so bare `Result` annotations remain valid)."""


class Result(BaseModel, Generic[ParsedT]):
    """Scan result."""

    uuid: str | None = Field(default_factory=_uuid)
    """Unique identifer for scan result."""

    value: JsonValue
    """Scan value."""

    answer: str | None = Field(default=None)
    """Answer extracted from model output (optional)"""

    explanation: str | None = Field(default=None)
    """Explanation of result (optional)."""

    metadata: dict[str, Any] | None = Field(default=None)
    """Additional metadata related to the result (optional)"""

    references: list[Reference] = Field(default_factory=list)
    """References to relevant messages or events."""

    label: str | None = Field(default=None)
    """Label for result to indicate its origin."""

    type: str | None = Field(default=None)
    """Type to designate contents of 'value' (used in `value_type` field in result data frames)."""

    parsed: SkipJsonSchema[ParsedT | None] = Field(default=None, exclude=True)
    """Typed parsed answer (in-memory only, excluded from serialization)."""


def as_resultset(results: list[Result]) -> Result:
    return Result(value=jsonable_python(results), type="resultset")
