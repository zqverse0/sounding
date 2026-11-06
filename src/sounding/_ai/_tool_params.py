# 摘自 inspect_ai 0.3.287.dev30+g7340817a6：inspect_ai/tool/_tool_params.py（MIT）。改动只限导入路径；同步上游时整文件替换。
from typing import (
    Literal,
    Optional,
    TypeAlias,
)

from pydantic import BaseModel, Field

from ._json import JSONSchema

ToolParam: TypeAlias = JSONSchema
"""Description of tool parameter in JSON Schema format."""


class ToolParams(BaseModel):
    """Description of tool parameters object in JSON Schema format."""

    type: Literal["object"] = Field(default="object")
    """Params type (always 'object')"""

    properties: dict[str, ToolParam] = Field(default_factory=dict)
    """Tool function parameters."""

    required: list[str] = Field(default_factory=list)
    """List of required fields."""

    additionalProperties: Optional[JSONSchema] | bool = Field(default=False)
    """Are additional object properties allowed?"""
