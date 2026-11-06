# 摘自 inspect_scout 7521262 src/inspect_scout/_util/refusal.py（MIT）。改动：导入改指 sounding 的替身。
from typing import Sequence

from .._ai.model import ChatMessage, GenerateConfig, Model, ModelOutput
from .._ai.tool import Tool, ToolChoice, ToolDef, ToolInfo, ToolSource


class RefusalError(RuntimeError):
    """Error indicating that the model refused a scan request."""

    pass


async def generate_retry_refusals(
    model: Model,
    input: str | list[ChatMessage],
    tools: Sequence[Tool | ToolDef | ToolInfo | ToolSource] | ToolSource,
    tool_choice: ToolChoice | None,
    config: GenerateConfig | None,
    retry_refusals: int,
) -> ModelOutput:
    refusals = 0
    while True:
        # run the generation
        output = await model.generate(
            input=input,
            tool_choice=tool_choice,
            tools=tools,
            config=config or GenerateConfig(),
        )

        # check for refusal and retry if needed (else raise error)
        if output.stop_reason == "content_filter":
            if refusals < retry_refusals:
                refusals += 1
            else:
                raise RefusalError(
                    f"Scanner request refused by content filter: {output.completion}"
                )
        else:
            return output
