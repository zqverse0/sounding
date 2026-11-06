"""跑一个场景：起假模型服务，记录请求，返回（请求列表, 归一化结果）。"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from mock_openai import MockOpenAI
from scenarios import MODEL, Scenario, transcript


def _strip_uuid(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_uuid(v) for k, v in value.items() if k != "uuid"}
    if isinstance(value, list):
        return [_strip_uuid(v) for v in value]
    return value


def normalize(result: Any) -> Any:
    if isinstance(result, list):
        return [normalize(r) for r in result]
    return _strip_uuid(json.loads(result.model_dump_json()))


def run(engine: Any, scenario: Scenario) -> dict[str, Any]:
    mock = MockOpenAI(scenario["replies"])
    os.environ["BERTH_BASE_URL"] = mock.base_url
    os.environ["BERTH_API_KEY"] = "test-key"
    # 默认模型（LLM 合并器等不带 model 的调用用它），与 scan(model=...) 的效果相同。
    os.environ["INSPECT_EVAL_MODEL"] = MODEL
    try:
        scanner = scenario["scanner"](engine)
        result = asyncio.run(scanner(transcript(engine)))
    finally:
        mock.close()
    requests = mock.requests
    if scenario.get("unordered"):
        # 分段并发扫描：请求到达顺序不定，按内容排序后再比。
        requests = sorted(
            requests, key=lambda r: json.dumps(r, ensure_ascii=False, sort_keys=True)
        )
    return {"requests": requests, "result": normalize(result)}
