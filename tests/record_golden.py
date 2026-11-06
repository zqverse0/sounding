"""用真 inspect_scout 跑全部场景，写 golden/<场景>.json。基线变化（升级上游）时重跑。"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import engines  # noqa: E402
from runner import run  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402

GOLDEN = pathlib.Path(__file__).parent / "golden"
MODEL_INFO_NAMES = [
    "deepseek-flash",
    "deepseek-v4-pro",
    "deepseek-v4-flash-0731",
    "gpt-4o",
    "gpt-5.5",
    "claude-opus-4-8",
    "claude-sonnet-4-5-20250929",
    "kimi-k3",
    "qwen3.8-flash",
    "glm-5.3",
    "Qwen/Qwen3-235B-A22B",
    "gemini-2.5-pro",
    "grok-4.5",
    "no-such-model-xyz",
]

if __name__ == "__main__":
    import importlib.metadata as md

    GOLDEN.mkdir(exist_ok=True)
    if len(sys.argv) > 1:
        # 子进程：跑一个场景。inspect_ai 的 HTTP 客户端跨 asyncio.run 复用会卡住，所以一个场景一个进程。
        scenario = next(s for s in SCENARIOS if s["name"] == sys.argv[1])
        record = run(engines.scout(), scenario)
        (GOLDEN / f"{scenario['name']}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=1) + "\n"
        )
        print(scenario["name"], len(record["requests"]), "requests")
        sys.exit(0)
    # 模型资料库查找（决定分段预算）：Model 实例按规范名查，与 llm_scanner 的用法一致。
    import os
    import subprocess

    os.environ.setdefault("BERTH_BASE_URL", "http://127.0.0.1:9/v1")
    os.environ.setdefault("BERTH_API_KEY", "k")
    from inspect_ai.model import get_model, get_model_info

    lookups = {}
    for name in MODEL_INFO_NAMES:
        info = get_model_info(get_model(f"openai-api/berth/{name}"))
        lookups[name] = info.input_tokens if info else None
    (GOLDEN / "model_info.json").write_text(json.dumps(lookups, indent=1) + "\n")

    for scenario in SCENARIOS:
        subprocess.run(
            [sys.executable, __file__, scenario["name"]], check=True, timeout=120
        )
    (GOLDEN / "VERSION").write_text(
        f"inspect_scout {md.version('inspect_scout')}\ninspect_ai {md.version('inspect_ai')}\n"
    )
