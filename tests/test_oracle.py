"""对照测试：sounding 跑同一组场景，请求与结果须与 inspect_scout 录下的 golden 完全一致。"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import engines  # noqa: E402
from runner import run  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402

GOLDEN = pathlib.Path(__file__).parent / "golden"


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["name"] for s in SCENARIOS])
def test_matches_inspect_scout(scenario: dict) -> None:
    golden = json.loads((GOLDEN / f"{scenario['name']}.json").read_text())
    actual = json.loads(json.dumps(run(engines.lite(), scenario), ensure_ascii=False))
    assert len(actual["requests"]) == len(golden["requests"])
    for index, (got, want) in enumerate(
        zip(actual["requests"], golden["requests"], strict=True)
    ):
        assert got == want, f"request {index} differs"
    assert actual["result"] == golden["result"]


def test_model_info_matches_inspect_ai(monkeypatch: pytest.MonkeyPatch) -> None:
    import sounding

    monkeypatch.setenv("BERTH_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("BERTH_API_KEY", "k")
    from sounding._ai.model import get_model_info

    golden = json.loads((GOLDEN / "model_info.json").read_text())
    for name, input_tokens in golden.items():
        info = get_model_info(sounding.get_model(f"openai-api/berth/{name}"))
        assert (info.input_tokens if info else None) == input_tokens, name
