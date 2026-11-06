# sounding

Analyze one AI agent session with your own questions. Whole-session questions run the scanner engine of
[inspect_scout](https://github.com/meridianlabs-ai/inspect_scout); per-item judging runs
[Jev](https://typesafe.ai) on every user message or tool call.

用你自己写的问题分析一个 AI Agent 会话：
- **整体问题**：通读整条会话、推理后作答，附依据的消息编号。扫描器摘自 inspect_scout，对同一会话发出的请求与结果和原版逐字段一致（`tests/golden`）。
- **逐条判断**：用 Jev 对每条用户消息或每个工具调用做局部判断，给出概率。快、便宜、同样输入结果稳定。
- **先筛后查**：整体问题可以 `focus` 一个逐条判断，把它命中的位置交给扫描器重点核对。

## 安装

需要 Python ≥ 3.11，推荐用 [uv](https://docs.astral.sh/uv/)：

```bash
uv tool install git+https://github.com/zqverse0/sounding
sounding --version
```

## 用法

会话输入是 OpenAI chat 格式的消息（`user` / 带 `tool_calls` 的 `assistant` / `tool`），
`[...]` 或 `{"messages": [...]}` 都可以：

```bash
export OPENAI_BASE_URL=https://api.deepseek.com/v1 OPENAI_API_KEY=…   # 整体问题用的对话模型
export TYPESAFE_API_KEY=…                                             # 逐条判断用的 Jev
sounding scan --file session.json --config questions.json --model openai-api/openai/deepseek-chat
```

`--file -` / `--config -` 从 stdin 读。结果 JSON 打到 stdout：`{"session", "model", "messages", "results": [{"id", "result", "error"}]}`，
一个问题出错不影响其他问题。

### 问题配置

```json
{
  "questions": [
    {"id": "corrected", "type": "jev", "unit": "user", "question": "是否在纠正 Agent？",
     "criteria": {"true": "指出 Agent 做错并要求改", "false": "新要求、提问或确认"}},
    {"id": "how", "type": "llm", "question": "列出用户纠正 Agent 的每一处", "focus": "corrected",
     "answer": {"fields": [{"name": "wrong", "type": "string", "description": "Agent 错在哪"}], "multiple": true},
     "exclude": {"toolUsage": true}},
    {"id": "outcome", "type": "llm", "question": "任务最后完成了吗？", "answer": "boolean"},
    {"id": "errors", "type": "grep", "patterns": ["error", "失败"]}
  ]
}
```

| `type` | 做什么 | 关键字段 |
|---|---|---|
| `llm` | 整体问题 | `answer`：`"boolean"` / `"numeric"` / `"string"`、`{"labels": [...], "multiple"?, "allowNone"?}`、`{"fields": [...], "multiple"?}`；`exclude`：`system` / `reasoning` / `toolUsage`；`focus`：一个 `jev` 问题的 id |
| `jev` | 逐条判断 | `unit`：`"user"`（每条用户消息）或 `"action"`（每个工具调用）；`criteria.true` / `criteria.false` 都要写清；`threshold`（默认 0.5） |
| `grep` | 关键词 | `patterns`、`regex`、`ignoreCase`、`wordBoundary` |

### 环境变量

| 变量 | 用途 |
|---|---|
| `<服务>_BASE_URL` / `<服务>_API_KEY` | 对话模型 `openai-api/<服务>/<模型>`，如 `openai-api/openai/…` 读 `OPENAI_*` |
| `TYPESAFE_API_KEY`、`TYPESAFE_BASE_URL`、`TYPESAFE_DEFAULT_MODEL` | Jev（与 TypeSafe 官方 SDK 同名；根地址默认 `https://api.typesafe.ai`，模型默认 `jev-latest`） |

### Python

```python
import asyncio, sounding

output = asyncio.run(sounding.analyze(messages, config, "openai-api/openai/deepseek-chat"))
```

## 和 Berth 一起用

[Berth](https://github.com/zqverse0/berth) 的「会话分析」插件（`plugins/sounding`）就是它的界面：Berth 把各 Agent 的会话转成上面的
消息格式，经 `berth provider run` 的本地代理注入 `OPENAI_*` 与 `TYPESAFE_*`（密钥不出 Berth）。
命令行里也可以直接分析 Berth 记录的会话：`sounding scan <会话 id> --project <项目> --agent <工具>`。

## 来源与许可

扫描器、提示模板与部分模型层代码摘自 inspect_scout 与 inspect_ai（MIT），逐文件对应关系与同步方法见 [UPSTREAM.md](UPSTREAM.md)。
本仓库同样以 MIT 许可发布（[LICENSE](LICENSE)）。
