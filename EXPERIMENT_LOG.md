# 实验记录 — Deep Research Agent

## V1 Baseline (2026-05-28)

| 项目 | 值 |
|------|-----|
| 版本 | V1 (简单 ReAct) |
| 数据集 | hard50 |
| 模型 | Qwen3-8B |
| 准确率 | **8.00%** (4/50) |
| 平均工具调用 | 1.24 |
| 平均检索文档 | 0.02 |
| 备注 | 初始 baseline，57% 的题模型未调用工具直接凭记忆回答 |

**错误模式：**
- insufficient_search (tc=0): 26 条 (57%)
- hallucination (搜了但答错): 20 条 (43%)

## V1 Retry Round1 (2026-05-28)

| 项目 | 值 |
|------|-----|
| 版本 | V1 + tool_choice="required" + retry 机制 |
| 模型 | Qwen3-8B |
| 测试 | 单条 q442 |
| 结果 | retry 机制生效，tool calls: 0→1 |
| 错误模式 | 模型仍然大量 thinking，但 retry 后成功调用了 search() |

**改动：**
- Round 1 若模型 thinking 未调工具，自动追加指令重试
- 重试时强制 `tool_choice={"type": "function", "function": {"name": "search"}}` + `max_tokens=128`
- 已验证 retry 可以解决 tc=0，待跑 hard50 全量评估看准确率提升

## V1 Retry + Eval 更新 + 高级工具 + Prompt 重写 (2026-05-28)

| 项目 | 值 |
|------|-----|
| 版本 | V1 + retry + decompose/verify 工具 + 新 prompt |
| 模型 | Qwen3-8B |
| 评估 | 新 eval (助教版，关 thinking + 提取最终答案 + 并行 + 严格判分) |
| 准确率 | 6.00% (3/50) — 新 eval 标准更严，待下轮提升 |

**问题诊断：**
- avg retrieved docs = 0.02：模型只看 snippet 不读全文
- avg tool calls = 1.96：搜一次就放弃
- 51% insufficient_search、49% hallucination
- ~9/50 条输出 `<think>` 污染答案格式

**本轮改动：**
1. `tools.py`: 新增 `decompose_question` + `verify_claim` 两个 LLM 驱动的高级工具
2. `eval.py`: 替换为助教新版（关 thinking + 提取最终答案 + 并行 + 严格 prompt）
3. `deep_research_agent.py`: 重写 system prompt — 强制读全文、至少搜2次、禁止 `<think>` 在答案中、引导使用新工具
4. `CLAUDE.md`: 同步文档更新
5. `EXPERIMENT_LOG.md`: 新增本条目
