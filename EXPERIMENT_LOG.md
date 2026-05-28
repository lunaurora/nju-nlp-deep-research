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

## V1 强制读全文 + 自动高级工具 (2026-05-28)

| 项目 | 值 |
|------|-----|
| 版本 | V1 + 强制 get_document + 自动 decompose + 强制 verify |
| 模型 | Qwen3-8B |
| 评估 | 新 eval (关闭 thinking + 提取最终答案 + 并行 + 严格判分) |
| 准确率 | **6.00%** (3/50) |
| 平均工具调用 | 3.22 |
| 平均检索文档 | 0.12 |
| 备注 | retry 解决了 tc=0，但模型仍然不读全文，98% 靠 snippet 猜答案 |

**错误模式：**
- hallucination (搜了但答错): 46 条 (98%)
- insufficient_search: 1 条 (2%)

**本轮改动：**
1. `deep_research_agent.py`: system prompt 重写为分步骤命令式
2. `deep_research_agent.py`: 自动调 decompose_question 拆题（loop 开始前注入）
3. `deep_research_agent.py`: 模型搜了不读全文 → 强制插入 user message 要求 get_document()
4. `deep_research_agent.py`: 模型想停 → 先强制 verify_claim() 再允许回答
5. `deep_research_agent.py`: snippet_max_chars 1500→800，逼模型读全文
6. `deep_research_agent.py`: 停止条件加 `not round_has_getdoc` 保护（刚读了文档就不停）
7. `EXPERIMENT_LOG.md`: 新增本条目

## V1 四项改进：自动读 top-1 / 放宽停止 / 多次强制读 / 搜索去重 (2026-05-28)

| 项目 | 值 |
|------|-----|
| 版本 | V1 + 自动 top-1 全文 + 3 轮停止 + 多次强制读 + 搜索去重 |
| 模型 | Qwen3-8B |
| 状态 | 代码完成，待 hard50 评估 |
| 备注 | 4 项非冲突改进全部实现 |

**改动清单：**

1. **自动加载 top-1 全文** — 首次 search() 后自动调 get_document() 获取 top-1 文档全文（前 3000 字符）并注入到 conversation，模型无需手动调用即可看到完整文本。解决 98% hallucination from snippet 的问题。

2. **停止条件放宽到 3 轮** — 原 stop 条件实质是"任意 1 轮无新文档就停"，过于激进。改为连续 3 轮无新文档才触发停止，给模型更多搜索空间。实现：`SimpleTracker.consecutive_no_new_docs` 计数器，每轮比较 docid 集合 subset 关系，累加连续轮数。

3. **多次强制读文档** — 原 `doc_read_forced` 是二值 flag（只强制 1 次）。改为 `unique_docs_read` 集合 + `max_docs_to_read=3`，模型搜了不读时重复强制，直到读完 3 篇不同文档。

4. **搜索去重** — `is_duplicate_query()` 从精确字符串匹配升级为 Jaccard token 重叠度 > 80% 检测。命中时插入 user message 提示换搜索角度。减少 BM25 重复检索的浪费。

**预期收益与风险：**
- 收益：模型读全文增多 → hallucination 减少；停止更宽松 → 更多搜索轮次
- 风险：自动加载 top-1 可能误导（top-1 不一定是相关文档）；停止放宽 → 更长推理时间；多次强制读可能打乱模型节奏

**改动的文件：**
- `agent/deep_research_agent.py` — SimpleTracker 重写 + solve() 方法扩展
