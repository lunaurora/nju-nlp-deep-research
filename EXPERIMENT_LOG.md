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
| 笔记文件 | runs_archive/0528_首次强制读/agent_vllm_deep_research.ipynb |
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

## V1 四项改进评估 — hard50 (2026-05-28 10:46)

| 项目 | 值 |
|------|-----|
| 版本 | V1 + 自动 top-1 全文 + 3 轮停止 + 多次强制读 + 搜索去重 |
| 模型 | Qwen3-8B |
| 数据集 | hard50 |
| 准确率 | **12.00% (6/50)** |
| 平均工具调用 | 4.96 |
| 平均检索文档 | 0.24 |
| 评估文件 | runs/eval_results_0528_1046.jsonl |
| 笔记文件 | runs_archive/0528_1046/agent_vllm_deep_research.ipynb |

**与历史对比：**

| 轮次 | 准确率 | tc/query | docs/query | hallucination |
|------|--------|----------|------------|---------------|
| Baseline | 8.00% | 1.24 | 0.02 | 43% |
| +retry+高级工具 | 6.00% | 1.96 | 0.02 | 49% |
| +强制读+拆分+验证 | 6.00% | 3.22 | 0.12 | 98% |
| +top-1全文+宽松停止+多次强读+去重 | **12.00%** | **4.96** | **0.24** | **95%** |

**分析：**
- **四项改进整体有效**，准确率从 6% 翻倍到 12%，刚好过及格线
- **工具调用显著增加**（3.22→4.96），模型更积极搜索了
- **检索文档仍然极低（0.24）**：即使自动注入 top-1 全文 + 最多 3 次强制读，模型还是倾向从 snippet 猜答案，不读完整文档
- **95% 错误仍是 hallucination**：核心问题不变——模型凭训练数据记忆回答，而非依据检索到的文档

**错误模式：**
- hallucination (搜了但答错): 42 条 (95%)
- wrong_query: 1 条 (2%)
- context_overload: 1 条 (2%)
- insufficient_search: 0 条 (0%) ✅ — tc=0 问题已彻底解决

**正确（6 题）：** q5, q53, q397, q432, q684, q1095

**存在问题：**
1. avg retrieved docs = 0.24 说明自动加载 top-1 全文的机制没有理想工作——可能是注入的全文只有前 3000 字符，关键信息在文档后半段
2. 模型即使在有全文的情况下仍然选择"不相信"文档，靠记忆回答
3. 接下来需要更强的"强制基于文档"机制，或者尝试更小的 snippet + 强制要求完整文档阅读

## V1 五项改进 (2026-05-29)

针对 12% 准确率中 95% hallucination 的根因，实施 5 项并行改进：

### Plan 1: Multi-Query 搜索扩展
**问题**: BM25 是纯关键词匹配，复杂自然语言查询的单次检索召回率极低。模型看不到相关文档 → 凭训练记忆回答 → hallucination。

**方案**: 单个 `search()` 调用内部自动扩展为 4 个不同角度的 BM25 查询（1 个原始改写 + 3 个 LLM 生成的多样化变体），全部检索后合并去重。

**改动**: `tools.py` 新增 `_expand_queries()` 函数，修改 `get_agent_tool_specs_and_registry` 中的 `search()` 闭包。

### Plan 2: 强制证据引用
**问题**: 答案格式 `Explanation + Exact Answer + Confidence` 允许模型"凭感觉"回答，不要求提供文档证据。

**方案**: 答案格式改为强制要求 `Evidence: <docid> "<verbatim quote>"`。模型输出缺少 Evidence 字段时自动阻断并提醒。

**改动**: `deep_research_agent.py` 更新 SYSTEM_PROMPT + solve() 中回答前格式检查。

### Plan 3: 自动 find_in_doc 精确定位
**问题**: `get_document()` 返回全文（或前 3000 字符），但对一本书来说开头只是目录和序言，答案在 300 页之后。

**方案**: Agent 启动时用 LLM 从问题中提取 5 个关键实体/短语（`_extract_key_terms`）。模型调用 `get_document()` 读长文档（>1500 字符）时，系统自动对每个关键实体执行 `find_in_doc()`，将匹配段落注入对话。

**改动**: `deep_research_agent.py` 新增 `_extract_key_terms()` 函数 + solve() 中的自动触发逻辑。

### Plan 4: 分层检索（搜索专用阶段）
**问题**: 模型搜一下就想回答，缺乏系统性证据收集。

**方案**: 前 3 轮强制为搜索阶段，任何回答尝试都被阻断并引导继续搜索。第 4 轮起才允许回答。

**改动**: `deep_research_agent.py` solve() 中 `tool_choice="required"` 覆盖 rounds 1-3 + 回答前轮次检查。

### Plan 5: 硬化 Verification 循环
**问题**: `verify_claim()` 依赖模型自主调用，模型可能忽视验证结果直接回答。

**方案**: 模型试图回答时系统自动调用 `verify_claim()`，解析 "Supported: YES/NO"。NO 时自动阻断、注入失败信息、强制重搜。YES + Evidence 检查通过才允许输出。

**改动**: `deep_research_agent.py` solve() 中的自动验证逻辑 + 阻断+重搜循环 + verify_passed 状态追踪。

---

**预期效果：**
- Plan 1: 检索召回率提升 3-4 倍，更多相关文档进入上下文
- Plan 2+5: 直接对抗 hallucination，强制答案必须基于文档
- Plan 3: 解决"读了但没读到关键段落"的问题
- Plan 4: 保证前 3 轮专注于证据收集

**待验证：** hard50 全量评估

## V1 五项改进 — 首次评估失败 (2026-05-29)

| 项目 | 值 |
|------|-----|
| 版本 | Plan 1-5 全部启用 (search_phase=3, verify-3000char, no safe valve) |
| 模型 | Qwen3-8B |
| 结果 | 跑 12 题后 HTTP 400 崩溃 (上下文溢出) |
| 平均工具调用 | **~11/题** (从 4.96 暴涨) |
| 平均单题耗时 | **~130s** (从 ~30s 暴涨) |
| 回答模式 | 大量 "cannot be answered" / "not supported by evidence" |

**失败根因：**

1. **Plan 4 搜索阶段太长 (3轮)**：第 1-3 轮强制 `tool_choice="required"`，模型想读文档也被迫搜索，浪费轮次
2. **Plan 5 验证只读 3000 字符**：`verify_claim()` 只取 `text[:3000]`，答案在文档后半段时验证总返回 NO，模型被逼继续搜直到 max_rounds
3. **最后无兜底**：第 7-8 轮验证还不过 → "cannot be answered"，不敢猜
4. **上下文爆炸**：11 次工具调用 × 8 轮 → 远超 Qwen3-8B 的 32K 上下文 → HTTP 400

**修复 (commit 90f623d):**
- `search_phase_max_round 3→1`：只有第 1 轮强制工具
- `verify_claim` 全文搜索：提取 claim 关键词定位全文档，不只读 3000 字符
- 最后轮 safe valve：`round_idx < max_rounds-1` 才验证，最后 1 轮直接答
- 预期 tc 从 ~11 降回 ~5-6，预计准确率回到 12%+ 基线

## V1 三项上下文 + 死胡同修复 (2026-05-29)

| 项目 | 值 |
|------|-----|
| 版本 | V1 + think剥离 + 提前压缩 + 死胡同检测 + verify瘦身 + tool_choice="auto" |
| 模型 | Qwen3-8B |
| 数据集 | hard50 |
| 准确率 | **6.00% (3/50)** |
| 平均工具调用 | 3.52 |
| 平均检索文档 | 0.02 |
| 错误模式 | hallucination 85%, insufficient_search 15% |
| 提交 | submission_0529_0021 |
| 评估 | eval_results_0529_0021 |

**与历史对比：**

| 轮次 | 准确率 | tc/query | docs/query | hallucination | context_overload |
|------|--------|----------|------------|---------------|-----------------|
| Baseline | 8.00% | 1.24 | 0.02 | 43% | 0% |
| +retry+高级工具 | 6.00% | 1.96 | 0.02 | 49% | 0% |
| +强制读+拆分+验证 | 6.00% | 3.22 | 0.12 | 98% | 0% |
| +top-1全文+宽松停止+多次强读+去重 | **12.00%** | 4.96 | 0.24 | 95% | 0% |
| Plam 1-5 (首次) | 崩溃 | ~11 | — | — | HTTP 400 |
| **v1.6 (三项修复+tool_choice)** | **6.00%** | **3.52** | **0.02** | **85%** | **0% ✅** |

**分析：**
- **上下文管理 ✅ 有效**：context_overload 归零，无 HTTP 400
- **死胡同检测 ✅ 有效**：验证循环不再撑爆上下文
- **STOP CONDITIONS ❌ 矫枉过正**：模型学会说 "cannot be determined"，放弃率极高
- **avg retrieved docs = 0.02**：几乎没读任何全文，所有答案基于 snippet + 训练记忆

**修复 (0529 v1.7):**
- 删除 STOP CONDITIONS，改为 FINAL ANSWER REQUIREMENT（禁止输出"cannot be determined"）
- 所有强制回答路径改为 "MUST output specific answer"
- 新增 `enforce_concrete_answer()` 最终答案检查，发现拒绝模式则替换为 "[Best Guess — model refused to answer]"
- 修正 notebook 3.5/4 标签翻转
- 移除 notebook 中暂不使用的 6/7 节

**针对 Component 1-3 的三项修复：**

### Component 1: 停止条件 — 死胡同检测
**问题**: 当 BM25 搜不到相关文档时，模型猜一个答案 → verify 拒绝 → 模型重复同一个答案 → verify 再拒绝 → 循环直到 HTTP 400。典型例子是 q815（1898年书+拍卖商父亲），模型猜 "The Picture of Dorian Gray"，verify 拒绝 3 次后还在循环。

**方案**: 在 verify 失败时检查 `len(unique_docs_read)` 是否增长。若无新文档 → 置 `force_final_answer=True` → 跳过后续所有 verify 和 Evidence 格式检查 → 强制输出 Best Guess。
- 最多浪费 1 轮而不是 7 轮
- 避免上下文撑爆

### Component 2: 上下文管理
- **剥离 `<think>` 块**: Qwen3 的 `<think>...</think>` 占每条 assistant 回复 30-50% 的 token，完全没用（模型下次生成时会重新推理）。每次加入 messages 前用 `re.sub(r'<think>.*?</think>', ...)` 剥离。
- **提前压缩**: round 5 → round 4，在上下文接近溢出前就裁剪。
- **verify_claim 瘦身**: max_tokens 1024 → 256，每次验证省 4 倍 token。
- **Evidence 检查死胡同绕过**: `force_final_answer=True` 时跳过格式检查直接输出。

### Component 3: Prompt — STOP CONDITIONS 章节
在 system prompt 新增 `## STOP CONDITIONS` 章节，明确告知模型三种停止条件：
1. 搜了 3+ 次没找到文档 → Best Guess
2. 读完所有文档 verify 还失败 → Best Guess
3. 禁止重复被 verify 拒绝的答案

**待上云验证**: 预期解决 HTTP 400 问题，工具调用从 ~5-11/题 降至 ~3-6/题，准确率保持 12%+ 或略有提升。
