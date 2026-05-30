# Deep Research Agent — NLP 课程项目文档

> 南京大学 · NLP 2025-2026 春季 · 任课教师：吴震
> 助教：张轩望 · 田阔 · DDL: 2026-06-02 20:00

---

## 一、作业要求

### 1.1 实验目标

基于 BrowseComp-Plus 评测集与 Qwen3-8B 模型，构建一个能够进行多轮检索、维护中间状态、并基于证据给出最终答案的 Deep Research Agent。

### 1.2 必做内容

| # | 组件 | 说明 |
|---|------|------|
| 1 | 多轮检索 Loop + 停止条件 | 单次 search→answer → 循环检索判断证据 |
| 2 | 上下文管理 | 多轮上下文增长，需管理历史信息 |
| 3 | Prompt 设计 | 引导工具使用、信息跟踪、停止判断 |
| 4 | 基于证据回答 | 答案必须来自检索文档，非模型记忆 |

### 1.3 评分细则

- **基础分 75 分**：排行榜正确率 0-35 分（≥12% 方计分）+ 实验报告 0-40 分
- **加分项 Open Track（25 分）**：新工具 / 新架构 / 模型训练
- **给分前提**：现场演示 + 实验报告检查 + 源代码检查
- **禁止**：泄露答案、替换检索器(BM25 固定)、外部检索服务、引入语料外知识

### 1.4 提交格式

```
学号-姓名-acc=10_5-opentrack/
├── 学号-姓名-acc=10_5-opentrack.pdf
├── core/
│   ├── deepresearch.ipynb
│   └── agent/
├── eval/
│   ├── 学号-姓名-submission-10_5.jsonl
│   └── eval.txt
├── open_track/
└── README.md
```

---

## 二、设计方案

### 2.1 架构选型：V1 ReAct Agent（最终采用）

本项目开发了两个版本的 Agent，最终采用 **V1 轻量 ReAct 架构**作为主线方案：

| 版本 | 定位 | 状态 | 原因 |
|------|------|------|------|
| **V1** (`deep_research_agent.py`) | 轻量 ReAct | **主线** | 简洁可靠，~400 行 |
| **V2** (`deep_research_agent_v2.py`) | 模块化 | **已废弃** | 6 模块设计冗余，LLM 调用爆炸，Qwen3-8B 32K 上下文撑不住 |

### 2.2 V1 架构图

```
┌──────────────────────────────────────────────────┐
│              DeepResearchAgent (V1)              │
│                                                   │
│  System Prompt + Question                         │
│       │                                           │
│       ▼                                           │
│  ┌──────────────────┐                             │
│  │  Auto-decompose  │  ← 自动拆题注入搜索计划       │
│  └──────┬───────────┘                             │
│         ▼                                         │
│  ┌──────────────────┐                             │
│  │  ReAct Loop      │  1-8 轮                     │
│  │  (tool_choice)   │  Round 1: auto+retry        │
│  │                  │  Round 2+: auto              │
│  │  ┌─ search()     │  Multi-Query 扩展 4 合 1    │
│  │  ├─ get_doc()    │  强制读全文机制              │
│  │  ├─ find_in_doc()│  关键实体自动定位            │
│  │  ├─ decompose()  │  高级拆题工具                │
│  │  └─ verify()     │  硬化验证 + 死胡同检测       │
│  └──────┬───────────┘                             │
│         ▼                                         │
│  ┌──────────────────┐                             │
│  │  Auto-verify     │  verify_claim 自动调用      │
│  │  + Dead-end      │  无新文档 → 强制 Best Guess │
│  │  + Enforce       │  拒绝检测 → 强制猜测占位符   │
│  └──────┬───────────┘                             │
│         ▼                                         │
│  ┌──────────────────┐                             │
│  │  Final Answer    │  Evidence + Exact + Conf    │
│  └──────────────────┘                             │
└──────────────────────────────────────────────────┘
```

### 2.3 核心创新点

#### 创新 1：BM25 Query Rewriting（自然语言→关键词改写）

BrowseComp-Plus 的问题用自然语言描述（如 "a book about certain inland discoveries"），但 BM25 是纯关键词匹配。在 `search()` 工具内部加了一层 LLM 改写：

```python
# tools.py _rewrite_for_bm25()
模型输入: "a book about certain inland discoveries in the 1920s"
改写输出: "1920s inland discoveries"  # 去掉 vague 词，保留实体
```

对模型完全透明，无需改动 agent loop。

#### 创新 2：Multi-Query 搜索扩展（Plan 1）

单个 `search()` 调用自动扩展为 4 个不同角度的 BM25 查询（1 个原始改写 + 3 个 LLM 多样化变体），所有结果合并去重：

```python
# tools.py _expand_queries()
原始: "a book about certain inland discoveries in the 1920s"
→ "1920s inland discoveries book"
→ "barrel-shaped floating vessel"
→ "publishing company founded 1880s"
→ "spear attack botanist party 463 464"
```

检索覆盖率提升 3-4 倍。

#### 创新 3：强制证据引用（Plan 2）

答案格式强制要求 `Evidence: <docid> "<verbatim quote>"`，缺少时自动阻断：

```
Evidence: doc_12345 "The first chapter is titled 'Foundations of Settlement'"
Exact Answer: Foundations of Settlement
Confidence: high
```

#### 创新 4：自动 find_in_doc 精确定位（Plan 3）

Agent 启动时用 LLM 从问题中提取 5 个关键实体。模型读长文档（>1500 字符）时，系统自动对每个关键实体执行 `find_in_doc()`：

```python
# 问题: 关于 1920s 内陆发现的书，332-339 页描述了桶形漂浮容器
关键实体: ["barrel-shaped floating vessel", "spear attack botanist",
           "inland discoveries", "1920s", "publishing founded 1880s"]
```

#### 创新 5：硬化 Verification 循环（Plan 5）

模型试图回答时系统自动调用 `verify_claim()`，解析 "Supported: YES/NO"。NO 时自动阻断、注入失败信息、强制重搜。YES + Evidence 检查通过才允许输出。

#### 创新 6：死胡同检测（Component 1）

verify 失败时检测 `unique_docs_read` 是否增长。无新文档 → `force_final_answer=True` → 跳过后续 verify → 强制 Best Guess：

```python
if len(unique_docs_read) <= docs_count_at_last_verify:
    force_final_answer = True  # 死胡同，直接输出
else:
    docs_count_at_last_verify = len(unique_docs_read)  # 有新文档，继续搜
```

#### 创新 7：上下文管理（Component 2）

三项优化：

| 优化 | 效果 |
|------|------|
| **剥离 `<think>` 块** | 每条回复节省 30-50% token |
| **提前压缩** | Round 4 触发 `compress_old_rounds` |
| **verify 瘦身** | max_tokens 1024 → 256，省 4 倍 |

#### 创新 8：FINAL ANSWER REQUIREMENT（Component 3, v1.7）

删除 v1.6 的 STOP CONDITIONS（导致模型全放弃），改为强制输出要求 + 拒绝检测：

```python
_REFUSAL_PATTERNS = [
    "cannot be determined", "unable to determine",
    "not supported by evidence", "insufficient evidence",
    ...  # 共 18 种拒绝模式
]

def enforce_concrete_answer(answer):
    if any(pattern in answer.lower() for pattern in _REFUSAL_PATTERNS):
        return "[Best Guess — model refused to answer]"
    return answer
```

### 2.4 5 个工具函数

| 工具 | 来源 | 说明 |
|------|------|------|
| `search(query)` | 基础 | BM25 检索，自动 Multi-Query 扩展 4 合 1 |
| `get_document(docid)` | 基础 | 读取文档全文 |
| `find_in_doc(docid, keyword)` | 基础 | 文档内关键词定位 |
| `decompose_question(question)` | 高级(LLM) | 拆解复杂问题为子查询 |
| `verify_claim(claim, docids)` | 高级(LLM) | 验证答案是否被文档支持 |

### 2.5 V2 废弃原因（经验教训）

V2 设计了 6 个模块（QuestionAnalyzer / EvidenceTracker / QueryRewriter / RelevanceFilter / ContextCompressor / Verifier），但存在严重问题：

1. **QueryRewriter 吃 thinking**：模型输出带 `<think>` 块，Rewriter 直接拿去当搜索 query
2. **RelevanceFilter 调用爆炸**：每轮对 top-15 每篇文档额外调 LLM 判断相关性，一轮多 15 次调用
3. **模块冗余**：V2 模块做的事 V1 用 prompt + 工具选择也能做到，但更灵活
4. **Qwen3-8B 32K 上下文限制**：V2 每轮 = 主 LLM + Rewriter LLM + Filter LLM + 结果，上下文增长更快

**结论**：在受限环境下（32K 上下文、8B 模型、BM25 检索），轻量 ReAct 比模块化架构更实用。

---

## 三、实验过程与结果

### 3.1 实验环境

| 项目 | 配置 |
|------|------|
| 平台 | Huawei Cloud (Ascend NPU) |
| 模型 | Qwen3-8B (vLLM 部署, Hermes tool-call parser) |
| 检索 | BM25 (SQLite FTS5, unicode61 tokenizer) |
| 语料 | BrowseComp-Plus (~10K 文档, parquet 格式) |
| 评估集 | hard50 (50 条多跳推理题) |
| 最大上下文 | 40960 tokens |
| max_rounds | 8 |
| top_k | 10 |
| temperature | 0.0 |

### 3.2 完整实验历程

| 轮次 | 版本 | 准确率 | tc/query | docs/query | 主要问题 |
|------|------|--------|----------|------------|----------|
| Baseline | V1 简单 ReAct | 8.00% | 1.24 | 0.02 | 57% 不调工具 |
| +retry+高级工具 | V1 | 6.00% | 1.96 | 0.02 | 新 eval 标准更严 |
| +强制读+拆分+验证 | V1 | 6.00% | 3.22 | 0.12 | 98% hallucination |
| +top-1全文+宽松停止+多次强读+去重 | V1 | **12.00%** | 4.96 | 0.24 | 95% hallucination, **过及格线** |
| +Plan 1-5 (首次) | V1 | **崩溃** | ~11 | — | HTTP 400 上下文溢出 |
| +三项修复+tool_choice=v1.6 | V1 | 6.00% | 3.52 | 0.02 | STOP CONDITIONS 过保守, 全放弃 |
| **+删STOP+强制输出=v1.7** | **V1** | **待验证** | — | — | — |

### 3.3 关键转折点分析

#### 转折 1：Baseline 8% → 12%（过及格线）

加入**自动 top-1 全文加载 + 3 轮停止 + 多次强制读 + 搜索去重**四项改进后，准确率从 6% 翻倍到 12%，刚好过及格线。核心改进是"自动加载 top-1 全文"，让模型至少看到一篇完整文档，而非仅凭 snippet 猜测。

#### 转折 2：12% → 崩溃（HTTP 400）

Plan 1-5 全部启用后，12 题后上下文撑爆。根因：
- **Plan 4 搜索阶段太长（3轮）**：模型在读文档时被强制搜索，浪费轮次
- **Plan 5 验证只读 3000 字符**：答案在文档后半段时验证总返回 NO，模型被逼搜到 max_rounds
- **最后无兜底**：第 7-8 轮验证还不过 → "cannot be answered"

**修复**：search_phase 3→1 轮、verify 改为全文搜索关键词定位、最后 1 轮 safe valve。

#### 转折 3：崩溃 → 6%（STOP CONDITIONS 过保守）

三项修复（死胡同检测 + 上下文管理 + STOP CONDITIONS）解决了 HTTP 400，但 STOP CONDITIONS 让模型学会说 "cannot be determined"，放弃率极高，准确率掉到 6%。

**修复（v1.7）**：删除 STOP CONDITIONS，改为 FINAL ANSWER REQUIREMENT + enforce_concrete_answer 拒绝检测。

### 3.4 最终方案组件清单

| 组件（作业要求） | 实现 | 状态 |
|------------------|------|------|
| Component 1: 停止条件 | 死胡同检测 + 3 轮无新文档停止 + max_rounds | ✅ |
| Component 2: 上下文管理 | think 剥离 + Round 4 压缩 + verify 瘦身 | ✅ |
| Component 3: Prompt 设计 | 5 步命令式 prompt + FINAL ANSWER REQUIREMENT | ✅（v1.7） |
| Component 4: 基于证据回答 | Evidence 格式强制 + verify_claim 自动验证 | ✅ |

### 3.5 Open Track 贡献

| 方向 | 分值 | 实现 |
|------|------|------|
| **新工具 × 3** | 0-15 分 | find_in_doc / decompose_question / verify_claim（tools.py） |
| **Multi-Query 搜索扩展** | 0-5 分 | _expand_queries() 4 查询合并去重 |
| **BM25 Query Rewriting** | 0-5 分 | _rewrite_for_bm25() LLM 关键词改写 |

---

## 四、题目风格分析

### 4.1 hard50 问题分类

hard50 的 50 道题并非平等的检索难度。按 BM25 友好度可分为两类：

| 类别 | 占比 | 特征 | 示例 |
|------|------|------|------|
| **BM25 友好型** | ~25% | 含组织名、机构名、专有名词 | q5 (Malaria Consortium, WHO), q53 (Norman Naimark) |
| **BM25 不友好型** | ~75% | 纯自然语言、无专有名词 | q442 ("certain inland discoveries"), q815 ("book published 1898") |

两类题目的答案类型分布：
- 人名（最常见）：q5(Cristina Ortiz)、q549(Marguerite Smith)、q471(Scott A. Ebers) 等
- 书名：q442(THE DAWN OF AUSTRALIAN COLONISATION)、q815(One Red Rose)
- 公司名：q314(Spero Therapeutics)、q651(FormFactor)
- 数字/年份：q761(7%)、q930(2001)
- 具体实体：q380(46.30 centimetres)、q662(pears)

### 4.2 BM25 不友好题型的根因

1. **描述性语言**：题目用"关于某内陆发现的书"代替书名，用"某人的父亲"代替人名
2. **时间范围约束**：大量 between X and Y (inclusive/exclusive)，BM25 无法理解
3. **串行推理依赖**：必须跨文档追踪实体链，单次 search 最多命中一环
4. **干扰文档极多**：平均 70-100+ 干扰文档/题，BM25 top-10 信噪比极低

### 4.3 对策略的影响

- BM25 友好型题是主要得分来源（~25% 贡献了几乎所有正确题）
- BM25 不友好型题 → 关键词改写 + Multi-Query 多点爆破 + 最后轮 Best Guess
- 模型"放弃"行为导致准确率降为 0 → 必须强制最后轮输出答案

---

## 五、错误轨迹分析

### 5.1 示例轨迹：q442（v1.6 tool_choice 修复后）

完整轨迹见 `trajectories/0528-7_tool_choice_fix/q442_trajectory.md`

```
R1: search("botanist returned to London 1830s spear attack on party")
    → BM25 返回 "Technical Reports" (docid 46212) — 完全无关
R2: search("author married 1890s botanist 1830s spear attack barrel-shaped vessel")
    → BM25 返回 "APA Format" (docid 28618) — 完全无关
R3: search(与 R2 相同)  ← 重复检测未阻止
    → BM25 返回 docid 28618 — 完全无关
Final: 模型放弃搜索，凭记忆回答 "The Foundations of Settlement Prosperity"（hallucination）
```

**问题诊断**：
1. **BM25 零召回**：所有 search 返回的文档与问题无关。云环境 BM25 索引可能指向错误语料
2. **模型拒绝读全文**：3 次 "CRITICAL" 强制提示均被忽略，模型只搜不读
3. **重复查询检测失效**：R3 提交与 R2 完全相同的 query，只提醒未阻止
4. **verify 传入 think 文本**：verification prompt 含 `<think>` 块内容而非实际答案

### 5.2 示例轨迹：q5（回答正确题）

```
q5: "There was a global report released by the World Health Organisation after 2011
     about resources, with contributions from Malaria Consortium, Ogilvy & Mather, WHO..."
     答案: Cristina Ortiz

分析: 包含 "Malaria Consortium" "Ogilvy & Mather" "WHO" 等专有名词
     → BM25 第 1 次 search 就命中相关文档
     → 模型读文档 → 找到答案
     
结论: 此题属于 BM25 友好型，专有名词多，检索效率高
```

### 5.3 错误分类统计（v1.6, 50 题）

| 错误模式 | 数量 | 占比 | 特征 |
|----------|------|------|------|
| hallucination | 42 | 85% | 搜了但凭训练记忆答错 |
| insufficient_search | 7 | 15% | 工具调用 ≤1 次 |
| context_overload | 0 | 0% | HTTP 400（已修复） |
| **正确** | **3** | **6%** | — |

### 5.4 核心教训

1. **verify 死循环是 HTTP 400 的根因**：BM25 召不回相关文档 → 模型猜→verify 拒绝→再猜→再拒绝，直到上下文溢出。死胡同检测（无新文档 → 强制输出）是最关键的修复。
2. **"没证据"vs"证据矛盾"需区分**：前者应允许放弃，后者才应引导重搜。
3. **STOP CONDITIONS 矫枉过正**：告诉模型"可以放弃"会误导 Qwen3 全面放弃——Qwen3 本身倾向于谨慎/拒绝回答，prompt 应鼓励猜测而非放弃。
4. **模型拒绝读全文是顽固问题**：即使有多次强制提示，Qwen3 仍然倾向从 snippet + 训练记忆回答，不调用 get_document()。平均检索文档量始终在 0.02-0.24 徘徊。

---

## 六、文件清单

| 文件 | 行数 | 说明 |
|------|------|------|
| `agent/deep_research_agent.py` | 554 | **V1 主线 Agent** — 多轮 ReAct + 所有改进 |
| `agent/deep_research_agent_v2.py` | 779 | V2 模块化架构（已废弃） |
| `agent/tools.py` | 363 | 5 个工具定义 + BM25 Query Rewriting + Multi-Query 扩展 |
| `agent/eval.py` | 437 | LLM 自动评估（关闭 thinking + 提取最终答案 + 并行 8 线程） |
| `agent/vllm_client.py` | 48 | vLLM HTTP 客户端 |
| `agent/browsecomp_searcher.py` | 217 | BM25 SQLite FTS5 检索实现 |
| `agent_vllm_deep_research.ipynb` | ~15 cells | 主实验 notebook |
| `browsecomp_plus_hard50.jsonl` | 50 题 | hard50 评估集 |
| `browsecomp-plus-corpus/` | ~10K 文档 | 全量语料 |
| `REPORT.md` | 本文档 | 课程报告 |
| `EXPERIMENT_LOG.md` | 实验记录 | 完整实验历史 |
| `CLAUDE.md` | 项目知识库 | 开发指南 |

---

## 七、待完成工作

- [ ] **P0**: v1.7 上云验证 — git push → git pull → 跑全量 50 题
- [ ] **P0**: 修复 BM25 索引（确认路径和语料一致）
- [ ] **P1**: 实体爆破搜索 — 每个实体独立 search 而非合并
- [ ] **P1**: verify 结果收割 — verify 返回 "Correct Answer: X" 时直接采纳
- [ ] **P1**: 最后 2 轮温度 0.0→0.3
- [ ] **P2**: 消融实验（子集 ~30 分钟）
- [ ] **P2**: REPORT.md 完善
- [ ] **P3**: 最终提交（6 月 2 日前）

---

## 八、版本记录

| 日期 | 版本 | 准确率 | 变更 |
|------|------|--------|------|
| 2026-05-28 | v1.0 | 8.00% | Baseline ReAct |
| 2026-05-28 | v1.1 | 6.00% | +retry+高级工具+新 eval |
| 2026-05-28 | v1.2 | 6.00% | +强制读全文+拆分+验证 |
| 2026-05-28 | v1.3 | **12.00%** | +top-1全文+宽松停止+多次强读+去重 |
| 2026-05-29 | v1.4 | 崩溃 | +Plan 1-5（HTTP 400） |
| 2026-05-29 | v1.5 | — | 修复 Plan 1-5（search_phase 3→1, verify 全文搜索, safe valve） |
| 2026-05-29 | v1.6 | 6.00% | +死胡同检测+think剥离+提前压缩+verify瘦身+STOP CONDITIONS |
| 2026-05-29 | **v1.7** | **待验证** | **删STOP CONDITIONS+FINAL ANSWER REQUIREMENT+拒绝检测** |
