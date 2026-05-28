# Deep Research Agent — NLP 课程项目文档

> 南京大学 · NLP 2025-2026 春季 · 任课教师：吴震
> 助教：张轩望 · 田阔 · DDL: 2026-06-02 20:00

---

## 一、作业要求（完整版）

### 1.1 实验目标

基于 BrowseComp-Plus 评测集与 Qwen3-8B 模型，构建一个能够进行多轮检索、维护中间状态、并基于证据给出最终答案的 Deep Research Agent。

### 1.2 必做内容

| # | 组件 | 说明 |
|---|------|------|
| 1 | 多轮检索 Loop + 停止条件 | 把单次 search→answer 改成循环：每轮检索后判断证据是否足够 | 
| 2 | 上下文管理 | 多轮检索上下文不断增长，需管理历史信息 |
| 3 | Prompt 设计 | 引导模型正确使用工具、跟踪信息、判断停止 |
| 4 | 基于证据回答 | 答案必须来自检索文档，不能靠模型记忆 |

### 1.3 评分细则

**基础分 75 分：**
- 排行榜正确率（0-35 分）：须达到 **12%** 方开始计分
- 实验报告（0-40 分）：含错误轨迹分析（至少 10 条）

**加分项 Open Track（25 分）：**
- 添加新工具（0-5 分）
- 设计新 Agent 架构（0-10 分）
- 涉及模型训练（0-10 分）

**给分前提：**
- 现场演示
- 实验报告与文档检查
- 源代码检查（含运行环境复现确认）

### 1.4 禁止行为

- 禁止在 LLM 输入中泄露答案
- 禁止使用测试数据作为训练样本
- 禁止替换检索器（统一 BM25）
- 禁止使用外部检索服务（Google/Bing 等）
- 禁止引入语料库以外的知识

### 1.5 提交要求

```
学号-姓名-acc=10_5-opentrack/
├── 学号-姓名-acc=10_5-opentrack.pdf
├── core/
│   ├── deepresearch.ipynb
│   └── agent/
├── eval/
│   ├── 学号-姓名-submission-10_5.jsonl
│   └── eval.txt
├── open_track/          # 仅 Open Track
│   ├── train.py
│   └── data/
└── README.md
```

---

## 二、设计方案

本项目提供两个版本的 Agent，供不同阶段使用：

| 版本 | 定位 | 适用场景 |
|------|------|----------|
| **V1** (`deep_research_agent.py`) | 轻量 ReAct | 快速验证 tool calling 链路，作为消融实验基线 |
| **V2** (`deep_research_agent_v2.py`) | 完整模块化 | 主力方案，集成 6 个独立模块追求最高正确率 |

---

### V1: 轻量 ReAct Agent

```
┌──────────────────────────────────────────────┐
│               DeepResearchAgent (V1)          │
│                                               │
│   User Question                               │
│        │                                      │
│        ▼                                      │
│   ┌──────────┐    ┌──────────┐               │
│   │  System   │    │  ReAct   │               │
│   │  Prompt   │───▶│  Loop    │───▶ Answer    │
│   └──────────┘    └────┬─────┘               │
│                         │                     │
│                  ┌──────┴──────┐              │
│                  ▼              ▼              │
│           ┌──────────┐  ┌──────────┐          │
│           │ search() │  │get_doc() │          │
│           └──────────┘  └──────────┘          │
└──────────────────────────────────────────────┘
```

- 基础 ReAct：模型每轮决定是调用工具还是输出答案
- 停止条件：模型自主停止 / 最大 8 轮 / 超限强制输出
- 8 行核心逻辑，适合做消融实验的 baseline 对比

---

### V2: 模块化多轮检索系统（主力方案）

```
┌─────────────────────────────────────────────────────────────┐
│               DeepResearchAgentV2                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │   Step 1: Question Analyzer                          │   │
│  │   用 LLM 从问题中提取关键实体、拆分子问题、生成初始    │   │
│  │   搜索计划（3 个 query：从具体到宽泛）                │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │   Step 2: ReAct Loop (最多 10 轮)                    │   │
│  │                                                      │   │
│  │  每轮：                                               │   │
│  │  1. Query Rewriter → 根据已有证据改写搜索 query      │   │
│  │  2. BM25 Search → 执行检索                           │   │
│  │  3. Relevance Filter → LLM 判断每篇文档是否相关      │   │
│  │  4. Evidence Tracker → 记录已知事实和待查子问题      │   │
│  │  5. 第 5 轮时 → Context Compressor 压缩旧轮内容      │   │
│  │  6. vLLM 决定下一步 → 继续搜或输出答案              │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │   Step 3: Verifier                                   │   │
│  │   答案输出前用 LLM 验证证据是否充分支持              │   │
│  │   → UNSUPPORTED 则让模型重新检查证据再回答           │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### V2 六大模块详解

#### 模块 1: Question Analyzer

```
输入: "一篇2018-2023年间发表于EMNLP的论文，第一作者本科达特茅斯，第四作者本科宾大..."

输出:
  Key Entities:
    - EMNLP (会议, 2018-2023)
    - 第一作者 (本科: 达特茅斯学院)
    - 第四作者 (本科: 宾夕法尼亚大学)
  
  Sub-questions:
    1. 哪篇 EMNLP 论文的第一作者本科就读达特茅斯？
    2. 这篇论文的第四作者本科是否就读宾大？
    3. 这篇论文的标题是什么？
  
  Search Plan:
    1. search: "Dartmouth EMNLP first author"
    2. search: "EMNLP 2018 2019 2020 2021 2022 2023 first author Dartmouth"
    3. search: "达特茅斯 EMNLP 论文"
```

作用：将非结构化问题转为结构化搜索计划，1 次 LLM 调用。

#### 模块 2: Query Rewriter

```
输入: 原始问题 + 已有证据 + 之前搜过的query + 还缺什么

输出: 一条精准的下一轮搜索 query

例子:
  第1轮搜了 "Dartmouth EMNLP" → 找到论文A
  第2轮 Query Rewriter → "Pennsylvania fourth author EMNLP 2020"
```

作用：每轮自动生成更精准的搜索词，避免搜重复内容。

#### 模块 3: Evidence Tracker

```python
class EvidenceTracker:
    confirmed_facts: []      # [{"fact": "论文A第一作者本科达特茅斯", "source": "doc_123", "round": 1}]
    sub_questions: []         # ["第一作者背景?", "第四作者背景?"]
    answered_subqs: set()     # 已答的子问题
    visited_docids: set()     # 已看过的文档，避免重复阅读
    search_history: []        # 历史查询，供 Rewriter 参考
```

作用：结构化的"工作记忆"，比纯靠模型上下文更可靠。

#### 模块 4: Relevance Filter

```
对每篇 BM25 返回的文档，用 LLM 判断:

  Question: "第一作者本科达特茅斯的 EMNLP 论文标题?"
  Snippet: "本文研究了 Transformer 的句法规则学习..."
  
  → LLM 判定: RELEVANT（因为是 EMNLP 论文，可能相关）
  
  Snippet: "澳大利亚殖民史的开端..."
  
  → LLM 判定: IRRELEVANT（完全不相关）
```

作用：BM25 的 score 并不等于语义相关性，LLM 过滤能大幅减少噪声。

#### 模块 5: Context Compressor

```
在第 5 轮触发，将前 3-4 轮的完整对话压缩为：

  [Context Summary — earlier rounds compressed]
  
  Findings So Far:
  - 论文标题为 "Frequency Effects on Syntactic Rule Learning in Transformers" (doc: 12345)
  - 第一作者本科就读于达特茅斯学院 (doc: 12345, 67890)
  - 第四作者本科就读于宾夕法尼亚大学 (doc: 12345, 11111)
  
  Evidence Status:
  - Sub-questions answered: 3/3
  - Documents examined: 12
  - Searches performed: 5
```

作用：解决长对话中"有用信息被淹没"的问题，节省 token。

#### 模块 6: Verifier

```
答案输出前做最后检查:

  Question: "这篇论文标题是什么?"
  Proposed Answer: "Frequency Effects on Syntactic Rule Learning in Transformers"
  
  Evidence: 论文A的标题、作者名单、学历背景...
  
  判定: ✓ SUPPORTED  → 输出最终答案
  判定: ✗ UNSUPPORTED → 让模型重新检查证据再回答
```

作用：防止模型在证据不足的情况下"硬编答案"。

---

### V2 完整执行流程

```
Question
   │
   ▼
┌──────────────┐
│ 1. Analyzer  │ 提取实体、子问题、搜索计划
└──────┬───────┘
       ▼
┌──────────────┐
│ 2. ReAct     │ ── 第1轮: 执行搜索计划(2-3个query)
│    Loop      │     └─ Relevance Filter 过滤
│    (1-10轮)  │     └─ Evidence Tracker 记录
│              │ ── 第2轮: Query Rewriter → 新query搜索
│              │     └─ ...同上
│              │ ── 第3-4轮: 继续搜索/阅读全文
│              │ ── 第5轮: Context Compressor 压缩历史
│              │ ── 第6-10轮: 继续搜索直到证据充分
└──────┬───────┘
       ▼
┌──────────────┐
│ 3. Verifier  │ 检查答案是否被证据充分支持
└──────┬───────┘
       │
   ┌───┴───┐
   │       │
   ▼       ▼
 SUPPORTED UNSUPPORTED
   │          │
   ▼          ▼
 输出答案   重新检查后输出
```

---

## 三、停止条件（V2）

| 条件 | 触发方式 |
|------|----------|
| **模型自主停止** | vLLM 不再输出 tool_call |
| **最大轮次** | 10 轮后强制输出 |
| **Verifier 否决** | 证据不足时打回重审 |
| **查询去重** | 自动跳过已搜过的 query |

## 四、上下文管理（V2）

| 策略 | 实现 |
|------|------|
| 保留完整历史 | 所有 messages 按时间顺序保留 |
| 工具结果截断 | >3000 chars 截断 |
| **Context Compression** | 第 5 轮起，用 LLM 把旧轮压缩为摘要 |
| **Evidence Tracker** | 结构化追踪已知事实，不依赖模型记忆 |

## 五、消融实验设计

为验证每个模块的贡献，依次关闭单个模块对比准确率：

| 配置 | Analyzer | Rewriter | Filter | Compressor | Verifier | 预期 ACC |
|------|----------|----------|--------|------------|----------|----------|
| Full V2 | ✓ | ✓ | ✓ | ✓ | ✓ | **最高** |
| w/o Analyzer | ✗ | ✓ | ✓ | ✓ | ✓ | ↓ |
| w/o Rewriter | ✓ | ✗ | ✓ | ✓ | ✓ | ↓ |
| w/o Filter | ✓ | ✓ | ✗ | ✓ | ✓ | ↓ |
| w/o Compressor | ✓ | ✓ | ✓ | ✗ | ✓ | ↓ (长对话时) |
| w/o Verifier | ✓ | ✓ | ✓ | ✓ | ✗ | ↓ (幻觉增加) |
| V1 (Baseline) | — | — | — | — | — | 最低 |

Notebook 第 7 节内置了自动消融实验脚本。

---

## 六、文件清单

### 6.1 新增文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `agent/deep_research_agent.py` | 267 | V1 轻量 ReAct Agent |
| `agent/deep_research_agent_text.py` | 284 | 文本版 Agent（备选） |
| `agent/deep_research_agent_v2.py` | 450+ | **V2 主力方案** — 6 大模块 |
| `agent_vllm_deep_research.ipynb` | ~20 cells | 主 Notebook，支持 V1/V2 切换 + 消融实验 |
| `REPORT.md` | 本文档 | 作业要求 + 完整设计方案 |

### 6.2 修改文件

| 文件 | 修改 |
|------|------|
| `agent/__init__.py` | 添加 V1 的 import |

### 6.3 项目已有（未动）

`agent/tools.py`、`agent/vllm_client.py`、`agent/browsecomp_searcher.py`、`agent/eval.py`、`agent/dataset_utils.py`、`agent/build_bm25_index.py`

---

## 七、推荐实验流程

```
1. 启动 vLLM 服务（terminal）
   vllm serve ./Qwen3-8B \
     --served-model-name qwen_auto \
     --enable-auto-tool-choice \
     --tool-call-parser hermes \
     --trust-remote-code \
     --host 0.0.0.0 --port 8000

2. 打开 agent_vllm_deep_research.ipynb

3. 用 V1 先跑 1 条 → 确认 tool calling 链路通

4. 切 V2 跑 1 条 → 看有没有报错

5. 跑 50 条批量 → 自动评估

6. 跑消融实验（可选）→ notebook 第 7 节

7. 错误轨迹分析 → 人工看 error_cases

8. 提交 results
```

---

## 八、版本记录

| 日期 | 版本 | 变更内容 |
|------|------|----------|
| 2026-05-26 | v1.0 | V1 ReAct Agent + Notebook + 文档 |
| 2026-05-26 | v2.0 | V2 模块化系统：6 大模块 + 消融实验 |
| 2026-05-29 | v1.5 | V1 五项并行改进：Multi-Query扩展 + 证据引用 + 自动定位 + 分层检索 + 硬化验证 |

---

## 九、V1 五项并行改进 (2026-05-29)

针对 V1 12% 准确率中 95% hallucination 的根因（BM25召回率低 → 模型看不到相关文档 → 凭训练记忆回答），实施 5 项并行改进：

### 1. Multi-Query 搜索扩展
- **问题**: BM25 是纯关键词匹配，单次检索对复杂自然语言查询召回率极低
- **方案**: 单个 `search()` 自动扩展为 4 个不同角度的 BM25 查询，合并去重
- **实现**: `tools.py` 的 `_expand_queries()` + 修改 `search()` 闭包
- **收益**: 检索覆盖率提升 3-4 倍

### 2. 强制证据引用
- **问题**: 答案格式允许模型"凭感觉"回答
- **方案**: 答案格式改为强制 `Evidence: <docid> "<verbatim quote>"`, 缺少时自动阻断
- **实现**: 更新 SYSTEM_PROMPT + solve() 中回答前格式检查

### 3. 自动 find_in_doc 精确定位
- **问题**: `get_document()` 读全书开头 3000 字符，但答案在 300 页之后
- **方案**: 提取问题中的 5 个关键实体，读长文档时自动 `find_in_doc` 定位相关段落
- **实现**: `_extract_key_terms()` + solve() 中自动触发

### 4. 分层检索（搜索专用阶段）
- **问题**: 模型搜一下就想回答，缺乏系统性证据收集
- **方案**: 前 3 轮强制搜索专用，阻断任何回答尝试，第 4 轮起才允许
- **实现**: `tool_choice="required"` 覆盖 rounds 1-3 + 轮次检查

### 5. 硬化 Verification 循环
- **问题**: `verify_claim()` 依赖模型自主调用，模型可能忽视验证结果
- **方案**: 系统自动验证 → 解析 Supported: YES/NO → NO 时阻断+强制重搜
- **实现**: solve() 中的自动验证 + 阻断+重搜 + verify_passed 追踪

---

## 十、Open Track 加分扩展

| 方向 | 分值 | 实现思路 |
|------|------|----------|
| **新工具** VerifyClaim | 0-5 分 | 已内置 Verifier 模块，可独立为 `verify_claim` 工具 |
| **新工具** DecomposeQuestion | 0-5 分 | 已内置 Question Analyzer，可独立为工具 |
| **Multi-Agent** | 0-10 分 | 将 Analyzer / Searcher / Verifier 拆成独立 agent 进程 |
| **模型训练 SFT** | 0-10 分 | 收集成功搜索轨迹微调 Qwen3-8B |
