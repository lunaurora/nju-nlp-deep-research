# Deep Research Agent — NLP 课程项目

## 项目概况

- **课程**: NLP @ NJU, 吴震, 助教张轩望/田阔
- **任务**: 基于 BrowseComp-Plus 构建多轮检索 Deep Research Agent
- **模型**: Qwen3-8B (vLLM 部署, Ascend NPU)
- **检索**: BM25 (SQLite FTS5, **固定不可替换**)
- **DDL**: 2026-06-02 20:00
- **及格线**: ≥12% 准确率

## 仓库

| 位置 | 路径/URL |
|------|----------|
| 官方 baseline | `fate-ubw/nju-nlp-deep-research` |
| 用户 fork | `lunaurora/nju-nlp-deep-research` |
| 本地代码 | `D:\nju-nlp-deep-research` |
| 华为云持久化 | `/opt/huawei/edu-apaas/src/init/deep-research-agent/` |
| 华为云工作区 | `/mnt/workspace/` |

## 工作流

```
本地编辑 → git push → 华为云 git pull → cp 到工作区 → 跑 notebook
```

### 具体步骤
1. 本地 Git 仓库 `D:\nju-nlp-deep-research` 修改代码
2. 暂存 → 提交 → 推送到 GitHub (`origin main`, fork `lunaurora/...`)
3. 华为云登录 science.lab.huaweicloud.com
4. 开终端: `cd /opt/huawei/edu-apaas/src/init/deep-research-agent && git pull origin main`
5. 同步到工作区: `cp -r * /mnt/workspace/`
6. 启动 vLLM → 运行 notebook

## Git 工作规范

1. **每次改动必须逐条解释** 改了哪些文件、改了什么、为什么改，得到许可后才提交
2. **只说暂存，不说提交** — 只 `git add`，等用户说"可以提交了"再 commit
3. **不说推送** — 等用户说"提交推送吧"再 `git push`
4. 不加 `--no-verify` / `--no-gpg-sign`
5. 每次新建 commit（不 amend）
6. 提交信息格式: 简短描述改动内容和原因

## 文件结构与关键代码

```
D:\nju-nlp-deep-research/
├── CLAUDE.md                        # 本文件
├── EXPERIMENT_LOG.md                # 实验记录
├── README.md                        # 实验文档
├── REPORT.md                        # 实验报告模板
├── browsecomp_plus_hard50.jsonl     # hard50 评估集
├── browsecomp-plus-corpus/          # 全量语料 (~10K 文档)
├── indexes/                         # BM25 索引 (.gitignore)
├── agent/
│   ├── deep_research_agent.py       # V1 ReAct Agent（主线）
│   ├── deep_research_agent_v2.py    # V2 模块化 Agent（有 bug 待修）
│   ├── deep_research_agent_text.py  # 纯文本版 V1（备选）
│   ├── tools.py                     # 工具定义：基础3个 + 高级2个 (LLM-powered)
│   │                                #   基础: search / get_document / find_in_doc
│   │                                #   高级: decompose_question / verify_claim
│   ├── vllm_client.py               # vLLM HTTP 客户端
│   ├── browsecomp_searcher.py       # BM25 检索实现
│   ├── build_bm25_index.py          # 构建索引
│   ├── dataset_utils.py             # 数据集加载
│   ├── eval.py                      # LLM 自动评估
│   ├── pangu_tool_parser.py         # Pangu 工具 parser（备选路线）
│   └── pangu_chat_template.jinja    # Pangu chat template
├── agent_vllm.ipynb                 # 单步 RAG baseline
├── agent_vllm_weather.ipynb         # 工具调用链路演示
└── agent_vllm_deep_research.ipynb   # 主实验 notebook
```

### agent/deep_research_agent.py (V1 主线)

多轮 ReAct loop:
```
system prompt → user question → model(tools) → execute tool → append result → repeat → answer
```

关键参数: `max_rounds=8`, `max_tokens=2048`, `top_k=10`, `temperature=0.0`

- Round 1 强制 `tool_choice="required"` (解决 Qwen3 不调工具的问题)
- `SimpleTracker` 去重 + 无新信息停止
- Round 5 触发 `compress_old_rounds()` 上下文裁剪
- 停止条件: clear evidence / 连续2轮无新文档 / max_rounds
- 输出格式: `Exact Answer: <answer>` + `Confidence: <high|medium|low>`

### agent/tools.py

五个工具函数（`get_agent_tool_specs_and_registry` 返回）:
1. `search(query)` → top-10 文档 snippet
2. `get_document(docid)` → 完整文档
3. `find_in_doc(docid, keyword)` → 文档内关键词搜索
4. `decompose_question(question)` → 分解复杂问题为子查询（LLM 驱动，需传 client）
5. `verify_claim(claim, docids)` → 验证候选答案是否被文档支持（LLM 驱动，需传 client）

高级工具（4-5）在 `client` 参数传入时自动启用，否则只注册基础 3 个。

### agent/deep_research_agent_v2.py (V2, 有 bug)

模块: Question Analyzer → Query Rewriter → Evidence Tracker → Relevance Filter → Context Compressor → Verifier

已知问题:
- Query Rewriter 会把 LLM thinking 文本当成搜索 query
- Relevance Filter 每轮对每篇文档额外调 LLM，调用量爆炸
- 模型在复杂 prompt 下大量 token 浪费在自我辩论
- **当前建议放弃 V2，主攻 V1 ReAct**

## Known Issues & Pitfalls

1. **Qwen3 不调工具**: 即使有 tool 定义 + prompt 要求，Qwen3 `tool_choice="auto"` 可能直接凭记忆回答。修复: Round 1 强制 `tool_choice="required"`
2. **Thinking 模式**: Qwen3 输出 `<think>...</think>` 推理过程后才回答。这些 think 块占 token 30-50% 且对后续推理无帮助，应在加入 messages 前剥离
3. **vLLM 路径**: 华为云上模型在 `/opt/huawei/edu-apaas/src/init/Qwen3-8B`，不是 `./Qwen3-8B`
4. **文件覆盖**: 多次运行 notebook 不覆盖 `submission.jsonl` — 已改为时间戳命名 `submission_MMDD_HHMM.jsonl`
5. **版本管理**: 每次同步按 `MMDD_N-描述` 建独立文件夹，不覆盖旧版本，方便横向对比
6. **工作区优先**: 先在版本文件夹里测试验证，确认没问题再复制到持久化存储
7. **Verify 死循环**: verify 失败后如果无新文档可读，模型会重复猜同一个答案导致 HTTP 400。修复: `docs_count_at_last_verify` 死胡同检测 + `force_final_answer` 强制输出
8. **上下文爆炸**: 主要来自 (a) `<think>` 块 (b) verify 死循环 (c) auto-load top-1 全文的 3000 字符。修复: 剥离 think + 死胡同检测 + 提前压缩
9. **BM25 零召回**: 复杂自然语言查询第一跳可能召回 0 篇相关文档。extract_key_terms + multi-query 扩展可以缓解但无法完全解决

## 华为云环境

### 环境概况
- **平台**: science.lab.huaweicloud.com (Jupyter + 终端)
- **系统**: Linux (notebook-46833dc5923... pod)
- **工作目录**: 持久化 `/opt/huawei/edu-apaas/src/init/` | 工作区 `/mnt/workspace/`
- **模型**: Qwen3-8B (39GB 含缓存), 路径 `/opt/huawei/edu-apaas/src/init/Qwen3-8B`
- **Claude Code**: 已安装，持久化目录 `/opt/huawei/edu-apaas/src/init/node/`
- **Claude Code 配置**: `~/.claude/settings.json`，使用 ccSwitch 中转 DeepSeek API

### 持久化存储文件结构 (/opt/huawei/edu-apaas/src/init/deep-research-agent/)
```
deep-research-agent/           # Git 仓库（origin→lunaurora/nju-nlp-deep-research）
├── CLAUDE.md                  # 项目文档（需 git pull 同步）
├── EXPERIMENT_LOG.md          # 实验记录
├── README.md / REPORT.md
├── agent/
│   ├── deep_research_agent.py    # V1 主线
│   ├── deep_research_agent_v2.py # V2（有 bug）
│   ├── tools.py                  # 工具定义
│   ├── eval.py                   # 评估脚本
│   ├── vllm_client.py / browsecomp_searcher.py / ...
├── agent_vllm.ipynb / agent_vllm_deep_research.ipynb / agent_vllm_weather.ipynb
├── browsecomp-plus-corpus/     # 全量语料
├── browsecomp_plus_hard50.jsonl
├── deep-research-agent/        # ⚠ 嵌套目录（旧的完整副本，可能是 cp -r 遗留）
├── runs/                       # 运行结果
│   ├── submission_MMDD_HHMM.jsonl
│   ├── eval_results_MMDD_HHMM.jsonl
│   └── ...
├── kernel_meta/                # GPU kernel 缓存
└── indexes/                    # BM25 索引
```

### 注意：持久化 vs 工作区
- **持久化** (`/opt/huawei/.../init/`)：git 仓库所在地，保存实验数据
- **工作区** (`/mnt/workspace/`)：Jupyter notebook 实际运行目录
- **版本管理**：每次同步时按 `MMDD_N-描述` 建子文件夹，代码和结果都在里面，方便横向对比
- **同步步骤**：git pull → 新建版本文件夹 → cp 到该文件夹 → 进该文件夹跑 notebook

### 常用命令

**启动 vLLM:**
```bash
cd /opt/huawei/edu-apaas/src/init/deep-research-agent
vllm serve /opt/huawei/edu-apaas/src/init/Qwen3-8B \
  --served-model-name qwen_auto \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```

**同步最新代码（保留历史结果）:**
```bash
cd /opt/huawei/edu-apaas/src/init/deep-research-agent
git pull origin main

# 按日期+序号建版本文件夹，方便横向对比
VERSION="0528_2-advanced-tools"
mkdir -p /mnt/workspace/$VERSION
cp -r * /mnt/workspace/$VERSION/

# 进该版本目录运行 notebook
cd /mnt/workspace/$VERSION
# 然后打开 notebook 或 jupyter lab
```

**进入 Claude Code（华为云终端）：**
```bash
source ~/.bashrc
claude
```

## 关键创新点

### BM25 Query Rewriting（自然语言→关键词改写）
**问题:** BrowseComp-Plus 的问题用自然语言描述，但 BM25 是纯关键词匹配，拿 "a book about certain inland discoveries" 这种模糊描述去搜基本召回不到相关文档。

**方案:** 在 `search()` 工具内部加了一层 LLM 改写。模型可以用自然语言写 query，工具自动提取具体实体（人名、日期、数字、独特术语），去掉模糊词（about, certain, something），输出 BM25 友好的关键词组合。

例如 q442 的查询：
```
模型输入 query: "a book about certain inland discoveries in the 1920s"
改写后实际检索: "1920s inland discoveries"
```

或者用问题里的具体细节直接撞:
```
模型输入 query: "a description of a barrel-shaped floating vessel on page 332-339"
改写后实际检索: "barrel-shaped floating vessel 332-339"
```

这解决了 BM25 的第一跳召回问题，让相关文档有机会进入 top-10。改写在 `tools.py` 的 `_rewrite_for_bm25()` 实现，对模型透明，无需修改 agent loop。

### 强制读全文机制
模型搜到 snippet 后常直接猜答案。Agent 在检测到模型连续搜了 2 次不读全文时，自动插入 user message 强制调用 `get_document()`。

### 自动拆题 + 回答前验证
Loop 开始前自动调 `decompose_question()` 将复杂问题拆成子查询；模型想回答时先强制 `verify_claim()` 核对证据。

### 自动加载 top-1 全文
首次搜索后自动调用 `get_document()` 获取排名第一文档的完整文本（前 3000 字符），直接注入到 conversation。解决模型只看 snippet 就猜答案的 hallucination 问题。

### 三次强制读文档
模型搜索后不调用 `get_document()` 时，agent 自动插入提示要求读全文。`unique_docs_read` 集合跟踪已读文档 ID，`max_docs_to_read=3` 控制目标，重复引导直到模型读完足够文档。

### 搜索去重（Token 重叠检测）
`SimpleTracker.is_duplicate_query()` 使用 Jaccard token 重叠度 > 80% 检测近似重复查询，命中时给出提示，减少 BM25 无效检索。

## V1 Five Improvements (2026-05-29)

### Plan 1: Multi-Query 搜索扩展
单个 search() 调用自动扩展为 4 个不同角度的 BM25 查询（1 个原始改写 + 3 个 LLM 生成的多样化变体），所有结果合并去重。大幅提高复杂问题的第一跳召回率，解决了"BM25 用自然语言描述搜不到匹配文档"的核心问题。

实现: `tools.py` 的 `_expand_queries()` + 修改 `search()` 闭包。

### Plan 2: 强制证据引用
答案格式改为强制要求 `Evidence: <docid> "<verbatim quote>"` + `Exact Answer` + `Confidence`。模型答案中缺少 Evidence 字段时自动阻断并提醒，直接对抗 hallucination。

实现: 更新 SYSTEM_PROMPT + 回答前格式检查。

### Plan 3: 自动 find_in_doc 精确定位
Agent 启动时用 LLM 从问题中提取 5 个关键实体/短语（`_extract_key_terms`）。模型调用 `get_document()` 读长文档（>1500 字符）时，系统自动对每个关键实体执行 `find_in_doc()`，将匹配段落注入对话。解决"读书的开头 3000 字符但答案在 300 页之后"的问题。

实现: `deep_research_agent.py` 的 `_extract_key_terms()` + solve() 中的自动触发逻辑。

### Plan 4: 分层检索（搜索专用阶段）
前 3 轮强制为搜索阶段，任何回答尝试都被阻断并引导继续搜索。第 4 轮起才允许回答。确保模型在试图回答前有足够的证据收集。

实现: `tool_choice="required"` 覆盖 rounds 1-3 + 回答前轮次检查。

### Plan 5: 硬化 Verification 循环
模型试图回答时，系统自动调用 `verify_claim()`（不依赖模型自主调用），解析返回的 "Supported: YES/NO"。NO 时自动阻断回答路径，注入失败信息强制继续搜索。YES 时才允许输出最终答案。

实现: solve() 中的自动验证逻辑 + 阻断+重搜循环 + verify_passed 状态追踪。

### 停止条件放宽
原停止逻辑实质是"任意 1 轮无新文档即停"。改为连续 3 轮无新文档才触发，给模型更多搜索空间。实现：`SimpleTracker.consecutive_no_new_docs` 计数器。

## 评估

- 自动评估: `agent/eval.py` 或 notebook cell 5
- LLM judge 判断 CORRECT/INCORRECT
- 准确率 ≥12% 开始计分
- 实验记录保存在 `EXPERIMENT_LOG.md`

## BrowseComp-Plus 题目风格分析 (hard50)

### 问题结构
所有 50 题都是**多跳推理链**，典型结构：
```
实体 A → 文档 X → 实体 B → 文档 Y → ... → 答案
```
答案类型：人名（最常见）、书名、公司名、数字/年份、具体实体名。

### 核心检索困难
BM25 搜索效果取决于问题中是否包含**可检索的专有名词**。按 BM25 友好度可分为两类：

| 类别 | 占比 | 特征 | 示例 |
|------|------|------|------|
| **BM25 友好型** | ~25% | 包含组织名、机构名、人名、具体数字等关键词 | q5 (Malaria Consortium, Ogilvy & Mather, WHO) |
| **BM25 不友好型** | ~75% | 全自然语言描述，无专有名词，高度抽象 | q442 ("certain inland discoveries", "barrel-shaped floating vessel") |

**模型得分主要来自 BM25 友好型**——这些题通过 1-2 次 search 就能定位到相关文档。

### BM25 不友好题型的根因
1. **描述性语言**：题目用"关于某内陆发现的书"代替书名，用"某人的父亲"代替人名
2. **时间范围约束**：大量 `between X and Y (inclusive/exclusive)`，BM25 无法理解数值范围
3. **串行推理依赖**：必须跨文档追踪实体链，单次 search 最多命中一环
4. **干扰文档极多**：平均 70-100+ 干扰文档/题，BM25 top-10 信噪比极低

### 对策略的影响
- BM25 友好型题是主要得分来源 → 前几轮快速识别并作答
- BM25 不友好型题 → 关键词改写 + Multi-Query 多点爆破 + 最后轮 best guess
- 模型"放弃"行为导致准确率降为 0 → 必须强制最后轮输出答案，不准说不知道

## 当前进度 (2026-05-29 全面归档)

### 已完成

- [x] **V1 Baseline**: 8.00% (4/50), 57% tc=0 错误
- [x] **V1 prompt 改进 + tool_choice**: tool_choice="required"(Qwen3 不支持) → auto + retry
- [x] **V2 bug 诊断**: 6 模块设计在 32K 上下文 + 8B 模型下不可行，已废弃
- [x] **5 个工具**: search(Multi-Query) / get_document / find_in_doc / decompose_question / verify_claim
- [x] **System prompt**: 5 步命令式流程（Search→Read→Cross-ref→Verify→Answer）
- [x] **自动 top-1 全文加载**: 首次搜索后自动注入 top 文档全文前 3000 字符
- [x] **停止条件放宽**: 1 轮 → 3 轮连续无新文档
- [x] **多次强制读**: 最多 3 次强制 get_document 提示
- [x] **搜索去重**: Jaccard token 重叠 > 80% 检测
- [x] **BM25 Query Rewriting**: search() 内部 LLM 关键词改写
- [x] **Plan 1: Multi-Query**: 单 search() 自动 4 查询合并（原始改写 + 3 多样化变体）
- [x] **Plan 2: 强制证据引用**: Evidence 格式 + 自动阻断
- [x] **Plan 3: 自动 find_in_doc**: 关键实体提取 + 长文档精确定位
- [x] **Plan 4: 分层检索**: 前 1 轮搜索专用（原 3 轮太严导致崩溃）
- [x] **Plan 5: 硬化 Verification**: 自动验证 + 阻断 + 重搜循环 + 死胡同检测
- [x] **Component 1: 死胡同检测**: verify 失败且无新文档 → force_final_answer
- [x] **Component 2: 上下文管理**: 剥离 think 块(省30-50%) + round 4 压缩 + verify 256 max_tokens
- [x] **Component 3(v1.6)**: STOP CONDITIONS（已删除 → 过保守导致全放弃）
- [x] **Component 3(v1.7)**: FINAL ANSWER REQUIREMENT + enforce_concrete_answer 拒绝检测
- [x] **上云验证 v1.6**: 6.00% (3/50), avg tc=3.52, 无 HTTP 400 — STOP CONDITIONS 矫枉过正
- [x] **v1.7 本地修复**: 删除 STOP CONDITIONS + enforce_concrete_answer + 18 种拒绝模式检测
- [x] **Notebook 清理**: 3.5/4 标签修正, 删除 6/7 节
- [x] **题目风格分析**: hard50 25% BM25友好 + 75% BM25不友好
- [x] **轨迹分析存档**: trajectories/0528-7_tool_choice_fix/ (q442 + analysis)
- [x] **REPORT.md 重写**: 反映真实架构(V1主线/V2废弃) + 完整实验历程
- [x] **记忆文件更新**: nlp_deep_research_status.md + MEMORY.md

### 待完成（按优先级）

- [ ] **P0: v1.7 上云验证** — git push → git pull → 修复 BM25 索引 → 跑全量 50 题 → 目标 ≥12%
- [ ] **P1: 实体爆破搜索** — _expand_queries() 改为每个实体独立 search，各取 top-5
- [ ] **P1: verify 结果收割** — verify_claim 返回 "Correct Answer: X" 时直接采纳
- [ ] **P1: 最后 2 轮温度 0.0→0.3** — 增加答案多样性
- [ ] **P2: 消融实验** — 子集 ~30 分钟
- [ ] **P2: 轨迹正例分析** — 收集 6 道正确题轨迹
- [ ] **P3: 最终提交** — 6 月 2 日前选定版本→跑 full 50→生成 submission→提交
