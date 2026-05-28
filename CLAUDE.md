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
│   ├── tools.py                     # 工具定义：search / get_document / find_in_doc
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

三个工具函数（`get_agent_tool_specs_and_registry` 返回）:
1. `search(query)` → top-10 文档 snippet
2. `get_document(docid)` → 完整文档
3. `find_in_doc(docid, keyword)` → 文档内关键词搜索

### agent/deep_research_agent_v2.py (V2, 有 bug)

模块: Question Analyzer → Query Rewriter → Evidence Tracker → Relevance Filter → Context Compressor → Verifier

已知问题:
- Query Rewriter 会把 LLM thinking 文本当成搜索 query
- Relevance Filter 每轮对每篇文档额外调 LLM，调用量爆炸
- 模型在复杂 prompt 下大量 token 浪费在自我辩论
- **当前建议放弃 V2，主攻 V1 ReAct**

## Known Issues & Pitfalls

1. **Qwen3 不调工具**: 即使有 tool 定义 + prompt 要求，Qwen3 `tool_choice="auto"` 可能直接凭记忆回答。修复: Round 1 强制 `tool_choice="required"`
2. **Thinking 模式**: Qwen3 输出 `<think>...</think>` 推理过程后才回答。需要确保 parser 正确处理 thinking 块
3. **vLLM 路径**: 华为云上模型在 `/opt/huawei/edu-apaas/src/init/Qwen3-8B`，不是 `./Qwen3-8B`
4. **文件覆盖**: 多次运行 notebook 不覆盖 `submission.jsonl` — 已改为时间戳命名 `submission_MMDD_HHMM.jsonl`
5. **云上代码同步**: git pull 后 `cp -r * /mnt/workspace/` 同步到工作区
6. **工作区优先**: 先在 `/mnt/workspace/` 测试验证，确认没问题再复制到持久化存储 `/opt/huawei/.../`

## 华为云配置

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

**同步代码:**
```bash
cd /opt/huawei/edu-apaas/src/init/deep-research-agent
git pull origin main
cp -r * /mnt/workspace/
```

## 评估

- 自动评估: `agent/eval.py` 或 notebook cell 5
- LLM judge 判断 CORRECT/INCORRECT
- 准确率 ≥12% 开始计分
- 实验记录保存在 `EXPERIMENT_LOG.md`

## 当前进度 (2026-05-28)

- [x] V1 baseline: 8.00% (4/50), 57% tc=0 错误
- [x] V1 prompt 改进 + tool_choice="required"
- [x] V2 bug 诊断
- [x] 新增 find_in_doc 工具
- [ ] tool_choice="required" 对 Qwen3 不够，已加 retry 机制
- [ ] 迭代 V1 提高准确率
- [ ] V2 修复或重构
- [ ] 消融实验
- [ ] 提交最终结果
