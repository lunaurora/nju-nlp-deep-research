# Deep Research Agent — 对话归档与交接文档

> 生成日期：2026-05-29 21:30
> 目的：下次启动新终端时快速恢复上下文

---

## 一、项目核心信息

| 项目 | 值 |
|------|-----|
| 任务 | BrowseComp-Plus Deep Research Agent |
| 课程 | NLP @ NJU, 吴震 |
| 助教 | 张轩望、田阔 |
| DDL | **2026-06-02 20:00** |
| 模型 | Qwen3-8B (vLLM, Ascend NPU, 32K 上下文) |
| 检索 | BM25 (SQLite FTS5, **固定不可替换**) |
| 评估集 | hard50 (50 题) |
| 及格线 | ≥12% |
| 历史最佳 | **12.00%** (v1.3 四项改进) |
| 当前版本 | **v1.7** (待上云验证) |

## 二、架构决策

### 最终方案：V1 ReAct Agent

**V2 模块化架构已废弃**（6 模块设计在 32K + 8B 下不可行）

### 5 个工具

| 工具 | 类型 | 关键特性 |
|------|------|----------|
| `search(query)` | 基础 | BM25 → _rewrite_for_bm25 → _expand_queries(4合1) |
| `get_document(docid)` | 基础 | 读全文 |
| `find_in_doc(docid, keyword)` | 基础 | 文档内关键词定位 |
| `decompose_question(q)` | 高级(LLM) | 拆解复杂问题 |
| `verify_claim(claim, docids)` | 高级(LLM) | 验证答案(max_tokens=256) |

### Agent 核心流程

```
1. 提取关键实体 (_extract_key_terms)
2. 自动拆题 (decompose_question)
3. ReAct Loop 1-8 轮:
   - Round 1: tool_choice="auto" + retry(强制search on failure)
   - Round 4: context compression
   - 每轮: 剥离 <think> 块, 执行工具, 强制读文档检测
   - 模型想回答时: auto-verify → 死胡同检测 → Evidence 格式检查
   - 停止检测: 3轮无新文档 / max_rounds / verify_passed
4. 所有返回路径: enforce_concrete_answer() 拒绝检测
```

## 三、实验数据

### 完整实验历程

| 轮次 | 版本 | 准确率 | tc/query | 关键问题 |
|------|------|--------|----------|----------|
| 1 | Baseline | 8.00% | 1.24 | 57% 不调工具 |
| 2 | +retry+高级工具 | 6.00% | 1.96 | 新 eval 更严 |
| 3 | +强制读+拆分+验证 | 6.00% | 3.22 | 98% hallucination |
| 4 | +top-1全文+宽松停止+多次强读+去重 | **12.00%** | 4.96 | **过及格线** |
| 5 | +Plan 1-5 (首次) | **崩溃** | ~11 | HTTP 400 |
| 6 | +三项修复+v1.6 | 6.00% | 3.52 | STOP CONDITIONS 过保守 |
| **7** | **v1.7 删STOP+强制输出** | **待验证** | — | — |

### v1.6 评估详情 (6.00%)

- 正确 3 题：id 待确认
- hallucination: 85% (40/47 错误)
- insufficient_search: 15% (7/47)
- HTTP 400: 0 ✅
- avg tc: 3.52
- avg docs: 0.02 (50 题只读了 1 篇全文)

### 12% 评估详情 (12.00%)

- 正确 6 题：q5, q53, q397, q432, q684, q1095
- hallucination: 95% (42/44 错误)
- avg tc: 4.96
- avg docs: 0.24

## 四、关键文件索引

### 核心代码

| 文件 | 说明 |
|------|------|
| `agent/deep_research_agent.py` | **V1 主线 Agent** — 所有改进都在这里 (554 行) |
| `agent/tools.py` | 5 个工具定义 + BM25 Query Rewriting + Multi-Query (363 行) |
| `agent/eval.py` | LLM 自动评估 (437 行) |
| `agent/vllm_client.py` | vLLM HTTP 客户端 (48 行) |
| `agent/browsecomp_searcher.py` | BM25 检索实现 (217 行) |
| `agent/deep_research_agent_v2.py` | V2 已废弃 (779 行) |

### 实验文档

| 文件 | 说明 |
|------|------|
| `EXPERIMENT_LOG.md` | 完整实验记录（所有版本的数据和错误分析） |
| `REPORT.md` | **课程报告**（架构设计 + 实验过程 + 错误分析，已重写） |
| `CLAUDE.md` | 项目知识库（开发指南 + 所有已知问题和修复） |
| `ARCHIVE.md` | **本文件**（对话归档与交接） |

### 轨迹存档

| 文件 | 说明 |
|------|------|
| `trajectories/README.md` | 存档结构说明 |
| `trajectories/0528-7_tool_choice_fix/q442_trajectory.md` | q442 完整 7 轮轨迹 |
| `trajectories/0528-7_tool_choice_fix/analysis.md` | 问题诊断与修复建议 |
| `trajectories/0528-7_tool_choice_fix/eval_analysis_v1.6.md` | v1.6 全量评估分析 |

### 数据文件

| 文件 | 说明 |
|------|------|
| `browsecomp_plus_hard50.jsonl` | hard50 评估集（50 题，含 query/answer/evidence docs） |
| `browsecomp-plus-corpus/` | 全量语料（~10K 文档，parquet 格式） |
| `indexes/browsecomp_plus_bm25.sqlite` | BM25 索引（构建后生成） |

## 五、未完成任务

### P0：恢复 12% 基线（最紧急）

```
1. git push v1.7 → 云上 git pull → cp 到版本文件夹
2. 修复 BM25 索引（确认索引路径和语料一致）
3. 建版本文件夹 → 跑 hard50 全量 → 目标 ≥12%
4. 记录 eval_results
```

**文件**：`agent/deep_research_agent.py`（已改好，已 git add 暂存）

### P1：针对性改进（有潜力提升 2-4%）

| 改进 | 文件 | 改动量 | 预期收益 |
|------|------|--------|----------|
| **实体爆破搜索** — 每个实体独立 search，各取 top-5 | `tools.py` ~20 行 | +2-4% |
| **verify 结果收割** — verify 返回正确答案时直接采纳 | `agent.py` ~15 行 | +1-2% |
| **最后 2 轮温度** — max_rounds-2 起 temperature=0.0→0.3 | `agent.py` 1 行 | +1-2% |
| **top_k=10→20** — 每次 search 合并更多文档 | `agent.py` 1 行 | +1-2% |

### P2：课程报告素材

- [ ] 消融实验（子集 ~30 分钟）
- [ ] 轨迹正例分析（收集 6 道正确题轨迹）
- [ ] REPORT.md 错误分类补充

### P3：最终提交（6 月 2 日前）

1. 选定最佳版本 → 跑 full 50
2. 生成 submission.jsonl
3. 完成 REPORT.md
4. 提交

## 六、已知问题和修复方案

### 问题 1：BM25 零召回（云环境）

**症状**：search 返回 "APA Format"、"Technical Reports"、"FDA Labeling Documents" 等完全无关文档
**原因**：BM25 索引路径不对，或索引文件指向错误语料
**修复**：云上验证 indexes/ 路径，必要时 rebuild 索引

### 问题 2：模型拒绝读全文

**症状**：即使 3 次 "CRITICAL" 提示，模型只搜不读
**原因**：Qwen3 在 BM25 返回无关文档后认为"搜不到"，倾向换角度搜而非读已有文档
**修复**：当前无完美方案。可以考虑注入 tool_call 而不是提示（但 vLLM 不支持强制 tool_call）

### 问题 3：重复查询检测不够严格

**症状**：R3 提交与 R2 完全相同 query，只提醒未阻止
**修复**：降低 Jaccard 阈值或直接字符串相等比较

### 问题 4：verify 可能传入 think 文本

**症状**：verification prompt 含 `<think>` 块内容
**修复**：确保 extract_answer() 在 verify 调用前先去 think。当前代码中剥离在 assistant 消息加入 messages 时已执行，需验证 verify 调用路径。

## 七、git 状态

当前暂存（已 git add，未 commit）：
- `agent/deep_research_agent.py` — v1.7 修复
- `agent_vllm_deep_research.ipynb` — 3.5/4 标签修复 + 删 6/7 节
- `CLAUDE.md` — 同步更新
- `EXPERIMENT_LOG.md` — v1.6+v1.7 记录
- `trajectories/` — 轨迹存档

## 八、华为云环境

- **平台**: science.lab.huaweicloud.com
- **vLLM 服务**: 端口 8000, 模型名 qwen_auto
- **模型路径**: `/opt/huawei/edu-apaas/src/init/Qwen3-8B`
- **持久化**: `/opt/huawei/edu-apaas/src/init/deep-research-agent/` (git)
- **工作区**: `/mnt/workspace/`
- **同步**: git pull → `cp -r * /mnt/workspace/MMDD_N-描述/`
- **Claude Code**: `/opt/huawei/edu-apaas/src/init/node/` → `claude` 命令

## 九、关键经验教训

1. **STOP CONDITIONS 在 Qwen3 上适得其反** — 模型本身就倾向于拒绝回答，应该永远禁止放弃
2. **死胡同检测是最关键的修复** — verify 失败 + 无新文档 = 直接强制输出，最多浪费 1 轮而非 7 轮
3. **轻量 ReAct > 模块化架构** — 在 32K + 8B + BM25 的限制下，简洁的循环比复杂模块更可靠
4. **BM25 友好度决定得分** — 25% 含专有名词的题贡献了几乎所有正确分，75% 自然语言题需要关键词改写
5. **模型幻觉是核心问题** — 即使有全文证据，Qwen3 仍然倾向凭训练记忆回答
6. **`<think>` 块消耗大量 token** — 剥离后每条回复省 30-50%，对后续轮次完全无影响
7. **`tool_choice="required"` Qwen3 不支持** — vLLM 的 Hermes parser 不接受 string 格式，需用 `{"type": "function", "function": {"name": "search"}}` 对象格式
