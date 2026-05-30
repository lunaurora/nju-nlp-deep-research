# v1.6 全量评估分析 — 6.00% (3/50)

## 基本信息

| 项目 | 值 |
|------|-----|
| 版本 | V1 v1.6 (think剥离 + 死胡同检测 + STOP CONDITIONS + tool_choice="auto") |
| 评估集 | hard50 (50 题) |
| 准确率 | **6.00%** (3/50) |
| 平均工具调用 | 3.52 |
| 平均检索文档 | 0.02 |
| 提交 | submission_0529_0021 |
| 时间 | 2026-05-29 00:21 |
| HTTP 400 | **0** (已修复 ✅) |

## 错误模式统计

```
正确:      3 (6.00%)
错误:     47 (94.00%)
├── hallucination:       40 (85%)  — 搜了但凭记忆答错
├── insufficient_search:  7 (15%)  — 仅 0-1 次工具调用
└── context_overload:     0 (0%)   — HTTP 400 已修复
```

## 与历史对比

| 轮次 | 准确率 | tc/query | docs/query | hallucination | 上下文溢出 |
|------|--------|----------|------------|---------------|-----------|
| Baseline | 8.00% | 1.24 | 0.02 | 43% | 0% |
| +retry+高级工具 | 6.00% | 1.96 | 0.02 | 49% | 0% |
| +强制读+拆分+验证 | 6.00% | 3.22 | 0.12 | 98% | 0% |
| +top-1全文+宽松停止(12%) | **12.00%** | 4.96 | 0.24 | 95% | 0% |
| Plan 1-5 (首次) | 崩溃 | ~11 | — | — | HTTP 400 |
| **v1.6** | **6.00%** | **3.52** | **0.02** | **85%** | **0% ✅** |

## 根因分析

### 为什么 12% → 6%？

STOP CONDITIONS 章节的引入是直接原因：

```
## STOP CONDITIONS (v1.6 新增, v1.7 已删除)
- If you searched 3+ different queries and found NO relevant documents → give Best Guess
- If you read all available documents and verification still fails → give Best Guess
- Do NOT repeat the same answer after verification says it is unsupported
```

Qwen3-8B 本身倾向于谨慎/拒绝回答。加上 STOP CONDITIONS 后，模型学会说 "cannot be determined"，即使有部分线索也不猜了。

对比：

| 版本 | 回答模式 | 效果 |
|------|----------|------|
| v1.3 (12%) | 总是猜一个答案，即使错了 | 6/50 正确 |
| v1.6 (6%) | 证据不足就说 cannot be determined | 3/50 正确，45/50 放弃 |

### 为什么 docs/query = 0.02？

50 题总共只读了 1 篇文档全文。模型倾向：
1. 只看 BM25 snippet（800 字符）就猜答案
2. 即使有 3 次 "CRITICAL: read get_document()" 提示，仍然忽略
3. Qwen3 在"搜不到相关文档"时倾向于不断换 query 搜，而不是去读已有文档

### STOP CONDITIONS 的教训

"告诉模型可以放弃"在 Qwen3 上适得其反。更好的策略是：
- **永远禁止放弃** — prompt 明确说 "wrong guess is better than no answer"
- **检测放弃模式** — enforce_concrete_answer() 检查 18 种拒绝措辞
- **强制猜测** — 发现拒绝时替换为 "[Best Guess — model refused to answer]"

## 三题正确的可能原因

v1.6 答对的 3 题（qids 待确认，推测为 BM25 友好型的子集）：
- 包含专有名词 → BM25 第一跳命中
- 模型读到了相关 snippet → 凭训练记忆也能答对部分
- 或者答案在 BM25 snippet 中直接可见

## 修复方向 (v1.7)

1. **删除 STOP CONDITIONS** → 改为 FINAL ANSWER REQUIREMENT
2. **enforce_concrete_answer()** → 拒绝检测 + 强制猜测占位符
3. **所有强制输出路径** → "cannot be determined is NOT allowed"
