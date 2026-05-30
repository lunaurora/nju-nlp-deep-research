# 分析报告 — v1.6 tool_choice 修复后首次云运行

## 实验概况

- **目标**: 验证 `tool_choice="required"` → `"auto"` 修复是否能解决 HTTP 400
- **测试题**: q442（hard50 中较复杂的一题）
- **结果**: ✅ HTTP 400 修复，但答案错误（hallucination）

## 问题诊断

### 问题 1: BM25 零召回（最严重）
search 返回的文档与问题完全无关：
- 搜 "botanist returned to London 1830s" → 返回 "Technical Reports" (docid 46212)
- 搜 "author married 1890s botanist" → 返回 "APA Format" (docid 28618)
- 搜 anything → 都返回无关文档

**根因推测**：云环境的 BM25 SQLite 索引 (`browsecomp_plus_bm25.sqlite`) 路径不对，或索引文件本身有问题（可能只索引了部分语料或索引损坏）。

### 问题 2: 模型拒绝读全文
即使有 3 次 "CRITICAL: read get_document('45781')" 强制提示，模型仍然只调 search 不调 get_document。

**根因推测**：Qwen3 在 BM25 返回无关文档后认为"搜不到"，倾向于反复换角度搜索而不是去读已有文档。prompt 的强制读指令优先级不够高。

### 问题 3: 重复查询检测不够严格
R2 和 R3 的 query 完全相同（"author married 1890s botanist 1830s spear attack barrel-shaped vessel"），但系统只提醒"有重叠"没阻止。

**根因推测**：SimpleTracker 的 Jaccard 相似度阈值 >80% 才判定完全重复，这段查询的 token 重叠度可能在 80% 以下。

### 问题 4: verify 传入 think 文本
Verification 的输入看起来包括了 `<think>` 块内容，导致 verify 判断的是模型的推理过程而非实际答案。

**根因推测**：`<think>` 剥离可能在 verify 调用之后才执行，或者 verify 函数拿到的 message 尚未经过 think 过滤。

## 修复建议

### 高优先级
1. **确认 BM25 索引路径正确** — 检查云上 `indexes/` 的位置，确保 SQLite 文件存在且包含完整语料
2. **在 notebook 中显示索引状态** — 在运行前 cell 打印 `os.listdir('indexes/')` 验证索引文件存在

### 中优先级
3. **加强 get_document 强制逻辑** — 3 次提示不读时，直接注入 tool_call 调用 get_document（而非提示）
4. **修复 verify 的 think 问题** — 确保 verify 调用前剥离所有 assistant 消息中的 think 块

### 低优先级
5. **强化重复检测** — 降低 Jaccard 阈值到 70% 或直接比较字符串相等
