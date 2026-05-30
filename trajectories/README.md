# Trajectories Archive — Deep Research Agent

## 文件夹结构说明

```
trajectories/
├── README.md                        # 本文件
├── 0528-7_tool_choice_fix/          # tool_choice="auto" 修复后首次云运行
│   ├── q442_trajectory.md           # 完整轨迹（轮次/工具调用/模型输出）
│   └── analysis.md                  # 问题诊断与修复建议
└── ...                              # 后续实验轨迹按日期-版本命名
```

## 命名规范

- 文件夹: `MMDD-N_描述`（与 workspace version 对应）
- 文件: `q{id}_trajectory.md` 或 `analysis.md`

## 收集目的

为课程实验报告提供原始轨迹数据和分析素材，包括：
1. 每轮模型思考与工具调用
2. BM25 检索结果
3. 错误模式和根因分析
4. 修复前后对比
