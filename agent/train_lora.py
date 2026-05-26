"""
LoRA 微调模块 — 用成功搜索轨迹微调 Qwen3-8B

流程:
  1. 用 V2 Agent 跑 hard50，收集正确案例的完整轨迹
  2. 格式化轨迹为 SFT 训练数据
  3. LoRA 微调 Qwen3-8B
  4. 加载微调后模型做推理

用法:
  # 1. 收集数据
  python -m agent.train_lora collect \
      --model qwen_auto --base-url http://127.0.0.1:8000/v1 \
      --dataset browsecomp_plus_hard50.jsonl \
      --output trajectories.jsonl

  # 2. 训练 (需要有 GPU)
  python -m agent.train_lora train \
      --data trajectories.jsonl \
      --base-model ./Qwen3-8B \
      --output-dir ./lora_checkpoints \
      --epochs 3

  # 3. 推理评估
  python -m agent.train_lora eval \
      --adapter ./lora_checkpoints/final \
      --base-model ./Qwen3-8B \
      --dataset browsecomp_plus_hard50.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════
#  第一步: 数据收集 — 从 V2 Agent 收集成功轨迹
# ═══════════════════════════════════════════════════════════════

def collect_trajectories(
    dataset_path: str,
    output_path: str,
    model_name: str = "qwen_auto",
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "dummy",
    max_samples: int = 0,
    use_v2: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    用 Agent 跑数据集，只保留正确的轨迹作为训练数据。

    需要 vLLM 服务已启动。
    """
    # 动态 import agent（只在有 vLLM 时才能调）
    sys.path.insert(0, str(Path.cwd()))
    from agent.vllm_client import VLLMClient
    from agent.tools import build_searcher
    from agent.dataset_utils import load_jsonl
    from agent.eval import run_evaluation

    if use_v2:
        from agent.deep_research_agent_v2 import DeepResearchAgentV2 as AgentClass
    else:
        from agent.deep_research_agent import DeepResearchAgent as AgentClass

    # 初始化
    client = VLLMClient(base_url=base_url, api_key=api_key)
    searcher = build_searcher(
        index_path=os.environ.get("BM25_INDEX", "indexes/browsecomp_plus_bm25.sqlite")
    )

    agent = AgentClass(
        client=client, searcher=searcher, model_name=model_name,
        max_rounds=10, max_tokens=2048, top_k=15, temperature=0.0,
    )

    rows = load_jsonl(dataset_path)
    if max_samples > 0:
        rows = rows[:max_samples]

    # 收集所有轨迹
    all_records = []
    for i, row in enumerate(rows):
        qid = row["query_id"]
        if verbose:
            print(f"[{i+1}/{len(rows)}] {qid}...", end=" ", flush=True)

        result = agent.solve(question=row["query"], query_id=qid)
        all_records.append({
            "query_id": qid,
            "question": row["query"],
            "gold_answer": row["answer"],
            "predicted_answer": result["predicted_answer"],
            "messages": result["messages"],
            "num_tool_calls": result["num_tool_calls"],
        })
        if verbose:
            print(f"tc={result['num_tool_calls']} ans={result['predicted_answer'][:50]}")

    # 评估
    if verbose:
        print("\nEvaluating to filter successful trajectories...")

    # 写临时提交文件让 eval 读
    tmp_sub = Path(output_path).with_suffix(".tmp_submission.jsonl")
    with open(tmp_sub, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps({
                "query_id": rec["query_id"],
                "predicted_answer": rec["predicted_answer"],
                "status": "completed",
                "messages": rec["messages"],
            }, ensure_ascii=False) + "\n")

    summary, details = run_evaluation(
        submission_path=str(tmp_sub),
        dataset_path=dataset_path,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        verbose=verbose,
    )

    # 标记成功/失败
    qid_to_judgment = {d["query_id"]: d["eval_judgment"] for d in details}
    successful = []
    for rec in all_records:
        rec["correct"] = qid_to_judgment.get(rec["query_id"], "INCORRECT") == "CORRECT"
        if rec["correct"]:
            successful.append(rec)

    # 保存全部 + 标注
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 清理临时文件
    tmp_sub.unlink(missing_ok=True)

    if verbose:
        print(f"\n{'='*50}")
        print(f"Total: {len(all_records)}  |  Successful: {len(successful)}  |  ACC: {summary['accuracy']:.2%}")
        print(f"Saved to: {output_path}")

    return successful


# ═══════════════════════════════════════════════════════════════
#  第二步: 数据格式化 — 轨迹 → SFT 训练格式
# ═══════════════════════════════════════════════════════════════

def format_trajectory_for_sft(messages: List[Dict[str, Any]]) -> str:
    """
    将 OpenAI 格式的 messages 转为 Qwen3 的 SFT 训练文本。

    格式:
      <|im_start|>system
      ...<|im_end|>
      <|im_start|>user
      ...<|im_end|>
      <|im_start|>assistant
      ...<|im_end|>
    """
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "") or ""

        # 处理 tool_calls
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            # 把 tool_calls 序列化成文本格式
            tc_lines = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                tc_lines.append(
                    f'{{"name": "{fn.get("name", "")}", '
                    f'"arguments": {fn.get("arguments", "{}")}}}'
                )
            content = content + "\n[TOOL_CALLS]\n" + "\n".join(tc_lines)

        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    return "\n".join(parts)


def prepare_sft_dataset(
    trajectories_path: str,
    output_path: str,
    max_examples: int = 0,
    only_correct: bool = True,
    val_split: float = 0.1,
):
    """
    将收集的轨迹转为 HuggingFace Dataset 格式的 JSONL。
    每条包含 "text" 字段，可直接用于 SFTTrainer。
    """
    records = []
    with open(trajectories_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    if only_correct:
        records = [r for r in records if r.get("correct", False)]

    if max_examples > 0:
        records = records[:max_examples]

    if not records:
        print("WARNING: No successful trajectories found. Cannot create training data.")
        return

    # 格式化
    formatted = []
    for rec in records:
        text = format_trajectory_for_sft(rec["messages"])
        formatted.append({
            "text": text,
            "query_id": rec["query_id"],
            "question": rec["question"],
            "gold_answer": rec["gold_answer"],
        })

    # 分 train/val
    n_val = max(1, int(len(formatted) * val_split))
    train_data = formatted[:-n_val]
    val_data = formatted[-n_val:]

    # 保存
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, data in [("train", train_data), ("val", val_data)]:
        path = output_dir / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                # 只保存 text 字段用于训练
                f.write(json.dumps({"text": item["text"]}, ensure_ascii=False) + "\n")
        print(f"  {split}: {len(data)} examples → {path}")

    print(f"\nTotal training examples: {len(train_data)}")
    print(f"Total validation examples: {len(val_data)}")

    return str(output_dir)


# ═══════════════════════════════════════════════════════════════
#  第三步: LoRA 训练
# ═══════════════════════════════════════════════════════════════

LORA_TRAINING_SCRIPT = r"""
import sys
sys.path.insert(0, ".")

import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer

# ── 配置 ──────────────────────────────────────────────────
BASE_MODEL = "{base_model}"
DATA_DIR = "{data_dir}"
OUTPUT_DIR = "{output_dir}"
EPOCHS = {epochs}
BATCH_SIZE = {batch_size}
LR = {lr}
LORA_R = {lora_r}
LORA_ALPHA = {lora_alpha}
LORA_DROPOUT = {lora_dropout}
USE_QLORA = {use_qlora}
MAX_SEQ_LEN = {max_seq_len}

# ── 加载 tokenizer ─────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── 加载模型 ───────────────────────────────────────────────
if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
else:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

# ── LoRA 配置 ──────────────────────────────────────────────
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── 加载数据 ───────────────────────────────────────────────
dataset = load_dataset("json", data_files={{
    "train": f"{DATA_DIR}/train.jsonl",
    "validation": f"{DATA_DIR}/val.jsonl",
}}, split="train")

# ── 训练参数 ──────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    bf16=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    report_to="none",
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=MAX_SEQ_LEN,
    dataset_text_field="text",
)

# ── 训练 ──────────────────────────────────────────────────
trainer.train()

# ── 保存 ──────────────────────────────────────────────────
final_path = f"{{OUTPUT_DIR}}/final"
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)
print(f"\\nModel saved to: {{final_path}}")
"""


def generate_training_script(
    base_model: str = "./Qwen3-8B",
    data_dir: str = "./sft_data",
    output_dir: str = "./lora_checkpoints",
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_qlora: bool = True,
    max_seq_len: int = 4096,
    output_script: str = "run_lora_train.py",
):
    """生成可独立运行的 LoRA 训练脚本。"""
    script = LORA_TRAINING_SCRIPT.format(
        base_model=base_model,
        data_dir=data_dir,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_qlora=str(use_qlora),
        max_seq_len=max_seq_len,
    )

    with open(output_script, "w", encoding="utf-8") as f:
        f.write(script)

    print(f"Training script generated: {output_script}")
    print(f"Run with: python {output_script}")
    return output_script


# ═══════════════════════════════════════════════════════════════
#  第四步: 用 LoRA 微调后的模型做推理
# ═══════════════════════════════════════════════════════════════

def create_lora_inference_script(
    base_model: str = "./Qwen3-8B",
    adapter_path: str = "./lora_checkpoints/final",
    output_script: str = "run_lora_inference.py",
    merged_model_path: str = "./Qwen3-8B-lora-merged",
):
    """
    生成 LoRA 推理脚本 — 加载微调后模型 + vLLM 服务。

    有两种方式:
      1. 合并 LoRA 权重到 base model，然后正常起 vLLM
      2. 直接起 vLLM 带 LoRA adapter（vLLM 支持 --lora-path）

    这里用方式 2（vLLM 原生支持 LoRA）。
    """
    script = f'''"""
启动合并 LoRA 权重的 vLLM 服务。

用法:
  python {output_script}

然后 notebook 里 model_name 设为 'qwen_lora' 即可。
"""

import subprocess
import sys

BASE_MODEL = "{base_model}"
ADAPTER_PATH = "{adapter_path}"
MODEL_NAME = "qwen_lora"

print(f"Starting vLLM with LoRA adapter...")
print(f"  Base model: {{BASE_MODEL}}")
print(f"  Adapter:    {{ADAPTER_PATH}}")
print(f"  Model name: {{MODEL_NAME}}")

cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", BASE_MODEL,
    "--served-model-name", MODEL_NAME,
    "--enable-auto-tool-choice",
    "--tool-call-parser", "hermes",
    "--trust-remote-code",
    "--enable-lora",
    "--lora-modules", f"lora={{ADAPTER_PATH}}",
    "--host", "0.0.0.0",
    "--port", "8000",
]

print(f"\\nCommand: {{' '.join(cmd)}}\\n")
subprocess.run(cmd)
'''
    with open(output_script, "w", encoding="utf-8") as f:
        f.write(script)

    print(f"Inference script generated: {output_script}")
    print(f"Run with: python {output_script}")
    print(f"Then in notebook set MODEL_NAME = 'qwen_lora'")
    return output_script


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LoRA 微调：收集→训练→推理")
    sub = parser.add_subparsers(dest="command")

    # ── collect ──
    p_collect = sub.add_parser("collect", help="用 Agent 收集成功轨迹")
    p_collect.add_argument("--model", default="qwen_auto")
    p_collect.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p_collect.add_argument("--dataset", default="browsecomp_plus_hard50.jsonl")
    p_collect.add_argument("--output", default="trajectories.jsonl")
    p_collect.add_argument("--max-samples", type=int, default=0)
    p_collect.add_argument("--use-v2", action="store_true", default=True)

    # ── prepare ──
    p_prep = sub.add_parser("prepare", help="轨迹 → SFT 训练数据")
    p_prep.add_argument("--input", default="trajectories.jsonl")
    p_prep.add_argument("--output-dir", default="sft_data")
    p_prep.add_argument("--max-examples", type=int, default=0)
    p_prep.add_argument("--all", action="store_true", help="包含错误轨迹")

    # ── gen-script ──
    p_gen = sub.add_parser("gen-script", help="生成 LoRA 训练脚本")
    p_gen.add_argument("--base-model", default="./Qwen3-8B")
    p_gen.add_argument("--data-dir", default="./sft_data")
    p_gen.add_argument("--output-dir", default="./lora_checkpoints")
    p_gen.add_argument("--epochs", type=int, default=3)
    p_gen.add_argument("--batch-size", type=int, default=4)
    p_gen.add_argument("--lr", type=float, default=2e-4)
    p_gen.add_argument("--lora-r", type=int, default=16)
    p_gen.add_argument("--lora-alpha", type=int, default=32)
    p_gen.add_argument("--qlora", action="store_true", default=True)
    p_gen.add_argument("--max-seq-len", type=int, default=4096)

    # ── gen-inference ──
    p_inf = sub.add_parser("gen-inference", help="生成 LoRA 推理脚本")
    p_inf.add_argument("--base-model", default="./Qwen3-8B")
    p_inf.add_argument("--adapter", default="./lora_checkpoints/final")

    args = parser.parse_args()

    if args.command == "collect":
        collect_trajectories(
            dataset_path=args.dataset,
            output_path=args.output,
            model_name=args.model,
            base_url=args.base_url,
            max_samples=args.max_samples,
            use_v2=args.use_v2,
        )

    elif args.command == "prepare":
        prepare_sft_dataset(
            trajectories_path=args.input,
            output_path=args.output_dir,
            max_examples=args.max_examples,
            only_correct=not args.all,
        )

    elif args.command == "gen-script":
        generate_training_script(
            base_model=args.base_model,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            use_qlora=args.qlora,
            max_seq_len=args.max_seq_len,
        )

    elif args.command == "gen-inference":
        create_lora_inference_script(
            base_model=args.base_model,
            adapter_path=args.adapter,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
