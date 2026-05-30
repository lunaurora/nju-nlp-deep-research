"""
单题快速测试：跑 1 道题 → 保存轨迹到 trajectories/

用法：
    python -m agent.run_one --query-id 26
    python -m agent.run_one --query-id 26 --temperature 0.3
    python -m agent.run_one --query-id 26 --dry-run  # 只打印问题信息
"""
import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

from .dataset_utils import load_jsonl
from .deep_research_agent import DeepResearchAgent
from .tools import build_searcher, get_agent_tool_specs_and_registry
from .vllm_client import VLLMClient


def find_question(dataset_path: str, query_id: str) -> dict:
    dataset = load_jsonl(dataset_path)
    for row in dataset:
        if str(row["query_id"]) == str(query_id):
            return row
    raise ValueError(f"query_id={query_id} not found in {dataset_path}")


def main():
    parser = argparse.ArgumentParser(description="单题快速测试")
    parser.add_argument("--query-id", required=True, help="要跑的题目 ID")
    parser.add_argument("--dataset", default="browsecomp_plus_hard50.jsonl")
    parser.add_argument("--corpus", default="browsecomp-plus-corpus")
    parser.add_argument("--index-dir", default="indexes")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="只打印问题信息，不跑 agent")
    parser.add_argument("--project-dir", default=".", help="项目根目录（包含 dataset/corpus）")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    dataset_path = project_dir / args.dataset
    corpus_dir = project_dir / args.corpus
    index_dir = project_dir / args.index_dir

    # 加载问题
    question_data = find_question(str(dataset_path), args.query_id)
    qid = str(question_data["query_id"])
    query = question_data["query"]
    answer = question_data.get("answer", "N/A")

    print(f"{'='*60}")
    print(f"query_id: {qid}")
    print(f"question: {query[:120]}...")
    print(f"gold answer: {answer}")
    print(f"difficulty: {question_data.get('difficulty_score', 'N/A')}")
    print(f"{'='*60}")

    if args.dry_run:
        return

    # 构建检索器和工具
    print("\n[init] Building searcher...")
    searcher = build_searcher("bm25", str(corpus_dir), str(index_dir))
    registry = get_agent_tool_specs_and_registry(
        searcher,
        str(corpus_dir),
        base_url=args.base_url,
        model_name=args.model,
    )
    tool_specs = registry["tool_specs"]

    # 初始化 client 和 agent
    client = VLLMClient(base_url=args.base_url, api_key="dummy")
    agent = DeepResearchAgent(
        client=client,
        model_name=args.model,
        searcher=searcher,
        corpus_dir=str(corpus_dir),
        tool_specs=tool_specs,
        registry=registry,
        max_rounds=args.max_rounds,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    # 跑 agent
    print(f"\n[run] Starting agent (max_rounds={args.max_rounds}, temperature={args.temperature})...")
    t0 = time.time()
    result = agent.solve(qid, query)
    elapsed = time.time() - t0
    print(f"[done] {elapsed:.1f}s")

    # 打印结果
    status = result.get("status", "unknown")
    pred = result.get("predicted_answer", "")
    tc = result.get("num_tool_calls", 0)
    print(f"\nstatus: {status}")
    print(f"tool_calls: {tc}")
    print(f"predicted_answer: {pred}")
    print(f"gold_answer: {answer}")
    match = pred.strip().lower() == answer.strip().lower()
    print(f"exact_match: {'YES' if match else 'NO'}")

    # 保存轨迹
    timestamp = datetime.now().strftime("%m%d_%H%M")
    traj_dir = project_dir / "trajectories" / f"q{qid}_{timestamp}"
    traj_dir.mkdir(parents=True, exist_ok=True)

    trajectory = {
        "query_id": qid,
        "question": query,
        "gold_answer": answer,
        "predicted_answer": pred,
        "status": status,
        "num_tool_calls": tc,
        "elapsed_seconds": round(elapsed, 1),
        "temperature": args.temperature,
        "max_rounds": args.max_rounds,
        "messages": result.get("messages", []),
    }
    traj_path = traj_dir / "trajectory.json"
    with open(traj_path, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, ensure_ascii=False, indent=2)
    print(f"\n[trajectory saved] {traj_path}")


if __name__ == "__main__":
    main()
