"""
自动评估脚本：使用 LLM 判断 agent 预测答案与标准答案是否一致。

用法：
    # 命令行
    python -m agent.eval \
        --submission runs/submission.jsonl \
        --dataset browsecomp_plus_hard50.jsonl \
        --model Qwen3-8B \
        --base-url http://127.0.0.1:8000/v1 \
        --output runs/eval_results.jsonl

    # notebook 中调用
    from agent.eval import run_evaluation
    summary, details = run_evaluation(
        submission_path="runs/submission.jsonl",
        dataset_path="browsecomp_plus_hard50.jsonl",
        model_name="Qwen3-8B",
        base_url="http://127.0.0.1:8000/v1",
        output_path="runs/eval_results.jsonl",
    )
"""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .dataset_utils import load_jsonl
from .vllm_client import VLLMClient


EVAL_SYSTEM_PROMPT = """You are an expert evaluator for question-answering systems.
Your task is to judge whether a predicted answer is semantically equivalent to the gold (reference) answer.

Rules:
- Mark CORRECT only when the predicted answer matches the gold answer exactly after normalization, or is an unambiguous equivalent that refers to the same entity, date, title, quantity, or fact.
- Ignore only harmless differences in case, punctuation, articles, spacing, and obvious formatting variants.
- Treat standard abbreviations and full forms as equivalent only when they clearly refer to the same answer.
- If the gold answer is specific but the predicted answer says the answer cannot be determined, is unknown, lacks evidence, or is insufficient, mark INCORRECT.
- If the predicted answer is broader, narrower, related, partially correct, speculative, or only thematically similar, mark INCORRECT.
- Different people, organizations, places, titles, dates, years, percentages, and numeric values are INCORRECT, even if they are plausible.
- For book/article/chapter/song/report titles, require the same title or a trivially formatted variant; thematic paraphrases are INCORRECT.
- If the predicted answer contains extra content, mark CORRECT only if the final answer is still explicit, unambiguous, and fully consistent with the gold answer.

Reply in exactly this format:
Judgment: CORRECT
Reasoning: <one sentence explaining your decision>"""


def _build_eval_user_message(gold_answer: str, predicted_answer: str, question: str = "") -> str:
    parts = []
    if question:
        parts.append(f"Question: {question}")
    parts.append(f"Gold answer: {gold_answer}")
    parts.append(f"Predicted answer: {predicted_answer}")
    return "\n".join(parts)


def _parse_eval_response(response_text: str) -> Tuple[str, str]:
    judgment = "INCORRECT"
    reasoning = ""

    jud_match = re.search(r"Judgment:\s*(CORRECT|INCORRECT)", response_text, re.IGNORECASE)
    if jud_match:
        judgment = jud_match.group(1).upper()

    reason_match = re.search(r"Reasoning:\s*(.+?)$", response_text, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reasoning = reason_match.group(1).strip()

    return judgment, reasoning


def _extract_submission_answer(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return str(msg["content"]).strip()
    return ""


def _strip_thinking(text: str) -> str:
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    stripped = re.sub(r"</?think>", "", stripped, flags=re.IGNORECASE)
    return stripped.strip()


def _extract_final_answer(text: str) -> str:
    if not text:
        return ""

    cleaned = str(text).strip()
    for label in ("Exact Answer", "Final Answer", "Answer"):
        match = re.search(
            rf"{label}\s*:\s*(.+?)(?:\n(?:Confidence|Explanation|Reasoning|Notes?)\s*:|$)",
            cleaned,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()

    cleaned = _strip_thinking(cleaned)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""
    if lines[-1].lower().startswith("confidence:") and len(lines) >= 2:
        return lines[-2]
    return lines[-1]


def _build_eval_extra_payload(model_name: str, disable_thinking: bool) -> Optional[Dict[str, Any]]:
    if disable_thinking and "qwen" in model_name.lower():
        return {
            "chat_template_kwargs": {
                "enable_thinking": False,
            }
        }
    return None


def _count_tool_calls(messages: List[Dict[str, Any]]) -> int:
    count = 0
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            count += len(msg["tool_calls"])
    return count


def _extract_retrieved_docids(messages: List[Dict[str, Any]]) -> List[str]:
    docids: List[str] = []
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "docid" in item:
                    docids.append(str(item["docid"]))
        elif isinstance(parsed, dict) and "docid" in parsed:
            docids.append(str(parsed["docid"]))
    return docids


def _compute_trajectory_stats(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    retrieved_docids = _extract_retrieved_docids(messages)
    unique_retrieved_docids = list(dict.fromkeys(retrieved_docids))

    return {
        "num_tool_calls": _count_tool_calls(messages),
        "num_assistant_messages": sum(1 for m in messages if m.get("role") == "assistant"),
        "num_tool_messages": sum(1 for m in messages if m.get("role") == "tool"),
        "num_retrieved_docs": len(retrieved_docids),
        "unique_retrieved_docids": len(unique_retrieved_docids),
        "retrieved_docids": unique_retrieved_docids,
    }


def _evaluate_one_submission(
    submission_index: int,
    submission: Dict[str, Any],
    gold_map: Dict[str, str],
    gold_question_map: Dict[str, str],
    model_name: str,
    base_url: str,
    api_key: str,
    eval_system_prompt: str,
    temperature: float,
    max_tokens: int,
    disable_thinking: bool,
    extract_final_answer_only: bool,
) -> Tuple[int, Optional[Dict[str, Any]]]:
    query_id = str(submission.get("query_id", ""))
    gold_answer = gold_map.get(query_id, "")
    question = gold_question_map.get(query_id, "")

    if not gold_answer:
        return submission_index, None

    messages = submission.get("messages", [])
    raw_predicted_answer = str(submission.get("predicted_answer", "")).strip() or _extract_submission_answer(messages)
    predicted_answer = _extract_final_answer(raw_predicted_answer) if extract_final_answer_only else raw_predicted_answer

    eval_text = ""
    if not predicted_answer:
        judgment = "INCORRECT"
        if extract_final_answer_only:
            reasoning = "No final answer found in submission."
        else:
            reasoning = "No predicted answer found in submission."
    else:
        client = VLLMClient(base_url=base_url, api_key=api_key)
        eval_messages = [
            {"role": "system", "content": eval_system_prompt},
            {"role": "user", "content": _build_eval_user_message(gold_answer, predicted_answer, question)},
        ]
        try:
            response = client.simple_chat(
                model=model_name,
                messages=eval_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_payload=_build_eval_extra_payload(model_name, disable_thinking),
            )
            eval_text = response["choices"][0]["message"]["content"]
            judgment, reasoning = _parse_eval_response(eval_text)
        except Exception as exc:
            eval_text = f"ERROR: {exc}"
            judgment = "INCORRECT"
            reasoning = str(exc)

    detail = {
        "query_id": query_id,
        "question": question,
        "gold_answer": gold_answer,
        "predicted_answer": predicted_answer,
        "raw_predicted_answer": raw_predicted_answer,
        "eval_judgment": judgment,
        "eval_reasoning": reasoning,
        "eval_model_response": eval_text,
        "trajectory_stats": _compute_trajectory_stats(messages),
        "status": submission.get("status", "unknown"),
    }
    return submission_index, detail


def run_evaluation(
    submission_path: str,
    dataset_path: str,
    model_name: str = "Qwen3-8B",
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "dummy",
    output_path: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    verbose: bool = True,
    max_workers: int = 8,
    disable_thinking: bool = True,
    extract_final_answer_only: bool = True,
    eval_system_prompt: str = EVAL_SYSTEM_PROMPT,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    运行自动评估。

    Parameters
    ----------
    submission_path : str
        学生提交的 trajectory 文件路径 (submission.jsonl)。
    dataset_path : str
        原始数据集路径（包含 gold answer）。
    model_name : str
        用于评估的模型名称。
    base_url : str
        vLLM 服务地址。
    api_key : str
        API key。
    output_path : str, optional
        评估结果输出路径。
    temperature : float
        评估模型 temperature。
    max_tokens : int
        评估模型 max_tokens。
    verbose : bool
        是否打印进度。
    max_workers : int
        并行评估线程数。
    disable_thinking : bool
        是否在支持的评估模型上默认关闭 thinking。
    extract_final_answer_only : bool
        是否只抽取最终答案字段送入评估模型。
    eval_system_prompt : str
        评估系统提示词，默认使用模块内更严格的版本。

    Returns
    -------
    summary : dict
        包含 accuracy、总体统计等。
    details : list[dict]
        每个 query 的详细评估结果。
    """
    submissions = load_jsonl(submission_path)
    dataset = load_jsonl(dataset_path)

    gold_map: Dict[str, str] = {}
    gold_question_map: Dict[str, str] = {}
    for row in dataset:
        query_id = str(row["query_id"])
        gold_map[query_id] = row["answer"]
        gold_question_map[query_id] = row.get("query", "")

    details_with_index: List[Tuple[int, Dict[str, Any]]] = []
    correct_count = 0
    total_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _evaluate_one_submission,
                index,
                submission,
                gold_map,
                gold_question_map,
                model_name,
                base_url,
                api_key,
                eval_system_prompt,
                temperature,
                max_tokens,
                disable_thinking,
                extract_final_answer_only,
            ): submission
            for index, submission in enumerate(submissions)
        }

        completed = 0
        for future in as_completed(futures):
            completed += 1
            submission = futures[future]
            query_id = str(submission.get("query_id", ""))
            submission_index, detail = future.result()

            if detail is None:
                if verbose:
                    print(f"[WARN] [{completed}/{len(submissions)}] query_id={query_id}: no gold answer found, skipping")
                continue

            details_with_index.append((submission_index, detail))
            if detail["eval_judgment"] == "CORRECT":
                correct_count += 1
            total_count += 1

            if verbose:
                print(
                    f"[{completed}/{len(submissions)}] {detail['query_id']} "
                    f"{detail['eval_judgment']:>9s} | pred={detail['predicted_answer'][:60]}..."
                )

    details = [detail for _, detail in sorted(details_with_index, key=lambda item: item[0])]
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    all_tool_calls = [d["trajectory_stats"]["num_tool_calls"] for d in details]
    all_retrieved = [d["trajectory_stats"]["num_retrieved_docs"] for d in details]

    summary: Dict[str, Any] = {
        "total_queries": total_count,
        "correct": correct_count,
        "incorrect": total_count - correct_count,
        "accuracy": round(accuracy, 4),
        "avg_tool_calls_per_query": round(sum(all_tool_calls) / total_count, 2) if total_count > 0 else 0,
        "avg_retrieved_docs_per_query": round(sum(all_retrieved) / total_count, 2) if total_count > 0 else 0,
        "total_tool_calls": sum(all_tool_calls),
        "total_retrieved_docs": sum(all_retrieved),
        "eval_model": model_name,
    }

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"type": "summary", **summary}, ensure_ascii=False) + "\n")
            for detail in details:
                f.write(json.dumps(detail, ensure_ascii=False) + "\n")

    if verbose:
        print(f"\n{'='*50}")
        print("Evaluation complete!")
        print(f"Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
        print(f"Avg tool calls/query: {summary['avg_tool_calls_per_query']}")
        print(f"Avg retrieved docs/query: {summary['avg_retrieved_docs_per_query']}")
        if output_path:
            print(f"Results saved to: {output_path}")

    return summary, details


def main() -> None:
    parser = argparse.ArgumentParser(description="自动评估 agent 预测结果")
    parser.add_argument("--submission", required=True, help="学生提交的 trajectory 文件 (submission.jsonl)")
    parser.add_argument("--dataset", required=True, help="原始数据集 (browsecomp_plus_hard50.jsonl)")
    parser.add_argument("--model", default="Qwen3-8B", help="用于评估的模型名称")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="vLLM 服务地址")
    parser.add_argument("--api-key", default="dummy", help="API key")
    parser.add_argument("--output", default=None, help="评估结果输出路径")
    parser.add_argument("--temperature", type=float, default=0.0, help="评估模型 temperature")
    parser.add_argument("--max-tokens", type=int, default=4096, help="评估模型 max_tokens")
    parser.add_argument("--max-workers", type=int, default=8, help="并行评估线程数")
    parser.add_argument(
        "--disable-thinking",
        dest="disable_thinking",
        action="store_true",
        default=True,
        help="在支持的模型上默认关闭 thinking（默认开启此行为）。",
    )
    parser.add_argument(
        "--enable-thinking",
        dest="disable_thinking",
        action="store_false",
        help="允许评估模型保留 thinking 输出。",
    )
    parser.add_argument(
        "--use-raw-predicted-answer",
        dest="extract_final_answer_only",
        action="store_false",
        default=True,
        help="直接使用 submission 中的原始 predicted_answer，而不是先抽取最终答案。",
    )
    args = parser.parse_args()

    if args.output is None:
        submission_stem = Path(args.submission).stem
        args.output = str(Path(args.submission).parent / f"{submission_stem}_eval.jsonl")

    run_evaluation(
        submission_path=args.submission,
        dataset_path=args.dataset,
        model_name=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        output_path=args.output,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers,
        disable_thinking=args.disable_thinking,
        extract_final_answer_only=args.extract_final_answer_only,
    )


if __name__ == "__main__":
    main()
