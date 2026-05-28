"""
Deep Research Agent — 多轮检索 ReAct Agent for BrowseComp-Plus.
改进版: 更好的 prompt + 无新信息停止 + 上下文管理
"""

import json
import re
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from .tools import build_searcher, get_agent_tool_specs_and_registry, retrieve_once
from .vllm_client import VLLMClient


SYSTEM_PROMPT = """You are a Deep Research Agent. You search a document corpus to answer questions.

## Available Tools

- **search(query: str)** — Search the corpus. Returns top-10 document snippets.
- **get_document(docid: str)** — Read a full document.

## How to Search

**Round 1** — Search using the most unique names, dates, or terms from the question.
**Round 2+** — Look at what you found. If you are missing a piece, search for it using new keywords from the snippets you already got.

Search Tips:
- Be specific: use quoted names ("Los Angeles"), exact years (1974), unique terms
- Vary keywords: if one query returns nothing useful, try synonyms
- Read full docs when a snippet looks promising

## When to Stop and Answer

Stop when EITHER:
(a) You have found clear evidence that directly answers the question, OR
(b) Your last 2 searches returned documents you already examined (no new info)

If stopping at (b), give your Best Guess with lower confidence.

## Output

When you are ready to answer:

Explanation: <one sentence showing your evidence>
Exact Answer: <concise answer>
Confidence: <high|medium|low>

Otherwise, call a tool to search or read more."""


def extract_answer(text: str) -> str:
    """Extract Exact Answer from model output."""
    match = re.search(r'Exact Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def truncate_content(content: Any, max_chars: int = 2000) -> str:
    """Truncate tool results to keep context lean."""
    text = json.dumps(content, ensure_ascii=False) if not isinstance(content, str) else content
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated {len(text) - max_chars} chars]"


class SimpleTracker:
    """Lightweight tracker — no LLM calls, just dedup and status."""

    def __init__(self):
        self.all_visited_docids: set = set()
        self.round_docids: List[set] = []          # docids per round
        self.seen_queries: set = set()
        self.last_round_no_new_docs: bool = False

    def record_search(self, query: str, results: List[Dict]) -> bool:
        """Returns True if any NEW doc was found."""
        self.seen_queries.add(query.lower().strip())
        new_docids = {d["docid"] for d in results if d["docid"] not in self.all_visited_docids}
        self.all_visited_docids.update(new_docids)
        return len(new_docids) > 0

    def record_round(self, round_docids: set):
        self.round_docids.append(round_docids)
        if len(self.round_docids) >= 2:
            prev = self.round_docids[-2]
            curr = self.round_docids[-1]
            self.last_round_no_new_docs = curr.issubset(prev)

    @property
    def should_stop(self) -> bool:
        """Stop if 2 consecutive rounds found no new documents."""
        if len(self.round_docids) < 3:
            return False
        return self.last_round_no_new_docs

    def is_duplicate_query(self, query: str) -> bool:
        return query.lower().strip() in self.seen_queries


def compress_old_rounds(messages: List[Dict], tracker: SimpleTracker) -> List[Dict]:
    """Keep system + question + last 4 rounds of conversation; drop the middle."""
    # Find the boundaries: system message + user question
    keep = messages[:2]  # system + question

    # Collect key facts from tracker
    if tracker.all_visited_docids:
        fact_line = f"[Session: searched {len(tracker.seen_queries)} queries, examined {len(tracker.all_visited_docids)} documents]"
        keep.append({"role": "user", "content": fact_line + "\nContinue from where you left off. What do you still need to find?"})

    # Keep the last 6 messages (≈ 2–3 rounds of back-and-forth)
    tail = messages[-6:] if len(messages) > 6 else messages[2:]
    keep.extend(tail)
    return keep


class DeepResearchAgent:
    """Multi-turn ReAct agent with smart stop conditions and context management."""

    def __init__(
        self,
        client: VLLMClient,
        searcher: Any,
        model_name: str = "qwen_auto",
        max_rounds: int = 8,
        max_tokens: int = 4096,
        top_k: int = 10,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ):
        self.client = client
        self.searcher = searcher
        self.model_name = model_name
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.temperature = temperature
        self.system_prompt = system_prompt or SYSTEM_PROMPT

        tool_specs, tool_registry = get_agent_tool_specs_and_registry(
            searcher=self.searcher, k=self.top_k, snippet_max_chars=1500,
        )
        self.tool_specs = tool_specs
        self.tool_registry = tool_registry

    def solve(self, question: str, query_id: Optional[str] = None) -> Dict[str, Any]:
        tracker = SimpleTracker()

        messages: List[Dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        num_tool_calls = 0
        compress_done = False

        for round_idx in range(1, self.max_rounds + 1):
            # ── Context compression (once, after round 4) ──
            if round_idx == 5 and not compress_done:
                messages = compress_old_rounds(messages, tracker)
                compress_done = True

            # ── Call vLLM ──
            response = self.client.simple_chat(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=self.tool_specs,
                tool_choice="auto",
            )

            choice = response["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []

            assistant_msg = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            # ── Model decided to answer directly ──
            if not tool_calls:
                return {
                    "query_id": query_id,
                    "question": question,
                    "predicted_answer": extract_answer(content),
                    "status": "completed",
                    "messages": messages,
                    "num_tool_calls": num_tool_calls,
                    "rounds_used": round_idx,
                }

            # ── Execute tool calls ──
            round_docids = set()
            for tc in tool_calls:
                num_tool_calls += 1
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = json.loads(fn.get("arguments", "{}"))
                try:
                    result = self.tool_registry[name](**args)

                    if name == "search":
                        query = args.get("query", "")
                        has_new = tracker.record_search(query, result)
                        for d in result:
                            round_docids.add(d["docid"])

                    elif name == "get_document":
                        docid = args.get("docid", "")
                        round_docids.add(docid)

                    truncated = truncate_content(result)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": truncated,
                    })
                except Exception as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps({"error": str(e)}),
                    })

            # ── Track round results and check stop condition ──
            tracker.record_round(round_docids)
            if tracker.should_stop and round_idx >= 3:
                messages.append({
                    "role": "user",
                    "content": "Your last 2 searches returned no new documents. Please give your Best Guess answer now.\n\nFormat:\nExplanation: <reasoning>\nExact Answer: <answer>\nConfidence: low",
                })
                response = self.client.simple_chat(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                final = response["choices"][0]["message"]["content"]
                messages.append({"role": "assistant", "content": final})
                return {
                    "query_id": query_id,
                    "question": question,
                    "predicted_answer": extract_answer(final),
                    "status": "no_new_info",
                    "messages": messages,
                    "num_tool_calls": num_tool_calls,
                    "rounds_used": round_idx,
                }

        # ── Max rounds — force answer ──
        messages.append({
            "role": "user",
            "content": "Maximum rounds reached. Give your Best Guess answer now.\n\nFormat:\nExplanation: <reasoning>\nExact Answer: <answer>\nConfidence: low",
        })
        response = self.client.simple_chat(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        final = response["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": final})
        return {
            "query_id": query_id,
            "question": question,
            "predicted_answer": extract_answer(final),
            "status": "max_rounds_reached",
            "messages": messages,
            "num_tool_calls": num_tool_calls,
            "rounds_used": self.max_rounds,
        }


def batch_solve(
    agent: DeepResearchAgent,
    questions: List[Dict[str, Any]],
    output_path: str,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run agent on batch and save results."""
    records = []
    for i, row in enumerate(questions):
        query_id = row.get("query_id", f"q{i}")
        question = row.get("query", row.get("question", ""))
        if not question:
            continue
        if verbose:
            print(f"[{i+1}/{len(questions)}] q={query_id}...", end=" ", flush=True)
        t0 = time.time()
        result = agent.solve(question=question, query_id=query_id)
        elapsed = time.time() - t0
        if verbose:
            print(f"r={result['rounds_used']} tc={result['num_tool_calls']} ans={result['predicted_answer'][:60]}... ({elapsed:.1f}s)")
        records.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if verbose:
        print(f"\nSaved {len(records)} results to {output_path}")
    return records
