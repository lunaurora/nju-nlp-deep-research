"""
Text-based Deep Research Agent — 不使用 vLLM auto tool calling，
模型在文本中输出工具调用指令，由客户端解析执行。

适用于 tool parser 不稳定的场景。
"""

import json
import re
import time
from typing import Any, Dict, List, Optional

from .tools import build_searcher, retrieve_once
from .vllm_client import VLLMClient


TEXT_REACT_SYSTEM_PROMPT = """You are a Deep Research Agent. Your task is to answer complex questions by searching a document corpus.

## Available Tools

You have access to the following tools. To use a tool, output exactly one tool call per line in this format:

SEARCH: <your search query>
READ: <docid>

## How to Work

1. Analyze the question and break it down.
2. Search for information using SEARCH.
3. When a snippet looks promising, READ the full document.
4. Track what you know and what's still missing.
5. When you have enough evidence, output your final answer.

## Format Rules

- Each line can contain at most ONE tool call
- You can output multiple tool call lines (they will be executed in parallel when possible)
- After all tool calls for this round, output THINK: <your reasoning>
- When ready to answer, output FINAL_ANSWER on its own line, followed by:
  Explanation: <brief explanation>
  Exact Answer: <final concise answer>
  Confidence: <percentage>%

## Example

SEARCH: 1920s book inland discoveries Australia
SEARCH: publisher founded 1880s
THINK: I need to find the author's marriage in 1890s and their other book published 1900-1910.

## Rules

- Always search at least once before answering.
- Base your answer ONLY on retrieved documents.
- If you cannot find the answer after thorough searching, say so.
- Search queries should be specific, using key terms from the question."""


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse tool calls from model text output."""
    tool_calls = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if line.upper().startswith('SEARCH:'):
            query = line[len('SEARCH:'):].strip()
            if query:
                tool_calls.append({'name': 'search', 'args': {'query': query}})
        elif line.upper().startswith('READ:'):
            docid = line[len('READ:'):].strip()
            if docid:
                tool_calls.append({'name': 'get_document', 'args': {'docid': docid}})
    return tool_calls


def has_final_answer(text: str) -> bool:
    """Check if the model has output a final answer."""
    return bool(re.search(r'FINAL_ANSWER', text, re.IGNORECASE))


def extract_final_answer(text: str) -> str:
    """Extract final answer from model output."""
    # Find everything after FINAL_ANSWER
    match = re.search(r'FINAL_ANSWER\s*\n(.*)', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try to find Exact Answer directly
    match = re.search(r'Exact Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def get_thinking(text: str) -> str:
    """Extract the THINK part from model output."""
    matches = re.findall(r'THINK:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    return ' | '.join(m.strip() for m in matches) if matches else text[:200]


class TextDeepResearchAgent:
    """Text-based ReAct agent that parses tool calls from model text output."""

    def __init__(
        self,
        client: VLLMClient,
        searcher: Any,
        model_name: str = "qwen_auto",
        max_rounds: int = 8,
        max_tokens: int = 2048,
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
        self.system_prompt = system_prompt or TEXT_REACT_SYSTEM_PROMPT

        self._search_cache = {}  # cache for search results by query

    def _search(self, query: str) -> List[Dict[str, Any]]:
        """Search with caching to avoid repeated searches."""
        if query in self._search_cache:
            return self._search_cache[query]
        results = retrieve_once(searcher=self.searcher, query=query, k=self.top_k)
        self._search_cache[query] = results
        return results

    def _get_document(self, docid: str) -> str:
        """Get full document text."""
        doc = self.searcher.get_document(docid)
        if doc is None:
            return f"Document {docid} not found."
        text = doc.get("text", "")
        # Truncate very long documents
        if len(text) > 5000:
            text = text[:5000] + "\n... [truncated]"
        return text

    def solve(self, question: str, query_id: Optional[str] = None) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Research Question: {question}"},
        ]

        num_tool_calls = 0
        final_status = "completed"

        for round_idx in range(1, self.max_rounds + 1):
            response = self.client.simple_chat(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response["choices"][0]["message"]["content"]

            # Check if the model gave a final answer
            if has_final_answer(content):
                messages.append({"role": "assistant", "content": content})
                final_answer = extract_final_answer(content)
                return {
                    "query_id": query_id,
                    "question": question,
                    "predicted_answer": final_answer,
                    "status": "completed",
                    "messages": messages,
                    "num_tool_calls": num_tool_calls,
                    "rounds_used": round_idx,
                }

            # Parse tool calls from text
            tool_calls = parse_tool_calls(content)
            thinking = get_thinking(content)

            if not tool_calls:
                # Model didn't call tools or give final answer — force it
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": (
                        "Please either:\n"
                        "1. Use SEARCH/READ tools to find information, or\n"
                        "2. Provide FINAL_ANSWER if you have enough evidence."
                    ),
                })
                continue

            # Execute tool calls
            messages.append({"role": "assistant", "content": content})
            tool_results = []

            for tc in tool_calls:
                num_tool_calls += 1
                name = tc["name"]
                args = tc["args"]

                if name == "search":
                    results = self._search(args["query"])
                    # Format results concisely
                    summary_lines = [f"--- Search results for '{args['query']}' ---"]
                    for i, r in enumerate(results[:5], 1):
                        snippet = r["snippet"][:300].replace('\n', ' ')
                        summary_lines.append(f"[{i}] docid={r['docid']} score={r['score']:.1f} | {snippet}")
                    tool_results.append("\n".join(summary_lines))

                elif name == "get_document":
                    full_text = self._get_document(args["docid"])
                    tool_results.append(f"--- Full document {args['docid']} ---\n{full_text}")

            messages.append({
                "role": "user",
                "content": "\n\n".join(tool_results) + "\n\nWhat would you like to do next? Continue searching or provide FINAL_ANSWER?",
            })

        # Max rounds — force final answer
        messages.append({
            "role": "user",
            "content": (
                "Maximum search rounds reached. "
                "Please provide your best final answer now.\n"
                "FINAL_ANSWER\nExplanation: <brief>\nExact Answer: <answer>\nConfidence: <percentage>%"
            ),
        })
        response = self.client.simple_chat(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": content})
        final_answer = extract_final_answer(content)

        return {
            "query_id": query_id,
            "question": question,
            "predicted_answer": final_answer,
            "status": "max_rounds_reached",
            "messages": messages,
            "num_tool_calls": num_tool_calls,
            "rounds_used": self.max_rounds,
        }


def batch_solve(
    agent: Any,
    questions: List[Dict[str, Any]],
    output_path: str,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run agent on a batch of questions and save results."""
    records = []
    for i, row in enumerate(questions):
        query_id = row.get("query_id", f"q{i}")
        question = row.get("query", row.get("question", ""))
        if not question:
            continue

        if verbose:
            print(f"[{i + 1}/{len(questions)}] query_id={query_id}...", end=" ", flush=True)

        t0 = time.time()
        result = agent.solve(question=question, query_id=query_id)
        elapsed = time.time() - t0

        if verbose:
            ans_preview = result["predicted_answer"][:60]
            print(f"rounds={result['rounds_used']} tc={result['num_tool_calls']} "
                  f"ans={ans_preview}... ({elapsed:.1f}s)")

        records.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if verbose:
        print(f"\nSaved {len(records)} results to {output_path}")

    return records
