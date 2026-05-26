"""
Deep Research Agent — 多轮检索 ReAct Agent for BrowseComp-Plus.

Usage:
    from agent.deep_research_agent import DeepResearchAgent

    agent = DeepResearchAgent(client, searcher, model_name="qwen_auto")
    result = agent.solve(question)
"""

import json
import re
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from .tools import (
    build_searcher,
    get_agent_tool_specs_and_registry,
    retrieve_once,
    format_rag_context,
)
from .vllm_client import VLLMClient


SYSTEM_PROMPT = """You are a Deep Research Agent. Your task is to answer complex questions by searching a document corpus.

## Available Tools

1. **search(query: str)** — Search the corpus and return top documents with snippets. Use this to find relevant documents.
2. **get_document(docid: str)** — Retrieve the full text of a specific document by its ID. Use this when a search snippet looks promising but you need the full context.

## How to Work

1. **Analyze** the question carefully. Break it down into the key facts you need to find.
2. **Search** for each piece of information. Start with broad queries, then refine.
3. **Read** full documents when snippets contain relevant information.
4. **Track** what you have found and what is still missing.
5. **Synthesize** all evidence to form the final answer.

## Search Strategy Tips

- Start with the most unique/ specific terms from the question
- If a search returns irrelevant results, try different keywords
- After finding some information, search for the next piece using newly discovered terms
- Use quotes for multi-word proper names when useful

## Output Format

When you have gathered sufficient evidence, provide your answer in this format:

Explanation: <brief explanation of your reasoning and evidence>
Exact Answer: <final concise answer>
Confidence: <percentage>%

If you need more information, call a tool instead.

## Rules

- Only answer when you have found clear evidence.
- Your answer must be based on the retrieved documents, not prior knowledge.
- If you cannot find the answer after thorough searching, say so.
- Always search at least once before attempting to answer."""


def extract_answer_from_text(text: str) -> str:
    """Extract the Exact Answer from the model's final response."""
    # Try to find "Exact Answer:" pattern
    match = re.search(r'Exact Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: return the whole text
    return text.strip()


def execute_tool_call(
    tool_call: Dict[str, Any],
    registry: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute a single tool call and return the result."""
    function = tool_call.get("function", {})
    name = function.get("name", "")
    arguments = function.get("arguments", "{}")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    result = registry[name](**arguments)
    return result


def truncate_tool_content(content: Any, max_chars: int = 3000) -> str:
    """Truncate tool content to avoid blowing the context window."""
    text = json.dumps(content, ensure_ascii=False) if not isinstance(content, str) else content
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated from {len(text)} total chars]"


class DeepResearchAgent:
    """Multi-turn ReAct agent for deep research on BrowseComp-Plus."""

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
        self.system_prompt = system_prompt or SYSTEM_PROMPT

        tool_specs, tool_registry = get_agent_tool_specs_and_registry(
            searcher=self.searcher,
            k=self.top_k,
            snippet_max_chars=1500,
        )
        self.tool_specs = tool_specs
        self.tool_registry = tool_registry

    def solve(self, question: str, query_id: Optional[str] = None) -> Dict[str, Any]:
        """Run the agent on a single question.

        Returns a dict with keys:
            - query_id
            - question
            - predicted_answer
            - status
            - messages (full conversation trajectory)
            - num_tool_calls
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        num_tool_calls = 0
        final_status = "completed"

        for round_idx in range(1, self.max_rounds + 1):
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

            # Build assistant message
            assistant_msg = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            # No tool calls -> final answer
            if not tool_calls:
                final_answer = content
                return {
                    "query_id": query_id,
                    "question": question,
                    "predicted_answer": extract_answer_from_text(final_answer),
                    "status": "completed",
                    "messages": messages,
                    "num_tool_calls": num_tool_calls,
                    "rounds_used": round_idx,
                }

            # Execute tool calls
            for tc in tool_calls:
                num_tool_calls += 1
                try:
                    result = execute_tool_call(tc, self.tool_registry)
                    truncated = truncate_tool_content(result, max_chars=3000)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": truncated if isinstance(truncated, str) else json.dumps(truncated, ensure_ascii=False),
                    })
                except Exception as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps({"error": str(e)}),
                    })

        # Max rounds reached without final answer
        # Force the model to answer
        messages.append({
            "role": "user",
            "content": (
                "You have reached the maximum number of search rounds. "
                "Please provide your best answer based on the evidence gathered so far. "
                "Format:\nExplanation: <brief>\nExact Answer: <answer>\nConfidence: <percentage>%"
            ),
        })
        response = self.client.simple_chat(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        final_answer = response["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": final_answer})

        return {
            "query_id": query_id,
            "question": question,
            "predicted_answer": extract_answer_from_text(final_answer),
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

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if verbose:
        print(f"\nSaved {len(records)} results to {output_path}")

    return records
