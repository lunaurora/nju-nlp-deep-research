"""
Deep Research Agent V2 — 模块化多轮检索系统

相较 V1 的核心改进：
  1. Question Analyzer     — 从问题中提取关键实体和子问题
  2. Query Rewriter        — 每轮基于证据状态改写搜索策略
  3. Evidence Tracker      — 结构化追踪已确认事实与待查子问题
  4. Relevance Filter      — 过滤 BM25 结果，只保留相关文档
  5. Context Compressor    — 对旧轮生成压缩摘要，节省 token
  6. Verifier              — 最终答案前验证证据是否充分
  7. Adaptive Search       — 根据结果质量动态调整搜索策略
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .tools import build_searcher, retrieve_once, get_agent_tool_specs_and_registry
from .vllm_client import VLLMClient


# ═══════════════════════════════════════════════════════════════
#  模块 1: Question Analyzer
# ═══════════════════════════════════════════════════════════════

QUESTION_ANALYZER_PROMPT = """You are a question analysis expert. Given a complex multi-hop question, extract:

1. KEY ENTITIES: List every named entity, date range, and unique term mentioned.
2. SUB-QUESTIONS: Break the question into independent facts that need to be verified.
3. SEARCH PLAN: Suggest 2-3 initial search queries (from most specific to most general).

Output format:
Key Entities:
- entity1 (type: person/organization/date/book/etc.)
- entity2 ...

Sub-questions:
1. <sub-question 1>
2. <sub-question 2>

Search Plan:
1. search: "<most specific query>"
2. search: "<broader fallback query>"
3. search: "<broadest query>"
"""


def analyze_question(client: VLLMClient, question: str, model_name: str) -> Dict[str, Any]:
    """Use LLM to analyze the question and extract entities and search plan."""
    messages = [
        {"role": "system", "content": QUESTION_ANALYZER_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
    ]
    try:
        response = client.simple_chat(
            model=model_name, messages=messages, temperature=0.0, max_tokens=1024
        )
        text = response["choices"][0]["message"]["content"]

        # Parse search plan
        search_plan = []
        in_plan = False
        for line in text.split("\n"):
            if "Search Plan" in line:
                in_plan = True
                continue
            if in_plan and line.strip().startswith("search"):
                match = re.search(r'"(.+?)"', line)
                if match:
                    search_plan.append(match.group(1))

        # Parse sub-questions
        sub_questions = []
        in_sub = False
        for line in text.split("\n"):
            if "Sub-questions" in line:
                in_sub = True
                continue
            if in_sub and ("Search Plan" in line or line.strip() == ""):
                if "Search Plan" in line:
                    break
                continue
            if in_sub and re.match(r"\d+\.", line.strip()):
                sq = re.sub(r"^\d+\.\s*", "", line.strip())
                sub_questions.append(sq)

        return {
            "raw_analysis": text,
            "search_plan": search_plan or [question],
            "sub_questions": sub_questions or ["Find the answer in the corpus"],
        }
    except Exception as e:
        return {
            "raw_analysis": f"Analysis failed: {e}",
            "search_plan": [question],
            "sub_questions": ["Find the answer in the corpus"],
        }


# ═══════════════════════════════════════════════════════════════
#  模块 2: Evidence Tracker
# ═══════════════════════════════════════════════════════════════

class EvidenceTracker:
    """Structured tracker for evidence collected during search."""

    def __init__(self):
        self.confirmed_facts: List[Dict[str, Any]] = []
        self.sub_questions: List[str] = []
        self.answered_subqs: set = set()
        self.visited_docids: set = set()
        self.search_history: List[str] = []
        self.seen_snippets: Dict[str, str] = {}  # docid -> first snippet

    def record_search(self, query: str, results: List[Dict[str, Any]]):
        """Record a search and its results."""
        self.search_history.append(query)
        for doc in results:
            docid = doc["docid"]
            self.visited_docids.add(docid)
            if docid not in self.seen_snippets:
                self.seen_snippets[docid] = doc.get("snippet", "")[:500]

    def add_fact(self, fact: str, source_docid: str, round_idx: int):
        """Record a confirmed fact with its source."""
        self.confirmed_facts.append({
            "fact": fact,
            "source": source_docid,
            "round": round_idx,
        })

    def mark_subq_answered(self, sub_q: str):
        self.answered_subqs.add(sub_q)

    def get_status_summary(self) -> str:
        """Generate a concise summary of current evidence state."""
        lines = ["=== Evidence Status ==="]
        if self.confirmed_facts:
            lines.append("Confirmed facts:")
            for f in self.confirmed_facts[-5:]:  # last 5 facts
                lines.append(f"  - {f['fact']} (doc: {f['source']})")
        lines.append(f"Sub-questions answered: {len(self.answered_subqs)}/{len(self.sub_questions) or '?'}")
        lines.append(f"Documents examined: {len(self.visited_docids)}")
        lines.append(f"Searches performed: {len(self.search_history)}")
        return "\n".join(lines)

    def get_unanswered_subqs(self) -> str:
        """Return unanswered sub-questions."""
        unanswered = [sq for sq in self.sub_questions if sq not in self.answered_subqs]
        if not unanswered:
            return "All sub-questions appear to be answered."
        return "Still need to find:\n" + "\n".join(f"  - {sq}" for sq in unanswered)


# ═══════════════════════════════════════════════════════════════
#  模块 3: Query Rewriter
# ═══════════════════════════════════════════════════════════════

QUERY_REWRITER_PROMPT = """Given the original question, current evidence state, and previous search queries,
generate ONE new search query that will help find the missing information.

Rules:
- Use specific terms (names, dates, places) from findings so far
- If previous queries were too broad, make the new query more specific
- If previous queries returned nothing, try synonyms or related terms
- Target the MOST critical missing piece of information

Output only the search query, nothing else."""


def rewrite_query(
    client: VLLMClient, model_name: str,
    question: str, tracker: EvidenceTracker,
    previous_queries: List[str],
) -> str:
    """Use LLM to generate a better search query based on current evidence."""
    prompt_parts = [
        f"Original question: {question}",
        f"Previous queries attempted: {json.dumps(previous_queries)}",
        f"Evidence found so far: {tracker.get_status_summary()}",
        f"Missing: {tracker.get_unanswered_subqs()}",
    ]
    messages = [
        {"role": "system", "content": QUERY_REWRITER_PROMPT},
        {"role": "user", "content": "\n\n".join(prompt_parts)},
    ]
    try:
        response = client.simple_chat(
            model=model_name, messages=messages, temperature=0.3, max_tokens=256
        )
        query = response["choices"][0]["message"]["content"].strip()
        # Clean up: remove quotes if present
        query = query.strip("\"'")
        return query
    except Exception:
        return question


# ═══════════════════════════════════════════════════════════════
#  模块 4: Relevance Filter
# ═══════════════════════════════════════════════════════════════

RELEVANCE_FILTER_PROMPT = """You are a document relevance judge. Given a question and a search result,
determine if the document is relevant.

Reply with exactly one word: RELEVANT or IRRELEVANT"""


def filter_relevant_docs(
    client: VLLMClient, model_name: str,
    question: str, results: List[Dict[str, Any]],
    max_docs: int = 5,
) -> List[Dict[str, Any]]:
    """Filter search results to keep only relevant documents."""
    relevant = []
    for doc in results:
        snippet = doc.get("snippet", "")[:500]
        messages = [
            {"role": "system", "content": RELEVANCE_FILTER_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\n\nDocument snippet:\n{snippet}",
            },
        ]
        try:
            response = client.simple_chat(
                model=model_name, messages=messages, temperature=0.0, max_tokens=16
            )
            judgment = response["choices"][0]["message"]["content"].strip().upper()
            if "RELEVANT" in judgment:
                relevant.append(doc)
                if len(relevant) >= max_docs:
                    break
        except Exception:
            relevant.append(doc)  # keep on error

    return relevant if relevant else results[:max_docs]  # fallback


# ═══════════════════════════════════════════════════════════════
#  模块 5: Context Compressor
# ═══════════════════════════════════════════════════════════════

CONTEXT_COMPRESSOR_PROMPT = """Summarize the following research conversation into a concise "Findings So Far" section.
Focus only on confirmed facts and direct evidence. Omit irrelevant search results.
Keep the summary under 200 words.

Output format:
Findings So Far:
- <fact 1> (source: docid)
- <fact 2> (source: docid)"""


def compress_context(
    client: VLLMClient, model_name: str,
    question: str, conversation: List[Dict[str, Any]],
    tracker: EvidenceTracker,
) -> str:
    """Compress old conversation turns into a concise summary."""
    # Take older turns (before the last 2 rounds)
    compressible = conversation[:-4] if len(conversation) > 6 else []
    if not compressible:
        return ""

    # Format the conversation to compress
    lines = [f"Question: {question}"]
    for msg in compressible:
        role = msg["role"]
        content = str(msg.get("content", ""))[:300]
        if content:
            lines.append(f"[{role}] {content}")
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                lines.append(f"[tool_call] {fn.get('name')}({fn.get('arguments', '')})")

    messages = [
        {"role": "system", "content": CONTEXT_COMPRESSOR_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]
    try:
        response = client.simple_chat(
            model=model_name, messages=messages, temperature=0.0, max_tokens=512
        )
        compressed = response["choices"][0]["message"]["content"]
        # Also add tracker status
        return compressed + "\n" + tracker.get_status_summary()
    except Exception:
        return tracker.get_status_summary()


# ═══════════════════════════════════════════════════════════════
#  模块 6: Verifier
# ═══════════════════════════════════════════════════════════════

VERIFIER_PROMPT = """You are an evidence verification expert. Given a question, a proposed answer, and the evidence found,
determine if the answer is fully supported by the evidence.

Check:
1. Does the evidence directly support the answer?
2. Are there any contradictions in the evidence?
3. Is there any missing piece that could change the answer?

Reply in format:
Verdict: SUPPORTED / PARTIAL / UNSUPPORTED
Reasoning: <one sentence>
Missing Evidence: <if any>"""


def verify_answer(
    client: VLLMClient, model_name: str,
    question: str, proposed_answer: str,
    tracker: EvidenceTracker,
) -> Dict[str, str]:
    """Verify if the proposed answer is supported by evidence."""
    evidence_summary = tracker.get_status_summary()
    messages = [
        {"role": "system", "content": VERIFIER_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Proposed Answer: {proposed_answer}\n\n"
                f"Evidence gathered:\n{evidence_summary}"
            ),
        },
    ]
    try:
        response = client.simple_chat(
            model=model_name, messages=messages, temperature=0.0, max_tokens=256
        )
        text = response["choices"][0]["message"]["content"]
        verdict = "UNSUPPORTED"
        if "SUPPORTED" in text:
            verdict = "SUPPORTED"
        elif "PARTIAL" in text:
            verdict = "PARTIAL"
        reasoning_match = re.search(r"Reasoning:\s*(.+?)$", text, re.IGNORECASE | re.MULTILINE)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        return {"verdict": verdict, "reasoning": reasoning, "raw": text}
    except Exception as e:
        return {"verdict": "UNSUPPORTED", "reasoning": str(e), "raw": ""}


# ═══════════════════════════════════════════════════════════════
#  V2 System Prompt
# ═══════════════════════════════════════════════════════════════

V2_SYSTEM_PROMPT = """You are a Deep Research Agent V2 — an advanced multi-hop retrieval system.

## Available Tools

1. **search(query: str)** — Search the corpus. Returns top documents with snippets.
2. **get_document(docid: str)** — Retrieve a full document by ID for detailed reading.

## How to Work

You operate in rounds. Each round you should:

1. **Review** the evidence status provided to you.
2. **Reason** about what is still missing.
3. **Search** or **Read** to fill the gaps.
4. **Track** what you've confirmed.

## Output Format

When calling tools:
- You may call multiple tools if they are independent.
- Each call helps you get closer to the answer.

When you have sufficient evidence, output:

Explanation: <step-by-step reasoning showing how the evidence supports your answer>
Exact Answer: <concise final answer>
Confidence: <percentage>%

## Rules

- Search at least 2-3 times to gather broad evidence before narrowing down.
- Use get_document to read full text when snippets show promise.
- Always cite specific evidence from the documents.
- If stuck, try different keywords or synonyms.
- Never fabricate evidence or answer from memory."""


# ═══════════════════════════════════════════════════════════════
#  Main Agent V2
# ═══════════════════════════════════════════════════════════════

class DeepResearchAgentV2:
    """Advanced multi-turn research agent with modular components."""

    def __init__(
        self,
        client: VLLMClient,
        searcher: Any,
        model_name: str = "qwen_auto",
        max_rounds: int = 10,
        max_tokens: int = 2048,
        top_k: int = 15,
        temperature: float = 0.0,
        use_analyzer: bool = True,
        use_rewriter: bool = True,
        use_filter: bool = True,
        use_compressor: bool = True,
        use_verifier: bool = True,
        compress_after_round: int = 5,
    ):
        self.client = client
        self.searcher = searcher
        self.model_name = model_name
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.temperature = temperature

        # Feature flags
        self.use_analyzer = use_analyzer
        self.use_rewriter = use_rewriter
        self.use_filter = use_filter
        self.use_compressor = use_compressor
        self.use_verifier = use_verifier
        self.compress_after_round = compress_after_round

        # Tools
        specs, registry = get_agent_tool_specs_and_registry(
            searcher=self.searcher, k=self.top_k, snippet_max_chars=1500,
        )
        self.tool_specs = specs
        self.tool_registry = registry

    def solve(self, question: str, query_id: Optional[str] = None) -> Dict[str, Any]:
        """Run the agent on a single question."""
        tracker = EvidenceTracker()

        # ── Step 1: Analyze the question ──────────────────────
        if self.use_analyzer:
            analysis = analyze_question(self.client, question, self.model_name)
            tracker.sub_questions = analysis["sub_questions"]
            initial_queries = analysis["search_plan"]
            analysis_note = f"[Question Analysis]\n{analysis['raw_analysis']}\n"
        else:
            initial_queries = [question]
            analysis_note = ""

        # ── Step 2: Build messages ────────────────────────────
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": V2_SYSTEM_PROMPT},
            {"role": "user", "content": f"Research Question: {question}"},
        ]

        if analysis_note:
            messages.append({
                "role": "assistant",
                "content": f"I'll analyze this question systematically.\n\n{analysis_note}\nLet me start by searching for key information.",
            })

        num_tool_calls = 0
        compress_triggered = False
        search_queries_used = []

        # ── Step 3: ReAct Loop ────────────────────────────────
        for round_idx in range(1, self.max_rounds + 1):

            # --- Context Compression ---
            if self.use_compressor and not compress_triggered and round_idx >= self.compress_after_round:
                compressed = compress_context(
                    self.client, self.model_name, question, messages, tracker
                )
                if compressed:
                    # Keep system + question + evidence summary + last 2 rounds
                    keep = messages[:2]  # system + question/analysis
                    keep.append({
                        "role": "user",
                        "content": f"[Context Summary — earlier rounds compressed]\n\n{compressed}",
                    })
                    # Add last 3 messages (the most recent round)
                    if len(messages) > 4:
                        keep.extend(messages[-4:])
                    messages = keep
                    compress_triggered = True

            # --- Decide query strategy ---
            if round_idx == 1:
                # Use initial queries
                current_queries = initial_queries[:2]
            elif self.use_rewriter and search_queries_used:
                # Generate a refined query
                new_query = rewrite_query(
                    self.client, self.model_name, question, tracker, search_queries_used
                )
                current_queries = [new_query]
            else:
                current_queries = [question]

            # --- Execute searches ---
            tool_calls_to_execute = []
            for q in current_queries:
                if q not in search_queries_used:
                    tool_calls_to_execute.append(q)

            if not tool_calls_to_execute:
                # Already searched everything — force a broader query
                tool_calls_to_execute = [question]

            # Build assistant message requesting tools
            assistant_content = (
                f"[Round {round_idx}] Searching for missing information...\n"
                f"Queries: {tool_calls_to_execute}"
            )
            assistant_msg = {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": [
                    {
                        "id": f"call_{round_idx}_{ti}",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": json.dumps({"query": q}),
                        },
                    }
                    for ti, q in enumerate(tool_calls_to_execute)
                ],
            }
            messages.append(assistant_msg)

            # Execute tools and collect results
            for ti, q in enumerate(tool_calls_to_execute):
                num_tool_calls += 1
                search_queries_used.append(q)

                results = retrieve_once(
                    searcher=self.searcher, query=q, k=self.top_k
                )
                tracker.record_search(q, results)

                # Relevance filter
                if self.use_filter and results:
                    filtered = filter_relevant_docs(
                        self.client, self.model_name, question, results
                    )
                else:
                    filtered = results

                # Format results
                result_text = json.dumps(
                    [
                        {
                            "docid": d["docid"],
                            "score": d["score"],
                            "snippet": d["snippet"][:600],
                        }
                        for d in filtered[:5]  # keep top 5 after filtering
                    ],
                    ensure_ascii=False,
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": f"call_{round_idx}_{ti}",
                    "content": result_text,
                })

            # --- Let model decide next step ---
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

            assistant_msg2 = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg2["tool_calls"] = tool_calls
            messages.append(assistant_msg2)

            # If model wants to call more tools, execute them
            if tool_calls:
                for tc in tool_calls:
                    num_tool_calls += 1
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args = json.loads(fn.get("arguments", "{}"))
                    try:
                        result = self.tool_registry[name](**args)
                        # Update tracker
                        if name == "search":
                            tracker.record_search(args.get("query", ""), result)
                            search_queries_used.append(args.get("query", ""))
                        elif name == "get_document":
                            docid = args.get("docid", "")
                            if result and isinstance(result, dict):
                                snippet = result.get("text", "")[:500]
                                tracker.seen_snippets[docid] = snippet

                        truncated = json.dumps(result, ensure_ascii=False)
                        if len(truncated) > 3000:
                            truncated = truncated[:3000] + "... [truncated]"
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
                continue  # Continue loop — model will decide again

            # No tool calls → this is the final answer
            final_answer_text = content

            # --- Verification step ---
            if self.use_verifier:
                extracted = extract_answer(final_answer_text)
                verification = verify_answer(
                    self.client, self.model_name, question, extracted, tracker
                )
                if verification["verdict"] == "UNSUPPORTED":
                    # Add a chance to reconsider
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Verification Warning: The evidence does not fully support your answer. "
                            f"Reason: {verification['reasoning']}\n\n"
                            f"Please review the evidence more carefully and provide a corrected answer.\n"
                            f"Format:\nExplanation: <reasoning>\nExact Answer: <answer>\nConfidence: <%>"
                        ),
                    })
                    response2 = self.client.simple_chat(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    final_answer_text = response2["choices"][0]["message"]["content"]
                    messages.append({"role": "assistant", "content": final_answer_text})

            return {
                "query_id": query_id,
                "question": question,
                "predicted_answer": extract_answer(final_answer_text),
                "status": "completed",
                "messages": messages,
                "num_tool_calls": num_tool_calls,
                "rounds_used": round_idx,
                "num_searches": len(search_queries_used),
                "num_docs_examined": len(tracker.visited_docids),
                "verification": verification if self.use_verifier else None,
            }

        # Max rounds reached
        messages.append({
            "role": "user",
            "content": (
                "Maximum rounds reached. Provide your best final answer:\n"
                "Explanation: <reasoning>\nExact Answer: <answer>\nConfidence: <%>"
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

        return {
            "query_id": query_id,
            "question": question,
            "predicted_answer": extract_answer(content),
            "status": "max_rounds_reached",
            "messages": messages,
            "num_tool_calls": num_tool_calls,
            "rounds_used": self.max_rounds,
            "num_searches": len(search_queries_used),
            "num_docs_examined": len(tracker.visited_docids),
        }


def extract_answer(text: str) -> str:
    """Extract Exact Answer from model output."""
    match = re.search(r'Exact Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def batch_solve(
    agent: Any,
    questions: List[Dict[str, Any]],
    output_path: str,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run agent on batch and save results."""
    records = []
    for i, row in enumerate(questions):
        query_id = row.get("query_id", f"q{i}")
        question = row.get("query", "")
        if not question:
            continue

        if verbose:
            print(f"[{i+1}/{len(questions)}] q={query_id}...", end=" ", flush=True)

        t0 = time.time()
        result = agent.solve(question=question, query_id=query_id)
        elapsed = time.time() - t0

        if verbose:
            ans = result["predicted_answer"][:50]
            print(f"r={result['rounds_used']} tc={result['num_tool_calls']} "
                  f"ans={ans}... ({elapsed:.1f}s)")
        records.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if verbose:
        print(f"\nSaved {len(records)} results to {output_path}")
    return records
