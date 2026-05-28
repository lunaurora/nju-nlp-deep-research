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
        raw = response["choices"][0]["message"]["content"]
        query = raw.strip()
        # Strip thinking markers (Qwen3 thinking mode may add these)
        if "Final Answer:" in query:
            query = query.split("Final Answer:")[-1].strip()
        # Remove quotes if present
        query = query.strip("\"'")
        # Remove newlines — must be a single-line query
        query = query.split("\n")[0].strip()
        # Fallback if empty after cleaning
        if not query:
            return question
        return query
    except Exception:
        return question


# ═══════════════════════════════════════════════════════════════
#  模块 4: Relevance Filter
# ═══════════════════════════════════════════════════════════════

RELEVANCE_FILTER_PROMPT = """You are a document relevance judge. Given a question and a list of search result snippets,
select which documents (by number) contain information relevant to answering the question.

Question:
__QUESTION__

Documents:
__DOC_LIST__

Reply with the comma-separated list of relevant document numbers only (e.g., "1, 3, 5").
If none are relevant, reply with "NONE"."""


def filter_relevant_docs(
    client: VLLMClient, model_name: str,
    question: str, results: List[Dict[str, Any]],
    max_docs: int = 5,
) -> List[Dict[str, Any]]:
    """Filter search results using a single batched LLM call."""
    if not results:
        return []

    if len(results) <= 3:
        return results[:max_docs]

    doc_lines = []
    for i, doc in enumerate(results, 1):
        snippet = doc.get("snippet", "")
        doc_lines.append(f"[{i}] (score={doc['score']:.2f}) {snippet[:300]}")

    try:
        prompt = RELEVANCE_FILTER_PROMPT.replace("__QUESTION__", question)
        prompt = prompt.replace("__DOC_LIST__", "\n".join(doc_lines))
        messages = [{"role": "system", "content": prompt}]
        response = client.simple_chat(
            model=model_name, messages=messages, temperature=0.0, max_tokens=128
        )
        text = response["choices"][0]["message"]["content"].strip()

        if text.upper() == "NONE":
            return results[:max_docs]

        indices = []
        for part in text.split(","):
            part = part.strip()
            if part.isdigit():
                idx = int(part) - 1
                if 0 <= idx < len(results):
                    indices.append(idx)

        if indices:
            return [results[i] for i in indices[:max_docs]]
    except Exception:
        pass

    return results[:max_docs]


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
#  Utility
# ═══════════════════════════════════════════════════════════════

def extract_answer(text: str) -> str:
    """Extract Exact Answer from model output."""
    match = re.search(r'Exact Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


# ═══════════════════════════════════════════════════════════════
#  V2 System Prompt
# ═══════════════════════════════════════════════════════════════

V2_SYSTEM_PROMPT = """You are a Deep Research Agent. Your task is to search a document corpus to answer complex questions by gathering evidence across multiple rounds.

## CRITICAL RULE
You MUST call search() at least once before giving any answer. Never answer from your training data — the correct answer is in the document corpus.

## Available Tools

- **search(query: str)** — Search the corpus. Returns top document snippets with docid and score.
- **get_document(docid: str)** — Retrieve a full document by ID for detailed reading.

## Research Process: Three Phases

### Phase 1 — Gather (Round 1-2)
1. Extract key entities from the question (names, dates, places, unique terms)
2. Start with specific queries targeting the most distinctive terms
3. If specific queries return little, broaden your terms

### Phase 2 — Analyze & Refine (Round 3-5)
1. After each search, identify what new information each result provides
2. Track what you know and what is still missing
3. When a snippet looks promising, use get_document() to read the full text
4. Cross-reference facts across multiple documents
5. If stuck, try synonyms or related terms — do NOT repeat the same query

### Phase 3 — Answer
Only stop searching when you have sufficient evidence to answer confidently.

## Search Strategies
- Use the most specific unique names, dates, IDs first
- Vary keywords: try different combinations of known entities
- Read full documents when snippets contain relevant information

## When to Stop

(a) **Clear evidence found** — You have direct evidence answering the core question → answer with confidence
(b) **No new information** — Your last 2 searches returned only documents you have already examined → give Best Guess
(c) **Maximum rounds** reached → give Best Guess

## Output Format

When you are ready to answer, output exactly:

Explanation: <one sentence showing what evidence supports your answer>
Exact Answer: <concise, complete answer>
Confidence: <high|medium|low>

Otherwise, call search() or get_document() to continue gathering evidence."""


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
        """Run the agent on a single question using native ReAct loop."""
        tracker = EvidenceTracker()
        search_queries_used = []

        # ── Step 1: Analyze the question ──────────────────────
        if self.use_analyzer:
            analysis = analyze_question(self.client, question, self.model_name)
            tracker.sub_questions = analysis["sub_questions"]
            initial_queries = analysis["search_plan"]
            analysis_context = (
                f"[Question Analysis]\n{analysis['raw_analysis']}\n\n"
                f"Initial search suggestions:\n" +
                "\n".join(f"  - search(\"{q}\")" for q in initial_queries[:3])
            )
        else:
            initial_queries = [question]
            analysis_context = ""

        # ── Step 2: Build messages ────────────────────────────
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": V2_SYSTEM_PROMPT},
            {"role": "user", "content": f"Research Question: {question}"},
        ]

        if analysis_context:
            messages.append({
                "role": "assistant",
                "content": f"I'll analyze the question systematically, then search for evidence.\n\n{analysis_context}",
            })

        num_tool_calls = 0
        compress_triggered = False
        last_round_no_tool = False
        no_new_doc_rounds = 0

        # ── Step 3: Native ReAct Loop ────────────────────────
        for round_idx in range(1, self.max_rounds + 1):

            # --- Context Compression (once, after round 5) ---
            if self.use_compressor and not compress_triggered and round_idx >= self.compress_after_round:
                compressed = compress_context(
                    self.client, self.model_name, question, messages, tracker
                )
                if compressed:
                    keep = messages[:2]  # system + question
                    keep.append({
                        "role": "user",
                        "content": f"[Context Summary — earlier rounds compressed]\n\n{compressed}",
                    })
                    if len(messages) > 4:
                        keep.extend(messages[-4:])
                    messages = keep
                    compress_triggered = True

            # --- Inject Query Rewriter hint if stuck (round 3+) ---
            if (self.use_rewriter and round_idx >= 3
                    and search_queries_used and round_idx % 2 == 1):
                hint = rewrite_query(
                    self.client, self.model_name, question, tracker, search_queries_used
                )
                if hint and hint not in search_queries_used:
                    messages.append({
                        "role": "user",
                        "content": f"Hint: The missing information might be found by searching for: \"{hint}\"",
                    })

            # --- Inject Evidence Tracker status (every 3 rounds) ---
            if round_idx > 1 and round_idx % 3 == 0:
                messages.append({
                    "role": "user",
                    "content": f"[Status Update]\n{tracker.get_status_summary()}\n\nContinue searching if you still need more evidence.",
                })

            # --- Call vLLM with tool choice (force on round 1) ---
            tool_choice = "required" if round_idx == 1 else "auto"
            response = self.client.simple_chat(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=self.tool_specs,
                tool_choice=tool_choice,
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
                final_answer_text = content

                # --- Verification ---
                if self.use_verifier:
                    extracted = extract_answer(final_answer_text)
                    if extracted:
                        verification = verify_answer(
                            self.client, self.model_name, question, extracted, tracker
                        )
                        if verification["verdict"] == "UNSUPPORTED":
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"Verification Warning: The evidence does not fully support your answer. "
                                    f"Reason: {verification['reasoning']}\n\n"
                                    f"Please review the evidence more carefully and provide a corrected answer.\n"
                                    f"Format:\nExplanation: <reasoning>\nExact Answer: <answer>\nConfidence: <percentage>"
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
                }

            # ── Execute tool calls ──
            round_docids = set()
            for tc in tool_calls:
                num_tool_calls += 1
                fn = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    continue

                try:
                    result = self.tool_registry[name](**args)

                    if name == "search":
                        query = args.get("query", "")
                        tracker.record_search(query, result)
                        search_queries_used.append(query)
                        for d in result:
                            round_docids.add(d["docid"])

                        # Apply relevance filter (batched LLM call)
                        if self.use_filter and len(result) > 3:
                            filtered = filter_relevant_docs(
                                self.client, self.model_name, question, result
                            )
                        else:
                            filtered = result

                        truncated = json.dumps(
                            [{"docid": d["docid"], "score": d["score"],
                              "snippet": d.get("snippet", "")[:600]}
                             for d in filtered[:5]],
                            ensure_ascii=False,
                        )

                    elif name == "get_document":
                        docid = args.get("docid", "")
                        round_docids.add(docid)
                        if result and isinstance(result, dict):
                            snippet = result.get("text", "")[:500]
                            tracker.seen_snippets[docid] = snippet
                        truncated = json.dumps(result, ensure_ascii=False)
                        if len(truncated) > 3000:
                            truncated = truncated[:3000] + "... [truncated]"

                    else:
                        truncated = json.dumps(result, ensure_ascii=False)

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

            # ── Check no-new-info stop ──
            if len(round_docids) == 0:
                no_new_doc_rounds += 1
            else:
                no_new_doc_rounds = 0

            if no_new_doc_rounds >= 2 and round_idx >= 3:
                messages.append({
                    "role": "user",
                    "content": "Your last searches returned no new documents. Please give your Best Guess answer now.\n\nFormat:\nExplanation: <reasoning>\nExact Answer: <answer>\nConfidence: low",
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
                    "num_searches": len(search_queries_used),
                    "num_docs_examined": len(tracker.visited_docids),
                }

        # ── Max rounds — force answer ──
        messages.append({
            "role": "user",
            "content": (
                "Maximum rounds reached. Provide your best final answer:\n"
                "Explanation: <reasoning>\nExact Answer: <answer>\nConfidence: <percentage>"
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
