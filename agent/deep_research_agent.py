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


SYSTEM_PROMPT = """You are a Deep Research Agent. Follow these steps STRICTLY in order:

STEP 1 — SEARCH: Call search() multiple times with different keywords covering different aspects of the question. Search at least 2-3 different queries.

STEP 2 — READ FULL DOCUMENTS: After finding relevant results, ALWAYS call get_document() to read the full text. Snippets are NEVER sufficient — always read the complete document.

STEP 3 — CROSS-REFERENCE: Read multiple documents and compare facts across them. Find direct evidence for each specific claim in the question.

STEP 4 — VERIFY: Call verify_claim() to check your answer against the documents before finalizing.

STEP 5 — ANSWER: Output in this EXACT format. You MUST include a verbatim quote from the document(s):

Evidence: <docid> "<direct quote from the document supporting your answer>"
Exact Answer: <concise, complete answer>
Confidence: <high|medium|low>

## STOP CONDITIONS
- If you searched 3+ different queries and found NO relevant documents → give Best Guess with LOW confidence.
- If you read all available documents and verification still fails → give Best Guess with LOW confidence.
- Do NOT repeat the same answer after verification says it is unsupported — try a different answer or give up.

## Available Tools

- **search(query)** — Search the corpus (BM25 keyword matching; queries auto-expanded to multiple variants for better coverage).
- **get_document(docid)** — Read a full document by its docid.
- **find_in_doc(docid, keyword)** — Search within a document for a keyword.
- **decompose_question(question)** — Break a complex question into sub-queries (auto-called at start).
- **verify_claim(claim, docids)** — Verify a candidate answer against specific documents.

## CRITICAL RULES
1. NEVER answer without reading at least 2 full documents first.
2. After each search, immediately call get_document() on any relevant result.
3. Your answer MUST include Evidence: with a verbatim quote and docid.
4. Call verify_claim() before your final answer to prevent mistakes.
5. Never include <think> tags in your final answer.
6. If evidence is insufficient, give Best Guess with low confidence."""


def extract_answer(text: str) -> str:
    """Extract Exact Answer from model output."""
    match = re.search(r'Exact Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def _extract_key_terms(question: str, client: Any, model_name: str, max_terms: int = 5) -> List[str]:
    """Extract key search terms/phrases from question for targeted within-document search (Plan 3)."""
    prompt = (
        f"Extract {max_terms} key search terms or short phrases from this question. "
        "These will be used to find relevant passages inside documents.\n"
        "Rules:\n"
        "- Focus on: proper names, dates, unique multi-word phrases, technical terms\n"
        "- Each term should be 1-4 words, likely to appear verbatim in a document\n"
        "- Avoid generic words: book, company, person, year, located, called, known\n"
        "- Output one term per line, no numbering, no explanation\n\n"
        "Example:\n"
        "Question: A book about inland discoveries in the 1920s, published by a company from the 1880s. A barrel-shaped floating vessel is described on pages 332-339.\n"
        "->\n"
        "barrel-shaped floating vessel\n"
        "spear attack botanist\n"
        "inland discoveries\n"
        "1920s\n"
        "publishing founded 1880s"
    )
    try:
        resp = client.simple_chat(
            model=model_name,
            messages=[
                {"role": "system", "content": "Extract key search terms from questions."},
                {"role": "user", "content": f"{prompt}\n\nQuestion: {question}"},
            ],
            temperature=0.0,
            max_tokens=128,
        )
        text = resp["choices"][0]["message"]["content"].strip()
        terms = [t.strip() for t in text.split("\n") if t.strip() and len(t.strip()) > 2]
        return terms[:max_terms]
    except Exception:
        return []


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
        self.consecutive_no_new_docs: int = 0

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
            if curr.issubset(prev):
                self.consecutive_no_new_docs += 1
            else:
                self.consecutive_no_new_docs = 0

    @property
    def should_stop(self) -> bool:
        """Stop after 3 consecutive rounds with no new documents."""
        return self.consecutive_no_new_docs >= 3

    def is_duplicate_query(self, query: str) -> bool:
        """Check if query is a duplicate via token overlap (Jaccard > 80%)."""
        query_tokens = set(query.lower().split())
        if not query_tokens:
            return False
        for seen in self.seen_queries:
            seen_tokens = set(seen.split())
            intersection = query_tokens & seen_tokens
            union = query_tokens | seen_tokens
            if intersection and len(intersection) / len(union) > 0.8:
                return True
        return False


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
            searcher=self.searcher, k=self.top_k, snippet_max_chars=800,
            client=self.client, model_name=self.model_name,
        )
        self.tool_specs = tool_specs
        self.tool_registry = tool_registry

    def solve(self, question: str, query_id: Optional[str] = None) -> Dict[str, Any]:
        tracker = SimpleTracker()

        messages: List[Dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        # ── Auto-decompose question at start ──
        if "decompose_question" in self.tool_registry:
            try:
                decomp = self.tool_registry["decompose_question"](question)
                messages.append({
                    "role": "assistant",
                    "content": f"I'll research this by searching for:\n{decomp}"
                })
            except Exception:
                pass

        num_tool_calls = 0
        compress_done = False
        unique_docs_read: set = set()
        max_docs_to_read = 3
        auto_loaded_top1 = False
        verify_forced = False
        search_phase_max_round = 1        # Plan 4: only round 1 forces tools (was 3 — too restrictive)
        auto_found_docids: set = set()    # Plan 3: track docs with auto find_in_doc
        verify_passed = False             # Plan 5: hardened verification status
        key_terms = _extract_key_terms(question, self.client, self.model_name)
        force_final_answer = False       # Dead-end: skip verify, force output
        docs_count_at_last_verify = 0    # Track whether new docs appeared since last verify

        for round_idx in range(1, self.max_rounds + 1):
            # ── Context compression (once, after round 4) ──
            if round_idx == 4 and not compress_done:
                messages = compress_old_rounds(messages, tracker)
                compress_done = True

            # ── Plan 4: Round 1 uses auto + retry (tool_choice="required" not supported by all vLLM) ──
            tool_choice = "auto"

            # ── Call vLLM ──
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
            # Strip <think> blocks to save context (~30-50% reduction)
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            tool_calls = message.get("tool_calls") or []

            assistant_msg = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            # ── Retry round 1: Qwen3 thinking mode often skips tool calls ──
            if not tool_calls and round_idx == 1:
                messages.append({
                    "role": "user",
                    "content": "CRITICAL: Call search() with relevant keywords. Never answer from training data.",
                })
                response = self.client.simple_chat(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=128,
                    tools=self.tool_specs,
                    tool_choice={"type": "function", "function": {"name": "search"}},
                )
                choice = response["choices"][0]
                message = choice["message"]
                content = message.get("content") or ""
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                tool_calls = message.get("tool_calls") or []

                assistant_msg = {"role": "assistant", "content": content}
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                messages.append(assistant_msg)

            # ── Model decided to answer directly ──
            if not tool_calls:
                # ── Plan 4: Search-only phase — block answers before round 4 ──
                if round_idx <= search_phase_max_round:
                    messages.append({
                        "role": "user",
                        "content": "SEARCH PHASE: You must search more before answering. Call search() with different keywords covering different aspects of the question. Do NOT answer yet."
                    })
                    continue

                # ── Plan 5: Hardened auto-verification with dead-end detection ──
                if "verify_claim" in self.tool_registry and not verify_passed and round_idx < self.max_rounds - 1 and unique_docs_read and not force_final_answer:
                    verify_forced = True
                    answer_text = extract_answer(content)
                    if answer_text and len(answer_text) > 3:
                        docids_str = ",".join(sorted(unique_docs_read)[:5])
                        verification = self.tool_registry["verify_claim"](answer_text, docids_str)
                        if "Supported: YES" in verification or "Supported:YES" in verification:
                            verify_passed = True
                        else:
                            # Dead-end: verify failed and no new docs appeared → give up
                            if len(unique_docs_read) <= docs_count_at_last_verify:
                                force_final_answer = True
                                messages.append({
                                    "role": "user",
                                    "content": (
                                        "Verification failed again with no new evidence available. "
                                        "Give your Best Guess now with LOW confidence.\n\n"
                                        "Evidence: <docids examined>\n"
                                        "Exact Answer: <your best guess>\nConfidence: low"
                                    )
                                })
                                continue
                            else:
                                docs_count_at_last_verify = len(unique_docs_read)
                                messages.append({
                                    "role": "user",
                                    "content": (
                                        f"Verification says your answer may NOT be supported by the evidence:\n"
                                        f"{verification[:500]}\n\n"
                                        f"Search for more specific evidence before answering. "
                                        f"Do NOT repeat the same answer without new evidence."
                                    )
                                })
                                continue

                # ── Plan 2: Enforce Evidence: format in final answer ──
                if "Evidence:" not in content:
                    if force_final_answer:
                        # Dead-end: accept answer even without Evidence format
                        return {
                            "query_id": query_id,
                            "question": question,
                            "predicted_answer": extract_answer(content),
                            "status": "best_guess",
                            "messages": messages,
                            "num_tool_calls": num_tool_calls,
                            "rounds_used": round_idx,
                        }
                    messages.append({
                        "role": "user",
                        "content": "Your answer is missing an Evidence section. Include:\n"
                                   "Evidence: <docid> \"<direct quote from the document>\"\n"
                                   "Exact Answer: <answer>\nConfidence: <level>"
                    })
                    verify_passed = False  # force re-verify if answer changes
                    continue

                return {
                    "query_id": query_id,
                    "question": question,
                    "predicted_answer": extract_answer(content),
                    "status": "completed" if verify_passed else "unverified",
                    "messages": messages,
                    "num_tool_calls": num_tool_calls,
                    "rounds_used": round_idx,
                }

            # ── Execute tool calls ──
            round_has_search = False
            round_has_getdoc = False
            round_docids = set()
            for tc in tool_calls:
                num_tool_calls += 1
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = json.loads(fn.get("arguments", "{}"))

                if name == "search":
                    round_has_search = True
                elif name == "get_document":
                    round_has_getdoc = True

                try:
                    result = self.tool_registry[name](**args)

                    if name == "search":
                        query = args.get("query", "")
                        has_new = tracker.record_search(query, result)

                        # Search dedup warning
                        if tracker.is_duplicate_query(query) and round_idx > 1:
                            messages.append({
                                "role": "user",
                                "content": "Note: This query closely overlaps with a previous one. Try a different search angle or read a document you haven't examined yet."
                            })

                        for d in result:
                            round_docids.add(d["docid"])

                        # Auto-load top-1 full text after first search
                        if result and not auto_loaded_top1:
                            auto_loaded_top1 = True
                            top_docid = result[0]["docid"]
                            top_doc = self.searcher.get_document(top_docid)
                            if top_doc:
                                unique_docs_read.add(top_docid)
                                round_docids.add(top_docid)
                                full_text = top_doc.get("text", "")
                                messages.append({
                                    "role": "user",
                                    "content": f"[Auto-loaded full text of top result {top_docid}]:\n{full_text[:3000]}"
                                })

                    elif name == "get_document":
                        docid = args.get("docid", "")
                        round_docids.add(docid)
                        unique_docs_read.add(docid)

                    truncated = truncate_content(result)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": truncated,
                    })

                    # ── Plan 3: Auto find_in_doc for long documents ──
                    if name == "get_document" and isinstance(result, dict) and result.get("text"):
                        docid = args.get("docid", "")
                        if docid not in auto_found_docids:
                            auto_found_docids.add(docid)
                            doc_text = result.get("text", "")
                            if len(doc_text) > 1500 and key_terms and "find_in_doc" in self.tool_registry:
                                for term in key_terms[:3]:
                                    try:
                                        find_result = self.tool_registry["find_in_doc"](docid, term)
                                        if isinstance(find_result, dict) and find_result.get("total_matches", 0) > 0:
                                            messages.append({
                                                "role": "user",
                                                "content": (
                                                    f"[Relevant section in {docid} for '{term}']:\n"
                                                    f"{json.dumps(find_result, ensure_ascii=False)[:1200]}"
                                                )
                                            })
                                    except Exception:
                                        pass
                except Exception as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps({"error": str(e)}),
                    })

            # ── Force get_document if model searches but hasn't read enough unique docs ──
            _just_forced_read = False
            if round_has_search and not round_has_getdoc and round_docids and round_idx >= 2:
                if len(unique_docs_read) < max_docs_to_read:
                    _just_forced_read = True
                    unread = [d for d in round_docids if d not in unique_docs_read]
                    target = unread[0] if unread else list(round_docids)[0]
                    messages.append({
                        "role": "user",
                        "content": f"CRITICAL: You have only read {len(unique_docs_read)}/{max_docs_to_read} documents so far. Call get_document('{target}') to read the full text. Snippets are not sufficient evidence."
                    })

            # ── Track round results and check stop condition ──
            tracker.record_round(round_docids)
            if tracker.should_stop and round_idx >= 3 and not round_has_getdoc and not _just_forced_read:
                messages.append({
                    "role": "user",
                    "content": "Your last 3 rounds found no new documents. Please give your Best Guess answer now.\n\nFormat:\nExplanation: <reasoning>\nExact Answer: <answer>\nConfidence: low",
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
