from typing import Any, Callable, Dict, List, Optional, Tuple

from .browsecomp_searcher import BrowseCompBM25Searcher, snippetize


def build_searcher(index_path: str) -> BrowseCompBM25Searcher:
    return BrowseCompBM25Searcher(index_path=index_path)


def retrieve_once(
    searcher: BrowseCompBM25Searcher,
    query: str,
    k: int = 5,
    snippet_max_chars: int = 1200,
) -> List[Dict[str, Any]]:
    docs = searcher.search(query, k=k)
    return [
        {
            "docid": doc["docid"],
            "score": doc["score"],
            "snippet": snippetize(doc["text"], snippet_max_chars),
            "url": doc.get("url", ""),
        }
        for doc in docs
    ]


def format_rag_context(results: List[Dict[str, Any]]) -> str:
    blocks = []
    for rank, item in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Document {rank}]",
                    f"docid: {item['docid']}",
                    f"score: {item['score']}",
                    f"url: {item.get('url', '')}",
                    item["snippet"],
                ]
            )
        )
    return "\n\n".join(blocks)


def _rewrite_for_bm25(natural_query: str, client: Optional[Any], model_name: str) -> str:
    """Rewrite natural language query into BM25-friendly keywords via LLM."""
    if client is None:
        return natural_query
    prompt = (
        "Extract 2-4 specific keywords for BM25 keyword search.\n"
        "Rules:\n"
        "- Keep ONLY: named entities, dates, numbers, unique terms\n"
        "- Remove vague words: about, certain, something, related\n"
        "- Output one line of space-separated keywords\n"
        "- No explanations, no numbering\n\n"
        "Examples:\n"
        "  barrel-shaped floating vessel mentioned in a book\n"
        "  -> barrel-shaped floating vessel\n\n"
        "  a company with 35 employees as of Dec 31 2022\n"
        "  -> 35 employees December 31 2022\n\n"
        "  book published in 1920s about inland discoveries\n"
        "  -> 1920s inland discoveries"
    )
    try:
        resp = client.simple_chat(
            model=model_name,
            messages=[
                {"role": "system", "content": "You extract concrete keywords for BM25 search. Output only the keywords, no explanation."},
                {"role": "user", "content": natural_query},
            ],
            temperature=0.0,
            max_tokens=128,
        )
        rewritten = resp["choices"][0]["message"]["content"].strip()
        if rewritten and len(rewritten) > 3 and rewritten != natural_query:
            return rewritten
    except Exception:
        pass
    return natural_query


def _expand_queries(natural_query: str, client: Any, model_name: str, num_queries: int = 3) -> List[str]:
    """Generate multiple diverse BM25 keyword queries targeting different aspects of the question."""
    if client is None:
        return [natural_query]
    prompt = (
        f"Generate {num_queries} different keyword searches for BM25 to find documents "
        "answering this question.\n"
        "Each query must focus on DIFFERENT entities, names, dates, numbers, or unique terms.\n"
        "Rules:\n"
        "- Keep ONLY concrete keywords (proper names, dates, numbers, unique phrases)\n"
        "- Remove vague words: about, certain, something, related, particular\n"
        "- Output one query per line — NO numbering, NO explanation\n\n"
        "Example:\n"
        "Question: A book first published in the 1920s that deals with certain inland discoveries, was published by a company founded in the 1880s. There's a description about a barrel-shaped floating vessel on pages 332-339.\n"
        "->\n"
        "1920s inland discoveries book\n"
        "barrel-shaped floating vessel\n"
        "publishing company founded 1880s\n"
        "spear attack botanist party 463 464\n\n"
        "Question:"
    )
    try:
        resp = client.simple_chat(
            model=model_name,
            messages=[
                {"role": "system", "content": "You extract diverse keyword queries for BM25 search. Output only the queries, no explanation."},
                {"role": "user", "content": f"{prompt}\n{natural_query}"},
            ],
            temperature=0.3,
            max_tokens=256,
        )
        text = resp["choices"][0]["message"]["content"].strip()
        queries = [q.strip() for q in text.split("\n") if q.strip() and len(q.strip()) > 3]
        return queries[:num_queries]
    except Exception:
        return [natural_query]


def get_search_tool_specs_and_registry(
    searcher: BrowseCompBM25Searcher,
    k: int = 5,
    snippet_max_chars: int = 1200,
) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., Any]]]:
    def search(query: str) -> List[Dict[str, Any]]:
        return retrieve_once(searcher=searcher, query=query, k=k, snippet_max_chars=snippet_max_chars)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    f"Search the BrowseComp-Plus BM25 index and return top-{k} results "
                    "with docid, score, and snippet."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        }
    ]
    return tools, {"search": search}


def get_agent_tool_specs_and_registry(
    searcher: BrowseCompBM25Searcher,
    k: int = 5,
    snippet_max_chars: int = 1200,
    client: Optional[Any] = None,
    model_name: str = "qwen_auto",
) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., Any]]]:
    def search(query: str) -> List[Dict[str, Any]]:
        # 1) Get BM25-optimized keywords from the natural language query
        optimized = _rewrite_for_bm25(query, client, model_name)
        # 2) Generate diverse query variants targeting different question aspects
        expanded = _expand_queries(query, client, model_name, num_queries=3)
        # 3) Deduplicate: optimized first, then diverse queries
        all_queries = [optimized] + [q for q in expanded if q.lower().strip() != optimized.lower().strip()]
        all_queries = list(dict.fromkeys(all_queries))[:4]  # max 4 unique queries
        # 4) Search with each query, merge unique results
        seen_docids = set()
        merged = []
        for q in all_queries:
            batch = retrieve_once(searcher=searcher, query=q, k=max(1, k // 2), snippet_max_chars=snippet_max_chars)
            for doc in batch:
                if doc["docid"] not in seen_docids:
                    seen_docids.add(doc["docid"])
                    merged.append(doc)
            if len(merged) >= k:
                break
        return merged[:k]

    def get_document(docid: str) -> Dict[str, Any]:
        doc = searcher.get_document(docid)
        if doc is None:
            return {"docid": docid, "error": "document not found"}
        return doc

    def find_in_doc(docid: str, keyword: str) -> Dict[str, Any]:
        """Search within a specific document for a keyword, return matching lines."""
        doc = searcher.get_document(docid)
        if doc is None:
            return {"docid": docid, "error": "document not found"}
        text = doc.get("text", "")
        lines = text.split("\n")
        matches = []
        for i, line in enumerate(lines):
            if keyword.lower() in line.lower():
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                context = "\n".join(lines[start:end])
                matches.append({"line": i, "context": context.strip()[:500]})
        return {"docid": docid, "keyword": keyword, "matches": matches, "total_matches": len(matches)}

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    f"Search the corpus and return top-{k} snippets with docid, score. "
                    "Note: queries are automatically optimized for BM25 keyword matching. "
                    "You can use natural language descriptions — the system extracts concrete keywords."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query — use natural language, the system auto-extracts BM25-friendly keywords"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_document",
                "description": "Retrieve a full document by its docid. Always call this when a search snippet looks relevant.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "docid": {"type": "string", "description": "Document id"},
                    },
                    "required": ["docid"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_in_doc",
                "description": "Search within a specific document for a keyword, returning matching lines with surrounding context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "docid": {"type": "string", "description": "Document id to search within"},
                        "keyword": {"type": "string", "description": "Keyword or phrase to find in the document"},
                    },
                    "required": ["docid", "keyword"],
                },
            },
        },
    ]
    registry = {"search": search, "get_document": get_document, "find_in_doc": find_in_doc}

    # ── Advanced LLM-powered tools ──
    if client is not None:
        def decompose_question(question: str) -> str:
            """Break a complex question into specific, searchable sub-queries."""
            prompt = (
                "You are a query decomposition assistant. Given a complex question, "
                "extract 2-4 specific, searchable queries that focus on concrete "
                "named entities (people, places, dates, titles, numbers).\n"
                "Return each query on a separate line, prefixed with '- '."
            )
            try:
                resp = client.simple_chat(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": question},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                )
                return resp["choices"][0]["message"]["content"]
            except Exception as e:
                return f"Decomposition failed: {e}"

        def verify_claim(claim: str, docids: str) -> str:
            """Verify a candidate answer against specific documents (searches full doc for claim terms, not just first 3000 chars)."""
            # Extract meaningful keywords from claim for targeted document search
            claim_words = claim.strip().split()
            stopwords = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by", "and", "or", "is", "was", "are", "were", "be", "been", "this", "that", "these", "those", "not", "no", "but", "from", "as", "it", "its", "they", "them", "their", "we", "you", "he", "she", "his", "her", "has", "had", "have", "do", "does", "did", "will", "would", "could", "should", "may", "might", "about", "between", "through", "during", "before", "after", "above", "below", "can", "what", "which", "who", "whom"}
            key_terms = [w.strip(".,;:!?\"'()[]{}") for w in claim_words if w.lower().strip(".,;:!?\"'()[]{}") not in stopwords and len(w.strip(".,;:!?\"'()[]{}")) > 2]
            key_terms = list(dict.fromkeys(key_terms))[:8]  # dedup, max 8

            doc_texts = []
            for did in docids.split(","):
                did = did.strip()
                doc = searcher.get_document(did)
                if doc:
                    text = doc.get("text", "")
                    if key_terms:
                        # Search document for claim-related passages
                        lines = text.split("\n")
                        relevant = []
                        found_lines = set()
                        for term in key_terms:
                            for i, line in enumerate(lines):
                                if term.lower() in line.lower() and i not in found_lines:
                                    found_lines.add(i)
                                    start = max(0, i - 2)
                                    end = min(len(lines), i + 3)
                                    relevant.append(f"...[line {i}]...\n" + "\n".join(lines[start:end]))
                        if relevant:
                            doc_texts.append(f"[Document {did}]:\n" + "\n\n".join(relevant[:5]))
                            continue
                    # Fallback: first 3000 chars if no key terms matched
                    doc_texts.append(f"[Document {did}]:\n{text[:3000]}")
            if not doc_texts:
                return "No valid documents found for the given docids."
            context = "\n\n---\n\n".join(doc_texts)
            prompt = (
                "You are a fact verification assistant. Given documents and a claim, "
                "determine whether the claim is supported.\n"
                "Format:\nSupported: YES/NO\nEvidence: <direct quote from document>\nCorrect Answer: <the correct answer if the claim was wrong>"
            )
            try:
                resp = client.simple_chat(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"Documents:\n{context}\n\nClaim: {claim}"},
                    ],
                    temperature=0.0,
                    max_tokens=1024,
                )
                return resp["choices"][0]["message"]["content"]
            except Exception as e:
                return f"Verification failed: {e}"

        tools.extend([
            {
                "type": "function",
                "function": {
                    "name": "decompose_question",
                    "description": "Break a complex question into 2-4 specific, searchable sub-queries focusing on named entities. Use this at the START of research to plan your search strategy.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "The complex question to decompose"},
                        },
                        "required": ["question"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "verify_claim",
                    "description": "Verify a candidate answer against specific documents. Provide your proposed answer and comma-separated docids. Use this BEFORE outputting your final answer.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "claim": {"type": "string", "description": "The candidate answer to verify"},
                            "docids": {"type": "string", "description": "Comma-separated document IDs containing evidence"},
                        },
                        "required": ["claim", "docids"],
                    },
                },
            },
        ])
        registry["decompose_question"] = decompose_question
        registry["verify_claim"] = verify_claim

    return tools, registry
