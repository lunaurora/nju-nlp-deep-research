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
        return retrieve_once(searcher=searcher, query=query, k=k, snippet_max_chars=snippet_max_chars)

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
                    "After finding a relevant snippet, call get_document() to read the full text."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query with specific keywords (names, dates, entities)"},
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
            """Verify a candidate answer against specific documents."""
            doc_texts = []
            for did in docids.split(","):
                did = did.strip()
                doc = searcher.get_document(did)
                if doc:
                    text = doc.get("text", "")
                    doc_texts.append(f"[Document {did}]:\n{text[:3000]}")
            if not doc_texts:
                return "No valid documents found for the given docids."
            context = "\n\n---\n\n".join(doc_texts)
            prompt = (
                "You are a fact verification assistant. Given documents and a claim, "
                "determine whether the claim is supported.\n"
                "Format:\nSupported: YES/NO\nEvidence: <quote>\nCorrect Answer: <the correct answer if the claim was wrong>"
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
