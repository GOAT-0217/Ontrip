import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from vectorizer.app.vectordb.vectordb import VectorDB
from customer_support_chat.app.core.settings import get_settings
from langchain_core.tools import tool
from customer_support_chat.app.services.retrieval import (
    RetrievalOrchestrator,
    OrchestratorConfig,
    RetrievalResult,
)
import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)

settings = get_settings()

_faq_vectordb = None
_faq_orchestrator = None

def _get_faq_vectordb():
    global _faq_vectordb
    if _faq_vectordb is None:
        try:
            _faq_vectordb = VectorDB(table_name="faq", collection_name="faq_collection")
            logger.info("FAQ VectorDB initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize FAQ VectorDB: {e}")
            return None
    return _faq_vectordb

def _get_faq_orchestrator():
    global _faq_orchestrator
    if _faq_orchestrator is None:
        vdb = _get_faq_vectordb()
        if vdb is not None:
            _faq_orchestrator = RetrievalOrchestrator(
                vectordb=vdb,
                table_name="faq",
                db_path="",
                config=OrchestratorConfig(
                    vector_weight=0.8,
                    keyword_weight=0.2,
                    rerank_top_k=3,
                    relevance_threshold=0.3,
                    enable_query_rewrite=True,
                    enable_rerank=True,
                    enable_hybrid=False,
                ),
            )
    return _faq_orchestrator

def _format_faq_results(results: List[RetrievalResult]) -> str:
    faq_entries = []
    for r in results:
        content = r.payload.get("content", "")
        question = "General FAQ Information"
        answer = content

        question_match = re.search(r'^\d+\. (.+?)(?=\n|$)', content, re.MULTILINE)
        if question_match:
            question = question_match.group(1).strip()
            answer_start = content.find(question) + len(question)
            answer = content[answer_start:].strip()
        elif content.startswith('##'):
            lines = content.split('\n', 1)
            question = lines[0].replace('##', '').strip()
            answer = lines[1] if len(lines) > 1 else "See section content for details."

        source_tag = f" [{r.source}]" if r.source != "vector" else ""
        faq_entries.append(f"Q: {question}\nA: {answer}{source_tag}")
    return "\n\n".join(faq_entries) if faq_entries else ""

@tool
async def search_faq(
    query: str,
    limit: int = 3,
) -> str:
    """Search for FAQ entries based on a natural language query. Supports semantic search with automatic query optimization and reranking."""
    try:
        orchestrator = _get_faq_orchestrator()
        if orchestrator is not None:
            results = await orchestrator.search(query, limit=limit)
            formatted = _format_faq_results(results)
            if formatted:
                return formatted
            return f"No FAQ entries found matching query: {query}"

        faq_vectordb = _get_faq_vectordb()
        if faq_vectordb is None:
            return "VectorDB not available (Qdrant not running). Please start Qdrant service."

        search_results = faq_vectordb.search(query, limit=limit)
        faq_entries = []
        for result in search_results:
            payload = result.payload
            content = payload.get("content", "")
            question = "General FAQ Information"
            answer = content
            question_match = re.search(r'^\d+\. (.+?)(?=\n|$)', content, re.MULTILINE)
            if question_match:
                question = question_match.group(1).strip()
                answer_start = content.find(question) + len(question)
                answer = content[answer_start:].strip()
            elif content.startswith('##'):
                lines = content.split('\n', 1)
                question = lines[0].replace('##', '').strip()
                answer = lines[1] if len(lines) > 1 else "See section content for details."
            faq_entries.append(f"Q: {question}\nA: {answer}")
        if not faq_entries:
            return f"No FAQ entries found matching query: {query}"
        return "\n\n".join(faq_entries)
    except Exception as e:
        logger.error(f"Error searching FAQ: {e}")
        return f"Error searching FAQ: {str(e)}"

@tool
async def lookup_policy(query: str) -> str:
    """Consult the company policies to check whether certain options are permitted.
    Use this before making any flight changes or performing other 'write' events."""
    try:
        faq_results = await search_faq.ainvoke({"query": query, "limit": 3})
        if not faq_results:
            return "Sorry, I couldn't find any relevant policy information. Please contact support for assistance."

        return f"Here's the relevant policy information:\n\n{faq_results}"
    except Exception as e:
        logger.error(f"Error looking up policy: {e}")
        return f"Sorry, an error occurred while looking up policy information: {str(e)}"
