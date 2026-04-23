import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from vectorizer.app.vectordb.vectordb import VectorDB
from customer_support_chat.app.core.settings import get_settings
from langchain_core.tools import tool
from customer_support_chat.app.core.humanloop_manager import humanloop_adapter
from customer_support_chat.app.services.retrieval import (
    RetrievalOrchestrator,
    OrchestratorConfig,
    RetrievalResult,
)
import sqlite3
from typing import Optional, List, Dict

settings = get_settings()
db = settings.SQLITE_DB_PATH

_excursions_vectordb = None
_excursions_orchestrator = None

def _get_excursions_vectordb():
    global _excursions_vectordb
    if _excursions_vectordb is None:
        try:
            _excursions_vectordb = VectorDB(table_name="trip_recommendations", collection_name="excursions_collection")
        except Exception as e:
            from customer_support_chat.app.core.logger import logger
            logger.warning(f"Failed to initialize excursions VectorDB: {e}")
            return None
    return _excursions_vectordb

def _get_excursions_orchestrator():
    global _excursions_orchestrator
    if _excursions_orchestrator is None:
        vdb = _get_excursions_vectordb()
        if vdb is not None:
            _excursions_orchestrator = RetrievalOrchestrator(
                vectordb=vdb,
                table_name="trip_recommendations",
                db_path=db,
                config=OrchestratorConfig(
                    vector_weight=0.7,
                    keyword_weight=0.3,
                    rerank_top_k=5,
                    relevance_threshold=0.25,
                    enable_query_rewrite=True,
                    enable_rerank=True,
                    enable_hybrid=True,
                ),
            )
    return _excursions_orchestrator

def _format_excursion_results(results: List[RetrievalResult]) -> str:
    recommendations = []
    for r in results:
        payload = r.payload
        booked = "Booked" if payload.get("booked") else "Available"
        source_tag = f" [{r.source}]" if r.source != "vector" else ""
        recommendations.append(
            f"Excursion: {payload.get('name')}, Location: {payload.get('location')}, "
            f"Keywords: {payload.get('keywords')}, Details: {payload.get('details')}, "
            f"Status: {booked}{source_tag}"
        )
    return "\n".join(recommendations) if recommendations else ""

@tool
async def search_trip_recommendations(
    query: str,
    limit: int = 5,
) -> str:
    """Search for trip recommendations based on a natural language query. Supports semantic, keyword, and hybrid search with automatic query optimization."""
    try:
        orchestrator = _get_excursions_orchestrator()
        if orchestrator is not None:
            results = await orchestrator.search(query, limit=limit)
            formatted = _format_excursion_results(results)
            if formatted:
                return formatted
            return f"No trip recommendations found matching query: {query}"

        vdb = _get_excursions_vectordb()
        if vdb is None:
            return "VectorDB not available (Qdrant not running)."
        search_results = vdb.search(query, limit=limit)
        recommendations = []
        for result in search_results:
            payload = result.payload
            booked = "Booked" if payload.get("booked") else "Available"
            recommendations.append(
                f"Excursion: {payload.get('name')}, Location: {payload.get('location')}, "
                f"Keywords: {payload.get('keywords')}, Details: {payload.get('details')}, "
                f"Status: {booked}"
            )
        if not recommendations:
            return f"No trip recommendations found matching query: {query}"
        return "\n".join(recommendations)
    except Exception as e:
        return f"Error searching trip recommendations: {str(e)}"

@tool
@humanloop_adapter.require_approval(execute_on_reject=False)
async def book_excursion(recommendation_id: int, approval_result=None) -> str:
    """Book an excursion by its ID."""
    try:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE trip_recommendations SET booked = 1 WHERE id = ?", (recommendation_id,)
        )
        conn.commit()

        if cursor.rowcount > 0:
            conn.close()
            return f"Excursion {recommendation_id} successfully booked."
        else:
            conn.close()
            return f"No excursion found with ID {recommendation_id}."
    except Exception as e:
        return f"Error booking excursion: {str(e)}"

@tool
@humanloop_adapter.require_approval(execute_on_reject=False)
async def update_excursion(recommendation_id: int, details: str, approval_result=None) -> str:
    """Update an excursion's details by its ID."""
    try:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE trip_recommendations SET details = ? WHERE id = ?",
            (details, recommendation_id),
        )
        conn.commit()

        if cursor.rowcount > 0:
            conn.close()
            return f"Excursion {recommendation_id} successfully updated."
        else:
            conn.close()
            return f"No excursion found with ID {recommendation_id}."
    except Exception as e:
        return f"Error updating excursion: {str(e)}"

@tool
@humanloop_adapter.require_approval(execute_on_reject=False)
async def cancel_excursion(recommendation_id: int, approval_result=None) -> str:
    """Cancel an excursion by its ID."""
    try:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE trip_recommendations SET booked = 0 WHERE id = ?", (recommendation_id,)
        )
        conn.commit()

        if cursor.rowcount > 0:
            conn.close()
            return f"Excursion {recommendation_id} successfully cancelled."
        else:
            conn.close()
            return f"No excursion found with ID {recommendation_id}."
    except Exception as e:
        return f"Error cancelling excursion: {str(e)}"
