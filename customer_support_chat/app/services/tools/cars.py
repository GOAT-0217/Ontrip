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
from customer_support_chat.app.services.api_clients import get_didi_client
import sqlite3
import re
from typing import List, Dict, Optional, Union
from datetime import datetime, date

settings = get_settings()
db = settings.SQLITE_DB_PATH

_cars_vectordb = None
_cars_orchestrator = None

def _get_cars_vectordb():
    global _cars_vectordb
    if _cars_vectordb is None:
        try:
            _cars_vectordb = VectorDB(table_name="car_rentals", collection_name="car_rentals_collection")
        except Exception as e:
            from customer_support_chat.app.core.logger import logger
            logger.warning(f"Failed to initialize cars VectorDB: {e}")
            return None
    return _cars_vectordb

def _get_cars_orchestrator():
    global _cars_orchestrator
    if _cars_orchestrator is None:
        vdb = _get_cars_vectordb()
        if vdb is not None:
            _cars_orchestrator = RetrievalOrchestrator(
                vectordb=vdb,
                table_name="car_rentals",
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
    return _cars_orchestrator

def _format_car_results(results: List[RetrievalResult]) -> str:
    rentals = []
    for r in results:
        payload = r.payload
        booked = "Booked" if payload.get("booked") else "Available"
        source_tag = f" [{r.source}]" if r.source != "vector" else ""
        rentals.append(
            f"Car: {payload.get('name')}, Location: {payload.get('location')}, "
            f"Price: {payload.get('price_tier')}, From: {payload.get('start_date')} "
            f"To: {payload.get('end_date')}, Status: {booked}{source_tag}"
        )
    return "\n".join(rentals) if rentals else ""

def _parse_ride_query(query: str) -> Dict[str, str]:
    result = {"from_name": "", "to_name": "", "from_lng": "", "from_lat": "", "to_lng": "", "to_lat": ""}
    patterns = [
        r'从(.+?)(?:到|去|至|→)(.+?)(?:的|打车|叫车|滴滴|出行|专车|快车|顺风车|多少钱|报价|估价|$)',
        r'(.+?)(?:到|去|至|→)(.+?)(?:打车|叫车|滴滴|出行|专车|快车|顺风车|多少钱|报价|估价)',
    ]
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            result["from_name"] = match.group(1).strip()
            result["to_name"] = match.group(2).strip()
            break
    return result

@tool
async def search_car_rentals(
    query: str,
    limit: int = 5,
) -> str:
    """Search for car rentals / ride-hailing based on a natural language query. Supports DiDi MCP API (滴滴出行) for real-time ride estimates, with fallback to local vector search."""
    try:
        didi_client = get_didi_client()
        if didi_client.is_configured():
            try:
                parsed = _parse_ride_query(query)
                if parsed["from_name"] and parsed["to_name"]:
                    from_place = didi_client.search_place(parsed["from_name"])
                    to_place = didi_client.search_place(parsed["to_name"])

                    from_data = didi_client.format_place_result(from_place)
                    to_data = didi_client.format_place_result(to_place)

                    from_lng = ""
                    from_lat = ""
                    to_lng = ""
                    to_lat = ""

                    if isinstance(from_data, dict):
                        loc = from_data.get("location", from_data.get("geometry", {}))
                        if isinstance(loc, dict):
                            from_lng = str(loc.get("lng", loc.get("lon", "")))
                            from_lat = str(loc.get("lat", ""))
                        if not from_lng:
                            results_list = from_data.get("results", from_data.get("pois", []))
                            if results_list and isinstance(results_list, list):
                                first = results_list[0]
                                loc = first.get("location", first.get("geometry", {}))
                                if isinstance(loc, dict):
                                    from_lng = str(loc.get("lng", loc.get("lon", "")))
                                    from_lat = str(loc.get("lat", ""))

                    if isinstance(to_data, dict):
                        loc = to_data.get("location", to_data.get("geometry", {}))
                        if isinstance(loc, dict):
                            to_lng = str(loc.get("lng", loc.get("lon", "")))
                            to_lat = str(loc.get("lat", ""))
                        if not to_lng:
                            results_list = to_data.get("results", to_data.get("pois", []))
                            if results_list and isinstance(results_list, list):
                                first = results_list[0]
                                loc = first.get("location", first.get("geometry", {}))
                                if isinstance(loc, dict):
                                    to_lng = str(loc.get("lng", loc.get("lon", "")))
                                    to_lat = str(loc.get("lat", ""))

                    if from_lng and from_lat and to_lng and to_lat:
                        estimate_result = didi_client.estimate_ride(
                            from_lng=from_lng,
                            from_lat=from_lat,
                            from_name=parsed["from_name"],
                            to_lng=to_lng,
                            to_lat=to_lat,
                            to_name=parsed["to_name"],
                        )
                        return didi_client.format_estimate_result(estimate_result)
                    else:
                        return (
                            f"⚠️ 无法解析地点坐标\n"
                            f"出发地: {parsed['from_name']} → {'✅' if from_lng else '❌'}\n"
                            f"目的地: {parsed['to_name']} → {'✅' if to_lng else '❌'}\n\n"
                            f"💡 请尝试更具体的地址，如：从北京天安门到北京首都机场打车"
                        )
            except Exception as e:
                from customer_support_chat.app.core.logger import logger
                logger.warning(f"滴滴MCP查询失败，回退到本地检索: {e}")

        orchestrator = _get_cars_orchestrator()
        if orchestrator is not None:
            results = await orchestrator.search(query, limit=limit)
            formatted = _format_car_results(results)
            if formatted:
                return formatted
            return f"No car rentals found matching query: {query}"

        vdb = _get_cars_vectordb()
        if vdb is None:
            return "VectorDB not available (Qdrant not running)."
        search_results = vdb.search(query, limit=limit)
        rentals = []
        for result in search_results:
            payload = result.payload
            booked = "Booked" if payload.get("booked") else "Available"
            rentals.append(
                f"Car: {payload.get('name')}, Location: {payload.get('location')}, "
                f"Price: {payload.get('price_tier')}, From: {payload.get('start_date')} "
                f"To: {payload.get('end_date')}, Status: {booked}"
            )
        if not rentals:
            return f"No car rentals found matching query: {query}"
        return "\n".join(rentals)
    except Exception as e:
        return f"Error searching car rentals: {str(e)}"

@tool
async def estimate_didi_ride(
    from_name: str,
    to_name: str,
) -> str:
    """Estimate ride-hailing prices using DiDi MCP API. Provide departure and destination names (in Chinese). Returns available car types with prices and ETAs."""
    try:
        didi_client = get_didi_client()
        if not didi_client.is_configured():
            return "滴滴 MCP 未配置，请设置 DID_MCP_KEY 环境变量"

        from_place = didi_client.search_place(from_name)
        to_place = didi_client.search_place(to_name)

        from_data = didi_client.format_place_result(from_place)
        to_data = didi_client.format_place_result(to_place)

        from_lng = ""
        from_lat = ""
        to_lng = ""
        to_lat = ""

        if isinstance(from_data, dict):
            loc = from_data.get("location", from_data.get("geometry", {}))
            if isinstance(loc, dict):
                from_lng = str(loc.get("lng", loc.get("lon", "")))
                from_lat = str(loc.get("lat", ""))
            if not from_lng:
                results_list = from_data.get("results", from_data.get("pois", []))
                if results_list and isinstance(results_list, list):
                    first = results_list[0]
                    loc = first.get("location", first.get("geometry", {}))
                    if isinstance(loc, dict):
                        from_lng = str(loc.get("lng", loc.get("lon", "")))
                        from_lat = str(loc.get("lat", ""))

        if isinstance(to_data, dict):
            loc = to_data.get("location", to_data.get("geometry", {}))
            if isinstance(loc, dict):
                to_lng = str(loc.get("lng", loc.get("lon", "")))
                to_lat = str(loc.get("lat", ""))
            if not to_lng:
                results_list = to_data.get("results", to_data.get("pois", []))
                if results_list and isinstance(results_list, list):
                    first = results_list[0]
                    loc = first.get("location", first.get("geometry", {}))
                    if isinstance(loc, dict):
                        to_lng = str(loc.get("lng", loc.get("lon", "")))
                        to_lat = str(loc.get("lat", ""))

        if not (from_lng and from_lat and to_lng and to_lat):
            return f"⚠️ 无法解析地点坐标，请尝试更具体的地址名称"

        estimate_result = didi_client.estimate_ride(
            from_lng=from_lng,
            from_lat=from_lat,
            from_name=from_name,
            to_lng=to_lng,
            to_lat=to_lat,
            to_name=to_name,
        )
        return didi_client.format_estimate_result(estimate_result)
    except Exception as e:
        return f"Error estimating ride: {str(e)}"

@tool
@humanloop_adapter.require_approval(execute_on_reject=False)
async def book_car_rental(rental_id: int, approval_result=None) -> str:
    """Book a car rental by its ID."""
    try:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        cursor.execute("UPDATE car_rentals SET booked = 1 WHERE id = ?", (rental_id,))
        conn.commit()

        if cursor.rowcount > 0:
            conn.close()
            return f"Car rental {rental_id} successfully booked."
        else:
            conn.close()
            return f"No car rental found with ID {rental_id}."
    except Exception as e:
        return f"Error booking car rental: {str(e)}"

@tool
@humanloop_adapter.require_approval(execute_on_reject=False)
async def update_car_rental(
    rental_id: int,
    start_date: Optional[Union[datetime, date]] = None,
    end_date: Optional[Union[datetime, date]] = None,
    approval_result=None
) -> str:
    """Update a car rental's start and end dates by its ID."""
    try:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        if start_date:
            cursor.execute(
                "UPDATE car_rentals SET start_date = ? WHERE id = ?",
                (start_date.strftime('%Y-%m-%d'), rental_id),
            )
        if end_date:
            cursor.execute(
                "UPDATE car_rentals SET end_date = ? WHERE id = ?",
                (end_date.strftime('%Y-%m-%d'), rental_id),
            )

        conn.commit()

        if cursor.rowcount > 0:
            conn.close()
            return f"Car rental {rental_id} successfully updated."
        else:
            conn.close()
            return f"No car rental found with ID {rental_id}."
    except Exception as e:
        return f"Error updating car rental: {str(e)}"

@tool
@humanloop_adapter.require_approval(execute_on_reject=False)
async def cancel_car_rental(rental_id: int, approval_result=None) -> str:
    """Cancel a car rental by its ID."""
    try:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        cursor.execute("UPDATE car_rentals SET booked = 0 WHERE id = ?", (rental_id,))
        conn.commit()

        if cursor.rowcount > 0:
            conn.close()
            return f"Car rental {rental_id} successfully cancelled."
        else:
            conn.close()
            return f"No car rental found with ID {rental_id}."
    except Exception as e:
        return f"Error cancelling car rental: {str(e)}"
