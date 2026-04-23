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
from customer_support_chat.app.services.api_clients import get_ctrip_hotel_client
import sqlite3
import re
from typing import Optional, Union, List, Dict
from datetime import datetime, date

settings = get_settings()
db = settings.SQLITE_DB_PATH

_hotels_vectordb = None
_hotels_orchestrator = None

def _get_hotels_vectordb():
    global _hotels_vectordb
    if _hotels_vectordb is None:
        try:
            _hotels_vectordb = VectorDB(table_name="hotels", collection_name="hotels_collection")
        except Exception as e:
            from customer_support_chat.app.core.logger import logger
            logger.warning(f"Failed to initialize hotels VectorDB: {e}")
            return None
    return _hotels_vectordb

def _get_hotels_orchestrator():
    global _hotels_orchestrator
    if _hotels_orchestrator is None:
        vdb = _get_hotels_vectordb()
        if vdb is not None:
            _hotels_orchestrator = RetrievalOrchestrator(
                vectordb=vdb,
                table_name="hotels",
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
    return _hotels_orchestrator

def _format_hotel_results(results: List[RetrievalResult]) -> str:
    hotels = []
    for r in results:
        payload = r.payload
        booked = "Booked" if payload.get("booked") else "Available"
        source_tag = f" [{r.source}]" if r.source != "vector" else ""
        hotels.append(
            f"Hotel: {payload.get('name')}, Location: {payload.get('location')}, "
            f"Price: {payload.get('price_tier')}, Check-in: {payload.get('checkin_date')}, "
            f"Check-out: {payload.get('checkout_date')}, Status: {booked}{source_tag}"
        )
    return "\n".join(hotels) if hotels else ""

def _parse_hotel_query(query: str) -> Dict[str, str]:
    result = {"city": "", "keyword": "", "checkin": "", "checkout": ""}
    city_patterns = [
        r'(?:在|去|到|找|搜索|查询)\s*([北京上海广州深圳成都杭州南京武汉长沙西安重庆昆明三亚海口厦门青岛大连天津哈尔滨济南福州南宁贵阳兰州呼和浩特太原长春合肥南昌温州宁波珠海桂林丽江拉萨银川西宁苏州无锡常州烟台威海泉州汕头][京沪穗深蓉杭宁汉长渝昆三亚厦青大连津哈济福南贵兰呼太长合南温宁珠桂丽拉银西苏锡常烟威泉汕][州城岛市]?)\s*(?:的|酒店|住宿|宾馆|民宿|旅馆)',
        r'([北京上海广州深圳成都杭州南京武汉长沙西安重庆昆明三亚海口厦门青岛大连天津哈尔滨济南福州南宁贵阳兰州呼和浩特太原长春合肥南昌温州宁波珠海桂林丽江拉萨银川西宁苏州无锡常州烟台威海泉州汕头]+)\s*(?:酒店|住宿|宾馆|民宿|旅馆)',
    ]
    for pattern in city_patterns:
        match = re.search(pattern, query)
        if match:
            result["city"] = match.group(1)
            break
    date_pattern = r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2})\s*(?:到|至|-|~|—)\s*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2})'
    date_match = re.search(date_pattern, query)
    if date_match:
        def normalize_date(d):
            d = d.replace('年', '-').replace('月', '-').replace('/', '-')
            return d
        result["checkin"] = normalize_date(date_match.group(1))
        result["checkout"] = normalize_date(date_match.group(2))
    keyword_patterns = [
        r'(?:便宜|经济|实惠|五星级|豪华|商务|海景|亲子|温泉|民宿)',
    ]
    for pattern in keyword_patterns:
        match = re.search(pattern, query)
        if match:
            result["keyword"] = match.group(0)
            break
    return result

@tool
async def search_hotels(
    query: str,
    limit: int = 5,
) -> str:
    """Search for hotels based on a natural language query. Supports Ctrip API (携程) for real-time hotel data, with fallback to local vector search."""
    try:
        ctrip_client = get_ctrip_hotel_client()
        if ctrip_client.is_configured():
            try:
                parsed = _parse_hotel_query(query)
                ctrip_results = ctrip_client.search_hotels(
                    city=parsed["city"] or query,
                    keyword=parsed["keyword"],
                    checkin=parsed["checkin"],
                    checkout=parsed["checkout"],
                    limit=limit,
                )
                if ctrip_results:
                    formatted = []
                    for hotel in ctrip_results:
                        formatted.append(ctrip_client.format_hotel_result(hotel))
                    return "\n\n".join(formatted)
            except Exception as e:
                from customer_support_chat.app.core.logger import logger
                logger.warning(f"携程酒店API查询失败，回退到本地检索: {e}")

        orchestrator = _get_hotels_orchestrator()
        if orchestrator is not None:
            results = await orchestrator.search(query, limit=limit)
            formatted = _format_hotel_results(results)
            if formatted:
                return formatted
            return f"No hotels found matching query: {query}"

        vdb = _get_hotels_vectordb()
        if vdb is None:
            return "VectorDB not available (Qdrant not running)."
        search_results = vdb.search(query, limit=limit)
        hotels = []
        for result in search_results:
            payload = result.payload
            booked = "Booked" if payload.get("booked") else "Available"
            hotels.append(
                f"Hotel: {payload.get('name')}, Location: {payload.get('location')}, "
                f"Price: {payload.get('price_tier')}, Check-in: {payload.get('checkin_date')}, "
                f"Check-out: {payload.get('checkout_date')}, Status: {booked}"
            )
        if not hotels:
            return f"No hotels found matching query: {query}"
        return "\n".join(hotels)
    except Exception as e:
        return f"Error searching hotels: {str(e)}"

@tool
@humanloop_adapter.require_approval(execute_on_reject=False)
async def book_hotel(hotel_id: int, approval_result=None) -> str:
    """Book a hotel by its ID."""
    try:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        cursor.execute("UPDATE hotels SET booked = 1 WHERE id = ?", (hotel_id,))
        conn.commit()

        if cursor.rowcount > 0:
            conn.close()
            return f"Hotel {hotel_id} successfully booked."
        else:
            conn.close()
            return f"No hotel found with ID {hotel_id}."
    except Exception as e:
        return f"Error booking hotel: {str(e)}"

@tool
@humanloop_adapter.require_approval(execute_on_reject=False)
async def update_hotel(
    hotel_id: int,
    checkin_date: Optional[Union[datetime, date]] = None,
    checkout_date: Optional[Union[datetime, date]] = None,
    approval_result=None
) -> str:
    """Update a hotel's check-in and check-out dates by its ID and mark it as booked."""
    try:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        cursor.execute("UPDATE hotels SET booked = 1 WHERE id = ?", (hotel_id,))

        if checkin_date:
            cursor.execute(
                "UPDATE hotels SET checkin_date = ? WHERE id = ?",
                (checkin_date.strftime('%Y-%m-%d'), hotel_id),
            )
        if checkout_date:
            cursor.execute(
                "UPDATE hotels SET checkout_date = ? WHERE id = ?",
                (checkout_date.strftime('%Y-%m-%d'), hotel_id),
            )

        conn.commit()

        if cursor.rowcount > 0:
            conn.close()
            return f"Hotel {hotel_id} successfully updated and booked."
        else:
            conn.close()
            return f"No hotel found with ID {hotel_id}."
    except Exception as e:
        return f"Error updating hotel: {str(e)}"

@tool
@humanloop_adapter.require_approval(execute_on_reject=False)
async def cancel_hotel(hotel_id: int, approval_result=None) -> str:
    """Cancel a hotel by its ID."""
    try:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        cursor.execute("UPDATE hotels SET booked = 0 WHERE id = ?", (hotel_id,))
        conn.commit()

        if cursor.rowcount > 0:
            conn.close()
            return f"Hotel {hotel_id} successfully cancelled."
        else:
            conn.close()
            return f"No hotel found with ID {hotel_id}."
    except Exception as e:
        return f"Error cancelling hotel: {str(e)}"
