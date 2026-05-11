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
    """
    获取汽车租赁向量数据库的单例实例（懒加载模式）。

    首次调用时初始化VectorDB连接，后续调用直接返回缓存的全局实例。

    Returns:
        VectorDB or None: VectorDB实例对象，如果初始化失败则返回None
    """
    global _cars_vectordb
    if _cars_vectordb is None:
        #  延迟初始化VectorDB实例，捕获异常避免程序崩溃
        try:
            _cars_vectordb = VectorDB(table_name="car_rentals", collection_name="car_rentals_collection")
        except Exception as e:
            from customer_support_chat.app.core.logger import logger
            logger.warning(f"Failed to initialize cars VectorDB: {e}")
            return None
    return _cars_vectordb

def _get_cars_orchestrator():
    """
    获取汽车租赁检索编排器的单例实例（懒加载模式）。

    首次调用时基于向量数据库创建RetrievalOrchestrator，配置混合检索、
    重排序和查询重写等功能，后续调用直接返回缓存的全局实例。

    Returns:
        RetrievalOrchestrator or None: 检索编排器实例对象，如果向量数据库未初始化则返回None
    """
    global _cars_orchestrator
    # 依赖向量数据库实例，仅在其成功初始化后创建编排器
    if _cars_orchestrator is None:
        vdb = _get_cars_vectordb()
        if vdb is not None:
            # 配置混合检索策略：向量权重0.7 + 关键词权重0.3，启用重排序和查询重写
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
    """
    将汽车租赁检索结果格式化为可读的字符串。

    遍历检索结果列表，提取每辆车的名称、位置、价格区间、租期状态等信息，
    并标注数据来源（如向量检索或关键词检索）。

    Args:
        results (List[RetrievalResult]): 检索结果列表，每个结果包含车辆租赁信息

    Returns:
        str: 格式化后的车辆信息字符串，每条记录占一行；若无结果则返回空字符串
    """
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
    """
    从打车/出行相关的自然语言查询中解析起点和终点信息。

    使用正则表达式匹配常见的中文出行表达模式（如"从A到B"、"A去B打车"等），
    提取起点和终点名称，其他字段（经纬度）保持为空待后续填充。

    Args:
        query (str): 用户的打车查询语句，例如"从北京站到上海虹桥多少钱"

    Returns:
        Dict[str, str]: 包含解析结果的字典，键包括from_name、to_name及经纬度字段，
                        未匹配到的字段值为空字符串
    """
    result = {"from_name": "", "to_name": "", "from_lng": "", "from_lat": "", "to_lng": "", "to_lat": ""}
    patterns = [
        r'从(.+?)(?:到|去|至|→)(.+?)(?:的|打车|叫车|滴滴|出行|专车|快车|顺风车|多少钱|报价|估价|$)',
        r'(.+?)(?:到|去|至|→)(.+?)(?:打车|叫车|滴滴|出行|专车|快车|顺风车|多少钱|报价|估价)',
    ]
    # 尝试多种正则模式匹配，提取起点和终点信息
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
    """
    基于自然语言查询搜索汽车租赁或网约车服务。

    优先使用滴滴出行MCP API获取实时打车估价，若失败则回退到本地向量检索。
    支持解析起点和终点信息，调用地理编码服务获取坐标，并估算行程费用。

    Args:
        query (str): 用户的自然语言查询，例如"从北京站到机场打车多少钱"
        limit (int): 返回结果的最大数量限制（默认5条）

    Returns:
        str: 格式化的搜索结果字符串，包含车辆信息或打车估价；
             若无匹配结果或发生错误则返回相应提示信息
    """
    """Search for car rentals / ride-hailing based on a natural language query. Supports DiDi MCP API (滴滴出行) for real-time ride estimates, with fallback to local vector search."""
    try:
        # 优先尝试使用滴滴MCP API进行实时打车估价
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
                    # 从滴滴API响应中提取出发地经纬度坐标
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
                    # 从滴滴API响应中提取目的地经纬度坐标
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
                    # 坐标解析成功后调用估价接口，失败则返回友好提示
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

        # 滴滴API不可用时，回退到本地向量检索编排器
        orchestrator = _get_cars_orchestrator()
        if orchestrator is not None:
            results = await orchestrator.search(query, limit=limit)
            formatted = _format_car_results(results)
            if formatted:
                return formatted
            return f"No car rentals found matching query: {query}"

        # 最后降级方案：直接使用VectorDB进行基础检索
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
