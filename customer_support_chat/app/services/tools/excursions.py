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
    """
    该函数实现游览推荐向量数据库的懒加载单例模式：
        1.延迟初始化：首次调用时创建VectorDB实例，后续复用全局对象
        2.错误处理：捕获初始化异常并记录日志，失败时返回None
    """
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
    """
       获取游览推荐检索编排器的单例实例（懒加载模式）。

       首次调用时基于向量数据库创建RetrievalOrchestrator，配置混合检索、
       重排序和查询重写等功能，后续调用直接返回缓存的全局实例。

       Returns:
           RetrievalOrchestrator or None: 检索编排器实例对象，如果向量数据库未初始化则返回None
       """
    global _excursions_orchestrator
    if _excursions_orchestrator is None:
        # 依赖向量数据库实例，仅在其成功初始化后创建编排器
        vdb = _get_excursions_vectordb()
        if vdb is not None:
            # 配置混合检索策略：向量权重0.7 + 关键词权重0.3，启用重排序和查询重写
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
    """
        将游览推荐检索结果格式化为可读的字符串。

        遍历检索结果列表，提取每个游览项目的名称、位置、关键词、详情及预订状态，
        并标注数据来源（如向量检索或关键词检索）。

        Args:
            results (List[RetrievalResult]): 检索结果列表，每个结果包含游览推荐信息

        Returns:
            str: 格式化后的游览信息字符串，每条记录占一行；若无结果则返回空字符串
        """
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
    """
    基于自然语言查询搜索旅行推荐（游览项目/景点）。

    优先使用检索编排器进行混合搜索（支持语义、关键词及自动查询优化），
    若不可用则回退到基础向量数据库检索。

    Args:
        query (str): 用户的自然语言查询，例如"推荐一些适合家庭游玩的景点"
        limit (int): 返回结果的最大数量限制（默认5条）

    Returns:
        str: 格式化后的推荐信息字符串；若无匹配结果或发生错误则返回相应提示
    """
    try:
        # 优先使用功能更强大的检索编排器
        orchestrator = _get_excursions_orchestrator()
        if orchestrator is not None:
            results = await orchestrator.search(query, limit=limit)
            formatted = _format_excursion_results(results)
            if formatted:
                return formatted
            return f"No trip recommendations found matching query: {query}"
        # 降级方案：直接使用VectorDB进行基础检索
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
    """
    根据ID预订游览项目。

    通过更新SQLite数据库中对应记录的booked字段为1来完成预订操作，
    并根据影响行数返回成功或失败的具体信息。

    Args:
        recommendation_id (int): 要预订的游览项目的唯一标识符
        approval_result: 可选的审批结果对象（当前逻辑中未使用）

    Returns:
        str: 预订操作的结果提示信息，包含成功确认或未找到对应ID的错误说明
    """
    try:
        # 连接数据库并执行更新操作，标记指定ID的项目为已预订
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE trip_recommendations SET booked = 1 WHERE id = ?", (recommendation_id,)
        )
        conn.commit()

        # 根据受影响的行数判断预订是否成功
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
    """
    根据ID更新游览项目的详细信息。

    通过更新SQLite数据库中对应记录的details字段来修改项目描述，
    并根据影响行数返回更新成功或失败的具体信息。

    Args:
        recommendation_id (int): 要更新的游览项目的唯一标识符
        details (str): 新的详细信息内容
        approval_result: 可选的审批结果对象（当前逻辑中未使用）

    Returns:
        str: 更新操作的结果提示信息，包含成功确认或未找到对应ID的错误说明
    """
    try:
        # 连接数据库并执行更新操作，修改指定ID项目的详细信息
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE trip_recommendations SET details = ? WHERE id = ?",
            (details, recommendation_id),
        )
        conn.commit()

        # 根据受影响的行数判断更新是否成功
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
    """
    根据ID取消已预订的游览项目。

    通过更新SQLite数据库中对应记录的booked字段为0来取消预订，
    并根据影响行数返回操作成功或失败的具体信息。

    Args:
        recommendation_id (int): 要取消的游览项目的唯一标识符
        approval_result: 可选的审批结果对象（当前逻辑中未使用）

    Returns:
        str: 取消操作的结果提示信息，包含成功确认或未找到对应ID的错误说明
    """
    try:
        # 连接数据库并执行更新操作，将指定ID的项目状态重置为未预订
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE trip_recommendations SET booked = 0 WHERE id = ?", (recommendation_id,)
        )
        conn.commit()

        # 根据受影响的行数判断取消操作是否成功
        if cursor.rowcount > 0:
            conn.close()
            return f"Excursion {recommendation_id} successfully cancelled."
        else:
            conn.close()
            return f"No excursion found with ID {recommendation_id}."
    except Exception as e:
        return f"Error cancelling excursion: {str(e)}"
