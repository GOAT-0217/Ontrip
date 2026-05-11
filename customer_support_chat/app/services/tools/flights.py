import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from vectorizer.app.vectordb.vectordb import VectorDB
from customer_support_chat.app.core.settings import get_settings
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from customer_support_chat.app.core.humanloop_manager import humanloop_adapter
import sqlite3
from typing import Optional, Union, List, Dict
from datetime import datetime, date
import pytz

settings = get_settings()
db = settings.SQLITE_DB_PATH

# Lazy initialization
_flights_vectordb = None

def _get_flights_vectordb():
    global _flights_vectordb
    if _flights_vectordb is None:
        # 延迟初始化VectorDB实例，捕获异常避免程序崩溃
        try:
            _flights_vectordb = VectorDB(table_name="flights", collection_name="flights_collection")
        except Exception as e:
            from customer_support_chat.app.core.logger import logger
            logger.warning(f"Failed to initialize flights VectorDB: {e}")
            return None
    return _flights_vectordb


@tool
def fetch_user_flight_information(*, config: RunnableConfig) -> str:
    """
    获取指定乘客的所有机票信息及对应的航班详情和座位分配。

    通过连接SQLite数据库，联查tickets、flights、ticket_flights及boarding_passes表，
    提取航班号、起降时间、座位号及舱位等级等关键信息并格式化返回。

    Args:
        config (RunnableConfig): 运行配置对象，需包含configurable字典，
                                 其中应提供passenger_id用于查询

    Returns:
        str: 格式化后的航班信息列表，每条记录包含票务、航班时刻及座位详情；
             若未提供乘客ID或无相关数据则返回相应提示
    """
    try:
        # 从配置中提取乘客ID，若缺失则直接返回错误提示
        configuration = config.get("configurable", {})
        passenger_id = configuration.get("passenger_id", None)
        if not passenger_id:
            return "No passenger ID configured."

        # 执行多表联查，获取机票、航班及座位信息
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        query = """
        SELECT 
            t.ticket_no, t.book_ref,
            f.flight_id, f.flight_no, f.departure_airport, f.arrival_airport, f.scheduled_departure, f.scheduled_arrival,
            bp.seat_no, tf.fare_conditions
        FROM 
            tickets t
            JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
            JOIN flights f ON tf.flight_id = f.flight_id
            LEFT JOIN boarding_passes bp ON bp.ticket_no = t.ticket_no AND bp.flight_id = f.flight_id
        WHERE 
            t.passenger_id = ?
        """
        cursor.execute(query, (passenger_id,))
        rows = cursor.fetchall()
        column_names = [column[0] for column in cursor.description]
        results = [dict(zip(column_names, row)) for row in rows]

        cursor.close()
        conn.close()

        # 处理查询结果，格式化为易读的字符串输出
        if not results:
            return f"No flight information found for passenger {passenger_id}."
        
        lines = []
        for r in results:
            lines.append(
                f"Ticket: {r.get('ticket_no')}, Flight: {r.get('flight_no')}, "
                f"From: {r.get('departure_airport')} To: {r.get('arrival_airport')}, "
                f"Departure: {r.get('scheduled_departure')}, Arrival: {r.get('scheduled_arrival')}, "
                f"Seat: {r.get('seat_no', 'N/A')}, Class: {r.get('fare_conditions', 'N/A')}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching flight information: {str(e)}"


@tool
def search_flights(
    query: str,
    limit: int = 2,
) -> str:
    """
    基于自然语言查询搜索航班信息。

    利用向量数据库进行语义检索，提取航班号、起降机场、时间及状态等关键信息并格式化返回。

    Args:
        query (str): 用户的自然语言查询，例如"明天早上从北京飞往上海的航班"
        limit (int): 返回结果的最大数量限制（默认2条）

    Returns:
        str: 格式化后的航班信息列表；若数据库不可用或无匹配结果则返回相应提示
    """
    try:
        # 获取航班向量数据库实例，若初始化失败则返回错误提示
        vdb = _get_flights_vectordb()
        if vdb is None:
            return "VectorDB not available (Qdrant not running)."

        # 执行向量搜索并格式化结果为易读的字符串
        search_results = vdb.search(query, limit=limit)

        flights = []
        for result in search_results:
            payload = result.payload
            flights.append(
                f"Flight {payload.get('flight_no')}: {payload.get('departure_airport')} -> {payload.get('arrival_airport')}, "
                f"Departure: {payload.get('scheduled_departure')}, Arrival: {payload.get('scheduled_arrival')}, "
                f"Status: {payload.get('status')}"
            )
        if not flights:
            return f"No flights found matching query: {query}"
        return "\n".join(flights)
    except Exception as e:
        return f"Error searching flights: {str(e)}"


@tool
@humanloop_adapter.require_approval(execute_on_reject=False)
async def update_ticket_to_new_flight(
    ticket_no: str, new_flight_id: int, *, config: RunnableConfig, approval_result=None
) -> str:
    """
    将用户的机票改签到新的有效航班。

    首先验证机票是否属于当前乘客，随后更新ticket_flights表中的航班ID。
    操作成功与否将通过返回的字符串信息进行反馈。

    Args:
        ticket_no (str): 需要改签的机票编号
        new_flight_id (int): 目标新航班的唯一标识符
        config (RunnableConfig): 运行配置对象，需包含configurable字典，其中应提供passenger_id
        approval_result: 可选的审批结果对象（当前逻辑中未使用）

    Returns:
        str: 改签操作的结果提示信息，包含成功确认、未找到票据或更新失败的说明
    """
    try:
        # 从配置中提取乘客ID以进行身份验证
        configuration = config.get("configurable", {})
        passenger_id = configuration.get("passenger_id", None)
        if not passenger_id:
            return "Error: No passenger ID configured."

        # 验证机票是否存在且归属于当前乘客
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        # 执行更新操作，将机票关联到新的航班ID
        cursor.execute(
            "SELECT * FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
            (ticket_no, passenger_id),
        )
        ticket = cursor.fetchone()
        if not ticket:
            conn.close()
            return f"Ticket {ticket_no} not found for passenger {passenger_id}."

        cursor.execute(
            "UPDATE ticket_flights SET flight_id = ? WHERE ticket_no = ?",
            (new_flight_id, ticket_no),
        )
        conn.commit()

        conn.close()
        # 根据受影响的行数判断改签是否成功
        if cursor.rowcount > 0:
            return f"Ticket {ticket_no} successfully updated to flight {new_flight_id}."
        else:
            return f"Failed to update ticket {ticket_no}."
    except Exception as e:
        return f"Error updating ticket: {str(e)}"


@tool
@humanloop_adapter.require_approval(execute_on_reject=False)
async def cancel_ticket(ticket_no: str, *, config: RunnableConfig, approval_result=None) -> str:
    """
    取消用户的机票并从数据库中移除相关记录。

    首先验证机票是否属于当前乘客，随后依次删除ticket_flights和tickets表中的关联数据，
    确保数据一致性。操作结果将通过返回的字符串信息进行反馈。

    Args:
        ticket_no (str): 需要取消的机票编号
        config (RunnableConfig): 运行配置对象，需包含configurable字典，其中应提供passenger_id
        approval_result: 可选的审批结果对象（当前逻辑中未使用）

    Returns:
        str: 取消操作的结果提示信息，包含成功确认、未找到票据或发生错误的说明
    """
    try:
        # 从配置中提取乘客ID以进行身份验证
        configuration = config.get("configurable", {})
        passenger_id = configuration.get("passenger_id", None)
        if not passenger_id:
            return "Error: No passenger ID configured."

        # 验证机票是否存在且归属于当前乘客
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        # 执行级联删除：先删除航班关联记录，再删除主票据记录
        cursor.execute(
            "SELECT * FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
            (ticket_no, passenger_id),
        )
        ticket = cursor.fetchone()
        if not ticket:
            conn.close()
            return f"Ticket {ticket_no} not found for passenger {passenger_id}."

        cursor.execute(
            "DELETE FROM ticket_flights WHERE ticket_no = ?",
            (ticket_no,),
        )
        cursor.execute(
            "DELETE FROM tickets WHERE ticket_no = ?",
            (ticket_no,),
        )
        conn.commit()

        conn.close()
        return f"Ticket {ticket_no} successfully cancelled."
    except Exception as e:
        return f"Error cancelling ticket: {str(e)}"
