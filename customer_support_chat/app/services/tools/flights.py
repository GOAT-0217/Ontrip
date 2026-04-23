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
        try:
            _flights_vectordb = VectorDB(table_name="flights", collection_name="flights_collection")
        except Exception as e:
            from customer_support_chat.app.core.logger import logger
            logger.warning(f"Failed to initialize flights VectorDB: {e}")
            return None
    return _flights_vectordb


@tool
def fetch_user_flight_information(*, config: RunnableConfig) -> str:
    """Fetch all tickets for the user along with corresponding flight information and seat assignments."""
    try:
        configuration = config.get("configurable", {})
        passenger_id = configuration.get("passenger_id", None)
        if not passenger_id:
            return "No passenger ID configured."

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
    """Search for flights based on a natural language query."""
    try:
        vdb = _get_flights_vectordb()
        if vdb is None:
            return "VectorDB not available (Qdrant not running)."
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
    """Update the user's ticket to a new valid flight."""
    try:
        configuration = config.get("configurable", {})
        passenger_id = configuration.get("passenger_id", None)
        if not passenger_id:
            return "Error: No passenger ID configured."

        conn = sqlite3.connect(db)
        cursor = conn.cursor()

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
        if cursor.rowcount > 0:
            return f"Ticket {ticket_no} successfully updated to flight {new_flight_id}."
        else:
            return f"Failed to update ticket {ticket_no}."
    except Exception as e:
        return f"Error updating ticket: {str(e)}"


@tool
@humanloop_adapter.require_approval(execute_on_reject=False)
async def cancel_ticket(ticket_no: str, *, config: RunnableConfig, approval_result=None) -> str:
    """Cancel the user's ticket and remove it from the database."""
    try:
        configuration = config.get("configurable", {})
        passenger_id = configuration.get("passenger_id", None)
        if not passenger_id:
            return "Error: No passenger ID configured."

        conn = sqlite3.connect(db)
        cursor = conn.cursor()

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
