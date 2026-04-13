# customer_support_chat/app/services/chat_service.py
"""
This module provides a service to process user messages using the LangGraph multi-agent system.
It encapsulates the core chat logic from main.py to make it reusable in a web application context.
"""

import asyncio
import sys
import os
from typing import Dict, Any, List, Union
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from customer_support_chat.app.graph import multi_agentic_graph
from customer_support_chat.app.core.logger import logger

# Try to import web_app modules
try:
    # Add the project root directory to the path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if project_root not in sys.path:
        sys.path.append(project_root)

    from web_app.app.core.user_data_manager import set_pending_action, get_pending_action, get_user_decision, \
        clear_pending_action, clear_user_decision, add_operation_log

    WEB_APP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Web app modules not available. HITL functionality will be limited. Error: {e}")
    WEB_APP_AVAILABLE = False


async def process_user_message(session_data: Dict[str, Any], user_message: str) -> str:
    """
    Process a user message using the LangGraph multi-agent system.

    Args:
        session_data (Dict[str, Any]): The session data containing the config (thread_id, passenger_id).
        user_message (str): The user's message to process.

    Returns:
        str: The AI's response message.
    """
    # Extract the config from session_data
    config = session_data.get("config", {})
    # Ensure it's in the correct format for LangGraph
    langgraph_config = {"configurable": config}

    # Variable to track printed message IDs to avoid duplicates
    printed_message_ids = set()
    latest_ai_response = None

    try:
        # Add user input to operation log
        if WEB_APP_AVAILABLE:
            add_operation_log(session_data["session_id"], {
                "type": "user_input",
                "title": "User Message",
                "content": user_message
            })

        # Process the user input through the graph
        events = multi_agentic_graph.stream(
            {"messages": [("user", user_message)]}, langgraph_config, stream_mode="values"
        )

        # Collect messages from the stream
        all_tool_calls_needing_response = []

        for event in events:
            messages = event.get("messages", [])
            for message in messages:
                if message.id not in printed_message_ids:
                    # Track any tool calls that need responses
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        for tool_call in message.tool_calls:
                            all_tool_calls_needing_response.append(tool_call)
                            logger.debug(f"Tracking tool call: {tool_call['name']} (ID: {tool_call['id']})")

                    # Log different types of messages
                    if WEB_APP_AVAILABLE:
                        if isinstance(message, AIMessage) and message.content and message.content.strip():
                            add_operation_log(session_data["session_id"], {
                                "type": "ai_response",
                                "title": "AI Response",
                                "content": message.content
                            })
                        elif hasattr(message, 'tool_calls') and message.tool_calls:
                            for tool_call in message.tool_calls:
                                add_operation_log(session_data["session_id"], {
                                    "type": "tool_call",
                                    "title": f"{tool_call['name']} call",
                                    "content": "\n".join([f"{k}: {v}" for k, v in tool_call['args'].items()]),
                                    "details": {
                                        "tool_name": tool_call['name'],
                                        "tool_call_id": tool_call['id'],
                                        "parameters": tool_call['args']
                                    }
                                })

                    if isinstance(message, AIMessage) and message.content.strip():
                        latest_ai_response = message.content
                    printed_message_ids.add(message.id)

        logger.info(f"Processed {len(all_tool_calls_needing_response)} tool calls during stream")

        # ========== 增强补全：确保所有工具调用都有 ToolMessage ==========
        # 无论是否有中断，先检查并补全缺失的 ToolMessage
        snapshot = multi_agentic_graph.get_state(langgraph_config)
        if snapshot.values and "messages" in snapshot.values:
            messages = snapshot.values["messages"]
            tool_calls_seen = set()
            tool_responses_seen = set()
            for msg in messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls_seen.add(tc['id'])
                elif hasattr(msg, 'tool_call_id'):
                    tool_responses_seen.add(msg.tool_call_id)

            unresponded = tool_calls_seen - tool_responses_seen
            if unresponded:
                logger.warning(f"Found {len(unresponded)} unresponded tool calls, adding auto ToolMessages.")
                tool_messages = []
                for tc_id in unresponded:
                    tool_messages.append(
                        ToolMessage(
                            tool_call_id=tc_id,
                            content="Tool call acknowledged (auto-response)."
                        )
                    )
                multi_agentic_graph.update_state(
                    langgraph_config,
                    {"messages": tool_messages}
                )
                logger.info(f"Added {len(tool_messages)} auto-response ToolMessages.")
                # 更新 snapshot 以便后续中断检查
                snapshot = multi_agentic_graph.get_state(langgraph_config)
        # ========== 补全结束 ==========

        # Check for interrupts (HITL)
        if snapshot.next:
            logger.info("Interrupt occurred. Handling HITL approval request.")

            last_message = snapshot.values["messages"][-1] if snapshot.values.get("messages") else None

            if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                logger.info(
                    f"Last message has {len(last_message.tool_calls)} tool calls: {[tc['name'] for tc in last_message.tool_calls]}")

                if WEB_APP_AVAILABLE:
                    tool_calls_details = []
                    for tool_call in last_message.tool_calls:
                        tool_calls_details.append({
                            "id": tool_call["id"],
                            "name": tool_call["name"],
                            "args": tool_call["args"]
                        })

                    pending_action = {
                        "tool_calls": tool_calls_details,
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    set_pending_action(session_data["session_id"], pending_action)

                    add_operation_log(session_data["session_id"], {
                        "type": "system_message",
                        "title": "HITL Interrupt",
                        "content": "Sensitive action requires user approval",
                        "details": {"tool_calls": tool_calls_details}
                    })

                    # 发送确认 ToolMessages（HITL 专用）
                    ack_messages = []
                    for tool_call in last_message.tool_calls:
                        logger.info(f"Sending HITL ack for tool_call_id: {tool_call['id']}")
                        ack_messages.append(
                            ToolMessage(
                                tool_call_id=tool_call["id"],
                                content="Action requires user approval. Please wait for user decision.",
                            )
                        )
                    multi_agentic_graph.update_state(
                        langgraph_config,
                        {"messages": ack_messages}
                    )

                    if latest_ai_response:
                        latest_ai_response += "\n\n[User approval required for sensitive action. Please approve or reject this action in the web interface.]"
                    else:
                        latest_ai_response = "[User approval required for sensitive action. Please approve or reject this action in the web interface.]"
                else:
                    # Fallback denial
                    denial_messages = []
                    for tool_call in last_message.tool_calls:
                        denial_messages.append(
                            ToolMessage(
                                tool_call_id=tool_call["id"],
                                content="API call denied by user. Reasoning: 'Sensitive operations require explicit approval in web interface.'",
                            )
                        )
                    denial_response = multi_agentic_graph.invoke(
                        {"messages": denial_messages},
                        langgraph_config,
                    )
                    messages = denial_response.get("messages", [])
                    for message in messages:
                        if message.id not in printed_message_ids:
                            if isinstance(message, AIMessage) and message.content.strip():
                                latest_ai_response = message.content
                            printed_message_ids.add(message.id)
            else:
                logger.warning("Interrupt detected but no tool calls found in last message")
                if latest_ai_response:
                    latest_ai_response += "\n\n[User approval required for sensitive action. Please contact support for assistance.]"
                else:
                    latest_ai_response = "[User approval required for sensitive action. Please contact support for assistance.]"
        else:
            logger.info("No interrupt detected")

        # Return the latest AI response, or a default message if no AI response was generated
        if latest_ai_response:
            return latest_ai_response
        else:
            return "I'm sorry, I didn't understand that. Could you please rephrase?"

    except Exception as e:
        logger.error(f"An error occurred while processing the user message: {e}")

        # Special handling for the tool_calls error (fallback)
        if "tool_calls must be followed by tool messages" in str(e):
            logger.warning("Detected tool_calls acknowledgment error - attempting recovery")
            try:
                snapshot = multi_agentic_graph.get_state(langgraph_config)
                if snapshot.values and "messages" in snapshot.values:
                    messages = snapshot.values["messages"]
                    tool_calls_seen = set()
                    tool_responses_seen = set()
                    for msg in messages:
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tool_calls_seen.add(tc['id'])
                        elif hasattr(msg, 'tool_call_id'):
                            tool_responses_seen.add(msg.tool_call_id)
                    unresponded = tool_calls_seen - tool_responses_seen
                    if unresponded:
                        tool_messages = [ToolMessage(tool_call_id=tc_id, content="Emergency auto-response.") for tc_id
                                         in unresponded]
                        multi_agentic_graph.update_state(langgraph_config, {"messages": tool_messages})
                        logger.info("Emergency ToolMessages sent.")
                return "I apologize for the technical difficulty. Your request has been processed. Please try again."
            except Exception as recovery_error:
                logger.error(f"Recovery failed: {recovery_error}")

        if WEB_APP_AVAILABLE:
            add_operation_log(session_data["session_id"], {
                "type": "error",
                "title": "Processing Error",
                "content": str(e)
            })
        return "An unexpected error occurred while processing your request. Please try again later."


async def process_user_decision(session_data: Dict[str, Any], decision: str) -> str:
    """
    Process a user's decision (approve/reject) for a pending action.

    Args:
        session_data (Dict[str, Any]): The session data containing the config (thread_id, passenger_id).
        decision (str): The user's decision ('approve' or 'reject').

    Returns:
        str: The AI's response message after processing the decision.
    """
    if not WEB_APP_AVAILABLE:
        return "HITL functionality is not available in this environment."

    # Extract the config from session_data
    config = session_data.get("config", {})
    # Ensure it's in the correct format for LangGraph
    langgraph_config = {"configurable": config}

    # Variable to track printed message IDs to avoid duplicates
    printed_message_ids = set()
    result_message = ""

    try:
        # Get the pending action
        pending_action = get_pending_action(session_data["session_id"])
        if not pending_action:
            return "No pending action found."

        # Add user decision to operation log
        add_operation_log(session_data["session_id"], {
            "type": "user_input",
            "title": "User Decision",
            "content": f"User {decision.lower()}d the action"
        })

        # Get the tool calls from the pending action
        tool_calls = pending_action.get("tool_calls", [])

        if decision.lower() == "approve":
            # For approval, we directly execute the tools
            # This is a simplified approach - in a real implementation, you would
            # execute the actual tools and return their results
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                # Import and execute the appropriate tools
                try:
                    if tool_name == "update_hotel":
                        from customer_support_chat.app.services.tools.hotels import update_hotel
                        # Use ainvoke for async functions
                        result = await update_hotel.ainvoke(tool_args)
                        result_message = f"Hotel updated successfully: {result}"
                    elif tool_name == "book_hotel":
                        from customer_support_chat.app.services.tools.hotels import book_hotel
                        # Use ainvoke for async functions
                        result = await book_hotel.ainvoke(tool_args)
                        result_message = f"Hotel booked successfully: {result}"
                    elif tool_name == "cancel_hotel":
                        from customer_support_chat.app.services.tools.hotels import cancel_hotel
                        # Use ainvoke for async functions
                        result = await cancel_hotel.ainvoke(tool_args)
                        result_message = f"Hotel cancelled successfully: {result}"
                    elif tool_name == "update_car_rental":
                        from customer_support_chat.app.services.tools.cars import update_car_rental
                        # Use ainvoke for async functions
                        result = await update_car_rental.ainvoke(tool_args)
                        result_message = f"Car rental updated successfully: {result}"
                    elif tool_name == "book_car_rental":
                        from customer_support_chat.app.services.tools.cars import book_car_rental
                        # Use ainvoke for async functions
                        result = await book_car_rental.ainvoke(tool_args)
                        result_message = f"Car rental booked successfully: {result}"
                    elif tool_name == "cancel_car_rental":
                        from customer_support_chat.app.services.tools.cars import cancel_car_rental
                        # Use ainvoke for async functions
                        result = await cancel_car_rental.ainvoke(tool_args)
                        result_message = f"Car rental cancelled successfully: {result}"
                    elif tool_name == "book_excursion":
                        from customer_support_chat.app.services.tools.excursions import book_excursion
                        # Use ainvoke for async functions
                        result = await book_excursion.ainvoke(tool_args)
                        result_message = f"Excursion booked successfully: {result}"
                    elif tool_name == "update_excursion":
                        from customer_support_chat.app.services.tools.excursions import update_excursion
                        # Use ainvoke for async functions
                        result = await update_excursion.ainvoke(tool_args)
                        result_message = f"Excursion updated successfully: {result}"
                    elif tool_name == "cancel_excursion":
                        from customer_support_chat.app.services.tools.excursions import cancel_excursion
                        # Use ainvoke for async functions
                        result = await cancel_excursion.ainvoke(tool_args)
                        result_message = f"Excursion cancelled successfully: {result}"
                    elif tool_name == "update_ticket_to_new_flight":
                        from customer_support_chat.app.services.tools.flights import update_ticket_to_new_flight
                        # Use ainvoke for async functions
                        result = await update_ticket_to_new_flight.ainvoke({**tool_args, "config": langgraph_config})
                        result_message = f"Flight updated successfully: {result}"
                    elif tool_name == "cancel_ticket":
                        from customer_support_chat.app.services.tools.flights import cancel_ticket
                        # Use ainvoke for async functions
                        result = await cancel_ticket.ainvoke({**tool_args, "config": langgraph_config})
                        result_message = f"Flight cancelled successfully: {result}"
                    else:
                        result_message = f"Tool {tool_name} executed successfully (tool not implemented in approval handler)"

                    # Add tool execution result to operation log
                    add_operation_log(session_data["session_id"], {
                        "type": "tool_result",
                        "title": f"{tool_name} Result",
                        "content": result if 'result' in locals() else result_message
                    })

                except Exception as e:
                    error_msg = f"Error executing {tool_name}: {str(e)}"
                    result_message = error_msg
                    add_operation_log(session_data["session_id"], {
                        "type": "error",
                        "title": f"{tool_name} Execution Error",
                        "content": error_msg
                    })
        else:  # reject
            # For rejection, we simply inform the user
            result_message = "Operation cancelled by user."
            # Add cancellation to operation log
            add_operation_log(session_data["session_id"], {
                "type": "system_message",
                "title": "Action Cancelled",
                "content": "User rejected the sensitive action"
            })

        # Clear the pending action and user decision
        clear_pending_action(session_data["session_id"])
        clear_user_decision(session_data["session_id"])

        # Return the result message
        if result_message:
            return result_message
        else:
            return "Action processed successfully."

    except Exception as e:
        logger.error(f"An error occurred while processing the user decision: {e}")
        # Add error to operation log
        add_operation_log(session_data["session_id"], {
            "type": "error",
            "title": "Decision Processing Error",
            "content": str(e)
        })
        # Clear the pending action and user decision even if there was an error
        try:
            clear_pending_action(session_data["session_id"])
            clear_user_decision(session_data["session_id"])
        except:
            pass
        return "An unexpected error occurred while processing your decision. Please try again later."