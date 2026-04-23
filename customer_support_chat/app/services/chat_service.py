# customer_support_chat/app/services/chat_service.py
"""
This module provides a service to process user messages using the LangGraph multi-agent system.
It encapsulates the core chat logic from main.py to make it reusable in a web application context.
"""

import asyncio
import sys
import os
import re
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


# ========== 推荐请求关键词检测 ==========
RECOMMENDATION_KEYWORDS = [
    '推荐', '建议', '有什么好', '哪里好玩', '好吃', '特色', 
    '小吃', '美食', '景点', '酒店推荐', '餐厅', '攻略',
    'recommend', 'suggest', 'best', 'top'
]

def _is_recommendation_request(message: str) -> bool:
    """检测是否为推荐/搜索类请求（绕过 LangGraph 工具系统）"""
    message_lower = message.lower()
    for keyword in RECOMMENDATION_KEYWORDS:
        if keyword.lower() in message_lower:
            return True
    return False


async def _handle_recommendation_directly(session_data: Dict[str, Any], user_message: str) -> Union[str, None]:
    """
    直接处理推荐请求：调用搜索 + LLM 生成回复，绕过 LangGraph 工具系统
    返回 None 表示应该走正常的 LangGraph 流程
    """
    if not _is_recommendation_request(user_message):
        return None
    
    logger.info(f"Detected recommendation request, using direct search mode: {user_message[:50]}...")
    
    try:
        from customer_support_chat.app.services.tools.web_search import web_search
        
        if WEB_APP_AVAILABLE:
            add_operation_log(session_data["session_id"], {
                "type": "system_message",
                "title": "Direct Search Mode",
                "content": f"Using direct search for: {user_message}"
            })
        
        # 1. 直接调用搜索工具（不经过 LangGraph）
        search_results = web_search.invoke({"query": user_message, "max_results": 5})
        
        # Remove emoji characters that may cause GBK encoding issues on Windows
        try:
            import re as _re
            search_results = _re.sub(r'[\U00010000-\U0010ffff]', '', search_results)
        except Exception:
            pass
        
        if WEB_APP_AVAILABLE:
            add_operation_log(session_data["session_id"], {
                "type": "tool_call",
                "title": "web_search (direct)",
                "content": f"Query: {user_message}\nResults length: {len(search_results)} chars"
            })
        
        logger.info(f"Direct search completed, results length: {len(search_results)}")
        
        # 2. 使用 LLM 生成回复（简单调用，不使用工具绑定）
        from langchain_openai import ChatOpenAI
        from customer_support_chat.app.core.settings import get_settings
        
        settings = get_settings()
        llm_simple = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_BASE_URL if settings.OPENAI_BASE_URL else None,
            temperature=0.7
        )
        
        prompt = f"""你是一个友好的旅行助手。根据以下搜索结果，为用户提供有用的推荐信息。

用户问题：{user_message}

搜索结果：
{search_results}

请根据搜索结果，用友好、自然的中文回答用户的问题。如果搜索结果中有具体的信息，请引用并整理。如果没有相关结果，请礼貌地告知。回答要简洁实用，不要太长。"""
        
        response = llm_simple.invoke(prompt)
        ai_response = response.content
        
        # Remove emoji from AI response to prevent GBK encoding issues on Windows
        try:
            ai_response = re.sub(r'[\U00010000-\U0010ffff]', '', ai_response)
        except Exception:
            pass
        
        if WEB_APP_AVAILABLE:
            add_operation_log(session_data["session_id"], {
                "type": "ai_response",
                "title": "AI Response (Direct)",
                "content": ai_response
            })
        
        logger.info(f"Direct response generated, length: {len(ai_response)}")
        return ai_response
        
    except Exception as e:
        logger.error(f"Direct recommendation handling failed: {e}")
        # 如果失败，返回 None 让它走正常流程
        return None


async def process_user_message(session_data: Dict[str, Any], user_message: str) -> str:
    """
    Process a user message using the LangGraph multi-agent system.

    Args:
        session_data (Dict[str, Any]): The session data containing the config (thread_id, passenger_id).
        user_message (str): The user's message to process.

    Returns:
        str: The AI's response message.
    """
    
    direct_response = await _handle_recommendation_directly(session_data, user_message)
    if direct_response:
        return direct_response
    
    config = session_data.get("config", {})
    original_config = dict(config)
    langgraph_config = {"configurable": config}

    printed_message_ids = set()
    latest_ai_response = None

    try:
        if WEB_APP_AVAILABLE:
            add_operation_log(session_data["session_id"], {
                "type": "user_input",
                "title": "User Message",
                "content": user_message
            })

        # Pre-stream check: detect orphaned tool calls and reset session if found
        # Do NOT try to fix by appending ToolMessages - that corrupts the message sequence
        try:
            pre_snapshot = multi_agentic_graph.get_state(langgraph_config)
            if pre_snapshot.values and "messages" in pre_snapshot.values:
                messages = pre_snapshot.values["messages"]
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
                    orphaned_names = []
                    for msg in messages:
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tc in msg.tool_calls:
                                if tc['id'] in unresponded:
                                    orphaned_names.append(tc['name'])
                    logger.warning(f"Pre-stream: Found {len(unresponded)} orphaned tool calls: {orphaned_names}. Resetting session.")
                    import uuid
                    new_thread_id = f"reset_{uuid.uuid4().hex[:12]}_{int(__import__('time').time())}"
                    config['thread_id'] = new_thread_id
                    session_data['config'] = config
                    langgraph_config = {"configurable": config}
                    logger.info(f"Session reset to: {new_thread_id}")
        except Exception as pre_check_error:
            logger.debug(f"Pre-stream check skipped: {pre_check_error}")

        # Process the user input through the graph
        events = multi_agentic_graph.stream(
            {"messages": [("user", user_message)]}, langgraph_config, stream_mode="values"
        )

        for event in events:
            messages = event.get("messages", [])
            for message in messages:
                if message.id not in printed_message_ids:
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

        # Continuation: if stream stopped but graph has next steps, continue execution
        try:
            post_snapshot = multi_agentic_graph.get_state(langgraph_config)
            if post_snapshot.next and not any(node in str(post_snapshot.next) for node in ['sensitive', 'interrupt']):
                logger.info(f"Stream stopped but graph has next steps: {post_snapshot.next}. Continuing execution...")
                continuation_events = multi_agentic_graph.stream(None, langgraph_config, stream_mode="values")
                for event in continuation_events:
                    messages = event.get("messages", [])
                    for message in messages:
                        if message.id not in printed_message_ids:
                            if WEB_APP_AVAILABLE:
                                if isinstance(message, AIMessage) and message.content and message.content.strip():
                                    add_operation_log(session_data["session_id"], {
                                        "type": "ai_response",
                                        "title": "AI Response (Continued)",
                                        "content": message.content
                                    })
                            if isinstance(message, AIMessage) and message.content.strip():
                                latest_ai_response = message.content
                            printed_message_ids.add(message.id)
                logger.info("Graph continuation completed")
        except Exception as cont_err:
            logger.warning(f"Graph continuation failed: {cont_err}")

        # Check for interrupts (HITL)
        snapshot = multi_agentic_graph.get_state(langgraph_config)
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

                    if latest_ai_response:
                        latest_ai_response += "\n\n[User approval required for sensitive action. Please approve or reject this action in the web interface.]"
                    else:
                        latest_ai_response = "[User approval required for sensitive action. Please approve or reject this action in the web interface.]"
                else:
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

        if latest_ai_response:
            logger.info(f"Returning AI response to chat window (length: {len(latest_ai_response)})")
            return latest_ai_response
        else:
            logger.warning("No AI response captured, returning default message")
            return "I'm sorry, I didn't understand that. Could you please rephrase?"

    except Exception as e:
        logger.error(f"An error occurred while processing the user message: {e}")

        if "tool_calls must be followed by tool messages" in str(e):
            logger.warning("=== DETECTED tool_calls ERROR - Starting Session Reset Recovery ===")
            
            import uuid
            
            try:
                new_thread_id = f"reset_{uuid.uuid4().hex[:12]}_{int(__import__('time').time())}"
                new_config = dict(original_config)
                new_config['thread_id'] = new_thread_id
                new_langgraph_config = {"configurable": new_config}
                
                logger.info(f"Created new session: {new_thread_id}")
                reset_events = multi_agentic_graph.stream(
                    {"messages": [("user", user_message)]}, 
                    new_langgraph_config, 
                    stream_mode="values"
                )
                
                reset_printed_ids = set()
                for event in reset_events:
                    messages = event.get("messages", [])
                    for message in messages:
                        if message.id not in reset_printed_ids:
                            if isinstance(message, AIMessage) and message.content and message.content.strip():
                                latest_ai_response = message.content
                                logger.info(f"SESSION RESET SUCCESS! AI response length: {len(latest_ai_response)}")
                                if WEB_APP_AVAILABLE:
                                    add_operation_log(session_data["session_id"], {
                                        "type": "ai_response",
                                        "title": "AI Response (Session Reset)",
                                        "content": message.content
                                    })
                            reset_printed_ids.add(message.id)
                
                if latest_ai_response:
                    config['thread_id'] = new_thread_id
                    session_data['config'] = config
                    return latest_ai_response
                    
            except Exception as reset_error:
                logger.error(f"Session reset failed: {reset_error}")
            
            if latest_ai_response:
                return latest_ai_response
            return "I apologize for the technical difficulty. Please try asking again."

        if WEB_APP_AVAILABLE:
            add_operation_log(session_data["session_id"], {
                "type": "error",
                "title": "Processing Error",
                "content": str(e)
            })
        if latest_ai_response:
            return latest_ai_response
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