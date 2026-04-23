from typing import Optional, List
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
from customer_support_chat.app.core.state import State
from customer_support_chat.app.core.logger import logger
from pydantic import BaseModel
from customer_support_chat.app.core.settings import get_settings
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

settings = get_settings()

llm = ChatOpenAI(
    model=settings.OPENAI_MODEL,
    openai_api_key=settings.OPENAI_API_KEY,
    openai_api_base=settings.OPENAI_BASE_URL if settings.OPENAI_BASE_URL else None,
    temperature=1,
    max_tokens=settings.MAX_TOKENS,
)


def _fix_orphaned_tool_calls(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Detect and fix orphaned tool calls in message history.

    When an assistant message with tool_calls is not followed by tool responses,
    we insert empty tool responses to satisfy the API requirement.

    Args:
        messages: List of messages from conversation history

    Returns:
        Fixed list of messages with tool responses added for orphaned tool calls
    """
    if not messages:
        return messages

    tool_calls_seen = {}
    tool_responses_seen = set()
    result_messages = []

    for msg in messages:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_seen[tc['id']] = {
                    'name': tc.get('name', 'Unknown'),
                    'args': tc.get('args', {}),
                    'msg': msg
                }
            result_messages.append(msg)
        elif hasattr(msg, 'tool_call_id') and msg.tool_call_id:
            tool_responses_seen.add(msg.tool_call_id)
            result_messages.append(msg)
        else:
            result_messages.append(msg)

    orphaned = set(tool_calls_seen.keys()) - tool_responses_seen
    if orphaned:
        logger.warning(f"Found {len(orphaned)} orphaned tool calls: {orphaned}")
        for tc_id in orphaned:
            tc_info = tool_calls_seen[tc_id]
            error_msg = (
                f"由于外部API暂时不可用（聚合数据额度已用尽/航空API连接超时），"
                f"无法完成航班查询。请稍后再试，或联系客服获取帮助。"
            )
            result_messages.append(ToolMessage(
                content=error_msg,
                tool_call_id=tc_id,
                name=tc_info['name']
            ))

    return result_messages


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: Optional[RunnableConfig] = None):
        msgs = state.get("messages", [])
        logger.info(f"=== Assistant.__call__: {len(msgs)} messages in state ===")

        has_orphaned = False
        for i, m in enumerate(msgs):
            tc = hasattr(m, 'tool_calls') and m.tool_calls
            tid = hasattr(m, 'tool_call_id')
            content = str(getattr(m, 'content', '') or '')[:50]
            logger.info(f"  [{i}] {type(m).__name__}: {content} | tool_calls={bool(tc)} | tool_call_id={tid}")

            if tc and not tid:
                has_orphaned = True

        if has_orphaned:
            logger.warning("Detected potentially orphaned tool_calls, fixing message sequence")
            msgs = _fix_orphaned_tool_calls(msgs)
            state = {**state, "messages": msgs}

        try:
            while True:
                result = self.runnable.invoke(state, config)

                if not result.tool_calls and (
                    not result.content
                    or isinstance(result.content, list)
                    and not result.content[0].get("text")
                ):
                    messages = state["messages"] + [("user", "Respond with a real output.")]
                    state = {**state, "messages": messages}
                else:
                    break
        except Exception as e:
            error_str = str(e)
            if "tool_calls must be followed by tool messages" in error_str or \
               "insufficient tool messages following tool_calls" in error_str:
                logger.warning(f"Caught tool_calls error: {error_str[:200]}")
                logger.warning("Attempting to fix orphaned tool_calls and retry...")

                msgs = _fix_orphaned_tool_calls(msgs)
                state = {**state, "messages": msgs}

                while True:
                    result = self.runnable.invoke(state, config)
                    if not result.tool_calls and (
                        not result.content
                        or isinstance(result.content, list)
                        and not result.content[0].get("text")
                    ):
                        messages = state["messages"] + [("user", "Respond with a real output.")]
                        state = {**state, "messages": messages}
                    else:
                        break
            else:
                raise

        return {"messages": result}

# Define the CompleteOrEscalate tool
@tool
def CompleteOrEscalate(reason: str) -> str:
    """A tool to mark the current task as completed or to escalate control to the main assistant.
    
    Args:
        reason: Reason for completion or escalation
        
    Returns:
        A message confirming the action
    """
    return f"Task completed/escalated to main assistant. Reason: {reason}"