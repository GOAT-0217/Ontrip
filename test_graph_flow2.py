import sys
sys.path.insert(0, '.')

from customer_support_chat.app.graph import multi_agentic_graph
from langchain_core.messages import HumanMessage

# Patch the Assistant.__call__ to print state
from customer_support_chat.app.services.assistants import assistant_base
original_call = assistant_base.Assistant.__call__

def patched_call(self, state, config=None):
    print(f'=== Assistant.__call__ invoked ===')
    print(f'  State messages: {len(state.get("messages", []))}')
    for i, msg in enumerate(state.get("messages", [])):
        msg_type = type(msg).__name__
        content = str(msg.content)[:60] if hasattr(msg, 'content') and msg.content else '(no content)'
        tc = hasattr(msg, 'tool_calls') and msg.tool_calls
        tid = hasattr(msg, 'tool_call_id')
        print(f'    [{i}] {msg_type}: {content} | has_tool_calls={bool(tc)} | has_tool_call_id={tid}')
    return original_call(self, state, config)

assistant_base.Assistant.__call__ = patched_call

# Also patch to see what the runnable receives
original_invoke = assistant_base.Assistant.runnable.invoke.__get__(lambda self, *args, **kwargs: print(f'  runnable.invoke called') or self.__class__.runnable.invoke(self, *args, **kwargs))

config = {'configurable': {'thread_id': 'test_fresh_004'}}

print('=== Invoking graph with flight query ===')
try:
    result = multi_agentic_graph.invoke(
        {'messages': [('user', '查询明天郑州到长沙的航班')]},
        config
    )
    print('Result messages:')
    for i, msg in enumerate(result.get('messages', [])):
        msg_type = type(msg).__name__
        content = str(msg.content)[:80] if hasattr(msg, 'content') and msg.content else '(no content)'
        tc_info = ''
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            tc_info = ' - tool_calls: ' + str([tc['name'] for tc in msg.tool_calls])
        tid_info = ''
        if hasattr(msg, 'tool_call_id'):
            tid_info = ' - tool_call_id: ' + msg.tool_call_id[:20] + '...'
        print(f'  [{i}] {msg_type}: {content}{tc_info}{tid_info}')
except Exception as e:
    print(f'ERROR: {e}')