import sys
sys.path.insert(0, '.')

from customer_support_chat.app.graph import multi_agentic_graph
from langchain_core.messages import HumanMessage

config = {'configurable': {'thread_id': 'test_fresh_003'}}

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
    import traceback
    traceback.print_exc()