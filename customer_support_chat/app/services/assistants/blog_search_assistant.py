# customer_support_chat/app/services/assistants/blog_search_assistant.py

from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from customer_support_chat.app.services.tools.blog import search_xiaohongshu
from customer_support_chat.app.services.assistants.assistant_base import Assistant, llm, CompleteOrEscalate
from pydantic import BaseModel, Field

# Define task delegation tool for Xiaohongshu search
class ToXiaohongshuSearch(BaseModel):
    """当用户需要查找旅行攻略、景点推荐、美食攻略、酒店测评等信息时，将任务委托给小红书搜索助手。"""
    keyword: str = Field(description="用于在小红书上搜索旅行攻略的关键词，例如'北京三日游攻略'、'三亚酒店推荐'。")

# Xiaohongshu search assistant prompt
xiaohongshu_search_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是小红书攻略搜索助手。你必须严格按照以下格式回复：\n\n"

            "══════════════════════════════════════\n"
            "📕 真实攻略（来自小红书 Top 3 热门笔记）\n"
            "══════════════════════════════════════\n\n"
            "[这里原样展示 search_xiaohongshu 工具返回的全部内容，一条不改]\n\n"

            "══════════════════════════════════════\n"
            "📝 AI 旅行建议（综合以上攻略整理）\n"
            "══════════════════════════════════════\n\n"
            "[这里写你自己的旅行建议——根据上面 3 条笔记的内容，总结出实用的行程推荐、避坑提示、交通住宿建议等]\n"
            "要求：用 emoji 分段、列点，便于快速阅读\n\n"

            "🔄 如果用户对某个城市/主题有兴趣，可以说：需要我帮你深入搜某个方向吗？例如「洛阳美食」「开封景点」\n\n"

            "───\n"
            "规则：\n"
            "  1. 必须先调用 search_xiaohongshu(keyword)\n"
            "  2. 把工具返回原样放入「真实攻略」区，保留所有链接\n"
            "  3. 再写「AI 旅行建议」区，结合笔记内容给出你的分析与建议\n"
            "  4. 两个区域用分隔线分开，缺一不可\n"
            "  5. 如果工具返回空结果，只写一个区：「💡 AI 旅行建议（小红书暂未搜到相关笔记）」\n\n"

            "如果用户的需求不是搜索小红书攻略，使用 CompleteOrEscalate 交还主助手。\n"
            "当前时间: {time}。",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

# Xiaohongshu search assistant tools
xiaohongshu_search_assistant_tools = [
    search_xiaohongshu,
    CompleteOrEscalate,
]

# Create the xiaohongshu search assistant runnable
xiaohongshu_search_assistant_runnable = xiaohongshu_search_assistant_prompt | llm.bind_tools(xiaohongshu_search_assistant_tools)

# Instantiate the xiaohongshu search assistant
xiaohongshu_search_assistant = Assistant(xiaohongshu_search_assistant_runnable)
