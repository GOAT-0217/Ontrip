from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from customer_support_chat.app.services.tools import (
    lookup_policy,
    web_search,
)
from customer_support_chat.app.services.assistants.assistant_base import Assistant, llm
from customer_support_chat.app.core.state import State
from pydantic import BaseModel, Field

# Import new delegation models
from customer_support_chat.app.services.assistants.woocommerce_assistant import ToWooCommerceProducts, ToWooCommerceOrders
from customer_support_chat.app.services.assistants.form_submission_assistant import ToFormSubmission
from customer_support_chat.app.services.assistants.blog_search_assistant import ToBlogSearch

# Define task delegation tools
class ToFlightBookingAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle flight updates and cancellations."""
    request: str = Field(description="Any necessary follow-up questions the update flight assistant should clarify before proceeding.")

class ToBookCarRental(BaseModel):
    """Transfers work to a specialized assistant to handle car rental bookings."""
    location: str = Field(description="The location where the user wants to rent a car.")
    start_date: str = Field(description="The start date of the car rental.")
    end_date: str = Field(description="The end date of the car rental.")
    request: str = Field(description="Any additional information or requests from the user regarding the car rental.")

class ToHotelBookingAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle hotel bookings, modifications, and cancellations."""
    location: str = Field(description="The location where the user wants to book a hotel. Use 'Unknown' if not specified for cancellation requests.", default="Unknown")
    checkin_date: str = Field(description="The check-in date for the hotel. Use 'Unknown' if not specified for cancellation requests.", default="Unknown")
    checkout_date: str = Field(description="The check-out date for the hotel. Use 'Unknown' if not specified for cancellation requests.", default="Unknown")
    request: str = Field(description="Any additional information or requests from the user regarding the hotel operation (booking, cancellation, modification).")

class ToBookExcursion(BaseModel):
    """Transfers work to a specialized assistant to handle trip recommendation and other excursion bookings."""
    location: str = Field(description="The location where the user wants to book a recommended trip.")
    request: str = Field(description="Any additional information or requests from the user regarding the trip recommendation.")

# Primary assistant prompt
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Chinese Airlines (中国航空). "
            "You primarily serve Chinese domestic travelers and provide support for flights within China and international routes from China. "
            "\n\n=== TOOL SELECTION GUIDE (CRITICAL - Choose the CORRECT tool) ==="
            "\n\n1. lookup_policy: ONLY for company policy questions (退改签、行李、退款规定等)"
            "   - Examples: '退改签政策', '行李额度', '退款规则'"
            "   - DO NOT use for: recommendations, food, attractions, general questions"
            
            "\n\n2. web_search: ONLY for recommendations and general travel information (NOT for flight operations)"
            "   - Food/restaurant: '推荐北京小吃', '有什么好吃的', '餐厅推荐'"
            "   - Attractions: '景点推荐', '好玩的地方', '旅游攻略'"
            "   - Hotels: '推荐酒店', '住宿建议' (for recommendations, NOT booking)"
            "   - Weather/traffic/news: '天气', '交通', '新闻'"
            "   - ANY question with 推荐/建议/有什么好/哪里好玩"
            "   - ⚠️ DO NOT use for: real-time flight queries, flight status, flight booking, flight search"
            
            "\n\n3. DELEGATION RULES (delegate to specialized assistants):"
            "- Flight operations (search/book/update/cancel/real-time queries/status) → ToFlightBookingAssistant"
            "  - Examples: '查询航班', '实时航班', '航班状态', '订票', '改签', '退票', '从X到Y的航班'"
            "  - Supports Chinese city names directly: '北京到上海', '广州飞成都', '深圳到杭州'"
            "  - Uses 聚合数据(Juhe) API for domestic flights, AviationStack as fallback"
            "- Hotel booking/modification/cancellation → ToHotelBookingAssistant"  
            "- Car rental/ride-hailing (租车/打车) → ToBookCarRental"
            "- Trip/excursion bookings → ToBookExcursion"
            "- Product searches → ToWooCommerceProducts"
            "- Order searches → ToWooCommerceOrders"
            "- Form submissions → ToFormSubmission"
            "- Blog searches → ToBlogSearch"
            
            "\n\n=== IMPORTANT RULES ==="
            "- For RECOMMENDATIONS (推荐/建议), ALWAYS use web_search FIRST, never lookup_policy"
            "- If user asks about food/restaurants/attractions/hotels recommendations → use web_search"
            "- For REAL-TIME FLIGHT QUERIES (实时航班/查询航班/航班状态) → ALWAYS delegate to ToFlightBookingAssistant, NEVER use web_search"
            "- Only use lookup_policy for official company policies and rules"
            "- Only delegate to ONE assistant at a time"
            "- The user doesn't know about different assistants, don't mention them"
            "- Always respond in Chinese unless the user writes in another language"
            
            "\n\nCurrent user flight information:\n<Flights>\n{user_info}\n</Flights>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

# Primary assistant tools
primary_assistant_tools = [
    lookup_policy,
    web_search,
    ToFlightBookingAssistant,
    ToBookCarRental,
    ToHotelBookingAssistant,
    ToBookExcursion,
    # New tools for delegation
    ToWooCommerceProducts,
    ToWooCommerceOrders,
    ToFormSubmission,
    ToBlogSearch,
]

# Create the primary assistant runnable
primary_assistant_runnable = primary_assistant_prompt | llm.bind_tools(primary_assistant_tools)

# Instantiate the primary assistant
primary_assistant = Assistant(primary_assistant_runnable)