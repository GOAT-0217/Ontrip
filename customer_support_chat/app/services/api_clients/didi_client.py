import json
import requests
from typing import Dict, List, Optional, Any
from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.core.logger import logger

settings = get_settings()

_ride_cache: Dict[str, Any] = {}
_CACHE_TTL = 120


class DiDiMCPClient:
    def __init__(self, use_sandbox: bool = True):
        self.mcp_key = settings.DID_MCP_KEY
        self.use_sandbox = use_sandbox
        if use_sandbox:
            self.base_url = settings.DID_MCP_SANDBOX_URL
        else:
            self.base_url = f"{settings.DID_MCP_BASE_URL}/mcp-servers"
        self._request_id = 0

    def is_configured(self) -> bool:
        return bool(self.mcp_key)

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _call(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        if not self.is_configured():
            raise ValueError("滴滴 MCP 未配置，请设置 DID_MCP_KEY")

        url = f"{self.base_url}?key={self.mcp_key}"
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self._next_id(),
            "params": params or {},
        }

        try:
            resp = requests.post(
                url,
                json=payload,
                timeout=15,
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                error = data["error"]
                raise Exception(f"滴滴MCP错误 [{error.get('code')}]: {error.get('message')}")

            return data

        except requests.exceptions.Timeout:
            raise Exception("滴滴MCP请求超时")
        except requests.exceptions.ConnectionError:
            raise Exception("滴滴MCP连接失败")
        except json.JSONDecodeError:
            raise Exception("滴滴MCP响应解析失败")

    def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return self._call("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })

    def search_place(self, keyword: str, city: str = "") -> Dict[str, Any]:
        params = {"keyword": keyword}
        if city:
            params["city"] = city
        return self._call_tool("maps_textsearch", params)

    def estimate_ride(
        self,
        from_lng: str,
        from_lat: str,
        from_name: str,
        to_lng: str,
        to_lat: str,
        to_name: str,
    ) -> Dict[str, Any]:
        return self._call_tool("taxi_estimate", {
            "from_lng": str(from_lng),
            "from_lat": str(from_lat),
            "from_name": from_name,
            "to_lng": str(to_lng),
            "to_lat": str(to_lat),
            "to_name": to_name,
        })

    def create_order(
        self,
        product_category: str,
        estimate_trace_id: str,
        caller_phone: str = "",
    ) -> Dict[str, Any]:
        params = {
            "product_category": product_category,
            "estimate_trace_id": estimate_trace_id,
        }
        if caller_phone:
            params["caller_car_phone"] = caller_phone
        return self._call_tool("taxi_create_order", params)

    def query_order(self, order_id: str) -> Dict[str, Any]:
        return self._call_tool("taxi_query_order", {"order_id": order_id})

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        return self._call_tool("taxi_cancel_order", {"order_id": order_id})

    def get_driver_location(self, order_id: str) -> Dict[str, Any]:
        return self._call_tool("taxi_get_driver_location", {"order_id": order_id})

    def generate_ride_link(
        self,
        from_lng: str,
        from_lat: str,
        from_name: str,
        to_lng: str,
        to_lat: str,
        to_name: str,
        product_categories: List[str] = None,
    ) -> Dict[str, Any]:
        params = {
            "from_lng": str(from_lng),
            "from_lat": str(from_lat),
            "from_name": from_name,
            "to_lng": str(to_lng),
            "to_lat": str(to_lat),
            "to_name": to_name,
            "product_category": product_categories or ["1"],
        }
        return self._call_tool("taxi_new_order", params)

    def driving_direction(
        self,
        from_lng: str,
        from_lat: str,
        to_lng: str,
        to_lat: str,
    ) -> Dict[str, Any]:
        return self._call_tool("maps_direction_driving", {
            "from_lng": str(from_lng),
            "from_lat": str(from_lat),
            "to_lng": str(to_lng),
            "to_lat": str(to_lat),
        })

    def format_estimate_result(self, result: Dict[str, Any]) -> str:
        content_list = result.get("result", {}).get("content", [])
        structured = result.get("result", {}).get("structuredContent", {})

        if content_list:
            text = content_list[0].get("text", "")
            if text:
                return text

        categories = structured.get("categories", [])
        if not categories:
            return "暂无可用车型报价"

        lines = ["🚗 **滴滴出行报价**\n"]
        for cat in categories:
            name = cat.get("name", "")
            price = cat.get("price", "")
            eta = cat.get("eta", "")
            category_id = cat.get("category", "")
            lines.append(f"  {name}: ¥{price} (预计{eta}分钟到达)")
            lines.append(f"    类别ID: {category_id}")

        trace_id = structured.get("traceId", "")
        if trace_id:
            lines.append(f"\n📋 预估ID: {trace_id}")

        return "\n".join(lines)

    def format_place_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        content_list = result.get("result", {}).get("content", [])
        if content_list:
            text = content_list[0].get("text", "")
            if text:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    pass
        return {}


_didi_client: Optional[DiDiMCPClient] = None


def get_didi_client() -> DiDiMCPClient:
    global _didi_client
    if _didi_client is None:
        _didi_client = DiDiMCPClient(use_sandbox=True)
    return _didi_client
