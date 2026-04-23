import time
import requests
from typing import Dict, List, Optional, Any
from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.core.logger import logger

settings = get_settings()

_flight_cache: Dict[str, Any] = {}
_hotel_cache: Dict[str, Any] = {}
_CACHE_TTL = 300


def _get_cache(cache: dict, key: str):
    if key in cache:
        data, ts = cache[key]
        if time.time() - ts < _CACHE_TTL:
            return data
        del cache[key]
    return None


def _set_cache(cache: dict, key: str, data: Any):
    cache[key] = (data, time.time())


class JuheFlightClient:
    def __init__(self):
        self.api_key = settings.JUHE_FLIGHT_KEY
        self.api_url = settings.JUHE_FLIGHT_API_URL

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def search_flights(
        self,
        departure: str,
        arrival: str,
        departure_date: str,
        flight_no: str = "",
        max_segments: str = "1",
    ) -> List[Dict[str, Any]]:
        if not self.is_configured():
            raise ValueError("聚合数据航班API未配置，请设置 JUHE_FLIGHT_KEY")

        cache_key = f"juhe_flight:{departure}:{arrival}:{departure_date}:{flight_no}"
        cached = _get_cache(_flight_cache, cache_key)
        if cached is not None:
            logger.info("📦 聚合数据航班缓存命中")
            return cached

        params = {
            "key": self.api_key,
            "departure": departure,
            "arrival": arrival,
            "departureDate": departure_date,
            "maxSegments": max_segments,
        }
        if flight_no:
            params["flightNo"] = flight_no

        try:
            resp = requests.get(self.api_url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            error_code = data.get("error_code", 0)
            if error_code != 0:
                reason = data.get("reason", "Unknown error")
                raise Exception(f"聚合数据航班API错误: {reason}")

            result = data.get("result", {})
            flight_info = result.get("flightInfo", [])

            if not isinstance(flight_info, list):
                flight_info = [flight_info] if flight_info else []

            _set_cache(_flight_cache, cache_key, flight_info)
            logger.info(f"✅ 聚合数据航班查询成功: {len(flight_info)} 条结果")
            return flight_info

        except requests.exceptions.Timeout:
            raise Exception("聚合数据航班API请求超时")
        except requests.exceptions.ConnectionError:
            raise Exception("聚合数据航班API连接失败")
        except Exception as e:
            logger.error(f"聚合数据航班API调用失败: {e}")
            raise

    def format_flight_result(self, flight: Dict[str, Any]) -> str:
        airline_name = flight.get("airlineName", "")
        flight_no = flight.get("flightNo", "")
        dep_name = flight.get("departureName", "")
        arr_name = flight.get("arrivalName", "")
        dep_date = flight.get("departureDate", "")
        dep_time = flight.get("departureTime", "")
        arr_date = flight.get("arrivalDate", "")
        arr_time = flight.get("arrivalTime", "")
        duration = flight.get("duration", "")
        equipment = flight.get("equipment", "")
        price = flight.get("ticketPrice", "")
        transfer_num = flight.get("transferNum", 1)

        transfer_text = "直飞" if transfer_num == 1 else f"{transfer_num}段(需中转)"

        lines = [
            f"✈️ **{flight_no}** · {airline_name}",
            f"   📍 {dep_name} → {arr_name}",
            f"   🕐 {dep_date} {dep_time} - {arr_date} {arr_time}",
            f"   ⏱️ 飞行时长: {duration} ({transfer_text})",
        ]
        if price:
            lines.append(f"   💰 参考票价: ¥{price}")
        if equipment:
            lines.append(f"   ✈️ 机型: {equipment}")

        return "\n".join(lines)


class CtripHotelClient:
    BASE_PATH = "/xiecheng/item_get_app"
    SEARCH_PATH = "/xiecheng/item_search_hotel"

    def __init__(self):
        self.app_key = settings.CTRIP_APP_KEY
        self.app_secret = settings.CTRIP_APP_SECRET
        self.base_url = settings.CTRIP_API_BASE_URL

    def is_configured(self) -> bool:
        return bool(self.app_key)

    def search_hotels(
        self,
        city: str = "",
        keyword: str = "",
        checkin: str = "",
        checkout: str = "",
        price_min: int = 0,
        price_max: int = 0,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        if not self.is_configured():
            raise ValueError("携程 API 未配置，请设置 CTRIP_APP_KEY")

        cache_key = f"hotel:{city}:{keyword}:{checkin}:{checkout}"
        cached = _get_cache(_hotel_cache, cache_key)
        if cached is not None:
            logger.info("📦 携程酒店缓存命中")
            return cached

        params = {
            "key": self.app_key,
            "cache": "yes",
            "result_type": "json",
            "lang": "cn",
        }
        if self.app_secret:
            params["secret"] = self.app_secret
        if city:
            params["city"] = city
        if keyword:
            params["q"] = keyword
        if checkin:
            params["checkin"] = checkin
        if checkout:
            params["checkout"] = checkout
        if price_min:
            params["price_min"] = price_min
        if price_max:
            params["price_max"] = price_max

        url = f"{self.base_url}{self.SEARCH_PATH}"

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            error_code = data.get("error_code", "")
            if str(error_code) not in ("200", "0000", "0"):
                error_msg = data.get("reason", data.get("error", "Unknown error"))
                raise Exception(f"携程酒店API错误: {error_msg}")

            hotels = data.get("result", data.get("items", []))
            if isinstance(hotels, dict):
                hotels = hotels.get("hotels", hotels.get("data", hotels.get("items", [])))

            if not isinstance(hotels, list):
                hotels = [hotels] if hotels else []

            _set_cache(_hotel_cache, cache_key, hotels[:limit])
            logger.info(f"✅ 携程酒店查询成功: {len(hotels)} 条结果")
            return hotels[:limit]

        except requests.exceptions.Timeout:
            raise Exception("携程酒店API请求超时")
        except requests.exceptions.ConnectionError:
            raise Exception("携程酒店API连接失败")
        except Exception as e:
            logger.error(f"携程酒店API调用失败: {e}")
            raise

    def get_hotel_detail(self, hotel_id: str) -> Optional[Dict[str, Any]]:
        if not self.is_configured():
            return None

        params = {
            "key": self.app_key,
            "num_iid": hotel_id,
            "cache": "yes",
            "result_type": "json",
            "lang": "cn",
        }
        if self.app_secret:
            params["secret"] = self.app_secret

        url = f"{self.base_url}{self.BASE_PATH}"

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            error_code = data.get("error_code", "")
            if str(error_code) not in ("200", "0000", "0"):
                return None

            result = data.get("result", data.get("item", {}))
            if isinstance(result, list) and result:
                return result[0]
            return result if isinstance(result, dict) else None

        except Exception as e:
            logger.error(f"携程酒店详情API调用失败: {e}")
            return None

    def format_hotel_result(self, hotel: Dict[str, Any]) -> str:
        name = hotel.get("HotelName", hotel.get("name", hotel.get("title", "")))
        address = hotel.get("Address", hotel.get("address", hotel.get("location", "")))
        if isinstance(address, dict):
            address = address.get("AddressLine", str(address))
        rating = hotel.get("Rating", hotel.get("rating", hotel.get("score", "")))
        price = hotel.get("Price", hotel.get("price", hotel.get("parPrice", "")))
        rooms = hotel.get("GuestRooms", hotel.get("rooms", []))

        lines = [
            f"🏨 **{name}**",
            f"   📍 {address}",
        ]
        if rating:
            lines.append(f"   ⭐ 评分: {rating}")
        if price:
            lines.append(f"   💰 参考价: ¥{price}")
        if rooms and isinstance(rooms, list):
            for room in rooms[:3]:
                if isinstance(room, dict):
                    room_type = room.get("RoomType", room.get("room_type", ""))
                    room_price = room.get("Price", room.get("price", ""))
                    lines.append(f"   🛏️ {room_type}: ¥{room_price}/晚")

        return "\n".join(lines)


_juhe_flight_client: Optional[JuheFlightClient] = None
_ctrip_hotel_client: Optional[CtripHotelClient] = None


def get_juhe_flight_client() -> JuheFlightClient:
    global _juhe_flight_client
    if _juhe_flight_client is None:
        _juhe_flight_client = JuheFlightClient()
    return _juhe_flight_client


def get_ctrip_hotel_client() -> CtripHotelClient:
    global _ctrip_hotel_client
    if _ctrip_hotel_client is None:
        _ctrip_hotel_client = CtripHotelClient()
    return _ctrip_hotel_client
