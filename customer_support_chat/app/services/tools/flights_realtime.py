import requests
import re
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from functools import lru_cache
from difflib import SequenceMatcher
from langchain_core.tools import tool
from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.core.logger import logger
from customer_support_chat.app.services.api_clients import get_juhe_flight_client

settings = get_settings()

AVIATIONSTACK_BASE_URL = "http://api.aviationstack.com/v1/flights"

JUHE_CITY_CODES = {
    "PEK": "BJS", "PKX": "BJS",
    "PVG": "SHA", "SHA": "SHA",
    "CAN": "CAN", "SZX": "SZX",
    "CTU": "CTU", "TFU": "CTU",
    "KMG": "KMG", "XIY": "SIA",
    "CKG": "CKG", "HGH": "HGH",
    "NKG": "NKG", "WUH": "WUH",
    "CSX": "CSX", "XMN": "XMN",
    "TAO": "TAO", "CGO": "CGO",
    "URC": "URC", "HAK": "HAK",
    "SYX": "SYX", "DLC": "DLC",
    "TSN": "TSN", "HRB": "HRB",
    "SHE": "SHE", "TNA": "TNA",
    "FOC": "FOC", "NNG": "NNG",
    "KWE": "KWE", "LHW": "LHW",
    "HET": "HET", "SJW": "SJW",
    "TYN": "TYN", "CGQ": "CGQ",
    "HFE": "HFE", "KHN": "KHN",
    "WNZ": "WNZ", "NGB": "NGB",
    "ZUH": "ZUH", "KWL": "KWL",
    "LJG": "LJG", "LXA": "LXA",
    "INC": "INC", "XNN": "XNN",
    "NRT": "TYO", "HND": "TYO",
    "KIX": "OSA", "ICN": "SEL",
    "SIN": "SIN", "BKK": "BKK",
    "HKG": "HKG", "MFM": "MFM",
    "TPE": "TPE", "LHR": "LON",
    "CDG": "PAR", "FRA": "FRA",
    "JFK": "NYC", "EWR": "NYC", "LGA": "NYC",
    "LAX": "LAX", "SFO": "SFO",
    "SYD": "SYD", "DXB": "DXB",
}

def _to_juhe_city_code(iata_code: str) -> str:
    if not iata_code:
        return iata_code
    upper = iata_code.upper()
    return JUHE_CITY_CODES.get(upper, upper)

flight_cache = {}
CACHE_TTL = 300
MAX_RETRIES = 3
RETRY_DELAY = 1

_city_resolve_cache = {}
_CITY_RESOLVE_CACHE_TTL = 3600

DOMESTIC_AIRPORTS = {
    "北京": {"iata": ["PEK", "PKX"], "name": "Beijing", "airports": "首都/大兴", "province": "北京"},
    "上海": {"iata": ["PVG", "SHA"], "name": "Shanghai", "airports": "浦东/虹桥", "province": "上海"},
    "广州": {"iata": ["CAN"], "name": "Guangzhou", "airports": "白云", "province": "广东"},
    "深圳": {"iata": ["SZX"], "name": "Shenzhen", "airports": "宝安", "province": "广东"},
    "成都": {"iata": ["CTU", "TFU"], "name": "Chengdu", "airports": "双流/天府", "province": "四川"},
    "昆明": {"iata": ["KMG"], "name": "Kunming", "airports": "长水", "province": "云南"},
    "西安": {"iata": ["XIY"], "name": "Xi'an", "airports": "咸阳", "province": "陕西"},
    "重庆": {"iata": ["CKG"], "name": "Chongqing", "airports": "江北", "province": "重庆"},
    "杭州": {"iata": ["HGH"], "name": "Hangzhou", "airports": "萧山", "province": "浙江"},
    "南京": {"iata": ["NKG"], "name": "Nanjing", "airports": "禄口", "province": "江苏"},
    "武汉": {"iata": ["WUH"], "name": "Wuhan", "airports": "天河", "province": "湖北"},
    "长沙": {"iata": ["CSX"], "name": "Changsha", "airports": "黄花", "province": "湖南"},
    "厦门": {"iata": ["XMN"], "name": "Xiamen", "airports": "高崎", "province": "福建"},
    "青岛": {"iata": ["TAO"], "name": "Qingdao", "airports": "胶东", "province": "山东"},
    "郑州": {"iata": ["CGO"], "name": "Zhengzhou", "airports": "新郑", "province": "河南"},
    "乌鲁木齐": {"iata": ["URC"], "name": "Urumqi", "airports": "地窝堡", "province": "新疆"},
    "海口": {"iata": ["HAK"], "name": "Haikou", "airports": "美兰", "province": "海南"},
    "三亚": {"iata": ["SYX"], "name": "Sanya", "airports": "凤凰", "province": "海南"},
    "大连": {"iata": ["DLC"], "name": "Dalian", "airports": "周水子", "province": "辽宁"},
    "天津": {"iata": ["TSN"], "name": "Tianjin", "airports": "滨海", "province": "天津"},
    "哈尔滨": {"iata": ["HRB"], "name": "Harbin", "airports": "太平", "province": "黑龙江"},
    "沈阳": {"iata": ["SHE"], "name": "Shenyang", "airports": "桃仙", "province": "辽宁"},
    "济南": {"iata": ["TNA"], "name": "Jinan", "airports": "遥墙", "province": "山东"},
    "福州": {"iata": ["FOC"], "name": "Fuzhou", "airports": "长乐", "province": "福建"},
    "南宁": {"iata": ["NNG"], "name": "Nanning", "airports": "吴圩", "province": "广西"},
    "贵阳": {"iata": ["KWE"], "name": "Guiyang", "airports": "龙洞堡", "province": "贵州"},
    "兰州": {"iata": ["LHW"], "name": "Lanzhou", "airports": "中川", "province": "甘肃"},
    "呼和浩特": {"iata": ["HET"], "name": "Hohhot", "airports": "白塔", "province": "内蒙古"},
    "石家庄": {"iata": ["SJW"], "name": "Shijiazhuang", "airports": "正定", "province": "河北"},
    "太原": {"iata": ["TYN"], "name": "Taiyuan", "airports": "武宿", "province": "山西"},
    "长春": {"iata": ["CGQ"], "name": "Changchun", "airports": "龙嘉", "province": "吉林"},
    "合肥": {"iata": ["HFE"], "name": "Hefei", "airports": "新桥", "province": "安徽"},
    "南昌": {"iata": ["KHN"], "name": "Nanchang", "airports": "昌北", "province": "江西"},
    "温州": {"iata": ["WNZ"], "name": "Wenzhou", "airports": "龙湾", "province": "浙江"},
    "宁波": {"iata": ["NGB"], "name": "Ningbo", "airports": "栎社", "province": "浙江"},
    "珠海": {"iata": ["ZUH"], "name": "Zhuhai", "airports": "金湾", "province": "广东"},
    "桂林": {"iata": ["KWL"], "name": "Guilin", "airports": "两江", "province": "广西"},
    "丽江": {"iata": ["LJG"], "name": "Lijiang", "airports": "三义", "province": "云南"},
    "拉萨": {"iata": ["LXA"], "name": "Lhasa", "airports": "贡嘎", "province": "西藏"},
    "银川": {"iata": ["INC"], "name": "Yinchuan", "airports": "河东", "province": "宁夏"},
    "西宁": {"iata": ["XNN"], "name": "Xining", "airports": "曹家堡", "province": "青海"},
    "无锡": {"iata": ["WUX"], "name": "Wuxi", "airports": "硕放", "province": "江苏"},
    "常州": {"iata": ["CZX"], "name": "Changzhou", "airports": "奔牛", "province": "江苏"},
    "烟台": {"iata": ["YNT"], "name": "Yantai", "airports": "蓬莱", "province": "山东"},
    "威海": {"iata": ["WEH"], "name": "Weihai", "airports": "大水泊", "province": "山东"},
    "泉州": {"iata": ["JJN"], "name": "Quanzhou", "airports": "晋江", "province": "福建"},
    "汕头": {"iata": ["SWA"], "name": "Shantou", "airports": "外砂", "province": "广东"},
    "揭阳": {"iata": ["SWA"], "name": "Jieyang", "airports": "潮汕", "province": "广东"},
    "张家界": {"iata": ["DYG"], "name": "Zhangjiajie", "airports": "荷花", "province": "湖南"},
    "黄山": {"iata": ["TXN"], "name": "Huangshan", "airports": "屯溪", "province": "安徽"},
    "九寨沟": {"iata": ["JZH"], "name": "Jiuzhaigou", "airports": "黄龙", "province": "四川"},
    "敦煌": {"iata": ["DNH"], "name": "Dunhuang", "airports": "机场", "province": "甘肃"},
    "喀什": {"iata": ["KHG"], "name": "Kashi", "airports": "国际机场", "province": "新疆"},
    "阿克苏": {"iata": ["AKU"], "name": "Aksu", "airports": "温宿", "province": "新疆"},
    "库尔勒": {"iata": ["KGQ"], "name": "Korla", "airports": "梨城", "province": "新疆"},
    "伊宁": {"iata": ["YIN"], "name": "Yining", "airports": "机场", "province": "新疆"},
    "绵阳": {"iata": ["MIG"], "name": "Mianyang", "airports": "南郊", "province": "四川"},
    "洛阳": {"iata": ["LYA"], "name": "Luoyang", "airports": "北郊", "province": "河南"},
    "宜昌": {"iata": ["YIH"], "name": "Yichang", "airports": "三峡", "province": "湖北"},
    "襄阳": {"iata": ["XFN"], "name": "Xiangyang", "airports": "刘集", "province": "湖北"},
    "赣州": {"iata": ["KOW"], "name": "Ganzhou", "airports": "黄金", "province": "江西"},
    "湛江": {"iata": ["ZHA"], "name": "Zhanjiang", "airports": "机场", "province": "广东"},
    "北海": {"iata": ["BHY"], "name": "Beihai", "airports": "福成", "province": "广西"},
    "三亚": {"iata": ["SYX"], "name": "Sanya", "airports": "凤凰", "province": "海南"},
    "博鳌": {"iata": ["BAR"], "name": "Boao", "airports": "机场", "province": "海南"},
    "遵义": {"iata": ["ZYI"], "name": "Zunyi", "airports": "新舟", "province": "贵州"},
    "铜仁": {"iata": ["TEN"], "name": "Tongren", "airports": "凤凰", "province": "贵州"},
    "恩施": {"iata": "ENH", "name": "Enshi", "airports": "许家坪", "province": "湖北"},
    "达州": {"iata": ["DAX"], "name": "Dazhou", "airports": "河市", "province": "四川"},
    "南充": {"iata": ["NAO"], "name": "Nanchong", "airports": "高坪", "province": "四川"},
    "泸州": {"iata": ["LZO"], "name": "Luzhou", "airports": "云龙", "province": "四川"},
    "宜宾": {"iata": ["YBP"], "name": "Yibin", "airports": "五粮液", "province": "四川"},
    "万州": {"iata": ["WXN"], "name": "Wanzhou", "airports": "五桥", "province": "重庆"},
    "黔江": {"iata": ["JIQ"], "name": "Qianjiang", "airports": "武陵山", "province": "重庆"},
    "大理": {"iata": ["DLU"], "name": "Dali", "airports": "凤仪", "province": "云南"},
    "西双版纳": {"iata": ["JHG"], "name": "Xishuangbanna", "airports": "嘎洒", "province": "云南"},
    "德宏": {"iata": ["LUM"], "name": "Dehong", "airports": "芒市", "province": "云南"},
    "保山": {"iata": ["BSD"], "name": "Baoshan", "airports": "云瑞", "province": "云南"},
    "昭通": {"iata": ["ZAT"], "name": "Zhaotong", "airports": "机场", "province": "云南"},
    "普洱": {"iata": ["SYM"], "name": "Pu'er", "airports": "思茅", "province": "云南"},
    "日喀则": {"iata": ["RKZ"], "name": "Shigatse", "airports": "和平", "province": "西藏"},
    "林芝": {"iata": ["LZY"], "name": "Nyingchi", "airports": "米林", "province": "西藏"},
    "昌都": {"iata": ["BPX"], "name": "Qamdo", "airports": "邦达", "province": "西藏"},
    "阿里": {"iata": ["NGQ"], "name": "Ngari", "airports": "昆莎", "province": "西藏"},
    "格尔木": {"iata": ["GOQ"], "name": "Golmud", "airports": "机场", "province": "青海"},
    "玉树": {"iata": ["YUS"], "name": "Yushu", "airports": "巴塘", "province": "青海"},
    "海西": {"iata": ["HXD"], "name": "Haixi", "airports": "花土沟", "province": "青海"},
    "固原": {"iata": ["GYU"], "name": "Guyuan", "airports": "六盘山", "province": "宁夏"},
    "中卫": {"iata": ["ZHY"], "name": "Zhongwei", "airports": "沙坡头", "province": "宁夏"},
    "嘉峪关": {"iata": ["JGN"], "name": "Jiayuguan", "airports": "机场", "province": "甘肃"},
    "金昌": {"iata": ["JIC"], "name": "Jinchang", "airports": "金昌", "province": "甘肃"},
    "张掖": {"iata": ["YZY"], "name": "Zhangye", "airports": "甘州", "province": "甘肃"},
    "陇南": {"iata": ["LNL"], "name": "Longnan", "airports": "成县", "province": "甘肃"},
    "庆阳": {"iata": ["IQN"], "name": "Qingyang", "airports": "西峰", "province": "甘肃"},
    "敦煌": {"iata": ["DNH"], "name": "Dunhuang", "airports": "机场", "province": "甘肃"},
    "吐鲁番": {"iata": ["TLQ"], "name": "Turpan", "airports": "交河", "province": "新疆"},
    "克拉玛依": {"iata": ["KRY"], "name": "Karamay", "airports": "机场", "province": "新疆"},
    "塔城": {"iata": ["TCG"], "name": "Tacheng", "airports": "机场", "province": "新疆"},
    "阿勒泰": {"iata": ["AAT"], "name": "Altay", "airports": "雪都", "province": "新疆"},
    "和田": {"iata": ["HTN"], "name": "Hotan", "airports": "机场", "province": "新疆"},
    "博乐": {"iata": ["BFL"], "name": "Bole", "airports": "阿拉山口", "province": "新疆"},
    "库车": {"iata": ["KCA"], "name": "Kuqa", "airports": "龟兹", "province": "新疆"},
    "且末": {"iata": ["IQM"], "name": "Qiemo", "airports": "玉都", "province": "新疆"},
    "富蕴": {"iata": ["FYN"], "name": "Fuyun", "airports": "可可托海", "province": "新疆"},
    "那拉提": {"iata": ["NLT"], "name": "Nalati", "airports": "机场", "province": "新疆"},
    "昭苏": {"iata": ["ZFL"], "name": "Zhaosu", "airports": "天马", "province": "新疆"},
    "塔什库尔干": {"iata": ["HQL"], "name": "Taxkorgan", "airports": "帕米尔", "province": "新疆"},
    "呼伦贝尔": {"iata": ["HLD"], "name": "Hulunbuir", "airports": "海拉尔", "province": "内蒙古"},
    "包头": {"iata": ["BAV"], "name": "Baotou", "airports": "二里半", "province": "内蒙古"},
    "赤峰": {"iata": ["CIF"], "name": "Chifeng", "airports": "玉龙", "province": "内蒙古"},
    "通辽": {"iata": ["TGO"], "name": "Tongliao", "airports": "机场", "province": "内蒙古"},
    "鄂尔多斯": {"iata": ["DSN"], "name": "Ordos", "airports": "伊金霍洛", "province": "内蒙古"},
    "乌海": {"iata": ["WUA"], "name": "Wuhai", "airports": "机场", "province": "内蒙古"},
    "锡林浩特": {"iata": ["XIL"], "name": "Xilinhot", "airports": "机场", "province": "内蒙古"},
    "二连浩特": {"iata": ["RLK"], "name": "Erenhot", "airports": "赛乌素", "province": "内蒙古"},
    "阿尔山": {"iata": ["YIE"], "name": "Arxan", "airports": "伊尔施", "province": "内蒙古"},
    "满洲里": {"iata": ["NZH"], "name": "Manzhouli", "airports": "西郊", "province": "内蒙古"},
    "乌兰浩特": {"iata": ["UCF"], "name": "Ulanhot", "airports": "义勒力特", "province": "内蒙古"},
    "扎兰屯": {"iata": ["NZL"], "name": "Zhalantun", "airports": "机场", "province": "内蒙古"},
}

CITY_ALIASES = {
    "京": "北京", "北平": "北京", "燕京": "北京",
    "沪": "上海", "申": "上海",
    "穗": "广州", "羊城": "广州", "花城": "广州",
    "深": "深圳", "鹏城": "深圳",
    "蓉": "成都", "蓉城": "成都", "锦城": "成都",
    "昆": "昆明", "春城": "昆明",
    "长安": "西安",
    "渝": "重庆", "山城": "重庆", "渝都": "重庆",
    "杭": "杭州", "临安": "杭州",
    "金陵": "南京",
    "江城": "武汉",
    "星城": "长沙", "潭州": "长沙",
    "鹭岛": "厦门",
    "青岛": "青岛", "岛城": "青岛",
    "商都": "郑州", "中原": "郑州",
    "乌市": "乌鲁木齐",
    "椰城": "海口",
    "鹿城": "三亚",
    "滨城": "大连",
    "津": "天津", "津门": "天津",
    "冰城": "哈尔滨",
    "盛京": "沈阳", "奉天": "沈阳",
    "泉城": "济南", "历城": "济南",
    "榕城": "福州",
    "绿城": "南宁",
    "筑城": "贵阳",
    "金城": "兰州",
    "青城": "呼和浩特",
    "并州": "太原", "晋阳": "太原",
    "春城": "长春", "北国春城": "长春",
    "庐州": "合肥",
    "洪城": "南昌",
    "甬城": "宁波",
    "珠": "珠海", "百岛之市": "珠海",
    "山水": "桂林",
    "古城": "丽江",
    "日光城": "拉萨",
    "凤城": "银川",
    "夏都": "西宁",
}

INTERNATIONAL_AIRPORTS = {
    "东京": {"iata": ["NRT", "HND"], "name": "Tokyo", "country": "Japan"},
    "大阪": {"iata": ["KIX"], "name": "Osaka", "country": "Japan"},
    "名古屋": {"iata": ["NGO"], "name": "Nagoya", "country": "Japan"},
    "福冈": {"iata": ["FUK"], "name": "Fukuoka", "country": "Japan"},
    "札幌": {"iata": ["CTS"], "name": "Sapporo", "country": "Japan"},
    "冲绳": {"iata": ["OKA"], "name": "Okinawa", "country": "Japan"},
    "首尔": {"iata": ["ICN", "GMP"], "name": "Seoul", "country": "South Korea"},
    "釜山": {"iata": ["PUS"], "name": "Busan", "country": "South Korea"},
    "新加坡": {"iata": ["SIN"], "name": "Singapore", "country": "Singapore"},
    "曼谷": {"iata": ["BKK", "DMK"], "name": "Bangkok", "country": "Thailand"},
    "清迈": {"iata": ["CNX"], "name": "Chiang Mai", "country": "Thailand"},
    "普吉": {"iata": ["HKT"], "name": "Phuket", "country": "Thailand"},
    "香港": {"iata": ["HKG"], "name": "Hong Kong", "country": "China"},
    "澳门": {"iata": ["MFM"], "name": "Macau", "country": "China"},
    "台北": {"iata": ["TPE", "TSA"], "name": "Taipei", "country": "Taiwan"},
    "高雄": {"iata": ["KHH"], "name": "Kaohsiung", "country": "Taiwan"},
    "台中": {"iata": ["RMQ"], "name": "Taichung", "country": "Taiwan"},
    "伦敦": {"iata": ["LHR", "LGW", "STN", "LTN"], "name": "London", "country": "UK"},
    "巴黎": {"iata": ["CDG", "ORY"], "name": "Paris", "country": "France"},
    "法兰克福": {"iata": ["FRA"], "name": "Frankfurt", "country": "Germany"},
    "慕尼黑": {"iata": ["MUC"], "name": "Munich", "country": "Germany"},
    "罗马": {"iata": ["FCO"], "name": "Rome", "country": "Italy"},
    "阿姆斯特丹": {"iata": ["AMS"], "name": "Amsterdam", "country": "Netherlands"},
    "马德里": {"iata": ["MAD"], "name": "Madrid", "country": "Spain"},
    "纽约": {"iata": ["JFK", "EWR", "LGA"], "name": "New York", "country": "USA"},
    "洛杉矶": {"iata": ["LAX"], "name": "Los Angeles", "country": "USA"},
    "旧金山": {"iata": ["SFO"], "name": "San Francisco", "country": "USA"},
    "芝加哥": {"iata": ["ORD", "MDW"], "name": "Chicago", "country": "USA"},
    "西雅图": {"iata": ["SEA"], "name": "Seattle", "country": "USA"},
    "拉斯维加斯": {"iata": ["LAS"], "name": "Las Vegas", "country": "USA"},
    "波士顿": {"iata": ["BOS"], "name": "Boston", "country": "USA"},
    "迈阿密": {"iata": ["MIA"], "name": "Miami", "country": "USA"},
    "多伦多": {"iata": ["YYZ"], "name": "Toronto", "country": "Canada"},
    "温哥华": {"iata": ["YVR"], "name": "Vancouver", "country": "Canada"},
    "悉尼": {"iata": ["SYD"], "name": "Sydney", "country": "Australia"},
    "墨尔本": {"iata": ["MEL"], "name": "Melbourne", "country": "Australia"},
    "奥克兰": {"iata": ["AKL"], "name": "Auckland", "country": "New Zealand"},
    "迪拜": {"iata": ["DXB"], "name": "Dubai", "country": "UAE"},
    "莫斯科": {"iata": ["SVO", "DME"], "name": "Moscow", "country": "Russia"},
    "伊斯坦布尔": {"iata": ["IST", "SAW"], "name": "Istanbul", "country": "Turkey"},
    "开罗": {"iata": ["CAI"], "name": "Cairo", "country": "Egypt"},
    "约翰内斯堡": {"iata": ["JNB"], "name": "Johannesburg", "country": "South Africa"},
    "圣保罗": {"iata": ["GRU"], "name": "São Paulo", "country": "Brazil"},
    "墨西哥城": {"iata": ["MEX"], "name": "Mexico City", "country": "Mexico"},
    "雅加达": {"iata": ["CGK"], "name": "Jakarta", "country": "Indonesia"},
    "马尼拉": {"iata": ["MNL"], "name": "Manila", "country": "Philippines"},
    "河内": {"iata": ["HAN"], "name": "Hanoi", "country": "Vietnam"},
    "胡志明市": {"iata": ["SGN"], "name": "Ho Chi Minh City", "country": "Vietnam"},
    "吉隆坡": {"iata": ["KUL"], "name": "Kuala Lumpur", "country": "Malaysia"},
    "新加坡": {"iata": ["SIN"], "name": "Singapore", "country": "Singapore"},
    "新德里": {"iata": ["DEL"], "name": "Delhi", "country": "India"},
    "孟买": {"iata": ["BOM"], "name": "Mumbai", "country": "India"},
    "班加罗尔": {"iata": ["BLR"], "name": "Bangalore", "country": "India"},
}

DATE_KEYWORDS = {
    "今天": 0,
    "明天": 1,
    "后天": 2,
    "大后天": 3,
    "昨天": -1,
    "前天": -2,
}

_IATA_TO_CITY_INDEX: Dict[str, dict] = {}
_CITY_PREFIX_INDEX: Dict[str, List[str]] = {}
_ALL_CITIES_SET: Set[str] = set()
_ALL_ALIASES_SET: Set[str] = set()

def _build_city_indexes():
    global _IATA_TO_CITY_INDEX, _CITY_PREFIX_INDEX, _ALL_CITIES_SET, _ALL_ALIASES_SET
    
    _IATA_TO_CITY_INDEX = {}
    _ALL_CITIES_SET = set(DOMESTIC_AIRPORTS.keys()) | set(INTERNATIONAL_AIRPORTS.keys())
    _ALL_ALIASES_SET = set(CITY_ALIASES.keys())
    
    all_airports = {**DOMESTIC_AIRPORTS, **INTERNATIONAL_AIRPORTS}
    
    for city_name, city_info in all_airports.items():
        for iata_code in city_info.get("iata", []):
            _IATA_TO_CITY_INDEX[iata_code.upper()] = {
                "chinese_name": city_name,
                **city_info
            }
        
        for prefix_len in range(1, min(len(city_name) + 1, 4)):
            prefix = city_name[:prefix_len]
            if prefix not in _CITY_PREFIX_INDEX:
                _CITY_PREFIX_INDEX[prefix] = []
            _CITY_PREFIX_INDEX[prefix].append(city_name)
        
        for alias, canonical in CITY_ALIASES.items():
            if canonical == city_name:
                for alias_prefix_len in range(1, min(len(alias) + 1, 3)):
                    alias_prefix = alias[:alias_prefix_len]
                    if alias_prefix not in _CITY_PREFIX_INDEX:
                        _CITY_PREFIX_INDEX[alias_prefix] = []
                    if city_name not in _CITY_PREFIX_INDEX[alias_prefix]:
                        _CITY_PREFIX_INDEX[alias_prefix].append(city_name)

_build_city_indexes()

_COMPILED_DATE_PATTERNS = [
    re.compile(r'(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})'),
    re.compile(r'(\d{1,2})[-/月](\d{1,2})日?'),
    re.compile(r'(\d{4})(\d{2})(\d{2})'),
]

_COMPILED_FLIGHT_PATTERNS = [
    re.compile(r'从\s*(.+?)\s*到\s*(.+?)\s*(?:的|飞|航班|机票|飞机)'),
    re.compile(r'(.+?)\s*飞(?:往|向|至|去)\s*(.+?)(?:\s|的|$)'),
    re.compile(r'(.+?)\s*[-–—到至]\s*(.+?)\s*(?:航班|机票|飞机)?'),
    re.compile(r'(?:出发|起飞|离开|从)\s*[:：]?\s*(.+?)(?:\s+(?:到达|抵达|前往|到)\s*[:：]?\s*(.+?))?'),
    re.compile(r'查询\s*(.+?)\s*(?:到|–|-)\s*(.+?)\s*(?:航班|实时)?'),
]

_COMPILED_IATA_PATTERN = re.compile(r'\b([A-Z]{3})\b')

_STOP_WORDS = {'的', '航班', '机票', '飞机', '实时', '查询', '搜索'}


def _generate_cache_key(params: dict) -> str:
    param_str = str(sorted(params.items()))
    return hashlib.md5(param_str.encode()).hexdigest()


def _get_from_cache(cache_key: str) -> Optional[list]:
    if cache_key in flight_cache:
        cached_data, timestamp = flight_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            logger.debug(f"Cache hit for key: {cache_key[:8]}...")
            return cached_data
        else:
            del flight_cache[cache_key]
            logger.debug(f"Cache expired for key: {cache_key[:8]}...")
    return None


def _set_to_cache(cache_key: str, data: list):
    flight_cache[cache_key] = (data, time.time())
    logger.debug(f"Cached result for key: {cache_key[:8]}...")


def resolve_city_to_iata(city_input: str) -> Tuple[List[str], str, bool]:
    """
    Enhanced city name resolution with fuzzy matching support.
    
    Supports:
    - Chinese: 郑州, 北京, 上海
    - Aliases: 京, 沪, 穗, 商都
    - English: Beijing, Shanghai, Zhengzhou
    - IATA: CGO, PEK, PVG (direct return)
    - Fuzzy matching: 郑洲 (typo), beijing (case insensitive)
    
    Features:
    - O(1) IATA code lookup using reverse index
    - Prefix-based fast search for Chinese cities
    - Intelligent caching for repeated queries
    - Advanced fuzzy matching with SequenceMatcher
    
    Returns:
        tuple: (iata_code_list, city_name_en, is_exact_match)
    """
    if not city_input:
        return ([], "", False)
    
    city_input = city_input.strip()
    
    cache_key = city_input.lower()
    if cache_key in _city_resolve_cache:
        cached_data, timestamp = _city_resolve_cache[cache_key]
        if time.time() - timestamp < _CITY_RESOLVE_CACHE_TTL:
            return cached_data
    
    result = _resolve_city_internal(city_input)
    
    _city_resolve_cache[cache_key] = (result, time.time())
    
    if len(_city_resolve_cache) > 1000:
        current_time = time.time()
        _city_resolve_cache.copy()
        _city_resolve_cache.clear()
        _city_resolve_cache[cache_key] = (result, current_time)
    
    return result


def _resolve_city_internal(city_input: str) -> Tuple[List[str], str, bool]:
    """Internal city resolution without caching."""
    
    if len(city_input) == 3 and city_input.isalpha() and city_input.isupper():
        if city_input in _IATA_TO_CITY_INDEX:
            info = _IATA_TO_CITY_INDEX[city_input]
            return (info["iata"], info.get("name", city_input), True)
        return ([city_input], city_input.upper(), True)
    
    if len(city_input) >= 2 and city_input.isalpha():
        upper_city = city_input.upper()
        if upper_city in _IATA_TO_CITY_INDEX:
            info = _IATA_TO_CITY_INDEX[upper_city]
            return (info["iata"], info.get("name", city_input), True)
        
        for iata_code, info in _IATA_TO_CITY_INDEX.items():
            if upper_city == iata_code:
                return (info["iata"], info.get("name", city_input), True)
    
    if city_input in DOMESTIC_AIRPORTS:
        info = DOMESTIC_AIRPORTS[city_input]
        return (info["iata"], info["name"], True)
    
    if city_input in INTERNATIONAL_AIRPORTS:
        info = INTERNATIONAL_AIRPORTS[city_input]
        return (info["iata"], info["name"], True)
    
    if city_input in CITY_ALIASES:
        actual_city = CITY_ALIASES[city_input]
        if actual_city in DOMESTIC_AIRPORTS:
            info = DOMESTIC_AIRPORTS[actual_city]
            return (info["iata"], info["name"], True)
        elif actual_city in INTERNATIONAL_AIRPORTS:
            info = INTERNATIONAL_AIRPORTS[actual_city]
            return (info["iata"], info["name"], True)
    
    for alias, city in CITY_ALIASES.items():
        if alias == city_input or city_input in alias or alias in city_input:
            if city in DOMESTIC_AIRPORTS:
                info = DOMESTIC_AIRPORTS[city]
                return (info["iata"], info["name"], False)
            elif city in INTERNATIONAL_AIRPORTS:
                info = INTERNATIONAL_AIRPORTS[city]
                return (info["iata"], info["name"], False)
    
    candidate_cities = []
    
    if len(city_input) >= 1:
        prefix = city_input[:min(len(city_input), 3)]
        if prefix in _CITY_PREFIX_INDEX:
            candidate_cities.extend(_CITY_PREFIX_INDEX[prefix])
        
        for prefix_len in range(1, min(len(city_input), 2)):
            shorter_prefix = city_input[:prefix_len]
            if shorter_prefix in _CITY_PREFIX_INDEX:
                for city in _CITY_PREFIX_INDEX[shorter_prefix]:
                    if city not in candidate_cities:
                        candidate_cities.append(city)
    
    best_match = None
    best_score = 0.0
    
    all_airports = {**DOMESTIC_AIRPORTS, **INTERNATIONAL_AIRPORTS}
    
    cities_to_check = set(candidate_cities) if candidate_cities else set(all_airports.keys())
    
    if len(cities_to_check) > 50:
        cities_to_check = set(candidate_cities[:50]) if candidate_cities else set(list(all_airports.keys())[:50])
    
    for chinese_name in cities_to_check:
        if chinese_name not in all_airports:
            continue
            
        info = all_airports[chinese_name]
        score = _calculate_match_score(city_input, chinese_name, info)
        
        if score > best_score and score >= 0.6:
            best_score = score
            best_match = (info["iata"], info.get("name", chinese_name))
    
    if best_match:
        is_exact = best_score >= 0.9
        if not is_exact:
            logger.info(f"Fuzzy matched '{city_input}' to {best_match[1]} (score: {best_score:.2%})")
        return (*best_match, is_exact)
    
    return ([], city_input, False)


def _calculate_match_score(user_input: str, city_name: str, city_info: dict) -> float:
    """Calculate match score between user input and city."""
    score = 0.0
    
    if user_input == city_name:
        return 1.0
    
    if city_name in user_input or user_input in city_name:
        score = max(score, 0.85)
    
    similarity = SequenceMatcher(None, user_input.lower(), city_name.lower()).ratio()
    if similarity > 0.7:
        score = max(score, similarity)
    
    eng_name = city_info.get("name", "").lower()
    user_lower = user_input.lower()
    
    if eng_name and eng_name == user_lower:
        score = max(score, 0.95)
    elif eng_name and (eng_name in user_lower or user_lower in eng_name):
        score = max(score, 0.8)
    
    if eng_name:
        eng_similarity = SequenceMatcher(None, user_lower, eng_name).ratio()
        if eng_similarity > 0.65:
            score = max(score, eng_similarity * 0.9)
    
    province = city_info.get("province", "")
    if province and (province in user_input or user_input in province):
        score = max(score, 0.65)
    
    country = city_info.get("country", "")
    if country and country.lower() in user_input.lower():
        score = max(score, 0.75)
    
    airports = city_info.get("airports", "")
    if airports and airports in user_input:
        score = max(score, 0.7)
    
    iata_codes = city_info.get("iata", [])
    for iata in iata_codes:
        if iata.lower() in user_input.lower():
            score = max(score, 0.9)
            break
    
    return score
    
    len_diff = abs(len(s1) - len(s2)) / max(len(s1), len(s2))
    length_penalty = 1 - (len_diff * 0.3)
    
    return min(1.0, jaccard * length_penalty)


def parse_flight_query(query: str) -> dict:
    """
    Enhanced natural language flight query parser with date recognition.
    
    Supported formats:
    - '从郑州到长沙的航班'
    - '明天从成都去杭州'
    - '查询 CGO 到 CSX 的实时航班'
    - '北京到上海的飞机'
    - '后天郑州飞长沙'
    
    Features:
    - Uses pre-compiled regex patterns for better performance
    - Supports filter keywords (e.g., "只看南航", "不要延误的")
    - Extracts sort preferences (e.g., "按时间排序", "最早的")
    
    Returns:
        dict with dep_iata, arr_iata, cities, confidence, date_offset, filters
    """
    result = {
        'dep_iata': None,
        'arr_iata': None,
        'departure_city': '',
        'arrival_city': '',
        'confidence': 0.0,
        'raw_query': query,
        'date_offset': None,
        'flight_date': None,
        'filters': {
            'airline': None,
            'status': None,
            'time_range': None,
            'exclude_delayed': False,
            'prefer_early': False,
            'prefer_late': False,
        },
        'sort_by': 'default',
    }
    
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['最早', '早班', '第一班', '首发']):
        result['filters']['prefer_early'] = True
        result['sort_by'] = 'departure_time'
    
    if any(word in query_lower for word in ['最晚', '晚班', '末班', '最后一班']):
        result['filters']['prefer_late'] = True
        result['sort_by'] = 'departure_time_desc'
    
    if any(word in query_lower for word in ['按时间', '时间排序', '时间顺序']):
        result['sort_by'] = 'departure_time'
    
    airline_keywords = {
        '国航': 'Air China', '中国国航': 'Air China', 'CA': 'Air China',
        '东航': 'China Eastern', '中国东航': 'China Eastern', 'MU': 'China Eastern',
        '南航': 'China Southern', '中国南航': 'China Southern', 'CZ': 'China Southern',
        '海航': 'Hainan Airlines', '海南航空': 'Hainan Airlines', 'HU': 'Hainan Airlines',
        '川航': 'Sichuan Airlines', '四川航空': 'Sichuan Airlines', '3U': 'Sichuan Airlines',
        '厦航': 'XiamenAir', '厦门航空': 'XiamenAir', 'MF': 'XiamenAir',
        '深航': 'Shenzhen Airlines', '深圳航空': 'Shenzhen Airlines', 'ZH': 'Shenzhen Airlines',
        '春秋': 'Spring Airlines', '春秋航空': 'Spring Airlines', '9C': 'Spring Airlines',
        '吉祥': 'Juneyao Airlines', '吉祥航空': 'Juneyao Airlines', 'HO': 'Juneyao Airlines',
    }
    
    for keyword, airline_name in airline_keywords.items():
        if keyword in query:
            result['filters']['airline'] = airline_name
            query = query.replace(keyword, '').strip()
            break
    
    if any(word in query_lower for word in ['正在飞', '飞行中', '空中', 'active']):
        result['filters']['status'] = 'active'
    
    if any(word in query_lower for word in ['已降落', '到达', '落地', 'landed']):
        result['filters']['status'] = 'landed'
    
    if any(word in query_lower for word in ['不要延误', '准时', '正点', '不延误']):
        result['filters']['exclude_delayed'] = True
    
    time_range_patterns = [
        (r'(\d{1,2})点.*?之前', 'before'),
        (r'(\d{1,2})点.*?之后', 'after'),
        (r'(上午|早上|清晨)', 'morning'),
        (r'(中午|下午)', 'afternoon'),
        (r'(晚上|夜间|夜里)', 'evening'),
    ]
    
    for pattern, range_type in time_range_patterns:
        match = re.search(pattern, query)
        if match:
            result['filters']['time_range'] = {
                'type': range_type,
                'value': match.group(1) if match.lastindex and match.group(1) else match.group(0),
            }
            break
    
    date_offset = None
    
    for keyword, offset in DATE_KEYWORDS.items():
        if keyword in query:
            date_offset = offset
            query_cleaned = query.replace(keyword, '').strip()
            break
    
    for pattern in _COMPILED_DATE_PATTERNS:
        match = pattern.search(query)
        if match:
            groups = match.groups()
            try:
                if len(groups) == 3:
                    year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                    if year < 100:
                        year += 2000
                    try:
                        target_date = datetime(year, month, day)
                        result['flight_date'] = target_date.strftime('%Y-%m-%d')
                        query_cleaned = query[:match.start()] + query[match.end():]
                        break
                    except ValueError:
                        continue
            except (ValueError, IndexError):
                continue
    
    if not hasattr(locals(), 'query_cleaned'):
        query_cleaned = query
    
    dep_city = ""
    arr_city = ""
    
    for pattern in _COMPILED_FLIGHT_PATTERNS:
        match = pattern.search(query_cleaned, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) >= 2 and groups[0] and groups[1]:
                dep_city = groups[0].strip().rstrip('的')
                arr_city = groups[1].strip().rstrip('的')
                
                for word in _STOP_WORDS:
                    if arr_city.endswith(word):
                        arr_city = arr_city[:-len(word)].strip()
                break
            elif len(groups) == 1 and groups[0]:
                dep_city = groups[0].strip()
    
    if not dep_city or not arr_city:
        all_cities_found = []
        
        if len(query_cleaned) <= 20:
            for chinese_name in _ALL_CITIES_SET:
                if chinese_name in query_cleaned:
                    all_cities_found.append(chinese_name)
        else:
            prefix = query_cleaned[:2] if len(query_cleaned) >= 2 else query_cleaned[:1]
            if prefix in _CITY_PREFIX_INDEX:
                for city in _CITY_PREFIX_INDEX[prefix]:
                    if city in query_cleaned:
                        all_cities_found.append(city)
            
            for alias in _ALL_ALIASES_SET:
                if alias in query_cleaned:
                    canonical = CITY_ALIASES.get(alias)
                    if canonical and canonical not in all_cities_found:
                        all_cities_found.append(canonical)
        
        iata_matches = _COMPILED_IATA_PATTERN.findall(query_cleaned.upper())
        all_cities_found.extend(iata_matches)
        
        if len(all_cities_found) >= 2:
            dep_city = all_cities_found[0]
            arr_city = all_cities_found[1]
    
    if dep_city and arr_city:
        dep_result = resolve_city_to_iata(dep_city)
        arr_result = resolve_city_to_iata(arr_city)
        
        if dep_result[0]:
            result['dep_iata'] = dep_result[0][0]
            result['departure_city'] = dep_result[1] or dep_city
        
        if arr_result[0]:
            result['arr_iata'] = arr_result[0][0]
            result['arrival_city'] = arr_result[1] or arr_city
        
        if result['dep_iata'] and result['arr_iata']:
            base_confidence = 0.9 if (dep_result[2] and arr_result[2]) else 0.7
            
            if date_offset is not None:
                base_confidence = min(1.0, base_confidence + 0.05)
            
            result['confidence'] = base_confidence
        elif result['dep_iata'] or result['arr_iata']:
            result['confidence'] = 0.5
        
        if date_offset is not None:
            result['date_offset'] = date_offset
            target_date = datetime.now() + timedelta(days=date_offset)
            result['flight_date'] = target_date.strftime('%Y-%m-%d')
    
    logger.info(f"Parsed flight query: dep={result['dep_iata']} ({result['departure_city']}), "
                f"arr={result['arr_iata']} ({result['arrival_city']}), "
                f"confidence={result['confidence']:.2f}, date={result['flight_date']}")
    
    return result


def _call_aviationstack_with_retry(params: dict) -> list:
    """
    Call AviationStack API with caching and retry mechanism.
    
    Features:
    - Memory cache with TTL (5 minutes)
    - Automatic retry on failure (3 attempts)
    - Exponential backoff
    - Rate limit handling
    """
    if not settings.AVIATIONSTACK_API_KEY:
        raise ValueError("AVIATIONSTACK_API_KEY is not configured. Please set it in .env file.")
    
    cache_key = _generate_cache_key(params)
    
    cached_result = _get_from_cache(cache_key)
    if cached_result is not None:
        return cached_result
    
    params["access_key"] = settings.AVIATIONSTACK_API_KEY
    
    last_exception = None
    
    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0:
                wait_time = RETRY_DELAY * (2 ** (attempt - 1))
                logger.info(f"Retry attempt {attempt + 1}/{MAX_RETRIES} after {wait_time}s...")
                time.sleep(wait_time)
            
            response = requests.get(
                AVIATIONSTACK_BASE_URL, 
                params=params, 
                timeout=15,
                headers={
                    'User-Agent': 'TravelAI-Assistant/1.0',
                    'Accept': 'application/json'
                }
            )
            
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue
            
            response.raise_for_status()
            
            data = response.json()
            
            if 'error' in data:
                error_msg = data.get('error', {}).get('message', 'Unknown API error')
                raise Exception(f"AviationStack API error: {error_msg}")
            
            results = data.get("data", [])
            
            pagination = data.get('pagination', {})
            if pagination:
                logger.info(f"API Response: {pagination.get('total', 0)} total results, "
                           f"showing {len(results)}")
            
            _set_to_cache(cache_key, results)
            
            return results
            
        except requests.exceptions.Timeout as e:
            last_exception = e
            logger.warning(f"Timeout on attempt {attempt + 1}: {e}")
            
        except requests.exceptions.ConnectionError as e:
            last_exception = e
            logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
            
        except requests.exceptions.HTTPError as e:
            last_exception = e
            if response.status_code >= 500:
                logger.warning(f"Server error {response.status_code} on attempt {attempt + 1}")
            else:
                raise Exception(f"HTTP Error {response.status_code}: {e}")
                
        except Exception as e:
            last_exception = e
            logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
            if attempt == MAX_RETRIES - 1:
                raise
    
    raise Exception(f"Failed to fetch flight data after {MAX_RETRIES} attempts. Last error: {last_exception}")


def _filter_flights(flights: List[dict], filters: dict) -> List[dict]:
    """
    Apply intelligent filters to flight results.
    
    Supported filters:
    - airline: Filter by airline name
    - status: Filter by flight status (active, landed, etc.)
    - time_range: Filter by departure time range
    - exclude_delayed: Exclude delayed flights
    
    Args:
        flights: List of flight dictionaries
        filters: Dictionary of filter criteria
        
    Returns:
        Filtered list of flights
    """
    if not flights:
        return []
    
    filtered = flights.copy()
    
    if filters.get('airline'):
        airline_lower = filters['airline'].lower()
        filtered = [
            f for f in filtered
            if airline_lower in f.get('airline', {}).get('name', '').lower()
        ]
        logger.info(f"Filtered by airline '{filters['airline']}': {len(filtered)} results")
    
    if filters.get('status'):
        status = filters['status'].lower()
        filtered = [
            f for f in filtered
            if f.get('flight_status', '').lower() == status
        ]
        logger.info(f"Filtered by status '{status}': {len(filtered)} results")
    
    if filters.get('exclude_delayed'):
        before_count = len(filtered)
        filtered = [
            f for f in filtered
            if not (f.get('departure', {}).get('delay', 0) or 0) > 15
               and not (f.get('arrival', {}).get('delay', 0) or 0) > 15
        ]
        logger.info(f"Excluded delayed flights: {before_count} → {len(filtered)}")
    
    if filters.get('time_range'):
        time_range = filters['time_range']
        range_type = time_range.get('type')
        value = time_range.get('value', '')
        
        def get_dep_hour(flight):
            sched = flight.get('departure', {}).get('scheduled', '')
            if sched:
                try:
                    dt = datetime.fromisoformat(sched.replace('Z', '+00:00'))
                    return dt.hour
                except:
                    pass
            return 12
        
        if range_type == 'morning':
            filtered = [f for f in filtered if 5 <= get_dep_hour(f) < 12]
        elif range_type == 'afternoon':
            filtered = [f for f in filtered if 12 <= get_dep_hour(f) < 18]
        elif range_type == 'evening':
            filtered = [f for f in filtered if 18 <= get_dep_hour(f) or get_dep_hour(f) < 2]
        elif range_type == 'before' and value.isdigit():
            hour = int(value)
            filtered = [f for f in filtered if get_dep_hour(f) < hour]
        elif range_type == 'after' and value.isdigit():
            hour = int(value)
            filtered = [f for f in filtered if get_dep_hour(f) >= hour]
        
        logger.info(f"Filtered by time range ({range_type}): {len(filtered)} results")
    
    return filtered


def _sort_flights(flights: List[dict], sort_by: str, prefer_early: bool = False, prefer_late: bool = False) -> List[dict]:
    """
    Sort flights based on specified criteria.
    
    Args:
        flights: List of flight dictionaries
        sort_by: Sort criteria ('default', 'departure_time', 'departure_time_desc')
        prefer_early: If True, prioritize early departures
        prefer_late: If True, prioritize late departures
        
    Returns:
        Sorted list of flights
    """
    if not flights or sort_by == 'default':
        return flights
    
    def get_sort_key(flight):
        sched = flight.get('departure', {}).get('scheduled', '')
        try:
            if sched:
                dt = datetime.fromisoformat(sched.replace('Z', '+00:00'))
                return dt.hour * 60 + dt.minute
        except:
            pass
        return 24 * 60
    
    if sort_by == 'departure_time':
        return sorted(flights, key=get_sort_key)
    elif sort_by == 'departure_time_desc':
        return sorted(flights, key=get_sort_key, reverse=True)
    
    return flights


def format_flight_display(flight_data: dict, include_details: bool = True) -> str:
    """
    Format a single flight record into a beautiful display string.
    
    Features:
    - Emoji icons for status
    - Structured layout
    - Chinese-friendly formatting
    - Highlight important information
    """
    fn = flight_data.get("flight", {}).get("iata", "") or flight_data.get("flight", {}).get("icao", "")
    airline = flight_data.get("airline", {}).get("name", "Unknown Airline")
    dep_iata = flight_data.get("departure", {}).get("iata", "")
    arr_iata = flight_data.get("arrival", {}).get("iata", "")
    
    sched_dep_raw = flight_data.get("departure", {}).get("scheduled", "")
    sched_arr_raw = flight_data.get("arrival", {}).get("scheduled", "")
    actual_dep = flight_data.get("departure", {}).get("actual", "N/A")
    actual_arr = flight_data.get("arrival", {}).get("actual", "N/A")
    
    status = flight_data.get("flight_status", "unknown").lower()
    dep_gate = flight_data.get("departure", {}).get("gate", "N/A")
    arr_gate = flight_data.get("arrival", {}).get("gate", "N/A")
    dep_terminal = flight_data.get("departure", {}).get("terminal", "")
    arr_terminal = flight_data.get("arrival", {}).get("terminal", "")
    dep_delay = flight_data.get("departure", {}).get("delay", 0) or 0
    arr_delay = flight_data.get("arrival", {}).get("delay", 0) or 0
    
    def format_time(time_str):
        if not time_str or time_str == "N/A":
            return "--:--"
        try:
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            return dt.strftime('%H:%M')
        except:
            return time_str[:5] if len(time_str) >= 5 else time_str
    
    sched_dep = format_time(sched_dep_raw)
    sched_arr = format_time(sched_arr_raw)
    actual_dep_fmt = format_time(actual_dep)
    actual_arr_fmt = format_time(actual_arr)
    
    status_icons = {
        "active": ("✈️", "正常飞行", "#2ecc71"),
        "landed": ("✅", "已降落", "#27ae60"),
        "scheduled": ("📋", "计划中", "#3498db"),
        "cancelled": ("❌", "已取消", "#e74c3c"),
        "delayed": ("⏰", "延误中", "#f39c12"),
        "diverted": ("🔄", "备降", "#9b59b6"),
        "unknown": ("❓", "未知", "#95a5a6"),
    }
    
    icon, status_text, color = status_icons.get(status, status_icons["unknown"])
    
    delay_info_parts = []
    if isinstance(dep_delay, (int, float)) and dep_delay > 0:
        delay_info_parts.append(f"出发延误{int(dep_delay)}分钟")
    if isinstance(arr_delay, (int, float)) and arr_delay > 0:
        delay_info_parts.append(f"到达延误{int(arr_delay)}分钟")
    delay_info = f" ⚠️ {' + '.join(delay_info_parts)}" if delay_info_parts else ""
    
    terminal_info = []
    if dep_terminal:
        terminal_info.append(f"T{dep_terminal}")
    if arr_terminal:
        terminal_info.append(f"T{arr_terminal}")
    terminal_str = f" [{','.join(terminal_info)}]" if terminal_info else ""
    
    lines = [
        f"{icon} **{fn}** · {airline}",
        f"   📍 {dep_iata} → {arr_iata}{terminal_str}",
        f"   🕐 计划: {sched_dep} - {sched_arr}",
        f"   ✨ 实际: {actual_dep_fmt} - {actual_arr_fmt}",
        f"   📊 状态: **{status_text}** | 🚪 登机口: {dep_gate}/{arr_gate}{delay_info}",
    ]
    
    if include_details:
        aircraft = flight_data.get("flight", {}).get("codeshared", {})
        if aircraft:
            lines.append(f"   ✈️ 机型: {aircraft}")
    
    return "\n".join(lines)


@tool
def search_realtime_flights(
    dep_iata: str = "",
    arr_iata: str = "",
    flight_date: str = "",
    flight_number: str = "",
    airline_name: str = "",
    limit: int = 10,
    natural_query: str = "",
) -> str:
    """
    🔍 Search real-time flights with SMART CHINESE CITY SUPPORT!
    
    ✨ FEATURES:
    - Auto-convert Chinese city names to IATA codes (e.g., 郑州→CGO)
    - Natural language parsing (e.g., "从郑州到长沙的航班")
    - Date recognition (今天/明天/后天)
    - Intelligent caching for performance
    - Beautiful formatted output with emojis
    
    📝 INPUT EXAMPLES (all work automatically):
    - IATA codes: dep_iata="CGO", arr_iata="CSX"
    - Chinese: dep_iata="郑州", arr_iata="长沙"
    - Aliases: dep_iata="商都", arr_iata="星城"
    - Natural: natural_query="明天从郑州到长沙的航班"
    - English: dep_iata="Zhengzhou", arr_iata="Changsha"
    
    Args:
        dep_iata: Departure city (Chinese name OR IATA code)
        arr_iata: Arrival city (Chinese name OR IATA code)
        flight_date: Date YYYY-MM-DD (optional, defaults to today)
        flight_number: Specific flight number (e.g., "CA1234")
        airline_name: Filter by airline (e.g., "China Southern")
        limit: Max results (default 10, max 100)
        natural_query: Full natural language query (recommended!)
    
    Returns:
        Beautifully formatted real-time flight information
    """
    try:
        resolved_dep = ""
        resolved_arr = ""
        used_date = flight_date
        query_source = ""
        
        if natural_query:
            parsed = parse_flight_query(natural_query)
            
            if parsed['confidence'] > 0.5:
                if parsed['dep_iata'] and not dep_iata:
                    dep_iata = parsed['dep_iata']
                if parsed['arr_iata'] and not arr_iata:
                    arr_iata = parsed['arr_iata']
                
                if parsed['flight_date'] and not flight_date:
                    used_date = parsed['flight_date']
                
                query_source = f"(解析自: \"{natural_query}\")"
        
        if dep_iata:
            dep_result = resolve_city_to_iata(dep_iata)
            if dep_result[0]:
                resolved_dep = dep_result[0][0]
                logger.info(f"✅ 出发地: '{dep_iata}' → {resolved_dep} ({dep_result[1]})")
            else:
                resolved_dep = dep_iata.upper()
                logger.warning(f"⚠️ 无法识别出发地: '{dep_iata}'，尝试直接使用")
        
        if arr_iata:
            arr_result = resolve_city_to_iata(arr_iata)
            if arr_result[0]:
                resolved_arr = arr_result[0][0]
                logger.info(f"✅ 目的地: '{arr_iata}' → {resolved_arr} ({arr_result[1]})")
            else:
                resolved_arr = arr_iata.upper()
                logger.warning(f"⚠️ 无法识别目的地: '{arr_iata}'，尝试直接使用")
        
        if not resolved_dep and not resolved_arr and not flight_number:
            user_query = natural_query or f"{dep_iata}/{arr_iata}"
            
            has_partial_info = bool(dep_iata or arr_iata)
            
            if has_partial_info:
                unresolved_city = dep_iata if dep_iata and not resolved_dep else (arr_iata if arr_iata and not resolved_arr else "")
                if unresolved_city:
                    return generate_error_recovery_message('city_not_found', {
                        'user_input': unresolved_city,
                        'departure_city': resolved_dep or dep_iata,
                        'arrival_city': resolved_arr or arr_iata,
                    })
            
            return generate_error_recovery_message('invalid_query', {
                'user_input': user_query,
            }) + (
                f"\n\n📊 **系统支持**:\n"
                f"  • 🏠 国内城市: **{len(DOMESTIC_AIRPORTS)}** 个\n"
                f"  • 🌍 国际城市: **{len(INTERNATIONAL_AIRPORTS)}** 个\n"
                f"  • 🔤 城市别名: **{len(CITY_ALIASES)}** 个\n\n"
                f"✨ 输入 `帮助` 或 `支持城市` 查看完整列表"
            )
        
        params = {"limit": min(limit, 100)}
        
        if resolved_dep:
            params["dep_iata"] = resolved_dep
        if resolved_arr:
            params["arr_iata"] = resolved_arr
        if used_date:
            params["flight_date"] = used_date
        if flight_number:
            params["flight_number"] = flight_number.upper().replace(" ", "")
        if airline_name:
            params["airline_name"] = airline_name
        
        logger.info(f"🔍 查询航班 {query_source}: {params}")
        
        juhe_client = get_juhe_flight_client()
        results = None
        data_source = "AviationStack"

        if juhe_client.is_configured() and resolved_dep and resolved_arr:
            try:
                dep_code = _to_juhe_city_code(resolved_dep)
                arr_code = _to_juhe_city_code(resolved_arr)

                juhe_results = juhe_client.search_flights(
                    departure=dep_code,
                    arrival=arr_code,
                    departure_date=used_date or datetime.now().strftime('%Y-%m-%d'),
                    flight_no=flight_number.upper().replace(" ", "") if flight_number else "",
                    max_segments="1",
                )
                if juhe_results:
                    results = []
                    for f in juhe_results:
                        mapped = {
                            "flight_date": f.get("departureDate", ""),
                            "flight_status": "scheduled",
                            "departure": {
                                "airport": f.get("departureName", ""),
                                "iata": f.get("departure", ""),
                                "scheduled": f"{f.get('departureDate', '')} {f.get('departureTime', '')}",
                            },
                            "arrival": {
                                "airport": f.get("arrivalName", ""),
                                "iata": f.get("arrival", ""),
                                "scheduled": f"{f.get('arrivalDate', '')} {f.get('arrivalTime', '')}",
                            },
                            "airline": {
                                "name": f.get("airlineName", ""),
                                "iata": f.get("airline", ""),
                            },
                            "flight": {
                                "iata": f.get("flightNo", ""),
                                "number": f.get("flightNo", ""),
                            },
                            "flight_type": "国内",
                            "duration": f.get("duration", ""),
                            "ticket_price": f.get("ticketPrice", ""),
                            "equipment": f.get("equipment", ""),
                            "transfer_num": f.get("transferNum", 1),
                        }
                        results.append(mapped)
                    data_source = "聚合数据"
                    logger.info(f"✅ 聚合数据API返回 {len(results)} 条航班")
            except Exception as e:
                logger.warning(f"聚合数据API查询失败，回退到AviationStack: {e}")

        if results is None:
            try:
                results = _call_aviationstack_with_retry(params)
                data_source = "AviationStack"
            except ValueError as e:
                return f"⚙️ **配置错误**: {str(e)}"
            except Exception as api_error:
                return (
                    f"❌ **查询失败**\n\n"
                    f"无法连接到航班数据服务。\n"
                    f"错误详情: {str(api_error)}\n\n"
                    f"💡 建议:\n"
                    f"• 请检查网络连接\n"
                    f"• 稍后重试\n"
                    f"• 尝试更简单的查询条件"
                )
        
        if not results:
            hint_lines = []
            if resolved_dep or resolved_arr:
                hint_lines.append(f"📍 航线: {resolved_dep or '*'} → {resolved_arr or '*'}")
            if used_date:
                hint_lines.append(f"📅 日期: {used_date}")
            
            hints = "\n".join(hint_lines) if hint_lines else ""
            
            error_context = {
                'departure_city': resolved_dep or dep_iata,
                'arrival_city': resolved_arr or arr_iata,
                'flight_date': used_date,
            }
            
            base_message = generate_error_recovery_message('no_results', error_context)
            
            if hints:
                base_message = base_message.replace(
                    "📋 **可能的原因**:",
                    f"📍 **查询条件**:\n{hints}\n\n📋 **可能的原因**:"
                )
            
            return base_message
        
        applied_filters = {}
        sort_method = 'default'
        
        if natural_query:
            parsed = parse_flight_query(natural_query)
            applied_filters = parsed.get('filters', {})
            sort_method = parsed.get('sort_by', 'default')
        
        if airline_name and not applied_filters.get('airline'):
            applied_filters['airline'] = airline_name
        
        if applied_filters:
            original_count = len(results)
            results = _filter_flights(results, applied_filters)
            
            if results:
                filter_summary = []
                if applied_filters.get('airline'):
                    filter_summary.append(f"航空公司: {applied_filters['airline']}")
                if applied_filters.get('status'):
                    status_map_cn = {
                        'active': '正在飞行',
                        'landed': '已降落',
                        'scheduled': '计划中',
                        'cancelled': '已取消',
                        'delayed': '延误中',
                    }
                    filter_summary.append(f"状态: {status_map_cn.get(applied_filters['status'], applied_filters['status'])}")
                if applied_filters.get('exclude_delayed'):
                    filter_summary.append("排除延误")
                if applied_filters.get('time_range'):
                    range_info = applied_filters['time_range']
                    filter_summary.append(f"时间: {range_info.get('value', range_info.get('type', ''))}")
                
                logger.info(f"🔽 Applied filters: {filter_summary} ({original_count} → {len(results)} results)")
        
        if sort_method != 'default':
            prefer_early = applied_filters.get('prefer_early', False)
            prefer_late = applied_filters.get('prefer_late', False)
            results = _sort_flights(results, sort_method, prefer_early, prefer_late)
            logger.info(f"📊 Sorted by: {sort_method}")
        
        active_flights = [f for f in results if f.get("flight_status", "").lower() == "active"]
        landed_flights = [f for f in results if f.get("flight_status", "").lower() == "landed"]
        other_flights = [f for f in results if f.get("flight_status", "").lower() not in ['active', 'landed']]
        
        formatted_results = []
        
        if applied_filters or sort_method != 'default':
            filter_info_parts = []
            if applied_filters.get('airline'):
                filter_info_parts.append(f"✈️ {applied_filters['airline']}")
            if sort_method == 'departure_time':
                filter_info_parts.append("📈 按时间升序")
            elif sort_method == 'departure_time_desc':
                filter_info_parts.append("📉 按时间降序")
            if applied_filters.get('prefer_early'):
                filter_info_parts.append("🌅 优先早班")
            if applied_filters.get('prefer_late'):
                filter_info_parts.append("🌆 优先晚班")
            
            if filter_info_parts:
                formatted_results.append(f"\n**应用条件**: {' | '.join(filter_info_parts)}")
        
        if active_flights:
            formatted_results.append(f"\n✈️ **正在飞行** ({len(active_flights)} 架航班)")
            for i, flight in enumerate(active_flights[:limit], 1):
                formatted_results.append(f"\n--- **航班 {i}** ---")
                formatted_results.append(format_flight_display(flight))
        
        if landed_flights:
            formatted_results.append(f"\n✅ **已降落** ({len(landed_flights)} 架航班)")
            for i, flight in enumerate(landed_flights[:limit], 1):
                formatted_results.append(f"\n--- **航班 {i}** ---")
                formatted_results.append(format_flight_display(flight))
        
        if other_flights:
            formatted_results.append(f"\n📋 **其他状态** ({len(other_flights)} 架航班)")
            for i, flight in enumerate(other_flights[:limit], 1):
                formatted_results.append(f"\n--- **航班 {i}** ---")
                formatted_results.append(format_flight_display(flight))
        
        summary_line = (
            f"\n{'='*50}\n"
            f"📊 共找到 **{len(results)}** 个航班结果"
            f" | ✈️ 正在飞行: **{len(active_flights)}**"
            f" | ✅ 已降落: **{len(landed_flights)}**"
            f" | 📡 数据来源: **{data_source}**"
        )
        
        if used_date:
            today = datetime.now().strftime('%Y-%m-%d')
            date_label = "今天" if used_date == today else used_date
            summary_line += f" | 📅 查询日期: **{date_label}**"
        
        formatted_results.append(summary_line)
        
        final_output = "\n".join(formatted_results)
        
        logger.info(f"✅ 成功返回 {len(results)} 条航班信息")
        
        return final_output

    except Exception as e:
        logger.error(f"❌ search_realtime_flights 异常: {e}", exc_info=True)
        return (
            f"❌ **系统异常**\n\n"
            f"处理查询时发生意外错误。\n"
            f"技术详情: {str(e)}\n\n"
            f"请稍后重试或联系技术支持。"
        )


@tool
def lookup_flight_status(flight_number: str, flight_date: str = "") -> str:
    """
    🔍 Look up detailed status of a specific flight by number.
    
    Args:
        flight_number: Flight number (e.g., "CA1234", "MU5678", "CZ1234")
        flight_date: Optional date YYYY-MM-DD (defaults to today)
    
    Returns:
        Detailed flight status with times, gates, delays
    """
    try:
        if not flight_number or not flight_number.strip():
            return "❓ **请提供航班号**\n\n示例: `CA1234`, `MU5678`, `CZ3501`"
        
        params = {
            "flight_number": flight_number.upper().replace(" ", ""),
            "limit": 1,
        }
        if flight_date:
            params["flight_date"] = flight_date

        logger.info(f"🔍 查询航班状态: {flight_number} @ {flight_date or '今天'}")
        
        try:
            results = _call_aviationstack_with_retry(params)
        except Exception as api_error:
            return f"❌ **查询失败**: {str(api_error)}"
        
        if not results:
            date_hint = f" 在 **{flight_date}**" if flight_date else "在今天"
            return (
                f"🔍 **未找到航班** `{flight_number}`{date_hint}\n\n"
                f"可能原因:\n"
                f"• 航班号不正确（请检查大小写）\n"
                f"• 该航班在指定日期没有计划\n"
                f"• 航班已经结束超过24小时\n\n"
                f"💡 提示: 航班号通常为 2位字母+3-4位数字 (如 CA1234)"
            )
        
        flight = results[0]
        
        fn = flight.get("flight", {}).get("iata", "")
        airline = flight.get("airline", {}).get("name", "")
        dep = flight.get("departure", {}).get("iata", "")
        dep_name = flight.get("departure", {}).get("airport", "")
        arr = flight.get("arrival", {}).get("iata", "")
        arr_name = flight.get("arrival", {}).get("airport", "")
        
        sched_dep_raw = flight.get("departure", {}).get("scheduled", "")
        sched_arr_raw = flight.get("arrival", {}).get("scheduled", "")
        actual_dep = flight.get("departure", {}).get("actual", "N/A")
        actual_arr = flight.get("arrival", {}).get("actual", "N/A")
        
        dep_gate = flight.get("departure", {}).get("gate", "N/A")
        dep_terminal = flight.get("departure", {}).get("terminal", "N/A")
        arr_gate = flight.get("arrival", {}).get("gate", "N/A")
        arr_terminal = flight.get("arrival", {}).get("terminal", "N/A")
        
        status = flight.get("flight_status", "unknown").lower()
        dep_delay = flight.get("departure", {}).get("delay", 0) or 0
        arr_delay = flight.get("arrival", {}).get("delay", 0) or 0
        
        def fmt(t):
            if not t or t == "N/A":
                return "--:--"
            try:
                dt = datetime.fromisoformat(t.replace('Z', '+00:00'))
                return dt.strftime('%H:%M')
            except:
                return t[:5] if len(t) >= 5 else t
        
        status_map = {
            "active": ("✈️", "正在飞行", "🟢"),
            "landed": ("✅", "已降落", "🟢"),
            "scheduled": ("📋", "按计划", "🔵"),
            "cancelled": ("❌", "已取消", "🔴"),
            "delayed": ("⏰", "延误中", "🟡"),
            "diverted": ("🔄", "已备降", "🟠"),
            "unknown": ("❓", "未知", "⚪"),
        }
        
        icon, status_text, status_emoji = status_map.get(status, status_map["unknown"])
        
        lines = [
            f"\n{icon} **航班 {fn}** · {airline}",
            f"{'─'*40}",
            f"📍 **航线**: {dep} ({dep_name}) → {arr} ({arr_name})",
            f"📊 **状态**: {status_emoji} **{status_text}**",
            f"",
            f"🛫 **出发**:",
            f"   计划时间: {fmt(sched_dep_raw)}",
            f"   实际时间: {fmt(actual_dep)}",
            f"   航站楼: T{dep_terminal} | 登机口: {dep_gate}",
        ]
        
        if isinstance(dep_delay, (int, float)) and dep_delay > 0:
            lines.append(f"   ⚠️ 延误: **{int(dep_delay)} 分钟**")
        
        lines.extend([
            f"",
            f"🛬 **到达**:",
            f"   计划时间: {fmt(sched_arr_raw)}",
            f"   实际时间: {fmt(actual_arr)}",
            f"   航站楼: T{arr_terminal} | 登机口: {arr_gate}",
        ])
        
        if isinstance(arr_delay, (int, float)) and arr_delay > 0:
            lines.append(f"   ⚠️ 延误: **{int(arr_delay)} 分钟**")
        
        total_duration = ""
        if sched_dep_raw and sched_arr_raw:
            try:
                dep_dt = datetime.fromisoformat(sched_dep_raw.replace('Z', '+00:00'))
                arr_dt = datetime.fromisoformat(sched_arr_raw.replace('Z', '+00:00'))
                duration = arr_dt - dep_dt
                hours, remainder = divmod(duration.total_seconds(), 3600)
                minutes = remainder // 60
                total_duration = f"{int(hours)}小时{int(minutes)}分钟"
            except:
                pass
        
        if total_duration:
            lines.append(f"", f"⏱️ **预计飞行时长**: ~{total_duration}")
        
        lines.append(f"\n{'─'*40}")
        lines.append(f"🕐 数据更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"❌ lookup_flight_status 异常: {e}", exc_info=True)
        return f"❌ **查询出错**: {str(e)}"


def get_smart_suggestions(user_input: str, max_suggestions: int = 5) -> List[dict]:
    """
    Generate smart suggestions for ambiguous or incorrect city names.
    
    Args:
        user_input: The user's input (possibly incorrect city name)
        max_suggestions: Maximum number of suggestions to return
        
    Returns:
        List of suggestion dictionaries with 'city', 'iata', 'score', and 'reason' keys
    """
    if not user_input or len(user_input) < 1:
        return []
    
    suggestions = []
    all_airports = {**DOMESTIC_AIRPORTS, **INTERNATIONAL_AIRPORTS}
    
    for city_name, info in all_airports.items():
        score = 0.0
        reason = ""
        
        if user_input.lower() == city_name.lower():
            score = 1.0
            reason = "完全匹配"
        elif user_input in city_name or city_name in user_input:
            similarity = SequenceMatcher(None, user_input.lower(), city_name.lower()).ratio()
            score = similarity
            reason = "包含关系"
        
        eng_name = info.get("name", "").lower()
        if eng_name:
            if user_input.lower() == eng_name:
                score = max(score, 0.98)
                reason = "英文名匹配"
            elif user_input.lower() in eng_name or eng_name in user_input.lower():
                eng_sim = SequenceMatcher(None, user_input.lower(), eng_name).ratio()
                if eng_sim > score:
                    score = eng_sim
                    reason = "英文名相似"
        
        province = info.get("province", "")
        if province and (province in user_input or user_input in province):
            prov_score = 0.7
            if prov_score > score:
                score = prov_score
                reason = f"位于{province}"
        
        iata_codes = info.get("iata", [])
        for iata in iata_codes:
            if iata.lower() in user_input.lower():
                iata_score = 0.95
                if iata_score > score:
                    score = iata_score
                    reason = f"IATA代码: {iata}"
        
        for alias, canonical in CITY_ALIASES.items():
            if alias == user_input:
                if canonical == city_name:
                    score = max(score, 0.95)
                    reason = f"别名: {alias}"
            
            if alias in user_input or user_input in alias:
                alias_sim = SequenceMatcher(None, user_input.lower(), alias.lower()).ratio()
                if alias_sim > 0.6 and alias_sim > score:
                    score = alias_sim
                    reason = f"与别名'{alias}'相似"
        
        if score >= 0.5:
            suggestions.append({
                'city': city_name,
                'iata': info.get('iata', [''])[0],
                'name_en': info.get('name', ''),
                'score': score,
                'reason': reason,
                'is_domestic': city_name in DOMESTIC_AIRPORTS,
            })
    
    suggestions.sort(key=lambda x: x['score'], reverse=True)
    
    seen_cities = set()
    unique_suggestions = []
    for sugg in suggestions:
        if sugg['city'] not in seen_cities:
            unique_suggestions.append(sugg)
            seen_cities.add(sugg['city'])
            if len(unique_suggestions) >= max_suggestions:
                break
    
    return unique_suggestions


def generate_error_recovery_message(error_type: str, context: dict = None) -> str:
    """
    Generate intelligent error recovery messages with actionable suggestions.
    
    Args:
        error_type: Type of error ('no_results', 'city_not_found', 'api_error', 'invalid_query')
        context: Additional context information
        
    Returns:
        User-friendly error message with suggestions
    """
    context = context or {}
    
    base_messages = {
        'no_results': {
            'title': '🔍 未找到航班',
            'possible_reasons': [
                "该航线今天可能没有航班",
                "航班可能已经全部降落或取消",
                "查询条件可能过于严格",
            ],
            'suggestions': [
                "尝试查询明天的航班",
                "移除筛选条件（如航空公司、时间范围）",
                "使用主要城市的IATA代码",
                "检查城市名称拼写是否正确",
            ],
        },
        'city_not_found': {
            'title': '🏙️ 城市未识别',
            'possible_reasons': [
                "城市名称可能有误",
                "该城市暂不支持",
                "可能是别名或简称",
            ],
            'suggestions': [
                "使用完整的城市名称（如：北京、上海）",
                "尝试使用城市的英文拼音",
                "使用IATA代码（如：PEK、PVG）",
                "查看支持的城市列表",
            ],
        },
        'api_error': {
            'title': '⚙️ 服务暂时不可用',
            'possible_reasons': [
                "航班数据服务可能正在维护",
                "网络连接不稳定",
                "API请求频率限制",
            ],
            'suggestions': [
                "稍等几秒后重试",
                "检查网络连接",
                "简化查询条件后重试",
                "如果问题持续，请联系技术支持",
            ],
        },
        'invalid_query': {
            'title': '❓ 查询格式不正确',
            'possible_reasons': [
                "缺少必要的查询参数",
                "查询格式无法识别",
            ],
            'suggestions': [
                "使用格式：从[出发地]到[目的地]",
                "示例：'从北京到上海的航班'",
                "示例：'明天郑州飞长沙'",
                "直接提供城市名：dep_iata='北京', arr_iata='上海'",
            ],
        },
    }
    
    if error_type not in base_messages:
        return "❌ 发生未知错误，请稍后重试。"
    
    msg_config = base_messages[error_type]
    
    lines = [f"\n{msg_config['title']}\n{'='*30}"]
    
    user_input = context.get('user_input', '')
    if error_type == 'city_not_found' and user_input:
        smart_suggs = get_smart_suggestions(user_input, max_suggestions=3)
        if smart_suggs:
            lines.append("\n💡 **您是否想查找以下城市？**:")
            for i, sugg in enumerate(smart_suggs, 1):
                flag = "🏠" if sugg['is_domestic'] else "🌍"
                iata_str = ", ".join(sugg['iata']) if isinstance(sugg.get('iata'), list) else sugg.get('iata', '')
                lines.append(f"  {i}. {flag} **{sugg['city']}** ({iata_str}) - {sugg['reason']} ({sugg['score']:.0%}匹配)")
    
    lines.append("\n📋 **可能的原因**:")
    for i, reason in enumerate(msg_config['possible_reasons'], 1):
        lines.append(f"  {i}. {reason}")
    
    lines.append("\n✨ **建议您尝试**:")
    for i, suggestion in enumerate(msg_config['suggestions'], 1):
        lines.append(f"  {i}. {suggestion}")
    
    dep_city = context.get('departure_city', '')
    arr_city = context.get('arrival_city', '')
    if dep_city and arr_city:
        lines.append(f"\n📝 **快速重试**: 查询\"{dep_city}到{arr_city}\"的航班")
    
    example_queries = [
        "从北京到上海的航班",
        "明天广州飞成都",
        "查询南航从深圳到杭州的航班",
        "最早从上海到西安的班机",
    ]
    
    lines.append(f"\n💬 **其他查询示例**:")
    for example in example_queries[:2]:
        lines.append(f"  • \"{example}\"")
    
    return "\n".join(lines)


@tool
def search_multi_route_flights(
    departure_cities: str = "",
    arrival_cities: str = "",
    flight_date: str = "",
    limit_per_route: int = 5,
    natural_query: str = "",
) -> str:
    """
    🔍 **Multi-Route Flight Search** - Search flights for multiple routes at once!
    
    ✨ Perfect for:
    - Comparing prices/times across different departure cities
    - Finding the best option from multiple origins
    - Planning trips with flexible departure points
    
    📝 EXAMPLES:
    - "从北京、上海、广州到成都的航班"
    - "西安/重庆飞往杭州"
    - "查询郑州和武汉到深圳的明天航班"
    
    Args:
        departure_cities: Comma-separated departure cities (e.g., "北京,上海,广州")
        arrival_cities: Comma-separated arrival cities (e.g., "成都,杭州")
        flight_date: Date YYYY-MM-DD (optional)
        limit_per_route: Max results per route (default 5)
        natural_query: Natural language query (recommended!)
    
    Returns:
        Comprehensive comparison of flights across all routes
    """
    try:
        dep_list = []
        arr_list = []
        
        if natural_query:
            parsed = parse_flight_query(natural_query)
            
            if parsed['departure_city']:
                dep_list = [parsed['departure_city']]
            if parsed['arrival_city']:
                arr_list = [parsed['arrival_city']]
            
            query_lower = natural_query.lower()
            
            multi_dep_pattern = r'从\s*(.+?)\s*(?:、|,|，|和|与|以及)\s*(.+?)\s*到'
            multi_arr_pattern = r'到\s*(.+?)\s*(?:、||,|，|和|与)\s*(.+?)(?:\s|$)'
            
            dep_match = re.search(multi_dep_pattern, natural_query)
            if dep_match:
                dep_part = dep_match.group(0).replace('从', '').split('到')[0]
                dep_list = re.split(r'[、,，和与及]', dep_part)
                dep_list = [d.strip() for d in dep_list if d.strip()]
            
            arr_match = re.search(r'到\s*(.+?)(?:\s*的航班|\s*$)', natural_query)
            if arr_match:
                arr_part = arr_match.group(1)
                if '、' in arr_part or ',' in arr_part or '，' in arr_part or '和' in arr_part:
                    arr_list = re.split(r'[、,，和与及]', arr_part)
                    arr_list = [a.strip() for a in arr_list if a.strip()]
                else:
                    arr_list = [arr_part.strip()]
            
            if parsed['flight_date'] and not flight_date:
                flight_date = parsed['flight_date']
        
        if departure_cities:
            user_deps = re.split(r'[、,，;；\s]+', departure_cities)
            dep_list = [d.strip() for d in user_deps if d.strip()]
        
        if arrival_cities:
            user_arrs = re.split(r'[、,，;；\s]+', arrival_cities)
            arr_list = [a.strip() for a in user_arrs if a.strip()]
        
        if not dep_list or not arr_list:
            return generate_error_recovery_message('invalid_query', {
                'user_input': natural_query or f"{departure_cities} → {arrival_cities}",
            }) + (
                "\n\n📝 **多航线查询格式**:\n"
                "• `从[城市1]、[城市2]到[目的地]` (如: 从北京、上海到广州)\n"
                "• `[出发地1],[出发地2] → [目的地1],[目的地2]`\n"
                "• `比较从北京和上海到成都的航班`"
            )
        
        resolved_deps = []
        for dep in dep_list:
            result = resolve_city_to_iata(dep)
            if result[0]:
                resolved_deps.append({
                    'original': dep,
                    'iata': result[0][0],
                    'name': result[1] or dep,
                    'resolved': True,
                })
            else:
                resolved_deps.append({
                    'original': dep,
                    'iata': dep.upper(),
                    'name': dep,
                    'resolved': False,
                })
        
        resolved_arrs = []
        for arr in arr_list:
            result = resolve_city_to_iata(arr)
            if result[0]:
                resolved_arrs.append({
                    'original': arr,
                    'iata': result[0][0],
                    'name': result[1] or arr,
                    'resolved': True,
                })
            else:
                resolved_arrs.append({
                    'original': arr,
                    'iata': arr.upper(),
                    'name': arr,
                    'resolved': False,
                })
        
        unresolved = [d['original'] for d in resolved_deps if not d['resolved']]
        unresolved += [a['original'] for a in resolved_arrs if not a['resolved']]
        
        if unresolved:
            warning = f"\n⚠️ 以下城市未能识别: {', '.join(unresolved)}\n"
            warning += "将尝试使用原始名称进行查询...\n\n"
        else:
            warning = ""
        
        total_routes = len(resolved_deps) * len(resolved_arrs)
        
        header = (
            f"\n{'='*60}\n"
            f"🗺️  **多航线航班对比**\n"
            f"{'='*60}\n"
            f"📍 出发地: {', '.join([d['name'] for d in resolved_deps])} ({len(resolved_deps)} 个)\n"
            f"🎯 目的地: {', '.join([a['name'] for a in resolved_arrs])} ({len(resolved_arrs)} 个)\n"
            f"📊 总计: **{total_routes}** 条航线 | 每条最多显示 **{limit_per_route}** 条结果\n"
            f"📅 日期: {flight_date or '今天'}\n"
            f"{warning}"
            f"{'─'*60}\n"
        )
        
        results_by_route = {}
        total_flights_found = 0
        routes_with_flights = 0
        
        for dep_info in resolved_deps:
            for arr_info in resolved_arrs:
                route_key = f"{dep_info['iata']}→{arr_info['iata']}"
                route_name = f"{dep_info['name']} → {arr_info['name']}"
                
                try:
                    params = {
                        "dep_iata": dep_info['iata'],
                        "arr_iata": arr_info['iata'],
                        "limit": limit_per_route,
                    }
                    
                    if flight_date:
                        params["flight_date"] = flight_date
                    
                    route_results = _call_aviationstack_with_retry(params)
                    
                    active_count = len([f for f in route_results if f.get("flight_status") == "active"])
                    landed_count = len([f for f in route_results if f.get("flight_status") == "landed"])
                    
                    results_by_route[route_key] = {
                        'name': route_name,
                        'flights': route_results[:limit_per_route],
                        'total': len(route_results),
                        'active': active_count,
                        'landed': landed_count,
                    }
                    
                    total_flights_found += len(route_results)
                    routes_with_flights += 1
                    
                except Exception as e:
                    logger.warning(f"Route {route_key} failed: {e}")
                    results_by_route[route_key] = {
                        'name': route_name,
                        'flights': [],
                        'total': 0,
                        'active': 0,
                        'landed': 0,
                        'error': str(e),
                    }
        
        output_sections = [header]
        
        sorted_routes = sorted(results_by_route.items(), key=lambda x: x[1]['total'], reverse=True)
        
        for i, (route_key, route_data) in enumerate(sorted_routes, 1):
            status_icon = "✈️" if route_data['total'] > 0 else "⭕"
            
            section_header = (
                f"\n{status_icon} **路线 {i}: {route_data['name']}**\n"
            )
            
            if route_data.get('error'):
                section_header += f"   ⚠️ 查询失败: {route_data['error'][:50]}...\n"
            elif route_data['total'] == 0:
                section_header += f"   🔍 未找到航班\n"
            else:
                section_header += (
                    f"   📊 共 {route_data['total']} 条"
                    f" | ✈️ 飞行中: {route_data['active']}"
                    f" | ✅ 已降落: {route_data['landed']}\n"
                )
            
            output_sections.append(section_header)
            
            if route_data['flights']:
                for j, flight in enumerate(route_data['flights'], 1):
                    output_sections.append(f"\n   **航班 {j}**:")
                    flight_display = format_flight_display(flight, include_details=False)
                    for line in flight_display.split('\n'):
                        output_sections.append(f"   {line}")
        
        summary = (
            f"\n{'='*60}\n"
            f"📈 **汇总统计**\n"
            f"{'='*60}\n"
            f"• 查询航线数: {total_routes}\n"
            f"• 有航班航线: {routes_with_flights}\n"
            f"• 航班总数: {total_flights_found}\n"
            f"• 空航线: {total_routes - routes_with_flights}\n"
        )
        
        if total_flights_found > 0:
            best_route = max(results_by_route.items(), key=lambda x: x[1]['total'])
            summary += (
                f"\n🏆 **最繁忙路线**: {best_route[1]['name']} ({best_route[1]['total']} 条航班)"
            )
        
        output_sections.append(summary)
        
        final_output = "\n".join(output_sections)
        
        logger.info(f"✅ Multi-route search completed: {total_routes} routes, {total_flights_found} flights")
        
        return final_output

    except Exception as e:
        logger.error(f"❌ search_multi_route_flights error: {e}", exc_info=True)
        return (
            f"❌ **多航线查询出错**\n\n"
            f"处理多航线查询时发生错误。\n"
            f"错误: {str(e)}\n\n"
            f"💡 建议:\n"
            f"• 减少查询的城市数量（建议每次不超过3个出发地和3个目的地）\n"
            f"• 使用简单的城市名称\n"
            f"• 尝试单航线查询作为替代"
        )


@tool
def analyze_flight_statistics(
    dep_iata: str = "",
    arr_iata: str = "",
    flight_date: str = "",
    natural_query: str = "",
) -> str:
    """
    📊 **Flight Statistics & Analytics** - Get insights about flight data!
    
    ✨ Provides:
    - Airline market share on a route
    - Flight status distribution (active, landed, delayed, etc.)
    - Time distribution analysis (morning/afternoon/evening flights)
    - Delay statistics and on-time performance
    - Busiest times and airline recommendations
    
    📝 EXAMPLES:
    - "分析北京到上海航线"
    - "统计郑州到长沙的航班情况"
    - "查看广州飞成都的航班统计"
    
    Args:
        dep_iata: Departure city (Chinese or IATA)
        arr_iata: Arrival city (Chinese or IATA)
        flight_date: Date YYYY-MM-DD (optional, defaults to today)
        natural_query: Natural language query (recommended!)
    
    Returns:
        Comprehensive statistical analysis with charts and insights
    """
    try:
        resolved_dep = ""
        resolved_arr = ""
        
        if natural_query:
            parsed = parse_flight_query(natural_query)
            if parsed['dep_iata']:
                dep_iata = parsed['dep_iata']
            if parsed['arr_iata']:
                arr_iata = parsed['arr_iata']
            if parsed['flight_date'] and not flight_date:
                flight_date = parsed['flight_date']
        
        if dep_iata:
            dep_result = resolve_city_to_iata(dep_iata)
            if dep_result[0]:
                resolved_dep = dep_result[0][0]
            else:
                resolved_dep = dep_iata.upper()
        
        if arr_iata:
            arr_result = resolve_city_to_iata(arr_iata)
            if arr_result[0]:
                resolved_arr = arr_result[0][0]
            else:
                resolved_arr = arr_iata.upper()
        
        if not resolved_dep and not resolved_arr:
            return generate_error_recovery_message('invalid_query', {
                'user_input': natural_query or f"{dep_iata} → {arr_iata}",
            }) + (
                "\n\n📊 **统计分析格式**:\n"
                "• `分析[城市]到[城市]的航线` (如: 分析北京到上海的航线)\n"
                "• `统计从郑州到长沙的航班`"
                "• `查看广州飞成都的情况`"
            )
        
        params = {"limit": 100}
        if resolved_dep:
            params["dep_iata"] = resolved_dep
        if resolved_arr:
            params["arr_iata"] = resolved_arr
        if flight_date:
            params["flight_date"] = flight_date
        
        logger.info(f"📊 Analyzing statistics for route: {resolved_dep} → {resolved_arr}")
        
        try:
            results = _call_aviationstack_with_retry(params)
        except Exception as e:
            return f"❌ **统计数据获取失败**: {str(e)}"
        
        if not results:
            return (
                f"📊 **无数据可分析**\n\n"
                f"航线 {resolved_dep} → {resolved_arr} 在{' ' + flight_date if flight_date else '今天'}没有找到航班数据。\n\n"
                f"💡 建议:\n"
                f"• 尝试其他日期\n"
                f"• 检查城市名称是否正确\n"
                f"• 该航线可能不是热门航线"
            )
        
        total_flights = len(results)
        
        status_counts = {}
        airline_counts = {}
        airline_delay_stats = {}
        hour_distribution = {}
        delay_flights = []
        on_time_flights = []
        
        for flight in results:
            status = flight.get("flight_status", "unknown").lower()
            status_counts[status] = status_counts.get(status, 0) + 1
            
            airline = flight.get("airline", {}).get("name", "Unknown")
            airline_counts[airline] = airline_counts.get(airline, 0) + 1
            
            dep_delay = flight.get("departure", {}).get("delay", 0) or 0
            arr_delay = flight.get("arrival", {}).get("delay", 0) or 0
            max_delay = max(dep_delay, arr_delay)
            
            if airline not in airline_delay_stats:
                airline_delay_stats[airline] = {'total': 0, 'delayed': 0, 'total_delay_minutes': 0}
            airline_delay_stats[airline]['total'] += 1
            if max_delay > 15:
                airline_delay_stats[airline]['delayed'] += 1
                airline_delay_stats[airline]['total_delay_minutes'] += max_delay
                delay_flights.append(flight)
            else:
                on_time_flights.append(flight)
            
            sched_dep = flight.get("departure", {}).get("scheduled", "")
            if sched_dep:
                try:
                    dt = datetime.fromisoformat(sched_dep.replace('Z', '+00:00'))
                    hour = dt.hour
                    if 5 <= hour < 12:
                        period = "早班 (05-12)"
                    elif 12 <= hour < 18:
                        period = "午间 (12-18)"
                    elif 18 <= hour < 22:
                        period = "晚间 (18-22)"
                    else:
                        period = "夜间 (22-05)"
                    hour_distribution[period] = hour_distribution.get(period, 0) + 1
                except:
                    pass
        
        status_emoji_map = {
            "active": "✈️",
            "landed": "✅",
            "scheduled": "📋",
            "cancelled": "❌",
            "delayed": "⏰",
        }
        
        header = (
            f"\n{'='*60}\n"
            f"📊 **航班数据分析报告**\n"
            f"{'='*60}\n"
            f"📍 航线: {resolved_dep} → {resolved_arr}\n"
            f"📅 日期: {flight_date or '今天'}\n"
            f"📈 样本量: **{total_flights}** 个航班\n"
            f"{'─'*60}\n"
        )
        
        sections = [header]
        
        sections.append("\n📊 **航班状态分布**:\n")
        for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
            emoji = status_emoji_map.get(status, "❓")
            percentage = (count / total_flights) * 100
            bar_len = int(percentage / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            status_cn = {
                "active": "正在飞行",
                "landed": "已降落",
                "scheduled": "计划中",
                "cancelled": "已取消",
                "delayed": "延误中",
                "unknown": "未知",
            }.get(status, status)
            sections.append(f"  {emoji} {status_cn}: {count:>3} ({percentage:>5.1f}%) {bar}")
        
        sections.append(f"\n✈️ **航空公司分布 (Top 10)**:\n")
        sorted_airlines = sorted(airline_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (airline, count) in enumerate(sorted_airlines, 1):
            percentage = (count / total_flights) * 100
            bar_len = int(percentage / 5)
            bar = "▓" * bar_len + "░" * (20 - bar_len)
            short_name = airline[:25] + "..." if len(airline) > 25 else airline
            sections.append(f"  {i:>2}. {short_name}: {count:>3} ({percentage:>5.1f}%) {bar}")
        
        if hour_distribution:
            sections.append(f"\n🕐 **时段分布**:\n")
            period_order = ["早班 (05-12)", "午间 (12-18)", "晚间 (18-22)", "夜间 (22-05)"]
            for period in period_order:
                if period in hour_distribution:
                    count = hour_distribution[period]
                    percentage = (count / total_flights) * 100
                    bar_len = int(percentage / 5)
                    bar = "◉" * bar_len + "○" * (20 - bar_len)
                    sections.append(f"  {period}: {count:>3} ({percentage:>5.1f}%) {bar}")
        
        total_on_time = len(on_time_flights)
        total_delayed = len(delay_flights)
        on_time_rate = (total_on_time / total_flights * 100) if total_flights > 0 else 0
        
        sections.append(f"\n⏱️ **准点率统计**:\n")
        sections.append(f"  ✅ 准点航班: {total_on_time} ({on_time_rate:.1f}%)")
        sections.append(f"  ⚠️ 延误航班: {total_delayed} ({100-on_time_rate:.1f}%)")
        
        if delay_flights:
            avg_delay = sum(max(
                f.get("departure", {}).get("delay", 0) or 0,
                f.get("arrival", {}).get("delay", 0) or 0
            ) for f in delay_flights) / len(delay_flights)
            max_delay_flight = max(delay_flights, key=lambda f: max(
                f.get("departure", {}).get("delay", 0) or 0,
                f.get("arrival", {}).get("delay", 0) or 0
            ))
            max_delay_val = max(
                max_delay_flight.get("departure", {}).get("delay", 0) or 0,
                max_delay_flight.get("arrival", {}).get("delay", 0) or 0
            )
            sections.append(f"  📊 平均延误: {avg_delay:.0f} 分钟")
            sections.append(f"  🔴 最大延误: {int(max_delay_val)} 分钟")
        
        if airline_delay_stats:
            sections.append(f"\n🏆 **航空公司准点率排名**:\n")
            airline_performance = []
            for airline, stats in airline_delay_stats.items():
                if stats['total'] > 0:
                    delay_rate = (stats['delayed'] / stats['total']) * 100
                    avg_delay = (stats['total_delay_minutes'] / stats['delayed']) if stats['delayed'] > 0 else 0
                    airline_performance.append({
                        'airline': airline,
                        'total': stats['total'],
                        'on_time_rate': 100 - delay_rate,
                        'avg_delay': avg_delay,
                    })
            
            airline_performance.sort(key=lambda x: x['on_time_rate'], reverse=True)
            
            for i, perf in enumerate(airline_performance[:8], 1):
                rating = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else "  "))
                short_name = perf['airline'][:20] + "..." if len(perf['airline']) > 20 else perf['airline']
                sections.append(
                    f"  {rating} {i}. {short_name}\n"
                    f"      准点率: {perf['on_time_rate']:.1f}% | "
                    f"平均延误: {perf['avg_delay']:.0f}分钟 | "
                    f"航班数: {perf['total']}"
                )
            
            best_airline = airline_performance[0] if airline_performance else None
            if best_airline and best_airline['on_time_rate'] >= 80:
                sections.append(
                    f"\n💡 **推荐**: **{best_airline['airline']}** 在该航线表现最佳，"
                    f"准点率达 {best_airline['on_time_rate']:.1f}%"
                )
        
        busiest_period = max(hour_distribution.items(), key=lambda x: x[1]) if hour_distribution else None
        if busiest_period:
            sections.append(
                f"\n🕐 **高峰时段**: {busiest_period[0]} ({busiest_period[1]} 个航班)"
            )
        
        footer = (
            f"\n{'─'*60}\n"
            f"📝 数据来源: AviationStack API | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"💡 提示: 统计基于当前可用数据，可能不完整"
        )
        sections.append(footer)
        
        final_output = "\n".join(sections)
        
        logger.info(f"✅ Statistical analysis completed for {resolved_dep} → {resolved_arr}: {total_flights} flights")
        
        return final_output

    except Exception as e:
        logger.error(f"❌ analyze_flight_statistics error: {e}", exc_info=True)
        return f"❌ **统计分析出错**: {str(e)}"


def get_supported_cities_summary() -> str:
    """Get a summary of supported cities for help text"""
    domestic_count = len(DOMESTIC_AIRPORTS)
    international_count = len(INTERNATIONAL_AIRPORTS)
    alias_count = len(CITY_ALIASES)
    
    major_domestic = sorted(list(DOMESTIC_AIRPORTS.keys()))[:15]
    major_international = sorted(list(INTERNATIONAL_AIRPORTS.keys()))[:10]
    
    text = (
        f"🌏 **智能航班查询系统**\n"
        f"{'='*35}\n\n"
        f"📊 **覆盖范围**:\n"
        f"  • 🏠 国内城市: **{domestic_count}** 个\n"
        f"  • 🌍 国际城市: **{international_count}** 个\n"
        f"  • 🔤 城市别名: **{alias_count}** 个\n\n"
        f"🏠 **热门国内城市**:\n"
    )
    
    for city in major_domestic:
        info = DOMESTIC_AIRPORTS[city]
        iata_str = "/".join(info["iata"][:2])
        text += f"  • {city} ({iata_str})\n"
    
    text += f"\n🌍 **主要国际城市**:\n"
    for city in major_international:
        info = INTERNATIONAL_AIRPORTS[city]
        iata_str = "/".join(info["iata"][:2])
        text += f"  • {city} ({iata_str})\n"
    
    text += (
        f"\n💡 **输入方式** (全部自动识别):\n"
        f"  • 中文全名: `北京`, `上海`, `广州`\n"
        f"  • 城市别名: `京`, `沪`, `穗`, `商都`\n"
        f"  • 英文名称: `Beijing`, `Shanghai`\n"
        f"  • IATA代码: `PEK`, `PVG`, `CAN`\n"
        f"  • 自然语言: `从郑州到长沙的航班`\n\n"
        f"📅 **日期支持**:\n"
        f"  • 关键词: `今天`, `明天`, `后天`, `大后天`\n"
        f"  • 具体日期: `2026-04-21`, `4月21日`\n\n"
        f"✨ 开始查询吧！例如: \"查询明天从北京到上海的航班\""
    )
    
    return text