from os import environ
from dotenv import load_dotenv

load_dotenv()
"""这段代码定义了配置管理类，从环境变量中加载各种API密钥和服务地址（如OpenAI、WooCommerce、数据库等），
提供统一的配置访问接口。
get_settings函数返回Config实例供其他模块使用。
"""

class Config:
    OPENAI_API_KEY: str = environ.get("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = environ.get("OPENAI_BASE_URL", "")
    
    OPENAI_MODEL: str = environ.get("OPENAI_MODEL", "deepseek-chat")
    MAX_TOKENS: int = int(environ.get("MAX_TOKENS", "1000"))

    EMBEDDING_API_KEY: str = environ.get("EMBEDDING_API_KEY", environ.get("OPENAI_API_KEY", ""))
    EMBEDDING_BASE_URL: str = environ.get("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
    EMBEDDING_MODEL: str = environ.get("EMBEDDING_MODEL", "bge-m3")
    
    DATA_PATH: str = "./customer_support_chat/data"
    LOG_LEVEL: str = environ.get("LOG_LEVEL", "DEBUG")
    SQLITE_DB_PATH: str = environ.get(
        "SQLITE_DB_PATH", "./customer_support_chat/data/travel2.sqlite"
    )
    QDRANT_URL: str = environ.get("QDRANT_URL", "http://localhost:6333")
    QDRANT_KEY: str = environ.get("QDRANT_KEY", "")
    RECREATE_COLLECTIONS: bool = environ.get("RECREATE_COLLECTIONS", "False")
    LIMIT_ROWS: int = environ.get("LIMIT_ROWS", "100")
    
    # WooCommerce API Settings
    # WOOCOMMERCE_API_URL should be the WordPress base URL (e.g., "https://yourstore.com")
    # The system will automatically append "/wp-json/wc/v3" to create the full API endpoint
    WOOCOMMERCE_CONSUMER_KEY: str = environ.get("WOOCOMMERCE_CONSUMER_KEY", "")
    WOOCOMMERCE_CONSUMER_SECRET: str = environ.get("WOOCOMMERCE_CONSUMER_SECRET", "")
    WOOCOMMERCE_API_URL: str = environ.get("WOOCOMMERCE_API_URL", "")
    
    # Form Submission API Settings
    FORM_SUBMISSION_API_URL: str = environ.get("FORM_SUBMISSION_API_URL", "")
    
    # Blog Search API Settings
    BLOG_SEARCH_API_URL: str = environ.get("BLOG_SEARCH_API_URL", "")

    # AviationStack Real-time Flight API (legacy, kept as fallback)
    AVIATIONSTACK_API_KEY: str = environ.get("AVIATIONSTACK_API_KEY", "")

    # Juhe (聚合数据) Flight API - 国内航班查询
    JUHE_FLIGHT_KEY: str = environ.get("JUHE_FLIGHT_KEY", "")
    JUHE_FLIGHT_API_URL: str = environ.get("JUHE_FLIGHT_API_URL", "https://apis.juhe.cn/flight/query")

    # Ctrip (携程) API Settings (via OneBound 万邦开放平台)
    CTRIP_APP_KEY: str = environ.get("CTRIP_APP_KEY", "")
    CTRIP_APP_SECRET: str = environ.get("CTRIP_APP_SECRET", "")
    CTRIP_API_BASE_URL: str = environ.get("CTRIP_API_BASE_URL", "https://api-gw.onebound.cn")

    # DiDi MCP API Settings (https://mcp.didichuxing.com)
    DID_MCP_KEY: str = environ.get("DID_MCP_KEY", "")
    DID_MCP_BASE_URL: str = environ.get("DID_MCP_BASE_URL", "https://mcp.didichuxing.com")
    DID_MCP_SANDBOX_URL: str = environ.get("DID_MCP_SANDBOX_URL", "https://mcp.didichuxing.com/mcp-servers-sandbox")

def get_settings():
    return Config()