# customer_support_chat/app/core/logger.py
import logging
import sys
import os
from .settings import get_settings

config = get_settings()

logger = logging.getLogger("customer_support_chat")
logger.setLevel(getattr(logging, config.LOG_LEVEL))

stream_handler = logging.StreamHandler(sys.stdout)

# Fix Windows GBK encoding issue with emoji/unicode characters
if sys.platform == 'win32':
    stream_handler.encoding = 'utf-8'
    # Reconfigure stdout for UTF-8 on Windows
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

stream_handler.setLevel(getattr(logging, config.LOG_LEVEL))

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s"
)
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
