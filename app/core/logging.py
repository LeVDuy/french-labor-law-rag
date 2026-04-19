"""
Logging structuré pour l'application RAG.
"""

import logging
import sys
import time
from functools import wraps
from typing import Callable

LOG_FORMAT = "[%(asctime)s] [%(levelname)-7s] [%(name)-20s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_execution_time(logger: logging.Logger) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info(f"{func.__name__} terminé en {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(f"{func.__name__} échoué après {elapsed:.2f}s : {e}")
                raise
        return wrapper
    return decorator

