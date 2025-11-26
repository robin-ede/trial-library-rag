"""
Centralized logging configuration for trial-library-rag.
Provides human-readable console logging with environment-based level control.
"""
import logging
import os
from typing import Optional

# Suppress Milvus/TensorFlow/GRPC C++ logging
os.environ["GLOG_minloglevel"] = "3"  # FATAL only
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def setup_logging(level: Optional[str] = None) -> None:
    """
    Configure application-wide logging.

    Args:
        level: Log level override. If None, reads from LOG_LEVEL env var (default: INFO)
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    numeric_level = getattr(logging, level, logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        force=True  # Override any existing configuration
    )

    # Suppress noisy third-party loggers
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('langchain').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module."""
    return logging.getLogger(name)
