"""
Structured Logging Utility
AI-Powered Multimodal Gesture Recognition and Hands-Free Computer Control Platform
"""

import logging
import sys
from typing import Optional


def setup_logger(name: str = "GestureControl", level: str = "INFO") -> logging.Logger:
    """
    Configures and returns a standardized logger.
    
    Args:
        name: Name of the logger or module.
        level: Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').
    
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
