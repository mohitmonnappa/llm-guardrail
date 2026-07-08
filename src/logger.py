"""
Logging Module
==============
Centralized logging configuration for LLM guardrails.
"""

import logging
import os
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "./output/guardrails.log",
    to_console: bool = True
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file
        to_console: Whether to log to console
    """
    # Create logger
    logger = logging.getLogger("guardrails")
    logger.setLevel(getattr(logging, log_level))

    # Create output directory if needed
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    # Console handler
    if to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


# Default logger
logger = setup_logging()
