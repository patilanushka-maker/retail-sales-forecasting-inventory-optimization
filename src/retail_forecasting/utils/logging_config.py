"""Logging configuration module for consistent console and file logging."""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "retail_forecasting",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """Configure and return a standardized logger instance.

    Parameters
    ----------
    name : str
        Logger identifier.
    level : int
        Logging level (e.g. logging.INFO, logging.DEBUG).
    log_file : Optional[str]
        Optional file path to persist logs.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
