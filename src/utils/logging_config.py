#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/utils/logging_config.py

import datetime
import os
from typing import Any, Dict, Optional


def get_log_file_path() -> str:
    """
    Generates log file path based on current date.

    Returns:
        str: Path to log file
    """
    # Current project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Logs directory
    logs_dir = os.path.join(project_dir, "logs")

    # Create directory if it doesn't exist
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)

    # File name format: stick_figure_YYYY-MM-DD_HH-MM-SS.log
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"stick_figure_{timestamp}.log"

    return os.path.join(logs_dir, log_filename)


def get_logger_config(debug: bool = False) -> Dict[str, Any]:
    """
    Returns logger configuration.

    Args:
        debug (bool): Whether to enable debug mode

    Returns:
        Dict[str, Any]: Logger configuration
    """
    log_file = get_log_file_path()

    return {
        "log_file": log_file,
        "console_level": "DEBUG" if debug else "INFO",
        "file_level": "DEBUG",
        "timezone": "Europe/Warsaw",
    }


def setup_logger(custom_logger_class, debug: bool = False) -> Optional[Any]:
    """
    Creates and configures logger instance.

    Args:
        custom_logger_class: Logger class (e.g., CustomLogger)
        debug (bool): Whether to enable debug mode

    Returns:
        Optional[Any]: Configured logger or None in case of error
    """
    try:
        logger_config = get_logger_config(debug)
        logger = custom_logger_class(**logger_config)

        logger.info(
            "LoggingConfig",
            f"Logger initialized. Logs saved to: {logger_config['log_file']}",
            log_type="CONFIG",
        )

        return logger
    except Exception as e:
        print(f"Error during logger initialization: {str(e)}")
        # Emergency creation of logger with console-only logging
        try:
            fallback_logger = custom_logger_class(log_file=None, console_level="INFO")
            fallback_logger.error(
                "LoggingConfig",
                f"Error during logger initialization: {str(e)}. Using emergency logger without file writing.",
                log_type="CONFIG",
            )
            return fallback_logger
        except:
            print("Critical error during emergency logger initialization.")
            return None
