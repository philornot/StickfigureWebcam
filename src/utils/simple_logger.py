#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simplified custom logger with color formatting."""

import datetime
import logging
import os
import re
from typing import Optional

import pytz
from colorama import init, Fore, Back, Style

# Initialize colorama
init(autoreset=True)


class CustomLogger:
    """Simplified logger with color formatting for console and clean file output.

    This logger provides:
    - Colored console output for better readability
    - Clean file logs without color codes
    - Standard logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Type-specific formatting (CAMERA, POSE, DRAWING, etc.)
    """

    # Logging levels with colors and symbols
    LEVELS = {
        "DEBUG": {"color": Fore.CYAN, "symbol": "🔍", "level": logging.DEBUG},
        "INFO": {"color": Fore.GREEN, "symbol": "ℹ️", "level": logging.INFO},
        "WARNING": {"color": Fore.YELLOW, "symbol": "⚠️", "level": logging.WARNING},
        "ERROR": {"color": Fore.RED, "symbol": "❌", "level": logging.ERROR},
        "CRITICAL": {"color": Fore.RED + Back.WHITE, "symbol": "🔥", "level": logging.CRITICAL}
    }

    # Log type colors
    TYPES = {
        "CAMERA": {"color": Fore.BLUE, "symbol": "📷"},
        "POSE": {"color": Fore.MAGENTA, "symbol": "👤"},
        "DRAWING": {"color": Fore.YELLOW, "symbol": "🖌️"},
        "VIRTUAL_CAM": {"color": Fore.LIGHTBLUE_EX, "symbol": "📺"},
        "CONFIG": {"color": Fore.GREEN, "symbol": "⚙️"},
        "PERFORMANCE": {"color": Fore.CYAN, "symbol": "⏱️"}
    }

    class ColoredFormatter(logging.Formatter):
        """Formatter for colored console output."""

        def format(self, record):
            return record.msg

    class PlainFormatter(logging.Formatter):
        """Formatter for plain file output without colors."""

        def format(self, record):
            if hasattr(record, 'plain_msg'):
                return record.plain_msg
            # Remove ANSI escape codes
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            return ansi_escape.sub('', record.msg)

    def __init__(
        self,
        log_file: Optional[str] = None,
        console_level: str = "INFO",
        file_level: str = "DEBUG",
        timezone: str = "Europe/Warsaw"
    ):
        """Initialize the logger.

        Args:
            log_file: Path to log file (None for console-only)
            console_level: Logging level for console output
            file_level: Logging level for file output
            timezone: Timezone for timestamps
        """
        self.timezone = pytz.timezone(timezone)
        self.console_level = console_level
        self.file_level = file_level
        self.log_file = log_file

        # Setup logger
        self.logger = logging.getLogger("StickFigureWebcam")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.LEVELS[console_level]["level"])
        console_handler.setFormatter(self.ColoredFormatter())
        self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(self.LEVELS[file_level]["level"])
            file_handler.setFormatter(self.PlainFormatter())
            self.logger.addHandler(file_handler)

        self.info("Logger", "Logger initialized successfully")

    def _format_message(
        self,
        level: str,
        module: str,
        message: str,
        log_type: Optional[str] = None,
        plain: bool = False
    ) -> str:
        """Format log message.

        Args:
            level: Log level
            module: Module name
            message: Log message
            log_type: Optional log type (CAMERA, POSE, etc.)
            plain: If True, return plain text without colors

        Returns:
            Formatted message
        """
        now = datetime.datetime.now(self.timezone)
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")

        level_info = self.LEVELS[level]

        if plain:
            formatted = f"[{time_str}] {level_info['symbol']} [{level}]"

            if log_type and log_type in self.TYPES:
                type_info = self.TYPES[log_type]
                formatted += f" {type_info['symbol']} [{log_type}]"

            formatted += f" [{module}] {message}"
        else:
            formatted = f"{level_info['color']}[{time_str}] {level_info['symbol']} [{level}]"

            if log_type and log_type in self.TYPES:
                type_info = self.TYPES[log_type]
                formatted += f" {type_info['color']}{type_info['symbol']} [{log_type}]"

            formatted += f" {Style.BRIGHT}{Fore.WHITE}[{module}]{Style.RESET_ALL} {message}"

        return formatted

    def _log(
        self,
        level: str,
        module: str,
        message: str,
        log_type: Optional[str] = None
    ):
        """Internal logging method.

        Args:
            level: Log level
            module: Module name
            message: Log message
            log_type: Optional log type
        """
        # Create colored and plain messages
        formatted = self._format_message(level, module, message, log_type, plain=False)
        plain_formatted = self._format_message(level, module, message, log_type, plain=True)

        # Create log record
        log_record = logging.LogRecord(
            name=self.logger.name,
            level=self.LEVELS[level]["level"],
            pathname="",
            lineno=0,
            msg=formatted,
            args=(),
            exc_info=None
        )
        log_record.plain_msg = plain_formatted

        # Send to handlers
        for handler in self.logger.handlers:
            if handler.level <= log_record.levelno:
                handler.handle(log_record)

    def debug(self, module: str, message: str, log_type: Optional[str] = None):
        """Log debug message."""
        self._log("DEBUG", module, message, log_type)

    def info(self, module: str, message: str, log_type: Optional[str] = None):
        """Log info message."""
        self._log("INFO", module, message, log_type)

    def warning(self, module: str, message: str, log_type: Optional[str] = None):
        """Log warning message."""
        self._log("WARNING", module, message, log_type)

    def error(self, module: str, message: str, log_type: Optional[str] = None):
        """Log error message."""
        self._log("ERROR", module, message, log_type)

    def critical(self, module: str, message: str, log_type: Optional[str] = None):
        """Log critical message."""
        self._log("CRITICAL", module, message, log_type)
