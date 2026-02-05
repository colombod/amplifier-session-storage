"""
Structured JSON logging utilities for cloud environments.

This module provides structured logging that works well with Azure Container Apps,
Log Analytics, and other cloud logging systems that expect JSON-formatted logs.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class StructuredJsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging in cloud environments.
    
    Outputs logs as single-line JSON objects with consistent fields:
    - timestamp: ISO 8601 format in UTC
    - level: Log level (INFO, WARNING, ERROR, etc.)
    - logger: Logger name
    - message: Log message
    - Additional context fields from extra dict
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        # Add extra fields (excluding standard LogRecord attributes)
        standard_attrs = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "exc_info", "exc_text", "thread", "threadName",
            "taskName", "message"
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                # Ensure value is JSON serializable
                try:
                    json.dumps(value)
                    log_obj[key] = value
                except (TypeError, ValueError):
                    log_obj[key] = str(value)

        return json.dumps(log_obj, default=str)


def configure_structured_logging(
    level: int = logging.INFO,
    logger_name: str | None = None,
) -> logging.Logger:
    """
    Configure structured JSON logging for cloud environments.
    
    Args:
        level: Logging level (default: INFO)
        logger_name: Specific logger to configure (default: root logger)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(logger_name)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create stdout handler with JSON formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredJsonFormatter())
    
    logger.addHandler(handler)
    logger.setLevel(level)
    
    return logger


def get_storage_logger(name: str) -> logging.Logger:
    """
    Get a logger for storage components with consistent naming.
    
    Args:
        name: Component name (e.g., 'cosmos', 'duckdb')
    
    Returns:
        Logger instance with name 'amplifier_session_storage.{name}'
    """
    return logging.getLogger(f"amplifier_session_storage.{name}")


class StorageLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds storage context to all log messages.
    
    Useful for adding consistent context like user_id, session_id, etc.
    """

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Add extra context to log record."""
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs
