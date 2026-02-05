"""
Logging utilities for amplifier-session-storage.

This module follows Python library logging best practices:
- Library code uses standard logging.getLogger()
- NullHandler is added to prevent "No handler found" warnings
- Applications configure their own handlers and formatters

The StructuredJsonFormatter is provided as a UTILITY for applications
that want JSON logging (e.g., for Azure Container Apps). The library
itself does NOT configure logging - that's the application's job.

Usage in library code:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Something happened", extra={"key": "value"})

Usage in applications:
    from amplifier_session_storage.logging_utils import StructuredJsonFormatter

    handler = logging.StreamHandler()
    handler.setFormatter(StructuredJsonFormatter())
    logging.getLogger("amplifier_session_storage").addHandler(handler)
    logging.getLogger("amplifier_session_storage").setLevel(logging.INFO)
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any


class StructuredJsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging in cloud environments.

    This is a UTILITY for applications to use when configuring logging.
    The library does NOT use this internally - it's provided for apps
    that want JSON output (e.g., Azure Container Apps, Log Analytics).

    Outputs logs as single-line JSON objects with consistent fields:
    - timestamp: ISO 8601 format in UTC
    - level: Log level (INFO, WARNING, ERROR, etc.)
    - logger: Logger name
    - message: Log message
    - module, function, line: Source location
    - Additional context fields from extra dict

    Example application usage:
        import logging
        from amplifier_session_storage.logging_utils import StructuredJsonFormatter

        # Configure root logger for JSON output
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredJsonFormatter())
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.INFO)
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            if record.exc_info[0]:
                log_obj["exception_type"] = record.exc_info[0].__name__

        # Add extra fields (excluding standard LogRecord attributes)
        standard_attrs = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "exc_info",
            "exc_text",
            "thread",
            "threadName",
            "taskName",
            "message",
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                try:
                    json.dumps(value)
                    log_obj[key] = value
                except (TypeError, ValueError):
                    log_obj[key] = str(value)

        return json.dumps(log_obj, default=str)


# Set up NullHandler for library - prevents "No handler found" warnings
# Applications should configure their own handlers
logging.getLogger("amplifier_session_storage").addHandler(logging.NullHandler())
