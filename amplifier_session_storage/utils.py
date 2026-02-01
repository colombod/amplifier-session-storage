"""Shared utility functions for session storage.

This module contains common utilities used across multiple storage backends.
"""

from __future__ import annotations

from typing import Any


def extract_event_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Extract safe summary fields from event data.

    CRITICAL: Never include 'data' or 'content' fields.
    Only extract small, known-safe fields that won't cause
    context overflow when returned to agents.

    Args:
        data: Raw event data dictionary.

    Returns:
        Dictionary containing only safe summary fields.
    """
    summary: dict[str, Any] = {}

    # Safe fields to extract (small, bounded values)
    safe_fields = [
        "model",
        "duration_ms",
        "has_tool_calls",
        "has_error",
        "error_type",
        "tool_name",
    ]

    for field in safe_fields:
        if field in data:
            summary[field] = data[field]

    # Extract usage summary (bounded structure)
    if "usage" in data:
        usage = data["usage"]
        summary["usage"] = {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }

    return summary
