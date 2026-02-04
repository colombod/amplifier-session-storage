"""
Data sanitization utilities for removing sensitive information.

Used when loading real Amplifier sessions for testing/demo purposes.
Removes API keys, secrets, personal data while preserving structure.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Patterns for detecting sensitive data
API_KEY_PATTERNS = [
    r"sk-[a-zA-Z0-9]{32,}",  # OpenAI keys
    r"sk-proj-[a-zA-Z0-9_-]{32,}",  # OpenAI project keys
    r"sk-ant-api03-[a-zA-Z0-9_-]{32,}",  # Anthropic keys
    r"AIza[a-zA-Z0-9_-]{35}",  # Google API keys
    r"ya29\.[a-zA-Z0-9_-]{100,}",  # Google OAuth tokens
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",  # Azure subscription IDs (if not session IDs)
]

SECRET_PATTERNS = [
    r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]?([a-zA-Z0-9_-]{20,})['\"]?",
    r"(?i)(secret|password|passwd)\s*[:=]\s*['\"]?([a-zA-Z0-9_-]{20,})['\"]?",
    r"(?i)(token)\s*[:=]\s*['\"]?([a-zA-Z0-9_-]{20,})['\"]?",
    r"(?i)(authorization|auth)\s*:\s*bearer\s+([a-zA-Z0-9_-]{20,})",
]

# Email patterns (personal data)
EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"


def sanitize_text(text: str, placeholder: str = "[REDACTED]") -> str:
    """
    Remove sensitive information from text content.

    Replaces:
    - API keys
    - Secrets and passwords
    - Email addresses
    - Bearer tokens

    Args:
        text: Text to sanitize
        placeholder: Replacement string for sensitive data

    Returns:
        Sanitized text with sensitive data removed
    """
    if not text or not isinstance(text, str):
        return text

    sanitized = text

    # Replace API keys
    for pattern in API_KEY_PATTERNS:
        sanitized = re.sub(pattern, placeholder, sanitized)

    # Replace secrets (with capture group handling)
    for pattern in SECRET_PATTERNS:
        sanitized = re.sub(pattern, rf"\1: {placeholder}", sanitized)

    # Replace email addresses
    sanitized = re.sub(EMAIL_PATTERN, "[EMAIL_REDACTED]", sanitized)

    return sanitized


def sanitize_dict(data: dict[str, Any], placeholder: str = "[REDACTED]") -> dict[str, Any]:
    """
    Recursively sanitize a dictionary.

    Removes sensitive data from string values while preserving structure.

    Args:
        data: Dictionary to sanitize
        placeholder: Replacement for sensitive data

    Returns:
        Sanitized dictionary (new instance, original unchanged)
    """
    if not isinstance(data, dict):
        return data

    sanitized = {}

    for key, value in data.items():
        # Sanitize the key name too (might contain sensitive info)
        clean_key = sanitize_text(key, placeholder) if isinstance(key, str) else key

        # Recursively sanitize values
        if isinstance(value, str):
            sanitized[clean_key] = sanitize_text(value, placeholder)
        elif isinstance(value, dict):
            sanitized[clean_key] = sanitize_dict(value, placeholder)
        elif isinstance(value, list):
            sanitized[clean_key] = [
                sanitize_dict(item, placeholder)
                if isinstance(item, dict)
                else sanitize_text(item, placeholder)
                if isinstance(item, str)
                else item
                for item in value
            ]
        else:
            # Numbers, booleans, None - keep as-is
            sanitized[clean_key] = value

    return sanitized


def sanitize_transcript_message(message: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitize a transcript message from transcript.jsonl.

    Handles complex assistant content arrays.

    Args:
        message: Transcript message

    Returns:
        Sanitized message (new instance)
    """
    sanitized = {}

    for key, value in message.items():
        if key == "content":
            # Handle content (string or array)
            if isinstance(value, str):
                sanitized[key] = sanitize_text(value)
            elif isinstance(value, list):
                # Assistant message content array
                sanitized[key] = []
                for block in value:
                    if isinstance(block, dict):
                        sanitized_block = {}
                        for block_key, block_value in block.items():
                            if isinstance(block_value, str) and block_key in ("text", "thinking"):
                                sanitized_block[block_key] = sanitize_text(block_value)
                            else:
                                sanitized_block[block_key] = block_value
                        sanitized[key].append(sanitized_block)
                    else:
                        sanitized[key].append(block)
            else:
                sanitized[key] = value
        elif isinstance(value, str):
            sanitized[key] = sanitize_text(value)
        else:
            sanitized[key] = value

    return sanitized


def sanitize_event(event: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitize an event from events.jsonl.

    Removes sensitive data from event data fields.

    Args:
        event: Event dictionary

    Returns:
        Sanitized event (new instance)
    """
    sanitized = {}

    for key, value in event.items():
        if key == "data" and isinstance(value, dict):
            # Recursively sanitize event data
            sanitized[key] = sanitize_dict(value)
        elif isinstance(value, str):
            sanitized[key] = sanitize_text(value)
        else:
            sanitized[key] = value

    return sanitized


def sanitize_session_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitize session metadata from metadata.json.

    Removes sensitive configuration data.

    Args:
        metadata: Session metadata

    Returns:
        Sanitized metadata (new instance)
    """
    return sanitize_dict(metadata)


def validate_sanitization(original: str, sanitized: str) -> dict[str, Any]:
    """
    Validate that sanitization removed sensitive data.

    Args:
        original: Original text
        sanitized: Sanitized text

    Returns:
        Validation report with detected patterns
    """
    report = {
        "original_length": len(original),
        "sanitized_length": len(sanitized),
        "api_keys_found": 0,
        "secrets_found": 0,
        "emails_found": 0,
        "is_clean": True,
    }

    # Check sanitized text for patterns that should have been removed
    for pattern in API_KEY_PATTERNS:
        if re.search(pattern, sanitized):
            report["api_keys_found"] += 1
            report["is_clean"] = False

    for pattern in SECRET_PATTERNS:
        if re.search(pattern, sanitized):
            report["secrets_found"] += 1
            report["is_clean"] = False

    if re.search(EMAIL_PATTERN, sanitized):
        report["emails_found"] += 1
        report["is_clean"] = False

    return report
