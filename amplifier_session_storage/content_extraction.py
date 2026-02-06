"""
Content extraction utilities for embedding generation.

Handles complex Amplifier transcript message structures:
- User messages: Simple string content
- Assistant messages: Complex content arrays (thinking, text, tool_call blocks)
- Tool messages: String or structured content

Extracts separate text for each embedding vector type.
"""

from __future__ import annotations

import logging
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)

# Embedding model token limit (text-embedding-3-large uses cl100k_base encoding)
EMBED_TOKEN_LIMIT = 8192
_TIKTOKEN_ENCODING = "cl100k_base"

# Lazy-initialized encoder (avoid import-time cost)
_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    """Get or create the tiktoken encoder (lazy singleton)."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding(_TIKTOKEN_ENCODING)
    return _encoder


def count_tokens(text: str) -> int:
    """Count tokens in text using the embedding model's tokenizer.

    Uses cl100k_base encoding (text-embedding-3-large, GPT-4, GPT-3.5-turbo).

    Args:
        text: The text to count tokens for.

    Returns:
        Number of tokens.
    """
    return len(_get_encoder().encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to fit within a token limit.

    Decodes back to a valid string after truncating the token sequence.

    Args:
        text: The text to truncate.
        max_tokens: Maximum number of tokens allowed.

    Returns:
        The truncated text (or original if already within limit).
    """
    encoder = _get_encoder()
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoder.decode(tokens[:max_tokens])


def extract_all_embeddable_content(message: dict[str, Any]) -> dict[str, str | None]:
    """
    Extract all embeddable content from a transcript message.

    Returns dict with keys for each vector type:
        - user_query: User questions/requests
        - assistant_response: Assistant text shown to user
        - assistant_thinking: Assistant internal reasoning
        - tool_output: Tool execution results

    Only the relevant key(s) for the message role will be populated.
    Others will be None.

    Args:
        message: Transcript message from JSONL

    Returns:
        Dict mapping vector type to extracted text (or None)

    Examples:
        >>> extract_all_embeddable_content({"role": "user", "content": "Hello"})
        {"user_query": "Hello", "assistant_response": None, ...}

        >>> extract_all_embeddable_content({
        ...     "role": "assistant",
        ...     "content": [
        ...         {"type": "thinking", "thinking": "User needs help..."},
        ...         {"type": "text", "text": "I can help with that"}
        ...     ]
        ... })
        {"user_query": None, "assistant_response": "I can help with that",
         "assistant_thinking": "User needs help...", "tool_output": None}
    """
    role = message.get("role")

    if role == "user":
        return _extract_user_content(message)
    elif role == "assistant":
        return _extract_assistant_content(message)
    elif role == "tool":
        return _extract_tool_content(message)
    else:
        logger.warning(f"Unknown role: {role}, skipping embedding extraction")
        return _empty_content()


def _empty_content() -> dict[str, str | None]:
    """Return empty content dict (all None)."""
    return {
        "user_query": None,
        "assistant_response": None,
        "assistant_thinking": None,
        "tool_output": None,
    }


def _extract_user_content(message: dict[str, Any]) -> dict[str, str | None]:
    """
    Extract content from user message.

    User messages have simple string content.
    """
    content = message.get("content", "")

    return {
        "user_query": content if content else None,
        "assistant_response": None,
        "assistant_thinking": None,
        "tool_output": None,
    }


def _extract_assistant_content(message: dict[str, Any]) -> dict[str, str | None]:
    """
    Extract content from assistant message.

    Assistant messages have complex content arrays with:
    - type: "thinking" -> thinking field (internal reasoning)
    - type: "text" -> text field (user-visible response)
    - type: "tool_call" -> Skip (not embeddable content)

    Returns separate text for thinking and response vectors.
    """
    content = message.get("content", "")

    # Handle simple string content (backward compatibility)
    if isinstance(content, str):
        return {
            "user_query": None,
            "assistant_response": content if content else None,
            "assistant_thinking": None,
            "tool_output": None,
        }

    # Handle complex content array (standard Amplifier format)
    if isinstance(content, list):
        thinking_parts: list[str] = []
        text_parts: list[str] = []

        for block in content:
            block_type = block.get("type")

            if block_type == "thinking":
                thinking_text = block.get("thinking", "")
                if thinking_text:
                    thinking_parts.append(thinking_text)

            elif block_type == "text":
                text_content = block.get("text", "")
                if text_content:
                    text_parts.append(text_content)

            # Skip tool_call blocks - these are metadata, not searchable content

        return {
            "user_query": None,
            "assistant_response": "\n\n".join(text_parts) if text_parts else None,
            "assistant_thinking": "\n\n".join(thinking_parts) if thinking_parts else None,
            "tool_output": None,
        }

    # Fallback for unknown content structure
    logger.warning(f"Unknown assistant content type: {type(content)}")
    return _empty_content()


def _extract_tool_content(message: dict[str, Any]) -> dict[str, str | None]:
    """
    Extract content from tool message.

    Tool messages have string content (tool output).
    May be very large (file contents, command output), so truncate.
    """
    content = message.get("content", "")

    # Convert to string if not already
    if not isinstance(content, str):
        content = str(content)

    # Truncate very large tool outputs (embeddings are expensive!)
    MAX_TOOL_OUTPUT_LENGTH = 10000

    if len(content) > MAX_TOOL_OUTPUT_LENGTH:
        content = content[:MAX_TOOL_OUTPUT_LENGTH]
        logger.debug(
            f"Truncated tool output from {len(message.get('content', ''))} "
            f"to {MAX_TOOL_OUTPUT_LENGTH} chars for embedding"
        )

    return {
        "user_query": None,
        "assistant_response": None,
        "assistant_thinking": None,
        "tool_output": content if content else None,
    }


def count_embeddable_content_types(messages: list[dict[str, Any]]) -> dict[str, int]:
    """
    Count how many of each content type will be embedded.

    Useful for cost estimation and batch size planning.

    Args:
        messages: List of transcript messages

    Returns:
        Dict with counts: {user_query: N, assistant_response: N, ...}
    """
    counts = {
        "user_query": 0,
        "assistant_response": 0,
        "assistant_thinking": 0,
        "tool_output": 0,
    }

    for message in messages:
        extracted = extract_all_embeddable_content(message)
        for key, value in extracted.items():
            if value:
                counts[key] += 1

    return counts
