"""
Tests for content extraction utilities.

Verifies correct extraction from real Amplifier message structures.
"""

from amplifier_session_storage.content_extraction import (
    count_embeddable_content_types,
    extract_all_embeddable_content,
)


class TestUserMessageExtraction:
    """Tests for user message content extraction."""

    def test_simple_user_message(self):
        """User message with simple string content."""
        message = {
            "role": "user",
            "content": "How do I implement vector search?",
            "turn": 0,
        }

        result = extract_all_embeddable_content(message)

        assert result["user_query"] == "How do I implement vector search?"
        assert result["assistant_response"] is None
        assert result["assistant_thinking"] is None
        assert result["tool_output"] is None

    def test_empty_user_content(self):
        """User message with empty content."""
        message = {"role": "user", "content": ""}

        result = extract_all_embeddable_content(message)

        assert result["user_query"] is None  # Empty -> None


class TestAssistantMessageExtraction:
    """Tests for assistant message content extraction."""

    def test_assistant_with_thinking_and_text(self):
        """Assistant message with both thinking and text blocks (real structure)."""
        message = {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "The user wants to understand vector search. I should explain embeddings and similarity metrics.",
                },
                {
                    "type": "text",
                    "text": "Vector search uses embeddings to find semantically similar content.",
                },
                {"type": "tool_call", "id": "toolu_123", "name": "bash", "input": {}},
            ],
        }

        result = extract_all_embeddable_content(message)

        assert result["user_query"] is None
        assert (
            result["assistant_response"]
            == "Vector search uses embeddings to find semantically similar content."
        )
        assert "The user wants to understand vector search" in result["assistant_thinking"]
        assert result["tool_output"] is None

    def test_assistant_with_only_thinking(self):
        """Assistant message with only thinking block (no text response yet)."""
        message = {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "I need to analyze the codebase first before responding.",
                }
            ],
        }

        result = extract_all_embeddable_content(message)

        assert result["assistant_thinking"] is not None
        thinking = result["assistant_thinking"]
        assert thinking is not None
        assert "analyze the codebase" in thinking
        assert result["assistant_response"] is None  # No text blocks

    def test_assistant_with_only_text(self):
        """Assistant message with only text block (no thinking)."""
        message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "Here's the answer you need."}],
        }

        result = extract_all_embeddable_content(message)

        assert result["assistant_response"] == "Here's the answer you need."
        assert result["assistant_thinking"] is None  # No thinking blocks

    def test_assistant_with_multiple_thinking_blocks(self):
        """Multiple thinking blocks are combined."""
        message = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "First thought process."},
                {"type": "thinking", "thinking": "Second thought process."},
                {"type": "text", "text": "Final answer."},
            ],
        }

        result = extract_all_embeddable_content(message)

        thinking = result["assistant_thinking"]
        assert thinking is not None
        assert "First thought process" in thinking
        assert "Second thought process" in thinking
        assert "\n\n" in thinking  # Joined with double newline

    def test_assistant_simple_string_content(self):
        """Backward compatibility: assistant with simple string content."""
        message = {"role": "assistant", "content": "Simple string response"}

        result = extract_all_embeddable_content(message)

        assert result["assistant_response"] == "Simple string response"
        assert result["assistant_thinking"] is None


class TestToolMessageExtraction:
    """Tests for tool message content extraction."""

    def test_tool_message_string_content(self):
        """Tool message with string output."""
        message = {
            "role": "tool",
            "content": "File contents:\n\nclass Example:\n    pass",
        }

        result = extract_all_embeddable_content(message)

        assert result["user_query"] is None
        assert result["assistant_response"] is None
        assert result["assistant_thinking"] is None
        tool_out = result["tool_output"]
        assert tool_out is not None
        assert "File contents" in tool_out

    def test_tool_message_large_output_truncation(self):
        """Large tool outputs are truncated."""
        large_content = "x" * 50000  # 50KB
        message = {"role": "tool", "content": large_content}

        result = extract_all_embeddable_content(message)

        assert result["tool_output"] is not None
        assert len(result["tool_output"]) == 10000  # Truncated to 10KB

    def test_tool_message_structured_content(self):
        """Tool message with structured content (dict/list)."""
        message = {"role": "tool", "content": {"result": "success", "data": [1, 2, 3]}}

        result = extract_all_embeddable_content(message)

        # Should be stringified
        assert result["tool_output"] is not None
        assert isinstance(result["tool_output"], str)


class TestRealAmplifierMessages:
    """Tests with actual Amplifier message structures."""

    def test_real_user_message(self):
        """Real user message from Amplifier session."""
        message = {
            "role": "user",
            "content": "Create a comprehensive smoke test for the lol-coaching bundle with SST knowledge graph",
            "turn": None,
            "timestamp": None,
        }

        result = extract_all_embeddable_content(message)

        assert result["user_query"] is not None
        assert "smoke test" in result["user_query"]
        assert "lol-coaching" in result["user_query"]

    def test_real_assistant_message_with_thinking(self):
        """Real assistant message with thinking and text blocks."""
        message = {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "The user wants me to create a comprehensive smoke test. This involves: 1. Creating shadow environment 2. Running tests 3. Validation.",
                    "signature": "...",
                },
                {
                    "type": "text",
                    "text": "I'll create a comprehensive smoke test for the lol-coaching bundle.",
                },
                {
                    "type": "tool_call",
                    "id": "toolu_01abc",
                    "name": "todo",
                    "input": {"action": "create"},
                },
            ],
            "turn": None,
        }

        result = extract_all_embeddable_content(message)

        thinking = result["assistant_thinking"]
        assert thinking is not None
        assert "shadow environment" in thinking
        assert "Running tests" in thinking

        response = result["assistant_response"]
        assert response is not None
        assert "comprehensive smoke test" in response

        # Tool calls not included
        assert "todo" not in thinking
        assert "tool_call" not in response


class TestContentCounting:
    """Tests for content type counting utility."""

    def test_count_mixed_messages(self):
        """Count embeddings needed for mixed message types."""
        messages = [
            {"role": "user", "content": "Question 1"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Reasoning..."},
                    {"type": "text", "text": "Response"},
                ],
            },
            {"role": "tool", "content": "Tool output"},
            {"role": "user", "content": "Question 2"},
        ]

        counts = count_embeddable_content_types(messages)

        assert counts["user_query"] == 2  # 2 user messages
        assert counts["assistant_response"] == 1  # 1 assistant text block
        assert counts["assistant_thinking"] == 1  # 1 thinking block
        assert counts["tool_output"] == 1  # 1 tool message

    def test_count_empty_content(self):
        """Empty content not counted."""
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": []},
        ]

        counts = count_embeddable_content_types(messages)

        assert counts["user_query"] == 0  # Empty content
        assert counts["assistant_response"] == 0
        assert counts["assistant_thinking"] == 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_role(self):
        """Message without role field."""
        message = {"content": "Some content"}

        result = extract_all_embeddable_content(message)

        # Should return empty (all None)
        assert all(v is None for v in result.values())

    def test_missing_content(self):
        """Message without content field."""
        message = {"role": "user"}

        result = extract_all_embeddable_content(message)

        assert result["user_query"] is None  # No content

    def test_none_content(self):
        """Message with None content."""
        message = {"role": "user", "content": None}

        result = extract_all_embeddable_content(message)

        assert result["user_query"] is None

    def test_assistant_empty_content_array(self):
        """Assistant with empty content array."""
        message = {"role": "assistant", "content": []}

        result = extract_all_embeddable_content(message)

        assert result["assistant_response"] is None
        assert result["assistant_thinking"] is None
