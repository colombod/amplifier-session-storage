"""
Tests for data sanitization utilities.
"""

import json

from amplifier_session_storage.sanitization import (
    sanitize_dict,
    sanitize_event,
    sanitize_text,
    sanitize_transcript_message,
    validate_sanitization,
)


class TestTextSanitization:
    """Tests for basic text sanitization."""

    def test_sanitize_openai_key(self):
        """OpenAI API keys are removed."""
        text = "Use this key: sk-proj-abc123xyz789def456ghi012jkl345mno678"
        result = sanitize_text(text)

        assert "sk-proj-" not in result
        assert "[REDACTED]" in result

    def test_sanitize_anthropic_key(self):
        """Anthropic API keys are removed."""
        text = "export ANTHROPIC_KEY=sk-ant-api03-abcdefghijklmnopqrstuvwxyz123456"
        result = sanitize_text(text)

        assert "sk-ant-api03-" not in result
        assert "[REDACTED]" in result

    def test_sanitize_google_key(self):
        """Google API keys are removed."""
        text = "GOOGLE_API_KEY=AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ123456"
        result = sanitize_text(text)

        assert "AIza" not in result
        assert "[REDACTED]" in result

    def test_sanitize_email(self):
        """Email addresses are removed."""
        text = "Contact me at user@example.com for details"
        result = sanitize_text(text)

        assert "user@example.com" not in result
        assert "[EMAIL_REDACTED]" in result

    def test_sanitize_secret_field(self):
        """Secret fields are redacted."""
        text = 'config = {"api_key": "secret123456789012345678"}'
        result = sanitize_text(text)

        assert "secret123456789012345678" not in result
        assert "[REDACTED]" in result


class TestDictSanitization:
    """Tests for dictionary sanitization."""

    def test_sanitize_nested_dict(self):
        """Nested dictionaries are sanitized recursively."""
        data = {
            "config": {
                "api_key": "sk-proj-abc123xyz789",
                "model": "gpt-4",
            },
            "user": "test@example.com",
        }

        result = sanitize_dict(data)

        assert "sk-proj-" not in json.dumps(result)
        assert "test@example.com" not in json.dumps(result)
        assert "[REDACTED]" in json.dumps(result)
        assert result["config"]["model"] == "gpt-4"  # Non-sensitive preserved

    def test_sanitize_list_in_dict(self):
        """Lists within dicts are sanitized."""
        data = {
            "keys": ["sk-proj-abc123", "sk-ant-api03-xyz789"],
            "models": ["gpt-4", "claude-3"],
        }

        result = sanitize_dict(data)

        assert all("[REDACTED]" in k for k in result["keys"])
        assert result["models"] == ["gpt-4", "claude-3"]  # Non-sensitive preserved


class TestTranscriptSanitization:
    """Tests for transcript message sanitization."""

    def test_sanitize_user_message(self):
        """User message with API key is sanitized."""
        message = {
            "role": "user",
            "content": "Use this key: sk-proj-test123456789",
            "turn": 0,
        }

        result = sanitize_transcript_message(message)

        assert "sk-proj-" not in result["content"]
        assert "[REDACTED]" in result["content"]
        assert result["role"] == "user"

    def test_sanitize_assistant_content_array(self):
        """Assistant message with complex content array."""
        message = {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "I'll use the API key sk-ant-api03-test123 to call the service",
                },
                {
                    "type": "text",
                    "text": "Contact support at support@example.com if needed",
                },
            ],
        }

        result = sanitize_transcript_message(message)

        thinking_text = result["content"][0]["thinking"]
        response_text = result["content"][1]["text"]

        assert "sk-ant-api03-" not in thinking_text
        assert "[REDACTED]" in thinking_text
        assert "support@example.com" not in response_text
        assert "[EMAIL_REDACTED]" in response_text


class TestEventSanitization:
    """Tests for event sanitization."""

    def test_sanitize_event_data(self):
        """Event data field is sanitized."""
        event = {
            "event": "llm.request",
            "ts": "2024-01-15T10:00:00Z",
            "data": {
                "api_key": "sk-test123",
                "model": "gpt-4",
                "user_email": "user@test.com",
            },
        }

        result = sanitize_event(event)

        assert "sk-test123" not in json.dumps(result)
        assert "user@test.com" not in json.dumps(result)
        assert result["data"]["model"] == "gpt-4"  # Non-sensitive preserved


class TestSanitizationValidation:
    """Tests for sanitization validation."""

    def test_validate_clean_text(self):
        """Clean text passes validation."""
        original = "This is clean text with no secrets"
        sanitized = original

        report = validate_sanitization(original, sanitized)

        assert report["is_clean"] is True
        assert report["api_keys_found"] == 0
        assert report["secrets_found"] == 0

    def test_validate_incompletely_sanitized(self):
        """Detects if sanitization missed something."""
        original = "Key: sk-proj-test123"
        sanitized = "Key: sk-proj-test123"  # Not actually sanitized!

        report = validate_sanitization(original, sanitized)

        assert report["is_clean"] is False
        assert report["api_keys_found"] > 0
