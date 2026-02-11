import pytest

from amplifier_session_storage.id_utils import (
    event_id,
    parse_event_sequence,
    parse_transcript_sequence,
    transcript_id,
)


class TestTranscriptId:
    def test_basic(self):
        assert transcript_id("abc123", 1) == "abc123_msg_1"

    def test_large_sequence(self):
        assert transcript_id("abc123", 500) == "abc123_msg_500"


class TestEventId:
    def test_basic(self):
        assert event_id("abc123", 1) == "abc123_evt_1"

    def test_large_sequence(self):
        assert event_id("abc123", 500) == "abc123_evt_500"


class TestParseTranscriptSequence:
    def test_basic(self):
        assert parse_transcript_sequence("abc123_msg_42") == 42

    def test_complex_session_id(self):
        assert parse_transcript_sequence("abc-123-def_msg_1") == 1

    def test_malformed_raises(self):
        with pytest.raises(ValueError):
            parse_transcript_sequence("not_a_valid_id")

    def test_missing_sequence_raises(self):
        with pytest.raises(ValueError):
            parse_transcript_sequence("abc123_msg_")


class TestParseEventSequence:
    def test_basic(self):
        assert parse_event_sequence("abc123_evt_42") == 42

    def test_malformed_raises(self):
        with pytest.raises(ValueError):
            parse_event_sequence("not_a_valid_id")
