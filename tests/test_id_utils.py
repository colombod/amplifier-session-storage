from unittest.mock import AsyncMock

import pytest

from amplifier_session_storage.backends.base import SyncGapResult
from amplifier_session_storage.id_utils import (
    event_id,
    find_missing_sequences,
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


class TestFindMissingSequences:
    @pytest.mark.asyncio
    async def test_no_gaps(self):
        backend = AsyncMock()
        backend.get_stored_transcript_ids.return_value = ["s1_msg_0", "s1_msg_1", "s1_msg_2"]
        backend.get_stored_event_ids.return_value = ["s1_evt_0", "s1_evt_1"]
        result = await find_missing_sequences(
            backend, "u1", "p1", "s1",
            transcript_line_count=3, event_line_count=2,
        )
        assert result.transcript_missing_sequences == []
        assert result.event_missing_sequences == []
        assert result.transcript_stored_count == 3
        assert result.event_stored_count == 2

    @pytest.mark.asyncio
    async def test_transcript_gaps(self):
        backend = AsyncMock()
        backend.get_stored_transcript_ids.return_value = ["s1_msg_0", "s1_msg_2"]
        result = await find_missing_sequences(
            backend, "u1", "p1", "s1", transcript_line_count=3,
        )
        assert result.transcript_missing_sequences == [1]
        assert result.transcript_stored_count == 2

    @pytest.mark.asyncio
    async def test_event_gaps(self):
        backend = AsyncMock()
        backend.get_stored_event_ids.return_value = ["s1_evt_0"]
        result = await find_missing_sequences(
            backend, "u1", "p1", "s1", event_line_count=3,
        )
        assert result.event_missing_sequences == [1, 2]
        assert result.event_stored_count == 1

    @pytest.mark.asyncio
    async def test_none_counts_skip_query(self):
        backend = AsyncMock()
        result = await find_missing_sequences(backend, "u1", "p1", "s1")
        assert result.transcript_missing_sequences == []
        assert result.event_missing_sequences == []
        assert result.transcript_stored_count == 0
        assert result.event_stored_count == 0
        backend.get_stored_transcript_ids.assert_not_called()
        backend.get_stored_event_ids.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_backend(self):
        backend = AsyncMock()
        backend.get_stored_transcript_ids.return_value = []
        result = await find_missing_sequences(
            backend, "u1", "p1", "s1", transcript_line_count=3,
        )
        assert result.transcript_missing_sequences == [0, 1, 2]
        assert result.transcript_stored_count == 0

    @pytest.mark.asyncio
    async def test_only_transcript_count(self):
        backend = AsyncMock()
        backend.get_stored_transcript_ids.return_value = ["s1_msg_0"]
        result = await find_missing_sequences(
            backend, "u1", "p1", "s1", transcript_line_count=2,
        )
        assert result.transcript_missing_sequences == [1]
        assert result.event_missing_sequences == []
        assert result.event_stored_count == 0
        backend.get_stored_event_ids.assert_not_called()

    @pytest.mark.asyncio
    async def test_only_event_count(self):
        backend = AsyncMock()
        backend.get_stored_event_ids.return_value = ["s1_evt_0"]
        result = await find_missing_sequences(
            backend, "u1", "p1", "s1", event_line_count=2,
        )
        assert result.event_missing_sequences == [1]
        assert result.transcript_missing_sequences == []
        assert result.transcript_stored_count == 0
        backend.get_stored_transcript_ids.assert_not_called()


class TestPublicExports:
    def test_sync_gap_result_importable(self):
        from amplifier_session_storage import SyncGapResult
        assert SyncGapResult is not None

    def test_id_utils_importable(self):
        from amplifier_session_storage import (
            transcript_id,
            event_id,
            parse_transcript_sequence,
            parse_event_sequence,
            find_missing_sequences,
        )
        assert callable(transcript_id)
        assert callable(event_id)
        assert callable(parse_transcript_sequence)
        assert callable(parse_event_sequence)
        assert callable(find_missing_sequences)
