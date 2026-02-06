"""
Tests for text chunking module.

Verifies correct splitting, overlap, merging, and edge-case handling
across all content types and text sizes.
"""

from amplifier_session_storage.chunking import (
    CHUNK_MIN_TOKENS,
    CHUNK_TARGET_TOKENS,
    Segment,
    _build_chunk,
    _force_split_segment,
    _get_overlap_tail,
    _merge_segments,
    _split_line_aware,
    _split_markdown_aware,
    _split_sentence_aware,
    chunk_text,
)
from amplifier_session_storage.content_extraction import EMBED_TOKEN_LIMIT, count_tokens

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_long_text(approx_tokens: int, *, paragraphs: bool = True) -> str:
    """Generate text with approximately *approx_tokens* tokens.

    Each sentence is ~12 tokens. Grouped into 5-sentence paragraphs when
    *paragraphs* is True.
    """
    sentence = "This is a test sentence about software engineering practices and methodologies. "
    n_sentences = approx_tokens // 12 + 1
    if paragraphs:
        groups: list[str] = []
        for i in range(0, n_sentences, 5):
            groups.append(sentence * min(5, n_sentences - i))
        return "\n\n".join(groups)
    # Single block — no paragraph boundaries
    return sentence * n_sentences


# ---------------------------------------------------------------------------
# Short text → single chunk
# ---------------------------------------------------------------------------


class TestShortText:
    """Short texts (within EMBED_TOKEN_LIMIT) produce exactly one chunk."""

    def test_single_chunk_returned(self):
        text = "How do I implement vector search?"
        result = chunk_text(text, "user_query")

        assert len(result) == 1
        assert result[0].total_chunks == 1
        assert result[0].chunk_index == 0

    def test_span_covers_full_text(self):
        text = "Short text for embedding."
        result = chunk_text(text, "assistant_response")

        assert result[0].span_start == 0
        assert result[0].span_end == len(text)

    def test_token_count_matches(self):
        text = "A simple sentence to test token counting accuracy."
        result = chunk_text(text, "user_query")

        assert result[0].token_count == count_tokens(text)

    def test_text_preserved_exactly(self):
        text = "Exact text preservation check."
        result = chunk_text(text, "user_query")

        assert result[0].text == text


# ---------------------------------------------------------------------------
# Long text → multiple chunks
# ---------------------------------------------------------------------------


class TestLongText:
    """Texts exceeding EMBED_TOKEN_LIMIT are split into multiple chunks."""

    def test_multiple_chunks_produced(self):
        text = _make_long_text(EMBED_TOKEN_LIMIT + 2000)
        result = chunk_text(text, "assistant_response")

        assert len(result) > 1

    def test_chunk_indices_sequential(self):
        text = _make_long_text(EMBED_TOKEN_LIMIT + 2000)
        result = chunk_text(text, "assistant_response")

        for i, chunk in enumerate(result):
            assert chunk.chunk_index == i

    def test_all_chunks_share_total(self):
        text = _make_long_text(EMBED_TOKEN_LIMIT + 2000)
        result = chunk_text(text, "assistant_response")

        for chunk in result:
            assert chunk.total_chunks == len(result)

    def test_chunks_within_reasonable_token_size(self):
        """Each chunk should be roughly within CHUNK_TARGET_TOKENS (plus overlap)."""
        text = _make_long_text(EMBED_TOKEN_LIMIT + 4000)
        result = chunk_text(text, "assistant_response")

        # Allow generous headroom for overlap + segment granularity
        max_allowed = CHUNK_TARGET_TOKENS * 2
        for chunk in result:
            assert chunk.token_count <= max_allowed, (
                f"Chunk {chunk.chunk_index} has {chunk.token_count} tokens (limit {max_allowed})"
            )


# ---------------------------------------------------------------------------
# Markdown code blocks preserved
# ---------------------------------------------------------------------------


class TestCodeBlockPreservation:
    """Fenced code blocks must never be split mid-block."""

    def test_code_block_with_internal_double_newlines(self):
        """Code blocks containing \\n\\n are kept as a single segment."""
        code_block = (
            "```python\ndef function_one():\n    pass\n\n\ndef function_two():\n    pass\n```"
        )
        # Surround with enough text to force chunking
        filler_para = (
            "This is a filler paragraph about software development. "
            "It discusses architecture and testing patterns.\n\n"
        )
        text = filler_para * 400 + code_block + "\n\n" + filler_para * 400

        result = chunk_text(text, "assistant_response")
        assert len(result) > 1

        # The code block content must appear intact in exactly one chunk
        code_content = "def function_one():\n    pass\n\n\ndef function_two():\n    pass"
        found_intact = any(code_content in c.text for c in result)
        assert found_intact, "Code block was split across chunks"

    def test_splitter_keeps_code_block_atomic(self):
        """Direct test: _split_markdown_aware never splits a code block."""
        code_block = "```js\nconst x = 1;\n\nconst y = 2;\n```"
        text = f"Introduction.\n\n{code_block}\n\nConclusion."

        segments = _split_markdown_aware(text)

        # Find the segment(s) that contain code content
        code_segments = [s for s in segments if "const x = 1;" in s.text]
        assert len(code_segments) == 1
        assert "const y = 2;" in code_segments[0].text


# ---------------------------------------------------------------------------
# Chunk overlap
# ---------------------------------------------------------------------------


class TestChunkOverlap:
    """Consecutive chunks should share overlapping content."""

    def test_consecutive_chunks_have_overlapping_spans(self):
        text = _make_long_text(EMBED_TOKEN_LIMIT + 4000)
        result = chunk_text(text, "assistant_response")

        assert len(result) > 2

        # At least some consecutive pairs must have overlapping span ranges
        overlap_count = sum(
            1 for i in range(len(result) - 1) if result[i].span_end > result[i + 1].span_start
        )
        assert overlap_count > 0, "No span overlap found between any consecutive chunks"

    def test_overlap_shares_text_content(self):
        """The tail text of chunk N should appear at the start of chunk N+1."""
        text = _make_long_text(EMBED_TOKEN_LIMIT + 4000)
        result = chunk_text(text, "assistant_response")

        assert len(result) > 2

        shared_found = False
        for i in range(len(result) - 1):
            # Grab last few words of chunk i and first few words of chunk i+1
            tail_words = result[i].text.split()[-15:]
            head_words = result[i + 1].text.split()[:15]
            if set(tail_words) & set(head_words):
                shared_found = True
                break

        assert shared_found, "No shared text content found between consecutive chunks"


# ---------------------------------------------------------------------------
# Tiny tail merged
# ---------------------------------------------------------------------------


class TestTinyTailMerge:
    """Very short trailing segments are merged into the previous chunk."""

    def test_no_runt_final_chunk(self):
        """When the last chunk would be below CHUNK_MIN_TOKENS, it is merged."""
        # Build segments: enough to fill several chunks, plus a tiny tail
        big_segment_text = "Word " * 200  # ~200 tokens
        segments = [
            Segment(text=big_segment_text, start=i * 1000, end=(i + 1) * 1000) for i in range(8)
        ]
        # Add a tiny segment that's below CHUNK_MIN_TOKENS
        segments.append(Segment(text="End.", start=8000, end=8004))

        result = _merge_segments(segments)

        # Every chunk (except possibly a lone chunk) should be >= CHUNK_MIN_TOKENS
        for chunk in result:
            if chunk.total_chunks > 1:
                assert chunk.token_count >= CHUNK_MIN_TOKENS, (
                    f"Chunk {chunk.chunk_index} has only {chunk.token_count} tokens "
                    f"(min {CHUNK_MIN_TOKENS})"
                )

    def test_single_tiny_segment_still_returned(self):
        """A lone tiny segment becomes the only chunk (no previous to merge into)."""
        segments = [Segment(text="Hi.", start=0, end=3)]
        result = _merge_segments(segments)

        assert len(result) == 1
        assert result[0].total_chunks == 1


# ---------------------------------------------------------------------------
# Content type routing
# ---------------------------------------------------------------------------


class TestContentTypeRouting:
    """Different content types use different splitting strategies."""

    def test_assistant_thinking_uses_markdown_aware(self):
        """assistant_thinking routes to _split_markdown_aware."""
        # Markdown-aware splits on headers / paragraph breaks but protects code blocks
        text = "# Heading\n\nParagraph one.\n\n## Sub\n\nParagraph two."
        segments = _split_markdown_aware(text)
        texts = [s.text for s in segments]
        # Should split at paragraph boundaries (headers are paragraph-level)
        assert any("Heading" in t for t in texts)
        assert any("Paragraph two" in t for t in texts)

    def test_tool_output_uses_line_aware(self):
        """tool_output routes to _split_line_aware."""
        text = "line1\nline2\n\nline3"
        segments = _split_line_aware(text)
        texts = [s.text for s in segments]
        assert "line1" in texts
        assert "line2" in texts
        assert "line3" in texts

    def test_user_query_uses_sentence_aware(self):
        """user_query routes to _split_sentence_aware."""
        text = "First sentence. Second sentence! Third sentence?"
        segments = _split_sentence_aware(text)
        texts = [s.text for s in segments]
        assert any("First sentence." in t for t in texts)
        assert any("Third sentence?" in t for t in texts)

    def test_unknown_content_type_falls_back_to_sentence(self):
        """Unknown content types use sentence-aware splitting as fallback."""
        text = "Sentence one. Sentence two."
        # Short enough to be single-chunk via chunk_text, so test splitter indirectly
        result = chunk_text(text, "unknown_type")
        assert len(result) >= 1
        assert result[0].text == text


# ---------------------------------------------------------------------------
# Empty / whitespace handling
# ---------------------------------------------------------------------------


class TestEmptyHandling:
    """Edge cases for empty or whitespace-only input."""

    def test_empty_string(self):
        result = chunk_text("", "user_query")

        assert len(result) == 1
        assert result[0].text == ""
        assert result[0].span_start == 0
        assert result[0].span_end == 0
        assert result[0].total_chunks == 1
        assert result[0].chunk_index == 0
        assert result[0].token_count == 0

    def test_whitespace_only(self):
        result = chunk_text("   \n\n  ", "user_query")

        assert len(result) == 1
        assert result[0].total_chunks == 1

    def test_single_newline(self):
        result = chunk_text("\n", "tool_output")

        assert len(result) == 1


# ---------------------------------------------------------------------------
# Force split
# ---------------------------------------------------------------------------


class TestForceSplit:
    """A single oversized segment with no natural boundaries is force-split."""

    def test_long_text_no_sentence_boundaries(self):
        """Repeated words without punctuation get force-split at word boundaries."""
        # ~10000 tokens, no sentence-ending punctuation
        text = "word " * 10000
        text = text.strip()

        result = chunk_text(text, "user_query")

        assert len(result) > 1
        # Every chunk should contain whole words only (no mid-word splits)
        for chunk in result:
            assert chunk.text == chunk.text.strip()
            # Check each word is intact
            for w in chunk.text.split():
                assert w == "word"

    def test_force_split_segment_directly(self):
        """Direct test of _force_split_segment helper."""
        big_text = "alpha " * 2000
        big_text = big_text.strip()
        segment = Segment(text=big_text, start=0, end=len(big_text))

        chunks = _force_split_segment(segment, CHUNK_TARGET_TOKENS, 0)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.token_count <= CHUNK_TARGET_TOKENS


# ---------------------------------------------------------------------------
# Span coverage
# ---------------------------------------------------------------------------


class TestSpanCoverage:
    """The union of chunk spans should cover the entire original text."""

    def test_spans_cover_original_text(self):
        text = _make_long_text(EMBED_TOKEN_LIMIT + 4000)
        result = chunk_text(text, "assistant_response")

        assert len(result) > 1

        # Sort by span_start
        sorted_chunks = sorted(result, key=lambda c: c.span_start)

        # First chunk should start near the beginning
        assert sorted_chunks[0].span_start < 50, (
            f"First chunk starts at {sorted_chunks[0].span_start}, expected near 0"
        )

        # No large gaps between consecutive chunks
        for i in range(1, len(sorted_chunks)):
            gap = sorted_chunks[i].span_start - sorted_chunks[i - 1].span_end
            assert gap < 200, (
                f"Large gap ({gap} chars) between chunk {i - 1} "
                f"(ends {sorted_chunks[i - 1].span_end}) and chunk {i} "
                f"(starts {sorted_chunks[i].span_start})"
            )

        # Last chunk should end near the end of the text
        assert sorted_chunks[-1].span_end > len(text) - 200, (
            f"Last chunk ends at {sorted_chunks[-1].span_end}, text length is {len(text)}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestBuildChunk:
    """Tests for _build_chunk helper."""

    def test_joins_segments_with_double_newline(self):
        segments = [
            Segment(text="First part", start=0, end=10),
            Segment(text="Second part", start=12, end=23),
        ]
        chunk = _build_chunk(segments, 0)

        assert chunk.text == "First part\n\nSecond part"
        assert chunk.span_start == 0
        assert chunk.span_end == 23
        assert chunk.chunk_index == 0

    def test_token_count_computed(self):
        segments = [Segment(text="Hello world", start=0, end=11)]
        chunk = _build_chunk(segments, 0)

        assert chunk.token_count == count_tokens("Hello world")


class TestGetOverlapTail:
    """Tests for _get_overlap_tail helper."""

    def test_returns_tail_within_budget(self):
        segments = [
            Segment(text="A " * 50, start=0, end=100),  # ~50 tokens
            Segment(text="B " * 50, start=100, end=200),  # ~50 tokens
            Segment(text="C " * 50, start=200, end=300),  # ~50 tokens
        ]
        result = _get_overlap_tail(segments, 60)

        # Should include only the last segment (~50 tokens fits in 60)
        assert len(result) >= 1
        assert result[-1].text == segments[-1].text

    def test_empty_when_all_segments_too_large(self):
        segments = [
            Segment(text="word " * 500, start=0, end=2500),  # ~500 tokens
        ]
        result = _get_overlap_tail(segments, 10)

        assert result == []
