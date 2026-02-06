"""
Text chunking for embedding generation.

Splits text into semantically coherent chunks for embedding. Short texts
(within the embedding token limit) produce a single chunk. Long texts are
split at structural boundaries (markdown headers, paragraph breaks, sentence
boundaries) and produce multiple chunks with overlap for continuity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .content_extraction import EMBED_TOKEN_LIMIT, count_tokens

# ---------------------------------------------------------------------------
# Chunking parameters
# ---------------------------------------------------------------------------

CHUNK_TARGET_TOKENS = 1024  # Target tokens per chunk
CHUNK_OVERLAP_TOKENS = 128  # Overlap between adjacent chunks
CHUNK_MIN_TOKENS = 64  # Don't create tiny chunks

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    """An atomic text segment (sentence, paragraph, code block, etc.)."""

    text: str
    start: int  # Character offset in original text
    end: int  # Character offset in original text


@dataclass
class ChunkResult:
    """A chunk of text ready for embedding."""

    text: str  # The chunk text
    span_start: int  # Character offset start in original text
    span_end: int  # Character offset end in original text
    chunk_index: int  # 0-based position in chunk sequence
    total_chunks: int  # Total chunks (set after all created)
    token_count: int  # Tokens in this chunk


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def chunk_text(text: str, content_type: str) -> list[ChunkResult]:
    """Chunk text for embedding. Always returns at least one ChunkResult.

    Short texts (within EMBED_TOKEN_LIMIT) produce a single chunk spanning
    the full text. Long texts are split at structural boundaries with overlap
    between chunks.

    Args:
        text: The text to chunk.
        content_type: One of "user_query", "assistant_response",
                      "assistant_thinking", "tool_output". Affects splitting
                      strategy.

    Returns:
        List of ChunkResult, always at least one.
    """
    # Handle empty / whitespace-only text
    if not text or not text.strip():
        return [
            ChunkResult(
                text=text or "",
                span_start=0,
                span_end=len(text) if text else 0,
                chunk_index=0,
                total_chunks=1,
                token_count=count_tokens(text) if text else 0,
            )
        ]

    # Short text: single chunk
    token_count = count_tokens(text)
    if token_count <= EMBED_TOKEN_LIMIT:
        return [
            ChunkResult(
                text=text,
                span_start=0,
                span_end=len(text),
                chunk_index=0,
                total_chunks=1,
                token_count=token_count,
            )
        ]

    # Long text: split based on content type
    if content_type in ("assistant_thinking", "assistant_response"):
        segments = _split_markdown_aware(text)
    elif content_type == "tool_output":
        segments = _split_line_aware(text)
    else:  # user_query and fallback
        segments = _split_sentence_aware(text)

    # Filter whitespace-only segments
    segments = [s for s in segments if s.text.strip()]

    if not segments:
        return [
            ChunkResult(
                text=text,
                span_start=0,
                span_end=len(text),
                chunk_index=0,
                total_chunks=1,
                token_count=count_tokens(text),
            )
        ]

    return _merge_segments(segments)


# ---------------------------------------------------------------------------
# Splitting strategies
# ---------------------------------------------------------------------------

# Regex for fenced code blocks: opening ``` at start of line, content, closing
# ``` at start of line.  MULTILINE makes ^ match line starts, DOTALL lets .
# match newlines so the lazy .*? can span multiple lines.
_CODE_BLOCK_RE = re.compile(r"^```[^\n]*\n.*?^```[^\n]*$", re.MULTILINE | re.DOTALL)


def _split_markdown_aware(text: str) -> list[Segment]:
    """Split markdown text preserving code blocks as atomic segments.

    1. Identify fenced code blocks and treat them as atomic (never split).
    2. Split remaining text on paragraph boundaries (double newlines).
    3. If a paragraph still exceeds CHUNK_TARGET_TOKENS, split at sentences.
    """
    segments: list[Segment] = []

    # Build alternating (non-code, code) regions
    regions: list[tuple[str, int, int, bool]] = []  # (text, start, end, is_code)
    last_end = 0

    for match in _CODE_BLOCK_RE.finditer(text):
        if match.start() > last_end:
            regions.append((text[last_end : match.start()], last_end, match.start(), False))
        regions.append((text[match.start() : match.end()], match.start(), match.end(), True))
        last_end = match.end()

    if last_end < len(text):
        regions.append((text[last_end:], last_end, len(text), False))

    if not regions:
        regions = [(text, 0, len(text), False)]

    for region_text, region_start, region_end, is_code in regions:
        if is_code:
            # Code blocks are atomic — never split inside
            segments.append(Segment(text=region_text, start=region_start, end=region_end))
        else:
            # Split non-code text on paragraph boundaries (\n\n+)
            parts = _split_preserving_positions(region_text, r"\n\n+", region_start)
            for part_text, start, end in parts:
                if not part_text.strip():
                    continue

                if count_tokens(part_text) <= CHUNK_TARGET_TOKENS:
                    segments.append(Segment(text=part_text, start=start, end=end))
                else:
                    # Oversized paragraph: split at sentence boundaries
                    sub_segments = _split_sentence_aware_with_offsets(part_text, start)
                    segments.extend(sub_segments)

    return segments


def _split_line_aware(text: str) -> list[Segment]:
    """Split at newline boundaries for tool output."""
    segments: list[Segment] = []
    pos = 0
    for line in text.split("\n"):
        end = pos + len(line)
        if line.strip():  # Skip empty lines
            segments.append(Segment(text=line, start=pos, end=end))
        pos = end + 1  # +1 for the \n
    return segments


def _split_sentence_aware(text: str) -> list[Segment]:
    """Split at sentence boundaries."""
    return _split_sentence_aware_with_offsets(text, 0)


# ---------------------------------------------------------------------------
# Position-preserving helpers
# ---------------------------------------------------------------------------


def _split_preserving_positions(
    text: str, pattern: str, base_offset: int = 0
) -> list[tuple[str, int, int]]:
    """Split *text* on *pattern* while tracking character positions.

    Returns a list of (part_text, start, end) tuples where *start* and *end*
    are character offsets relative to the original document (adjusted by
    *base_offset*).
    """
    parts: list[tuple[str, int, int]] = []
    last_end = 0

    for match in re.finditer(pattern, text):
        if match.start() > last_end:
            part_text = text[last_end : match.start()]
            parts.append((part_text, base_offset + last_end, base_offset + match.start()))
        last_end = match.end()

    if last_end < len(text):
        parts.append((text[last_end:], base_offset + last_end, base_offset + len(text)))

    # If the pattern never matched, return the whole text as one part
    if not parts and text:
        parts.append((text, base_offset, base_offset + len(text)))

    return parts


def _split_sentence_aware_with_offsets(text: str, base_offset: int) -> list[Segment]:
    """Split text at sentence boundaries with offset tracking."""
    # Split after sentence-ending punctuation followed by whitespace
    pattern = r"(?<=[.!?])\s+"
    parts = _split_preserving_positions(text, pattern, base_offset)
    segments: list[Segment] = []
    for part_text, start, end in parts:
        if part_text.strip():
            segments.append(Segment(text=part_text, start=start, end=end))
    return segments


# ---------------------------------------------------------------------------
# Segment merging with overlap
# ---------------------------------------------------------------------------


def _merge_segments(segments: list[Segment]) -> list[ChunkResult]:
    """Merge small segments into chunks up to CHUNK_TARGET_TOKENS with overlap."""
    chunks: list[ChunkResult] = []
    current_segments: list[Segment] = []
    current_tokens = 0

    for segment in segments:
        seg_tokens = count_tokens(segment.text)

        # Oversized single segment: force-split at word boundaries
        if seg_tokens > CHUNK_TARGET_TOKENS:
            # Flush current accumulator
            if current_segments:
                chunks.append(_build_chunk(current_segments, len(chunks)))
                current_segments = []
                current_tokens = 0
            # Force-split the oversized segment
            sub_chunks = _force_split_segment(segment, CHUNK_TARGET_TOKENS, len(chunks))
            chunks.extend(sub_chunks)
            continue

        # Would exceed target: flush and start new chunk with overlap
        if current_tokens + seg_tokens > CHUNK_TARGET_TOKENS and current_segments:
            chunk = _build_chunk(current_segments, len(chunks))
            chunks.append(chunk)

            # Overlap: carry tail segments from previous chunk
            overlap_segs = _get_overlap_tail(current_segments, CHUNK_OVERLAP_TOKENS)
            current_segments = overlap_segs + [segment]
            current_tokens = sum(count_tokens(s.text) for s in current_segments)
            continue

        current_segments.append(segment)
        current_tokens += seg_tokens

    # Flush remaining segments
    if current_segments:
        if current_tokens < CHUNK_MIN_TOKENS and chunks:
            # Tiny tail: extend previous chunk instead of creating a runt
            chunks[-1] = _extend_chunk(chunks[-1], current_segments)
        else:
            chunks.append(_build_chunk(current_segments, len(chunks)))

    # Set total_chunks on every result
    for chunk in chunks:
        chunk.total_chunks = len(chunks)

    return chunks


# ---------------------------------------------------------------------------
# Chunk construction helpers
# ---------------------------------------------------------------------------


def _build_chunk(segments: list[Segment], index: int) -> ChunkResult:
    """Build a ChunkResult from a list of segments."""
    text = "\n\n".join(s.text for s in segments)
    return ChunkResult(
        text=text,
        span_start=segments[0].start,
        span_end=segments[-1].end,
        chunk_index=index,
        total_chunks=0,  # Set later
        token_count=count_tokens(text),
    )


def _extend_chunk(chunk: ChunkResult, extra_segments: list[Segment]) -> ChunkResult:
    """Extend an existing chunk with additional segments."""
    extra_text = "\n\n".join(s.text for s in extra_segments)
    new_text = chunk.text + "\n\n" + extra_text
    return ChunkResult(
        text=new_text,
        span_start=chunk.span_start,
        span_end=extra_segments[-1].end,
        chunk_index=chunk.chunk_index,
        total_chunks=0,
        token_count=count_tokens(new_text),
    )


def _get_overlap_tail(segments: list[Segment], target_tokens: int) -> list[Segment]:
    """Get tail segments that fit within *target_tokens* for overlap."""
    result: list[Segment] = []
    tokens = 0
    for seg in reversed(segments):
        seg_tokens = count_tokens(seg.text)
        if tokens + seg_tokens > target_tokens:
            break
        result.insert(0, seg)
        tokens += seg_tokens
    return result


def _force_split_segment(segment: Segment, max_tokens: int, start_index: int) -> list[ChunkResult]:
    """Force-split an oversized segment at word boundaries."""
    words = segment.text.split()
    chunks: list[ChunkResult] = []
    current_words: list[str] = []
    current_char_start = segment.start
    idx = start_index

    for word in words:
        test_text = " ".join(current_words + [word])
        if count_tokens(test_text) > max_tokens and current_words:
            chunk_txt = " ".join(current_words)
            chunks.append(
                ChunkResult(
                    text=chunk_txt,
                    span_start=current_char_start,
                    span_end=current_char_start + len(chunk_txt),
                    chunk_index=idx,
                    total_chunks=0,
                    token_count=count_tokens(chunk_txt),
                )
            )
            idx += 1
            current_char_start = current_char_start + len(chunk_txt) + 1  # +1 for space
            current_words = [word]
        else:
            current_words.append(word)

    # Flush remaining words
    if current_words:
        chunk_txt = " ".join(current_words)
        chunks.append(
            ChunkResult(
                text=chunk_txt,
                span_start=current_char_start,
                span_end=current_char_start + len(chunk_txt),
                chunk_index=idx,
                total_chunks=0,
                token_count=count_tokens(chunk_txt),
            )
        )

    return chunks
