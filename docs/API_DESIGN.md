# API Design: Enhanced Discovery and Navigation

## Overview

This document outlines the API enhancements for the storage backend to support:
1. Discovery APIs (find users, sessions, projects)
2. Optional filtering (user_id optional for team-wide queries)
3. Sequence-based navigation (when turn is null)

## Current State Analysis

### What Exists

| API | user_id | Description |
|-----|---------|-------------|
| `search_sessions()` | Required | Search sessions with filters |
| `search_transcripts()` | Required | Search transcript content |
| `get_turn_context()` | Required | Get context around a turn |
| `get_session_statistics()` | Required | Aggregate statistics |

### What's Missing

1. **Discovery APIs**
   - `list_users()` - Get all unique user IDs
   - `list_projects()` - Get all unique project slugs
   - `list_sessions()` - Simple session listing without search

2. **Optional user_id**
   - All search APIs should accept empty user_id for team-wide queries

3. **Sequence-based navigation**
   - `get_message_context()` - Navigate by sequence when turn is null

## Proposed API Additions

### 1. Discovery APIs

```python
async def list_users(
    self,
    filters: SearchFilters | None = None,
) -> list[str]:
    """
    List all unique user IDs in the storage.
    
    Args:
        filters: Optional filters (project_slug, date range)
    
    Returns:
        List of unique user IDs
    """
    pass

async def list_projects(
    self,
    user_id: str = "",  # Empty = all users
    filters: SearchFilters | None = None,
) -> list[str]:
    """
    List all unique project slugs.
    
    Args:
        user_id: Filter by user (empty = all users)
        filters: Optional filters (date range, bundle)
    
    Returns:
        List of unique project slugs
    """
    pass

async def list_sessions(
    self,
    user_id: str = "",  # Empty = all users
    project_slug: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """
    List sessions with pagination.
    
    Simpler than search_sessions - just lists recent sessions.
    
    Args:
        user_id: Filter by user (empty = all users)
        project_slug: Filter by project
        limit: Max results
        offset: Pagination offset
    
    Returns:
        List of session metadata dicts
    """
    pass
```

### 2. Sequence-Based Navigation

```python
@dataclass
class MessageContext:
    """
    Context window around a specific message by sequence.
    
    Used when turn is null or for precise sequence-based navigation.
    """
    session_id: str
    project_slug: str
    target_sequence: int
    
    # Messages around the target
    previous: list[TranscriptMessage]  # Messages before (oldest first)
    current: TranscriptMessage | None  # The target message
    following: list[TranscriptMessage]  # Messages after (oldest first)
    
    # Navigation metadata
    has_more_before: bool
    has_more_after: bool
    first_sequence: int
    last_sequence: int
    
    @property
    def total_messages(self) -> int:
        count = len(self.previous) + len(self.following)
        return count + 1 if self.current else count

async def get_message_context(
    self,
    user_id: str = "",  # Empty = search all users
    session_id: str,
    sequence: int,
    before: int = 5,
    after: int = 5,
    include_tool_outputs: bool = True,
) -> MessageContext:
    """
    Get context window around a specific message by sequence.
    
    Use this when:
    - Turn is null in the transcript
    - You need precise sequence-based navigation
    - You want to expand search results by sequence
    
    Args:
        user_id: User ID (empty = search all users for session)
        session_id: Session identifier
        sequence: Target sequence number
        before: Messages to include before
        after: Messages to include after
        include_tool_outputs: Include tool outputs
    
    Returns:
        MessageContext with surrounding messages
    """
    pass
```

### 3. Enhanced Existing APIs (Optional user_id)

The following APIs already support empty user_id after our previous changes:
- `search_transcripts()` - ✅ Updated
- `vector_search()` - ✅ Updated

Need to update:
- `search_sessions()` - Allow empty user_id
- `search_events()` - Allow empty user_id
- `get_session_statistics()` - Allow empty user_id
- `get_turn_context()` - Allow empty user_id (find session across all users)

## Data Model Considerations

### Transcripts Container Schema

```json
{
  "id": "unique-doc-id",
  "user_id": "SC-dc174",
  "session_id": "abc123",
  "project_slug": "my-project",
  "sequence": 42,
  "turn": 5,           // CAN BE NULL
  "role": "assistant",
  "content": "...",
  "ts": "2024-01-15T10:30:00Z",
  
  // Vector fields (optional)
  "user_query_vector": [...],
  "assistant_response_vector": [...],
  "assistant_thinking_vector": [...],
  "tool_output_vector": [...]
}
```

**Key insight**: `turn` CAN BE NULL. When null:
- Use `sequence` for ordering
- Use `sequence` for context navigation
- Don't rely on turn-based grouping

### Sessions Container Schema

```json
{
  "id": "unique-doc-id",
  "user_id": "SC-dc174",
  "session_id": "abc123",
  "project_slug": "my-project",
  "bundle": "foundation",
  "created": "2024-01-15T10:00:00Z",
  "updated": "2024-01-15T12:00:00Z",
  "turn_count": 15,
  "message_count": 45
}
```

## Implementation Priority

1. **High Priority** (Core discovery)
   - `list_users()` - Essential for team-wide queries
   - `list_projects()` - Essential for filtering
   - `get_message_context()` - Essential when turn is null

2. **Medium Priority** (Enhanced filtering)
   - Update `search_sessions()` for optional user_id
   - Update `get_turn_context()` for optional user_id
   - `list_sessions()` - Simpler listing API

3. **Lower Priority** (Convenience)
   - Update `search_events()` for optional user_id
   - Update remaining APIs

## Testing Strategy

Each backend (Cosmos, DuckDB, SQLite) needs tests for:

1. **Discovery Tests**
   - `test_list_users_returns_unique_users`
   - `test_list_users_with_project_filter`
   - `test_list_projects_all_users`
   - `test_list_projects_specific_user`
   - `test_list_sessions_pagination`

2. **Optional User ID Tests**
   - `test_search_sessions_empty_user_id`
   - `test_search_transcripts_empty_user_id`
   - `test_get_turn_context_empty_user_id`

3. **Sequence Navigation Tests**
   - `test_get_message_context_by_sequence`
   - `test_get_message_context_when_turn_null`
   - `test_get_message_context_boundary_conditions`

4. **Integration Tests**
   - `test_end_to_end_discovery_workflow`
   - `test_team_wide_search_workflow`
