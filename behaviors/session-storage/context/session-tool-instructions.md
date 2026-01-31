# Session Tool Instructions

You have access to the `session` tool for safe, structured access to Amplifier session data.

## Why Use This Tool

**CRITICAL**: Amplifier session files (especially `events.jsonl`) can contain lines with 100k+ tokens. Standard file tools (grep, cat, read_file) that output full lines will:
- Overflow your context window
- Cause session failures
- Miss important data due to truncation

The `session` tool provides **safe projections** that never return full event payloads.

## Available Operations

### list_sessions

List sessions with optional filtering.

```
session list_sessions [project=<slug>] [date_range=<range>] [limit=<n>]
```

| Parameter | Description | Example |
|-----------|-------------|---------|
| `project` | Filter by project slug | `project=default` |
| `date_range` | Filter by date | `today`, `last_week`, `2025-01-01:2025-01-31` |
| `limit` | Max results (default 50) | `limit=10` |

**Returns**: List of session summaries with ID, project, dates, bundle, model, turn count.

### get_session

Get detailed information about a session.

```
session get_session session_id=<id> [include_transcript=true] [include_events_summary=true]
```

| Parameter | Description |
|-----------|-------------|
| `session_id` | Full or unique partial session ID |
| `include_transcript` | Include conversation messages |
| `include_events_summary` | Include event statistics |

**Returns**: Session metadata, optionally with transcript and events summary.

### search_sessions

Search sessions for matching content.

```
session search_sessions query=<text> [scope=<where>] [limit=<n>]
```

| Parameter | Description | Values |
|-----------|-------------|--------|
| `query` | Search term (case-insensitive) | Any text |
| `scope` | Where to search | `metadata`, `transcript`, `all` (default) |
| `limit` | Max results | Default 50 |

**Returns**: List of matches with session ID, match type, and truncated excerpt.

### get_events

Get event summaries from a session. **SAFE** - never returns full payloads.

```
session get_events session_id=<id> [event_types=<list>] [errors_only=true] [limit=<n>] [offset=<n>]
```

| Parameter | Description |
|-----------|-------------|
| `session_id` | Session ID |
| `event_types` | Filter by types (e.g., `["llm:request", "tool:call"]`) |
| `errors_only` | Only return error events |
| `limit` | Max events (default 100) |
| `offset` | Skip first N events (for pagination) |

**Returns**: Event summaries (timestamp, type, turn, has_error) with pagination info.

**IMPORTANT**: This method returns **summaries only**, not full event data. This is intentional to prevent context overflow.

### analyze_events

Analyze session events without loading full data.

```
session analyze_events session_id=<id> [analysis_type=<type>]
```

| Analysis Type | Returns |
|---------------|---------|
| `summary` | Total events, types breakdown, duration, LLM/tool counts |
| `errors` | List of errors with truncated messages |
| `timeline` | First/last event timestamps, duration |
| `usage` | LLM request count, tool call count |

**Returns**: Analysis results appropriate to the requested type.

### rewind_session

Preview or execute a session rewind.

```
session rewind_session session_id=<id> [to_turn=<n>] [to_message=<n>] [dry_run=true]
```

| Parameter | Description |
|-----------|-------------|
| `session_id` | Session ID |
| `to_turn` | Rewind to after this turn number |
| `to_message` | Rewind to after this message index |
| `dry_run` | Preview only (default: true) |

**Returns**: Preview of what would be removed (messages, events, new turn count).

**SAFETY**: Defaults to dry_run=true. Always preview before executing.

## Common Patterns

### Investigating a Failed Session

```
# 1. Get session overview
session get_session session_id=<id> include_events_summary=true

# 2. Find errors
session get_events session_id=<id> errors_only=true

# 3. Analyze error patterns
session analyze_events session_id=<id> analysis_type=errors
```

### Finding Past Conversations

```
# Search by topic
session search_sessions query="authentication" scope=transcript

# Search by date
session list_sessions date_range=last_week
```

### Repairing a Corrupted Session

```
# 1. Preview what would be removed
session rewind_session session_id=<id> to_turn=5 dry_run=true

# 2. If preview looks correct, execute
session rewind_session session_id=<id> to_turn=5 dry_run=false
```

## What This Tool Does NOT Provide

- **Full event data**: Use `get_events` for summaries. If you need full event data for a specific event, that requires special handling.
- **File system paths**: The tool abstracts storage. Use the `path` field in results if you need to point users to local files.
- **Real-time updates**: The tool reads current state. For live sessions, data may change.

## Backward Compatibility

This tool is designed to work alongside existing file-based approaches. If this tool is not available, fall back to file tools with these safety measures:

1. **Never read full events.jsonl lines**: Use `grep -n ... | cut -d: -f1` for line numbers only
2. **Extract specific fields**: Use `jq -c '{small_field}'` to extract only needed fields
3. **Paginate manually**: Use `head`, `tail`, `sed` to read specific line ranges

When this tool IS available, prefer it over file-based approaches for safety and simplicity.
