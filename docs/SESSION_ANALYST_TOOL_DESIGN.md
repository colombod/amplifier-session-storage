# Session Analyst Tool Design

## Problem Statement

The session-analyst agent currently uses file-based tools to access session data:
- `tool-filesystem` for reading files
- `tool-search` for grep operations  
- `tool-bash` for jq/sed surgical extraction

This approach has several limitations:
1. **No cloud support** - Can only access local files
2. **Safety risks** - Must manually avoid reading large events.jsonl lines
3. **Multi-machine blind** - Can't see sessions from other devices
4. **No abstraction** - Agent must know file paths and formats

## Solution: Dedicated Session Tool

Provide a `tool-session` module that:
1. Abstracts storage backend (local files OR Cosmos DB)
2. Provides safe operations (never returns full events.jsonl lines)
3. Enables cross-device session access when connected
4. Maintains all current capabilities (search, read, repair)

## Tool Operations

### 1. List Sessions

```yaml
operation: list
parameters:
  project: string (optional) - Filter by project slug
  date_range: string (optional) - "today", "last_week", "2025-01-01:2025-01-31"
  top_level_only: bool (default: true) - Exclude spawned sub-sessions
  include_cloud: bool (default: true) - Include sessions from cloud
  limit: int (default: 50) - Max sessions to return
  
returns:
  sessions:
    - session_id: string
      project: string
      created: ISO timestamp
      modified: ISO timestamp
      bundle: string
      model: string
      turn_count: int
      source: "local" | "cloud" | "both"
      name: string (optional)
```

### 2. Get Session Details

```yaml
operation: get
parameters:
  session_id: string - Full or partial session ID
  include_transcript: bool (default: false) - Include conversation
  include_events_summary: bool (default: false) - Include events stats
  
returns:
  session_id: string
  project: string
  metadata: {...}  # Full metadata.json content
  transcript: [...] (if requested)  # Full transcript messages
  events_summary: {...} (if requested)  # Event counts, timestamps, errors
  source: "local" | "cloud"
  path: string (if local)  # For manual inspection
```

### 3. Search Sessions

```yaml
operation: search
parameters:
  query: string - Search term
  scope: "metadata" | "transcript" | "all" (default: "all")
  project: string (optional) - Limit to project
  date_range: string (optional)
  limit: int (default: 20)
  context_lines: int (default: 2) - Lines around matches
  
returns:
  matches:
    - session_id: string
      project: string
      created: ISO timestamp
      match_type: "metadata" | "transcript"
      excerpt: string  # Surrounding context
      line_number: int (if transcript match)
```

### 4. Get Events (Safe)

```yaml
operation: get_events
parameters:
  session_id: string
  event_types: list[string] (optional) - Filter by type
  fields: list[string] (optional) - Only return these fields
  limit: int (default: 100)
  offset: int (default: 0)
  errors_only: bool (default: false)
  
returns:
  events:
    - ts: ISO timestamp
      event: string
      # Only requested fields, NEVER full data payload
  total_count: int
  has_more: bool
```

**Safety**: This operation NEVER returns the full `data` field. Agent must specify which fields to extract.

### 5. Analyze Events

```yaml
operation: analyze_events
parameters:
  session_id: string
  analysis_type: "summary" | "errors" | "timeline" | "usage"
  
returns:
  # For summary:
  total_events: int
  event_types: {type: count}
  duration_ms: int
  first_event: ISO timestamp
  last_event: ISO timestamp
  
  # For errors:
  errors: [{ts, event, message (truncated)}]
  
  # For timeline:
  turns: [{turn_num, user_ts, assistant_ts, tool_calls: int}]
  
  # For usage:
  llm_requests: int
  total_input_tokens: int
  total_output_tokens: int
  tool_calls: int
```

### 6. Rewind Session

```yaml
operation: rewind
parameters:
  session_id: string
  to_turn: int (optional) - Rewind to after this turn number
  to_message: int (optional) - Rewind to after this message index
  before_timestamp: string (optional) - Rewind to before this time
  dry_run: bool (default: true) - Preview only, don't modify
  
returns:
  would_remove:
    messages: int
    events: int
  backup_created: bool (if not dry_run)
  new_turn_count: int
```

### 7. Sync Status

```yaml
operation: sync_status
parameters:
  session_id: string (optional) - Specific session, or all
  
returns:
  connected: bool
  last_sync: ISO timestamp
  pending_uploads: int
  pending_downloads: int
  sessions:  # If specific session requested
    - session_id: string
      local_sequence: int
      cloud_sequence: int
      status: "synced" | "ahead" | "behind" | "conflict"
```

## Implementation Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    tool-session                              │
│  (Amplifier Tool Module - amplifier-module-tool-session)    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐      │
│   │    list     │   │    get      │   │   search    │      │
│   │  sessions   │   │  session    │   │  sessions   │      │
│   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘      │
│          │                 │                  │              │
│   ┌──────┴─────────────────┴──────────────────┴──────┐      │
│   │              SessionAnalyst                       │      │
│   │         (from amplifier-session-storage)         │      │
│   └──────────────────────┬───────────────────────────┘      │
│                          │                                   │
│   ┌──────────────────────┴───────────────────────────┐      │
│   │              HybridBlockStorage                   │      │
│   │         (local + cloud transparent)              │      │
│   └─────────────┬─────────────────────┬──────────────┘      │
│                 │                     │                      │
│   ┌─────────────┴─────────┐ ┌────────┴──────────────┐      │
│   │   LocalBlockStorage   │ │   CosmosBlockStorage  │      │
│   │   (SessionStore +     │ │   (cloud sync)        │      │
│   │    block format)      │ │                       │      │
│   └───────────────────────┘ └───────────────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Session Analyst Behavior Update

The session-analyst agent definition will be updated to:

1. **Remove file tools** - No more tool-filesystem, tool-bash
2. **Add tool-session** - Single tool for all session operations
3. **Update instructions** - Use tool operations instead of file commands

### Before (Current)

```yaml
tools:
  - module: tool-filesystem
  - module: tool-search
  - module: tool-bash
```

```markdown
# Instructions
Use grep -n events.jsonl | cut -d: -f1 to safely extract...
```

### After (New)

```yaml
tools:
  - module: tool-session
    source: git+https://github.com/colombod/amplifier-session-storage
```

```markdown
# Instructions
Use the session tool's operations:
- `list` to find sessions
- `search` to find by content
- `get_events` with specific fields for safe extraction
- `analyze_events` for summaries
```

## Safety Guarantees

1. **No full events.jsonl lines** - `get_events` requires field specification
2. **Truncated excerpts** - Search results have length limits
3. **Pagination** - Large result sets are paginated
4. **Dry-run repairs** - Rewind defaults to preview mode

## Migration Path

### Phase 1: Parallel Support
- Keep file-based tools available
- Add tool-session as additional option
- Session-analyst can use either

### Phase 2: Tool Preference  
- Update session-analyst to prefer tool-session
- Fall back to file tools if tool-session unavailable
- Gradual migration

### Phase 3: Tool Only
- Remove file tools from session-analyst
- tool-session is the only interface
- File access still possible via path in `get` response

## Configuration

The tool reads configuration from:
1. Environment variables (COSMOS_ENDPOINT, etc.)
2. `~/.amplifier/settings.yaml` (cosmos section)
3. Tool initialization parameters

```yaml
# ~/.amplifier/settings.yaml
session_storage:
  cosmos:
    endpoint: https://xxx.cosmos.azure.com
    database: amplifier-sessions
    auth: managed_identity  # or connection_string
  sync:
    enabled: true
    auto_sync: true
    interval_seconds: 60
```

## Open Questions

1. **Should search work offline?** - Yes for local sessions, cloud search requires connection
2. **How to handle conflicts?** - Show both versions, let agent/user decide
3. **Should rewind sync to cloud?** - Yes, rewind creates a REWIND block that syncs
4. **Rate limiting?** - Cloud operations may need throttling for large searches
