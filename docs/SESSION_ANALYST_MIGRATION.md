# Session Analyst Migration Guide

This document describes how to migrate the session-analyst agent from file-based tools to the new `session` tool provided by amplifier-session-storage.

## Overview

The session-analyst agent currently uses file-based tools (tool-filesystem, tool-bash) to access session data. This approach has limitations:

1. **Context overflow risk**: `events.jsonl` lines can be 100k+ tokens
2. **File path knowledge required**: Agent must know Amplifier's directory structure
3. **No cloud support**: Cannot access sessions stored in Cosmos DB
4. **Safety concerns**: Raw file access can accidentally read massive payloads

The new `session` tool provides:

1. **Safe projections**: Never returns full event payloads
2. **Abstracted storage**: Works with local files or Cosmos DB
3. **Structured operations**: Purpose-built for session analysis
4. **Pagination**: Large result sets are bounded

## Migration Strategy

### Phase 1: Add Behavior (Non-Breaking)

Add the `session-storage` behavior to session-analyst's includes:

```yaml
# In session-analyst's bundle definition
includes:
  - session-storage  # Add this line
```

This makes the `session` tool available alongside existing file tools.

### Phase 2: Update Context (Non-Breaking)

Update session-analyst's context to prefer the `session` tool:

```markdown
# In session-analyst's context file

## Session Access

**Preferred**: Use the `session` tool for all session operations.

**Fallback**: If `session` tool is not available, use file-based access
with these safety measures:
- Never read full events.jsonl lines
- Use `grep -n ... | cut -d: -f1` for line numbers only
- Use `jq -c '{field}'` to extract specific fields
```

### Phase 3: Remove File Dependencies (Breaking)

Once all users have updated, remove file tool dependencies:

```yaml
# Remove from session-analyst's tools
tools:
  # - tool-filesystem  # Remove
  # - tool-bash        # Remove (or keep for other purposes)
  - session-storage    # Keep
```

## Operation Mapping

### List Sessions

**Old (file-based):**
```bash
ls -la ~/.amplifier/projects/*/sessions/
find ~/.amplifier/projects -name "metadata.json" -exec dirname {} \;
```

**New (session tool):**
```
session list_sessions [project=<slug>] [date_range=<range>] [limit=<n>]
```

### Get Session Metadata

**Old (file-based):**
```bash
cat ~/.amplifier/projects/<project>/sessions/<session_id>/metadata.json
```

**New (session tool):**
```
session get_session session_id=<id>
```

### Get Transcript

**Old (file-based):**
```bash
cat ~/.amplifier/projects/<project>/sessions/<session_id>/transcript.jsonl
```

**New (session tool):**
```
session get_session session_id=<id> include_transcript=true
```

### Search Sessions

**Old (file-based):**
```bash
grep -r "keyword" ~/.amplifier/projects/*/sessions/*/transcript.jsonl
```

**New (session tool):**
```
session search_sessions query="keyword" scope=transcript
```

### Get Events (CRITICAL CHANGE)

**Old (file-based) - DANGEROUS:**
```bash
# This can output 100k+ tokens per line!
cat ~/.amplifier/projects/<project>/sessions/<session_id>/events.jsonl
```

**New (session tool) - SAFE:**
```
session get_events session_id=<id> [event_types=<list>] [limit=<n>]
```

The new approach **never returns full event payloads**. It returns summaries:
- Timestamp
- Event type
- Turn number
- Error flag

### Analyze Events

**Old (file-based):**
```bash
# Count events by type
jq -r '.event' events.jsonl | sort | uniq -c

# Find errors
grep '"lvl":"ERROR"' events.jsonl
```

**New (session tool):**
```
session analyze_events session_id=<id> analysis_type=summary
session analyze_events session_id=<id> analysis_type=errors
```

### Rewind Session

**Old (file-based):**
```bash
# Dangerous manual truncation
head -n 10 transcript.jsonl > transcript.jsonl.new
mv transcript.jsonl.new transcript.jsonl
```

**New (session tool):**
```
# Preview first (dry_run=true is default)
session rewind_session session_id=<id> to_turn=5

# Execute
session rewind_session session_id=<id> to_turn=5 dry_run=false
```

## Backward Compatibility

### Detection Pattern

Session-analyst can detect which approach to use:

```markdown
# In context instructions

## Session Access Strategy

1. **Check for session tool**: Try `session list_sessions limit=1`
2. **If available**: Use session tool for all operations
3. **If not available**: Fall back to file-based access with safety measures

### File-Based Fallback Safety

When using file tools, NEVER:
- Read full lines from events.jsonl
- Use `cat` on events.jsonl
- Pipe events.jsonl content to other commands

Instead:
- Use `grep -n ... | cut -d: -f1` for line numbers
- Use `jq -c '{ts, event}'` to extract only needed fields
- Use `head -n 1` and `tail -n 1` for specific lines
```

### Gradual Rollout

The session tool can be added gradually:

1. **Week 1**: Add behavior to session-analyst (available but not required)
2. **Week 2**: Update context to prefer session tool
3. **Week 3**: Monitor usage, gather feedback
4. **Week 4**: Consider removing file tool dependencies

## Testing the Migration

### Verify Tool Availability

```python
# In session-analyst tests
def test_session_tool_available():
    """Session tool should be available when behavior is composed."""
    # Tool should respond to list_sessions
    result = session.execute(operation="list_sessions", limit=1)
    assert "sessions" in result
```

### Verify Operation Parity

```python
def test_list_sessions_parity():
    """Session tool lists same sessions as file discovery."""
    # File-based count
    file_count = len(glob("~/.amplifier/projects/*/sessions/*/metadata.json"))
    
    # Tool-based count
    result = session.execute(operation="list_sessions")
    tool_count = result["count"]
    
    assert file_count == tool_count
```

### Verify Safety

```python
def test_events_never_overflow():
    """get_events should never return large payloads."""
    result = session.execute(
        operation="get_events",
        session_id=session_with_large_events,
    )
    
    # Result should be small regardless of event sizes
    assert len(json.dumps(result)) < 50000  # 50KB max
```

## Cloud Support (Future)

When Cosmos DB support is enabled, the same operations work transparently:

```yaml
# In behavior configuration
tools:
  - module: tool-session
    config:
      enable_cloud: true  # Enable Cosmos DB
```

Session-analyst doesn't need to change its operations - the tool handles:
- Querying local and cloud sessions
- Merging results
- User isolation enforcement
- Sync status reporting

## Rollback Plan

If issues are discovered:

1. **Immediate**: Users can remove `session-storage` from includes
2. **Session-analyst**: Revert context to prefer file-based access
3. **No data loss**: All session data remains in local files

## Timeline

| Phase | Duration | Change Type | Risk |
|-------|----------|-------------|------|
| Add behavior | 1 week | Non-breaking | Low |
| Update context | 1 week | Non-breaking | Low |
| Deprecation notice | 2 weeks | Non-breaking | Low |
| Remove file deps | 1 week | Breaking | Medium |

Total migration: ~5 weeks with safety margins.

## Questions?

For questions about this migration:
1. Check the behavior documentation in `behaviors/session-storage/`
2. Review the tool implementation in `amplifier_session_storage/tools/`
3. Run the test suite: `pytest tests/test_session_tool.py -v`
