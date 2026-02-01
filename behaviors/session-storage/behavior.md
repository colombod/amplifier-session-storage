---
bundle:
  name: session-storage
  version: "0.1.0"
  description: |
    Provides safe, structured access to Amplifier session data.
    Use this behavior for agents that need to analyze, search, or manage sessions
    without direct file system access.

context:
  - session-storage:context/session-tool-instructions.md

tools:
  - module: tool-session
    source: git+https://github.com/colombod/amplifier-session-storage
    config:
      enable_cloud: false  # Set to true when Cosmos is configured
      max_results: 50
      max_excerpt_length: 500
---

# Session Storage Behavior

This behavior provides safe session access capabilities for Amplifier agents.

## Capabilities

- **List sessions** - Find sessions by project, date, or other criteria
- **Get session details** - Retrieve metadata, transcript, or events summary
- **Search sessions** - Full-text search across metadata and transcripts
- **Analyze events** - Get event statistics without loading full payloads
- **Rewind sessions** - Safely truncate session history (with preview)

## Safety Guarantees

This behavior ensures agents cannot accidentally overflow their context by:

1. **Never returning full event payloads** - Events can be 100k+ tokens
2. **Paginating large result sets** - Default limit of 50 items
3. **Truncating excerpts** - Search results are length-bounded
4. **Dry-run by default** - Rewind operations preview first

## When to Use

Compose this behavior when your agent needs to:

- Investigate session failures or issues
- Search for past conversations
- Analyze session patterns or errors
- Safely repair/rewind corrupted sessions
- Access session data without file system knowledge

## Migration from File Tools

If your agent previously used file-based tools (tool-filesystem, tool-bash) for
session access, this behavior provides equivalent functionality through the
`session` tool operations:

| Old Approach | New Approach |
|--------------|--------------|
| `grep -r "keyword" ~/.amplifier/projects/` | `session search query="keyword"` |
| `cat metadata.json` | `session get session_id=X` |
| `jq '.event' events.jsonl` | `session get_events session_id=X` |
| `head -n 100 transcript.jsonl` | `session get session_id=X include_transcript=true` |

The new approach is:
- **Safer** - No risk of reading 100k+ token lines
- **Simpler** - No need to know file paths or formats
- **Cloud-ready** - Same API works with Cosmos DB when enabled
