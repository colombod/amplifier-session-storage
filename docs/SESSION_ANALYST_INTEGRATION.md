# Session Analyst Integration Design

## Research Findings

### Current Session Storage Architecture

#### Storage Location
Sessions are stored at: `~/.amplifier/projects/PROJECT_NAME/sessions/SESSION_ID/`

#### Session Files

| File | Purpose | Used for Resume? | Size Risk |
|------|---------|------------------|-----------|
| `metadata.json` | Session metadata (id, created, bundle, model, turn_count) | Yes (info) | Small |
| `transcript.jsonl` | Conversation messages (user, assistant, tool) | **YES - Source of Truth** | Medium |
| `events.jsonl` | Full audit log (API calls, tool executions) | No (debug only) | **HUGE (100k+ tokens/line)** |
| `config.md` | Bundle config snapshot | No | Small |

#### Key Insight: transcript.jsonl is the Source of Truth
When a session resumes, `transcript.jsonl` is loaded to restore conversation context. The `events.jsonl` is purely an audit log and is NOT used for replay.

### Current Implementation Locations

```
amplifier-app-cli
├── session_store.py      # SessionStore class - persistence to filesystem
├── session_runner.py     # Session initialization and lifecycle
└── session_spawner.py    # Sub-session spawning

amplifier-foundation
├── session/
│   ├── events.py         # Events slicing utilities
│   ├── fork.py           # Session fork operations
│   └── slice.py          # Turn boundary detection
└── utilities
    ├── sanitize.py       # sanitize_message, sanitize_for_json
    └── file_ops.py       # write_with_backup
```

### session-analyst Agent Behavior

The session-analyst agent (defined in `amplifier-foundation/agents/session-analyst.md`):

1. **Searches in**: `~/.amplifier/projects/*/sessions/`
2. **Uses tools**: tool-filesystem, tool-search, tool-bash
3. **Reads**: 
   - `metadata.json` for filtering (always safe)
   - `transcript.jsonl` for content search (medium size)
   - `events.jsonl` with **surgical extraction only** (never full lines)
4. **Can repair**: Truncate transcript.jsonl and events.jsonl for rewind operations

### Critical Constraints for Compatibility

1. **File paths must be exact**: `~/.amplifier/projects/{project}/sessions/{session_id}/`
2. **File names must match**: `metadata.json`, `transcript.jsonl`, `events.jsonl`
3. **JSONL format for transcript**: One JSON object per line with `role`, `content`, `timestamp`
4. **JSONL format for events**: One JSON object per line with `ts`, `event`, `session_id`, `data`

---

## Design Decision

### Approach: Unified Session Storage Library

This library (`amplifier-session-storage`) will:

1. **Provide the canonical SessionStore implementation** that matches current app-cli behavior
2. **Maintain exact file format and location** for session-analyst compatibility
3. **Add Cosmos DB sync as a transparent layer** via HybridStorage
4. **Be imported by app-cli** (not the other way around) to avoid circular dependencies

### Dependency Direction

```
amplifier-session-storage (this library)
         ↑
amplifier-app-cli (imports SessionStore from this library)
         ↑
amplifier-foundation (utilities: sanitize_message, write_with_backup)
```

**Note**: This library depends on `amplifier-foundation` for utilities, which is the same dependency app-cli already has.

---

## Implementation Plan

### Phase 1: Compatible Local Storage

Create a `SessionStore` class that is **drop-in compatible** with `amplifier-app-cli/session_store.py`:

```python
from amplifier_session_storage import SessionStore

# Same interface as current app-cli SessionStore
store = SessionStore(base_dir=Path("~/.amplifier/projects/myproj/sessions"))
store.save(session_id, transcript, metadata)
transcript, metadata = store.load(session_id)
```

#### Files to Create/Modify

1. `amplifier_session_storage/local/session_store.py` - Compatible SessionStore
2. `amplifier_session_storage/local/events_log.py` - Events.jsonl writer (audit log)

### Phase 2: Hybrid Storage with Cosmos Sync

Add transparent Cosmos DB sync:

```python
from amplifier_session_storage import HybridSessionStore

# Local + cloud sync
store = HybridSessionStore(
    base_dir=Path("~/.amplifier/projects/myproj/sessions"),
    cosmos_config=cosmos_config,  # Optional - if None, local only
)

# Same interface - writes locally, syncs to cloud in background
store.save(session_id, transcript, metadata)
```

### Phase 3: Block-Based Internal Format

Internally, convert to blocks for efficient sync, but **always write compatible files**:

```
On save():
1. Convert transcript to MESSAGE blocks (internal)
2. Write transcript.jsonl (compatible format)
3. Write metadata.json (compatible format)
4. Sync blocks to Cosmos (if configured)

On load():
1. Read transcript.jsonl (compatible format)
2. Read metadata.json (compatible format)
3. Return standard format to caller
```

---

## API Design

### SessionStore (Drop-in Replacement)

```python
class SessionStore:
    """Drop-in replacement for amplifier-app-cli SessionStore."""
    
    def __init__(self, base_dir: Path | None = None):
        """Initialize with base directory for sessions."""
        
    def save(self, session_id: str, transcript: list, metadata: dict) -> None:
        """Save session state atomically with backup."""
        
    def load(self, session_id: str) -> tuple[list, dict]:
        """Load session state with corruption recovery."""
        
    def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        
    def list_sessions(self, *, top_level_only: bool = True) -> list[str]:
        """List session IDs sorted by modification time."""
        
    def find_session(self, partial_id: str, *, top_level_only: bool = True) -> str:
        """Find session by partial ID prefix."""
        
    def get_metadata(self, session_id: str) -> dict:
        """Get session metadata without loading transcript."""
        
    def update_metadata(self, session_id: str, updates: dict) -> dict:
        """Update specific fields in session metadata."""
        
    def save_config_snapshot(self, session_id: str, config: dict) -> None:
        """Save config snapshot used for session."""
        
    def cleanup_old_sessions(self, days: int = 30) -> int:
        """Remove sessions older than specified days."""
```

### EventsLog (Audit Log Writer)

```python
class EventsLog:
    """Writer for events.jsonl audit log."""
    
    def __init__(self, session_dir: Path):
        """Initialize events log for a session."""
        
    def append(self, event: dict) -> None:
        """Append an event to the log."""
        
    def close(self) -> None:
        """Close the log file."""
```

### HybridSessionStore (With Cloud Sync)

```python
class HybridSessionStore(SessionStore):
    """SessionStore with transparent Cosmos DB sync."""
    
    def __init__(
        self,
        base_dir: Path | None = None,
        cosmos_config: CosmosConfig | None = None,
        sync_enabled: bool = True,
    ):
        """Initialize with optional cloud sync."""
        
    async def sync(self) -> SyncResult:
        """Manually trigger sync to cloud."""
        
    async def pull(self) -> PullResult:
        """Pull changes from cloud."""
```

---

## File Format Specifications

### metadata.json

```json
{
  "session_id": "uuid-string",
  "created": "2025-01-31T12:00:00.000Z",
  "bundle": "bundle:foundation",
  "model": "claude-sonnet-4-20250514",
  "turn_count": 5,
  "name": "Optional session name",
  "parent_id": "parent-session-uuid (if sub-session)"
}
```

### transcript.jsonl

```jsonl
{"role": "user", "content": "Hello", "timestamp": "2025-01-31T12:00:00.000Z"}
{"role": "assistant", "content": "Hi there!", "timestamp": "2025-01-31T12:00:01.000Z", "tool_calls": [...]}
{"role": "tool", "tool_call_id": "call_123", "content": "{...}", "timestamp": "2025-01-31T12:00:02.000Z"}
```

### events.jsonl

```jsonl
{"ts": "2025-01-31T12:00:00.000Z", "lvl": "INFO", "event": "session:start", "session_id": "uuid", "data": {...}}
{"ts": "2025-01-31T12:00:01.000Z", "lvl": "INFO", "event": "llm:request", "session_id": "uuid", "data": {...}}
{"ts": "2025-01-31T12:00:02.000Z", "lvl": "INFO", "event": "llm:response", "session_id": "uuid", "data": {...}}
```

---

## Migration Path for amplifier-app-cli

### Step 1: Add Dependency
```toml
# amplifier-app-cli/pyproject.toml
dependencies = [
    "amplifier-session-storage",
    # ... other deps
]
```

### Step 2: Update Import
```python
# Before (amplifier-app-cli/session_runner.py)
from .session_store import SessionStore

# After
from amplifier_session_storage import SessionStore
```

### Step 3: Enable Cloud Sync (Optional)
```python
# For cloud-enabled installations
from amplifier_session_storage import HybridSessionStore

store = HybridSessionStore(
    cosmos_config=load_cosmos_config_from_settings(),
)
```

---

## session-analyst Compatibility Checklist

- [ ] Files at `~/.amplifier/projects/{project}/sessions/{session_id}/`
- [ ] `metadata.json` with required fields (session_id, created, bundle)
- [ ] `transcript.jsonl` with message objects (role, content, timestamp)
- [ ] `events.jsonl` with event objects (ts, event, session_id, data)
- [ ] Atomic writes with `.backup` files
- [ ] Session ID validation (no path traversal)
- [ ] Top-level vs sub-session detection (`_` in session_id)
- [ ] Modification time preserved for sorting
