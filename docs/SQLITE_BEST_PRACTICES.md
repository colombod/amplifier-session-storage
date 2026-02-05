# SQLite Best Practices for Session Storage

> Research document covering query optimization, data setup, and vector exclusion patterns for the SQLite backend.

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Query Best Practices](#query-best-practices)
3. [Excluding Vectors from Query Results](#excluding-vectors-from-query-results)
4. [Data Setup and Schema Design](#data-setup-and-schema-design)
5. [FTS5 Full-Text Search](#fts5-full-text-search)
6. [Vector Similarity Search](#vector-similarity-search)
7. [Schema Evolution and Migrations](#schema-evolution-and-migrations)
8. [WAL Mode and Concurrency](#wal-mode-and-concurrency)
9. [Implementation Recommendations](#implementation-recommendations)

---

## Executive Summary

SQLite provides lightweight, embedded session storage with excellent reliability. While it lacks native vector indexes, extensions like sqlite-vss/sqlite-vec enable similarity search. Careful schema design and query patterns are essential for performance with large BLOB columns.

**Key Findings:**
- **Explicit column selection** prevents reading large BLOB/vector data unnecessarily
- **Covering indexes** can satisfy queries without table access
- **Views provide stable API** for schema evolution
- **External-content FTS5** enables full-text search without data duplication
- **WAL mode essential** for concurrent access
- **sqlite-vss/sqlite-vec** extensions enable vector search

---

## Query Best Practices

### Explicit Column Selection vs SELECT *

Unlike some databases, SQLite's query planner doesn't significantly change based on column selection. However, **data transfer and memory usage** are dramatically affected when tables contain large BLOBs.

**Performance Impact with Large Columns:**
- `SELECT *` reads and transfers all columns, including large vectors
- Explicit selection skips reading unused BLOB columns
- Covering indexes can eliminate table access entirely

```sql
-- BAD: Reads all columns including ~48KB of vector data per row
SELECT * FROM transcripts WHERE session_id = ?

-- GOOD: Reads only needed columns
SELECT id, session_id, sequence, turn, role, content, ts, embedding_model
FROM transcripts 
WHERE session_id = ?

-- BEST: If all columns are in an index (covering index)
SELECT id, session_id, sequence 
FROM transcripts 
WHERE session_id = ?
-- Shows "USING COVERING INDEX" in EXPLAIN QUERY PLAN
```

### Covering Indexes

Create indexes that include all queried columns to avoid table lookups:

```sql
-- Create covering index for common query pattern
CREATE INDEX idx_transcripts_session_covering 
ON transcripts(session_id, sequence, id, turn, role, ts);

-- This query uses covering index (no table access)
SELECT id, sequence, turn, role, ts 
FROM transcripts 
WHERE session_id = ?
ORDER BY sequence;
```

### Query Plan Analysis

```sql
-- Check query execution plan
EXPLAIN QUERY PLAN
SELECT id, session_id, content, ts 
FROM transcripts 
WHERE session_id = ?;

-- Look for:
-- SEARCH ... USING INDEX ... (good)
-- USING COVERING INDEX ... (best)
-- SCAN ... (bad - full table scan)
```

### BLOB Handling Patterns

When BLOBs must sometimes be accessed:

```sql
-- Get metadata only (fast)
SELECT id, session_id, ts, length(content) AS content_length
FROM transcripts WHERE session_id = ?;

-- Get BLOB preview (moderate)
SELECT id, session_id, substr(content, 1, 1000) AS content_preview
FROM transcripts WHERE session_id = ?;

-- Get full BLOB only when needed (separate query)
SELECT content FROM transcripts WHERE id = ?;
```

---

## Excluding Vectors from Query Results

### Strategy 1: Explicit Column Lists (Most Reliable)

Define column lists in application code:

```python
# Define once, use everywhere
TRANSCRIPT_COLUMNS_NO_VECTORS = """
    id, user_id, session_id, sequence, turn, role, content, ts,
    embedding_model, vector_metadata
"""

# Query without vectors
GET_MESSAGES_SQL = f"""
    SELECT {TRANSCRIPT_COLUMNS_NO_VECTORS}
    FROM transcripts
    WHERE session_id = ?
    ORDER BY sequence
"""

# When vectors needed (explicit request)
GET_MESSAGES_WITH_VECTORS_SQL = """
    SELECT * FROM transcripts
    WHERE session_id = ?
    ORDER BY sequence
"""
```

**Advantages:**
- Complete control over returned data
- Works with all SQLite versions
- Easy to understand and maintain

### Strategy 2: Views as Abstraction Layer (Recommended)

Create views that exclude vector columns:

```sql
-- View for general queries (no vectors)
CREATE VIEW IF NOT EXISTS transcript_messages AS
SELECT 
    id,
    user_id,
    session_id,
    sequence,
    turn,
    role,
    content,
    ts,
    embedding_model,
    vector_metadata
FROM transcripts;

-- Application uses view
SELECT * FROM transcript_messages WHERE session_id = ?;
```

**Advantages:**
- `SELECT *` on view is safe
- Schema changes hidden from application
- Can add computed columns
- Centralized projection logic

**Important:** Views in SQLite are expanded at query time, so performance is equivalent to the underlying query.

### Strategy 3: Vertical Partitioning

Store vectors in a separate table:

```sql
-- Main transcript table (no vectors)
CREATE TABLE transcripts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    turn INTEGER,
    role TEXT NOT NULL,
    content TEXT,
    ts TEXT NOT NULL,
    embedding_model TEXT,
    UNIQUE (user_id, session_id, sequence)
);

-- Separate vector table
CREATE TABLE transcript_vectors (
    transcript_id TEXT PRIMARY KEY REFERENCES transcripts(id),
    user_query_vector BLOB,
    assistant_response_vector BLOB,
    assistant_thinking_vector BLOB,
    tool_output_vector BLOB
);

-- Join only when vectors needed
SELECT t.*, v.user_query_vector
FROM transcripts t
JOIN transcript_vectors v ON t.id = v.transcript_id
WHERE t.session_id = ?;
```

**Advantages:**
- Most queries never touch vector table
- Clear separation of concerns
- Smaller main table = faster scans

**Disadvantages:**
- Requires JOIN for vector operations
- More complex schema
- Must maintain referential integrity

### Strategy 4: Prepared Statements with Column Discipline

Use prepared statements consistently:

```python
import sqlite3

class TranscriptQueries:
    """Prepared statements with explicit columns."""
    
    # Define column groups
    METADATA_COLS = "id, user_id, session_id, sequence, turn, role, ts"
    CONTENT_COLS = "content, embedding_model, vector_metadata"
    VECTOR_COLS = """user_query_vector, assistant_response_vector, 
                     assistant_thinking_vector, tool_output_vector"""
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._prepare_statements()
    
    def _prepare_statements(self):
        # Prepare without vectors (default)
        self._get_messages = self.conn.cursor()
        self._get_messages.execute(f"""
            SELECT {self.METADATA_COLS}, {self.CONTENT_COLS}
            FROM transcripts
            WHERE session_id = ?
            ORDER BY sequence
        """)
        
    def get_messages(self, session_id: str) -> list:
        self._get_messages.execute((session_id,))
        return self._get_messages.fetchall()
```

---

## Data Setup and Schema Design

### Core Schema

```sql
-- Enable WAL mode for concurrent access
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    host_id TEXT,
    project_slug TEXT,
    bundle TEXT,
    created TEXT NOT NULL DEFAULT (datetime('now')),
    updated TEXT NOT NULL DEFAULT (datetime('now')),
    turn_count INTEGER DEFAULT 0,
    message_count INTEGER DEFAULT 0,
    event_count INTEGER DEFAULT 0,
    visibility TEXT DEFAULT 'private',
    tags TEXT,  -- JSON array as text
    PRIMARY KEY (user_id, session_id)
);

-- Transcripts table
CREATE TABLE IF NOT EXISTS transcripts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    turn INTEGER,
    role TEXT NOT NULL,
    content TEXT,
    ts TEXT NOT NULL,
    -- Vector columns as BLOB (for sqlite-vss compatibility)
    user_query_vector BLOB,
    assistant_response_vector BLOB,
    assistant_thinking_vector BLOB,
    tool_output_vector BLOB,
    embedding_model TEXT,
    vector_metadata TEXT,  -- JSON
    UNIQUE (user_id, session_id, sequence)
);

-- Events table
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    ts TEXT NOT NULL,
    turn INTEGER,
    level TEXT DEFAULT 'INFO',
    data TEXT,  -- JSON
    data_truncated INTEGER DEFAULT 0,
    data_size_bytes INTEGER,
    UNIQUE (user_id, session_id, sequence)
);
```

### Index Strategy

```sql
-- Primary access patterns
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(user_id, project_slug);
CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(user_id, updated DESC);

-- Transcript indexes
CREATE INDEX IF NOT EXISTS idx_transcripts_session 
ON transcripts(user_id, session_id);

CREATE INDEX IF NOT EXISTS idx_transcripts_session_seq 
ON transcripts(session_id, sequence);

-- Covering index for common queries (excludes content and vectors)
CREATE INDEX IF NOT EXISTS idx_transcripts_metadata
ON transcripts(session_id, sequence, id, turn, role, ts, embedding_model);

-- Event indexes
CREATE INDEX IF NOT EXISTS idx_events_session ON events(user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(session_id, event_type);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(session_id, ts);
```

---

## FTS5 Full-Text Search

### External-Content FTS5 Table

Use external-content FTS5 to avoid data duplication:

```sql
-- Create FTS5 table linked to transcripts
CREATE VIRTUAL TABLE IF NOT EXISTS transcripts_fts USING fts5(
    content,              -- Message text (searchable)
    session_id UNINDEXED, -- For filtering (not tokenized)
    user_id UNINDEXED,    -- For filtering (not tokenized)
    role UNINDEXED,       -- For filtering (not tokenized)
    content='transcripts',
    content_rowid='rowid',
    tokenize='porter unicode61',
    detail='column'       -- Balance of features vs size
);
```

### FTS5 Configuration Options

| Option | Description | Trade-off |
|--------|-------------|-----------|
| `detail=full` | Supports phrase/proximity queries, snippets | Largest index |
| `detail=column` | Per-column ranking, limited phrase support | Medium index |
| `detail=none` | Only rowid matches, no ranking features | Smallest index |
| `columnsize=1` | Store token counts (faster BM25) | Small overhead |
| `columnsize=0` | Recompute on demand | Slower BM25 |

### Maintaining FTS5 Sync

```sql
-- Triggers to keep FTS in sync with main table
CREATE TRIGGER IF NOT EXISTS transcripts_ai AFTER INSERT ON transcripts BEGIN
    INSERT INTO transcripts_fts(rowid, content, session_id, user_id, role) 
    VALUES (new.rowid, new.content, new.session_id, new.user_id, new.role);
END;

CREATE TRIGGER IF NOT EXISTS transcripts_ad AFTER DELETE ON transcripts BEGIN
    INSERT INTO transcripts_fts(transcripts_fts, rowid, content, session_id, user_id, role) 
    VALUES('delete', old.rowid, old.content, old.session_id, old.user_id, old.role);
END;

CREATE TRIGGER IF NOT EXISTS transcripts_au AFTER UPDATE ON transcripts BEGIN
    INSERT INTO transcripts_fts(transcripts_fts, rowid, content, session_id, user_id, role) 
    VALUES('delete', old.rowid, old.content, old.session_id, old.user_id, old.role);
    INSERT INTO transcripts_fts(rowid, content, session_id, user_id, role) 
    VALUES (new.rowid, new.content, new.session_id, new.user_id, new.role);
END;
```

### FTS5 Query Patterns

```sql
-- Basic search
SELECT t.id, t.session_id, t.content, t.ts
FROM transcripts t
JOIN transcripts_fts fts ON t.rowid = fts.rowid
WHERE transcripts_fts MATCH 'authentication error'
ORDER BY rank;

-- Search with filters
SELECT t.id, t.content, bm25(transcripts_fts) AS score
FROM transcripts t
JOIN transcripts_fts fts ON t.rowid = fts.rowid
WHERE transcripts_fts MATCH 'python AND async'
  AND fts.user_id = ?
ORDER BY score;

-- Snippet extraction
SELECT t.id, snippet(transcripts_fts, 0, '<b>', '</b>', '...', 20) AS snippet
FROM transcripts_fts fts
JOIN transcripts t ON t.rowid = fts.rowid
WHERE transcripts_fts MATCH ?;
```

### FTS5 Maintenance

```sql
-- Optimize index (merge segments)
INSERT INTO transcripts_fts(transcripts_fts) VALUES('optimize');

-- Rebuild if out of sync
INSERT INTO transcripts_fts(transcripts_fts) VALUES('rebuild');

-- Check integrity
INSERT INTO transcripts_fts(transcripts_fts, rank) VALUES('integrity-check', 1);
```

---

## Vector Similarity Search

### Option 1: sqlite-vss Extension (Faiss-based)

```sql
-- Load extensions
.load ./vector0
.load ./vss0

-- Create virtual table for vector index
CREATE VIRTUAL TABLE IF NOT EXISTS vss_transcripts USING vss0(
    user_query_vector(3072),
    assistant_response_vector(3072)
);

-- Index existing vectors
INSERT INTO vss_transcripts(rowid, user_query_vector, assistant_response_vector)
SELECT rowid, user_query_vector, assistant_response_vector
FROM transcripts
WHERE user_query_vector IS NOT NULL;

-- Search
SELECT t.id, t.session_id, t.content, vss.distance
FROM vss_transcripts vss
JOIN transcripts t ON t.rowid = vss.rowid
WHERE vss_search(vss.user_query_vector, ?)
LIMIT 20;
```

**sqlite-vss characteristics:**
- Based on Faiss library
- Optimized for read-heavy workloads
- May rewrite index on updates
- "Bring your own vectors" - works with any embedding

### Option 2: sqlite-vec Extension (Newer)

```sql
-- Load extension
.load ./vec0

-- Create vector table
CREATE VIRTUAL TABLE IF NOT EXISTS vec_transcripts USING vec0(
    embedding FLOAT[3072]
);

-- Insert vectors
INSERT INTO vec_transcripts(rowid, embedding)
SELECT rowid, user_query_vector FROM transcripts
WHERE user_query_vector IS NOT NULL;

-- Search
SELECT t.id, t.content, vec.distance
FROM vec_transcripts vec
JOIN transcripts t ON t.rowid = vec.rowid
WHERE vec.embedding MATCH ?
ORDER BY vec.distance
LIMIT 20;
```

**sqlite-vec characteristics:**
- Faster writes than sqlite-vss
- Better for OLTP workloads
- Updates only affected vectors

### Option 3: Application-Level Vector Search

Without extensions, compute similarity in Python:

```python
import numpy as np
from typing import List, Tuple

def vector_search(
    conn,
    query_vector: np.ndarray,
    user_id: str,
    limit: int = 20
) -> List[Tuple[str, float]]:
    """Brute-force vector search (no index)."""
    
    # Fetch all vectors for user
    cursor = conn.execute("""
        SELECT id, user_query_vector 
        FROM transcripts 
        WHERE user_id = ? AND user_query_vector IS NOT NULL
    """, (user_id,))
    
    results = []
    for row in cursor:
        doc_id, vec_bytes = row
        doc_vector = np.frombuffer(vec_bytes, dtype=np.float32)
        
        # Cosine similarity
        similarity = np.dot(query_vector, doc_vector) / (
            np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
        )
        results.append((doc_id, similarity))
    
    # Sort by similarity descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:limit]
```

**Use when:**
- Small datasets (<10,000 vectors)
- Extensions not available
- Offline/batch processing

---

## Schema Evolution and Migrations

### ALTER TABLE Limitations

SQLite's `ALTER TABLE` is intentionally limited:

| Operation | Support | Notes |
|-----------|---------|-------|
| `ADD COLUMN` | ✅ Full | With DEFAULT or NULL |
| `RENAME TABLE` | ✅ Full | |
| `RENAME COLUMN` | ✅ SQLite 3.25+ | |
| `DROP COLUMN` | ⚠️ SQLite 3.35+ | Fails if column has dependencies |
| `ALTER COLUMN TYPE` | ❌ None | Must recreate table |
| `ADD/DROP CONSTRAINT` | ❌ None | Must recreate table |

### Safe Migration Pattern: Recreate Table

The canonical SQLite migration pattern:

```sql
-- 1. Begin transaction
BEGIN TRANSACTION;

-- 2. Disable foreign keys temporarily
PRAGMA foreign_keys=OFF;

-- 3. Create new table with desired schema
CREATE TABLE transcripts_new (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    -- ... new schema
    new_column TEXT DEFAULT 'default_value',
    UNIQUE (user_id, session_id, sequence)
);

-- 4. Copy data (with any transformations)
INSERT INTO transcripts_new (id, user_id, session_id, sequence, ...)
SELECT id, user_id, session_id, sequence, ...
FROM transcripts;

-- 5. Drop old table
DROP TABLE transcripts;

-- 6. Rename new table
ALTER TABLE transcripts_new RENAME TO transcripts;

-- 7. Recreate indexes
CREATE INDEX idx_transcripts_session ON transcripts(user_id, session_id);
-- ... other indexes

-- 8. Recreate triggers
CREATE TRIGGER transcripts_ai AFTER INSERT ON transcripts BEGIN
    -- ...
END;

-- 9. Re-enable foreign keys
PRAGMA foreign_keys=ON;

-- 10. Verify and commit
PRAGMA foreign_key_check;
COMMIT;
```

### Views for API Stability

Use views to provide stable interface:

```sql
-- Internal table can evolve
CREATE TABLE transcripts_internal (
    id TEXT PRIMARY KEY,
    -- ... actual schema
);

-- Public view provides stable API
CREATE VIEW transcripts AS
SELECT 
    id,
    user_id,
    session_id,
    sequence,
    turn,
    role,
    content,
    ts,
    -- Computed/transformed columns
    COALESCE(embedding_model, 'unknown') AS embedding_model
FROM transcripts_internal;

-- Application always uses view
SELECT * FROM transcripts WHERE session_id = ?;
```

### Migration Tooling

For Python applications, use Alembic's batch mode:

```python
# alembic migration
def upgrade():
    with op.batch_alter_table('transcripts') as batch_op:
        batch_op.add_column(sa.Column('new_field', sa.Text()))
        batch_op.drop_column('old_field')
```

Alembic automatically handles the create-copy-rename pattern for SQLite.

---

## WAL Mode and Concurrency

### Enabling WAL Mode

```sql
-- Enable WAL mode (persistent setting)
PRAGMA journal_mode=WAL;

-- Verify
PRAGMA journal_mode;  -- Should return 'wal'
```

### WAL Mode Benefits

| Feature | Rollback Journal | WAL Mode |
|---------|-----------------|----------|
| Readers block writer | Yes | No |
| Writer blocks readers | Yes | No |
| Concurrent readers | Limited | Many |
| Concurrent writers | No | No (still single) |
| Crash recovery | Slower | Faster |

### Busy Timeout Configuration

```sql
-- Set busy timeout (milliseconds)
PRAGMA busy_timeout=5000;  -- 5 seconds
```

```python
import sqlite3

# In Python
conn = sqlite3.connect('sessions.db', timeout=5.0)
# OR
conn.execute('PRAGMA busy_timeout=5000')
```

### Connection Patterns

```python
import sqlite3
from contextlib import contextmanager
from threading import local

class SQLiteConnectionPool:
    """Thread-local connections with proper WAL configuration."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = local()
    
    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn'):
            conn = sqlite3.connect(
                self.db_path,
                timeout=5.0,
                check_same_thread=False
            )
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA busy_timeout=5000')
            conn.execute('PRAGMA synchronous=NORMAL')  # Faster with WAL
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn
    
    @contextmanager
    def connection(self):
        conn = self._get_connection()
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
```

### Best Practices for Concurrency

1. **Keep transactions short** - Long transactions block other writers
2. **Use WAL mode** - Essential for any concurrent access
3. **Set busy_timeout** - Prevent immediate failures on contention
4. **Don't upgrade read → write** - Causes deadlocks; start with write intent
5. **One connection per thread** - SQLite connections aren't fully thread-safe

```python
# BAD: Read then upgrade to write
conn.execute("SELECT * FROM sessions WHERE id = ?", (id,))
# ... later ...
conn.execute("UPDATE sessions SET ...")  # May deadlock

# GOOD: Start transaction with write intent
conn.execute("BEGIN IMMEDIATE")
conn.execute("SELECT * FROM sessions WHERE id = ?", (id,))
conn.execute("UPDATE sessions SET ...")
conn.commit()
```

---

## Implementation Recommendations

### 1. Initialization Script

```python
def initialize_sqlite(db_path: str) -> sqlite3.Connection:
    """Initialize SQLite with optimal settings."""
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    
    # Optimal pragmas
    pragmas = [
        "PRAGMA journal_mode=WAL",
        "PRAGMA busy_timeout=5000",
        "PRAGMA synchronous=NORMAL",
        "PRAGMA cache_size=-64000",  # 64MB cache
        "PRAGMA foreign_keys=ON",
        "PRAGMA temp_store=MEMORY",
    ]
    
    for pragma in pragmas:
        conn.execute(pragma)
    
    # Create tables
    conn.executescript(SCHEMA_SQL)
    
    # Create views
    conn.execute("""
        CREATE VIEW IF NOT EXISTS transcript_messages AS
        SELECT id, user_id, session_id, sequence, turn, role, 
               content, ts, embedding_model, vector_metadata
        FROM transcripts
    """)
    
    return conn
```

### 2. Query Layer with Views

```python
class SQLiteQueryLayer:
    """Query layer using views for vector exclusion."""
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._ensure_views()
    
    def _ensure_views(self):
        self.conn.execute("""
            CREATE VIEW IF NOT EXISTS transcript_messages AS
            SELECT id, user_id, session_id, sequence, turn, role,
                   content, ts, embedding_model, vector_metadata
            FROM transcripts
        """)
    
    def get_session_messages(self, user_id: str, session_id: str) -> list:
        """Get messages without vectors."""
        cursor = self.conn.execute("""
            SELECT * FROM transcript_messages
            WHERE user_id = ? AND session_id = ?
            ORDER BY sequence
        """, (user_id, session_id))
        return [dict(row) for row in cursor]
    
    def search_content(self, user_id: str, query: str, limit: int = 20) -> list:
        """Full-text search using FTS5."""
        cursor = self.conn.execute("""
            SELECT t.id, t.session_id, t.content, t.ts,
                   bm25(transcripts_fts) AS score
            FROM transcripts t
            JOIN transcripts_fts fts ON t.rowid = fts.rowid
            WHERE transcripts_fts MATCH ?
              AND fts.user_id = ?
            ORDER BY score
            LIMIT ?
        """, (query, user_id, limit))
        return [dict(row) for row in cursor]
```

### 3. Vector Storage Helper

```python
import struct
from typing import Optional
import numpy as np

def encode_vector(vector: list[float]) -> bytes:
    """Encode float vector as BLOB."""
    return struct.pack(f'{len(vector)}f', *vector)

def decode_vector(blob: bytes) -> np.ndarray:
    """Decode BLOB to numpy array."""
    return np.frombuffer(blob, dtype=np.float32)

def store_transcript_with_vectors(
    conn: sqlite3.Connection,
    transcript: dict,
    vectors: dict[str, Optional[list[float]]]
):
    """Store transcript with encoded vectors."""
    conn.execute("""
        INSERT INTO transcripts (
            id, user_id, session_id, sequence, turn, role, content, ts,
            user_query_vector, assistant_response_vector,
            assistant_thinking_vector, tool_output_vector,
            embedding_model
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        transcript['id'],
        transcript['user_id'],
        transcript['session_id'],
        transcript['sequence'],
        transcript.get('turn'),
        transcript['role'],
        transcript.get('content'),
        transcript['ts'],
        encode_vector(vectors['user_query']) if vectors.get('user_query') else None,
        encode_vector(vectors['assistant_response']) if vectors.get('assistant_response') else None,
        encode_vector(vectors['assistant_thinking']) if vectors.get('assistant_thinking') else None,
        encode_vector(vectors['tool_output']) if vectors.get('tool_output') else None,
        transcript.get('embedding_model'),
    ))
```

### 4. Best Practices Checklist

- [ ] Enable WAL mode (`PRAGMA journal_mode=WAL`)
- [ ] Set busy_timeout (`PRAGMA busy_timeout=5000`)
- [ ] Use explicit column selection (never `SELECT *` with vectors)
- [ ] Create views as abstraction layer
- [ ] Create covering indexes for common queries
- [ ] Use external-content FTS5 for full-text search
- [ ] Keep transactions short
- [ ] Use `BEGIN IMMEDIATE` when write intent is known
- [ ] Consider vertical partitioning for large vectors
- [ ] Use migrate-via-recreate pattern for schema changes
- [ ] Profile with `EXPLAIN QUERY PLAN`

---

## References

- [SQLite Query Planning](https://sqlite.org/queryplanner.html)
- [SQLite FTS5 Extension](https://www.sqlite.org/fts5.html)
- [SQLite ALTER TABLE](https://www.sqlite.org/lang_altertable.html)
- [Internal vs External BLOBs](https://www.sqlite.org/intern-v-extern-blob.html)
- [sqlite-vss Extension](https://github.com/asg017/sqlite-vss)
- [sqlite-vec Extension](https://github.com/asg017/sqlite-vec)
- [Alembic Batch Migrations](https://alembic.sqlalchemy.org/en/latest/batch.html)
- [SQLite WAL Mode](https://www.sqlite.org/wal.html)
