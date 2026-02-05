# DuckDB Best Practices for Session Storage

> Research document covering query optimization, data setup, and vector exclusion patterns for the DuckDB backend.

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Query Best Practices](#query-best-practices)
3. [Excluding Vectors from Query Results](#excluding-vectors-from-query-results)
4. [Data Setup and Schema Design](#data-setup-and-schema-design)
5. [VSS HNSW Index Configuration](#vss-hnsw-index-configuration)
6. [Schema Evolution and Migrations](#schema-evolution-and-migrations)
7. [Concurrency and Connection Patterns](#concurrency-and-connection-patterns)
8. [Implementation Recommendations](#implementation-recommendations)

---

## Executive Summary

DuckDB provides excellent analytical performance for session storage with native vector search via the VSS extension. Its columnar storage model makes column projection especially impactful for performance.

**Key Findings:**
- **Column projection matters significantly** in DuckDB due to columnar storage
- **`SELECT * EXCLUDE (...)` syntax** provides elegant vector exclusion
- **Views as abstraction layers** protect against schema changes
- **HNSW indexes** require careful configuration and persistence caveats
- **Single-writer model** requires connection management strategy

---

## Query Best Practices

### Explicit Column Selection vs SELECT *

DuckDB's columnar storage means unused columns add measurable overhead:

**Performance Impact:**
- Issue #14729 shows `SELECT *` can be significantly slower than explicit column selection
- Runtime grows roughly linearly with the number of extra columns
- Large ARRAY columns (embeddings) amplify this effect

```sql
-- BAD: Reads all columns including large vector arrays
SELECT * FROM transcripts WHERE session_id = ?

-- GOOD: Explicit column list
SELECT id, session_id, sequence, turn, role, content, ts, embedding_model
FROM transcripts 
WHERE session_id = ?
```

### DuckDB Star Expression Variants

DuckDB provides powerful star expression syntax:

```sql
-- Basic star (includes everything - avoid with vectors)
SELECT * FROM transcripts

-- EXCLUDE: Omit specific columns
SELECT * EXCLUDE (user_query_vector, assistant_response_vector, 
                  assistant_thinking_vector, tool_output_vector)
FROM transcripts WHERE session_id = ?

-- REPLACE: Transform columns inline
SELECT * REPLACE (length(content) AS content)
FROM transcripts WHERE session_id = ?

-- COLUMNS with regex: Select matching columns only
SELECT COLUMNS('^(id|session_id|sequence|turn|role|content|ts)$')
FROM transcripts WHERE session_id = ?

-- Combine patterns
SELECT * EXCLUDE (user_query_vector), 
       array_cosine_distance(user_query_vector, ?) AS similarity
FROM transcripts
WHERE user_query_vector IS NOT NULL
ORDER BY similarity
LIMIT 10
```

### Query Plan Analysis

```sql
-- Analyze query execution
EXPLAIN ANALYZE 
SELECT id, session_id, content, ts 
FROM transcripts 
WHERE session_id = ?;

-- Check if indexes are used
EXPLAIN 
SELECT * FROM transcripts WHERE user_id = ? AND session_id = ?;
```

---

## Excluding Vectors from Query Results

### Strategy 1: SELECT * EXCLUDE (Recommended for Ad-hoc Queries)

The most DuckDB-idiomatic approach:

```sql
-- Exclude all vector columns
SELECT * EXCLUDE (
    user_query_vector,
    assistant_response_vector,
    assistant_thinking_vector,
    tool_output_vector
)
FROM transcripts
WHERE session_id = ?
ORDER BY sequence;
```

**Advantages:**
- Automatically includes new non-vector columns
- Clean, readable syntax
- DuckDB-native optimization

**Disadvantages:**
- Must remember to add new vector columns to EXCLUDE list
- Not portable to other SQL databases

### Strategy 2: Views as Abstraction Layer (Recommended for Production)

Create views that hide vector columns:

```sql
-- View for metadata and content (no vectors)
CREATE OR REPLACE VIEW transcript_messages AS
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

-- View for vector search (includes vectors + similarity calculation)
CREATE OR REPLACE VIEW transcript_vectors AS
SELECT 
    id,
    user_id,
    session_id,
    sequence,
    user_query_vector,
    assistant_response_vector,
    assistant_thinking_vector,
    tool_output_vector
FROM transcripts;

-- Application queries the view
SELECT * FROM transcript_messages WHERE session_id = ?;
```

**Advantages:**
- Schema changes hidden from application code
- Can evolve table structure without breaking queries
- Centralizes projection logic
- Enables access control patterns

### Strategy 3: Explicit Column Lists in Prepared Statements

For maximum control and portability:

```python
TRANSCRIPT_COLUMNS = """
    id, user_id, session_id, sequence, turn, role, content, ts,
    embedding_model, vector_metadata
"""

# Prepared statement with explicit columns
GET_MESSAGES_SQL = f"""
    SELECT {TRANSCRIPT_COLUMNS}
    FROM transcripts
    WHERE session_id = ?
    ORDER BY sequence
"""
```

### Strategy 4: COLUMNS Regex Pattern

Select columns matching a pattern:

```sql
-- Select only columns that don't end with '_vector'
SELECT COLUMNS(c -> NOT c LIKE '%_vector')
FROM transcripts
WHERE session_id = ?;

-- Or use positive matching
SELECT COLUMNS('^(id|user_id|session_id|sequence|turn|role|content|ts|embedding_model)$')
FROM transcripts;
```

### Combining Patterns: Vector Search with Metadata

```sql
-- Get similarity score but not the actual vectors
SELECT 
    t.id,
    t.session_id,
    t.sequence,
    t.content,
    t.ts,
    GREATEST(
        CASE WHEN t.user_query_vector IS NOT NULL 
             THEN array_cosine_similarity(t.user_query_vector, ?) 
             ELSE 0.0 END,
        CASE WHEN t.assistant_response_vector IS NOT NULL 
             THEN array_cosine_similarity(t.assistant_response_vector, ?)
             ELSE 0.0 END
    ) AS similarity
FROM transcripts t
WHERE t.user_id = ?
  AND (t.user_query_vector IS NOT NULL OR t.assistant_response_vector IS NOT NULL)
ORDER BY similarity DESC
LIMIT ?;
```

---

## Data Setup and Schema Design

### Table Schema

```sql
CREATE TABLE IF NOT EXISTS sessions (
    user_id VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    host_id VARCHAR,
    project_slug VARCHAR,
    bundle VARCHAR,
    created TIMESTAMP NOT NULL DEFAULT current_timestamp,
    updated TIMESTAMP NOT NULL DEFAULT current_timestamp,
    turn_count INTEGER DEFAULT 0,
    message_count INTEGER DEFAULT 0,
    event_count INTEGER DEFAULT 0,
    visibility VARCHAR DEFAULT 'private',
    tags VARCHAR[],
    PRIMARY KEY (user_id, session_id)
);

CREATE TABLE IF NOT EXISTS transcripts (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    sequence INTEGER NOT NULL,
    turn INTEGER,
    role VARCHAR NOT NULL,
    content VARCHAR,  -- Can be TEXT for large content
    ts TIMESTAMP NOT NULL,
    -- Multi-vector embedding columns
    user_query_vector FLOAT[3072],
    assistant_response_vector FLOAT[3072],
    assistant_thinking_vector FLOAT[3072],
    tool_output_vector FLOAT[3072],
    embedding_model VARCHAR,
    vector_metadata JSON,
    UNIQUE (user_id, session_id, sequence)
);

CREATE TABLE IF NOT EXISTS events (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    sequence INTEGER NOT NULL,
    event_type VARCHAR NOT NULL,
    ts TIMESTAMP NOT NULL,
    turn INTEGER,
    level VARCHAR DEFAULT 'INFO',
    data JSON,
    data_truncated BOOLEAN DEFAULT FALSE,
    data_size_bytes INTEGER,
    UNIQUE (user_id, session_id, sequence)
);
```

### Index Strategy

```sql
-- Primary access patterns
CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_sessions_project ON sessions(user_id, project_slug);
CREATE INDEX idx_sessions_updated ON sessions(user_id, updated DESC);

CREATE INDEX idx_transcripts_session ON transcripts(user_id, session_id);
CREATE INDEX idx_transcripts_session_seq ON transcripts(session_id, sequence);
CREATE INDEX idx_transcripts_ts ON transcripts(user_id, ts DESC);

CREATE INDEX idx_events_session ON events(user_id, session_id);
CREATE INDEX idx_events_type ON events(session_id, event_type);
CREATE INDEX idx_events_ts ON events(session_id, ts);
```

**Note:** DuckDB may not always use compound indexes as expected (Issue #14764). Profile your queries.

---

## VSS HNSW Index Configuration

### Installation and Setup

```sql
-- Install and load the VSS extension
INSTALL vss;
LOAD vss;
```

### Creating HNSW Indexes

```sql
-- Enable experimental persistence (CAUTION: see warnings below)
SET hnsw_enable_experimental_persistence = true;

-- Create HNSW indexes for each vector column
CREATE INDEX idx_user_query_vector 
ON transcripts USING HNSW (user_query_vector)
WITH (metric = 'cosine');

CREATE INDEX idx_assistant_response_vector 
ON transcripts USING HNSW (assistant_response_vector)
WITH (metric = 'cosine');

CREATE INDEX idx_assistant_thinking_vector 
ON transcripts USING HNSW (assistant_thinking_vector)
WITH (metric = 'cosine');

CREATE INDEX idx_tool_output_vector 
ON transcripts USING HNSW (tool_output_vector)
WITH (metric = 'cosine');
```

### HNSW Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `metric` | `l2sq` | Distance metric: `l2sq`, `cosine`, `ip` |
| `ef_construction` | 128 | Build-time recall vs speed |
| `ef_search` | 64 | Query-time recall vs speed |
| `M` | 16 | Max connections per node |
| `M0` | 2×M | Max connections at layer 0 |

```sql
-- Adjust search accuracy at runtime
SET hnsw_ef_search = 100;  -- Higher = better recall, slower

-- Create with custom parameters
CREATE INDEX idx_vectors ON transcripts 
USING HNSW (user_query_vector)
WITH (metric = 'cosine', ef_construction = 200, M = 32);
```

### Query Patterns That Use HNSW

HNSW acceleration requires specific query patterns:

```sql
-- WORKS: Constant vector, ORDER BY distance, LIMIT
SELECT id, content, array_cosine_distance(user_query_vector, [0.1, 0.2, ...]::FLOAT[3072]) AS dist
FROM transcripts
ORDER BY dist
LIMIT 10;

-- WORKS: Using min_by aggregate
SELECT min_by(id, array_cosine_distance(user_query_vector, ?), 10)
FROM transcripts;

-- DOES NOT USE HNSW: Non-constant vector (e.g., join)
SELECT * FROM t1
ORDER BY array_distance(t1.vec, t2.vec)  -- Both are columns
LIMIT 10;
```

### Critical Persistence Warnings

**For disk-backed databases:**

1. **Experimental flag required:** `SET hnsw_enable_experimental_persistence = true`
2. **WAL recovery incomplete:** Crashes can cause index data loss or corruption
3. **Not recommended for production** without careful backup strategy
4. **Full serialization:** Index serializes entirely on checkpoint, deserializes on first access
5. **Memory:** Index must fit entirely in RAM; not governed by `memory_limit`

**Recommendation for production:**
- Use in-memory databases for vector search, OR
- Rebuild indexes on startup, OR
- Accept experimental status with robust backups

### Index Maintenance

```sql
-- Compact index after many deletes
PRAGMA hnsw_compact_index('idx_user_query_vector');

-- Or drop and recreate after bulk updates
DROP INDEX idx_user_query_vector;
CREATE INDEX idx_user_query_vector ON transcripts USING HNSW (user_query_vector);
```

---

## Schema Evolution and Migrations

### ALTER TABLE Capabilities

DuckDB supports:
- `ADD COLUMN` - Always safe
- `DROP COLUMN` - Blocked if indexes/views depend on it
- `RENAME COLUMN` - Safe
- `ALTER COLUMN TYPE` - Limited support

```sql
-- Safe: Add new column with default
ALTER TABLE transcripts ADD COLUMN processed_at TIMESTAMP;

-- Safe: Rename column
ALTER TABLE transcripts RENAME COLUMN ts TO created_at;

-- May fail: Drop column with dependencies
ALTER TABLE transcripts DROP COLUMN old_column;  -- Fails if indexed
```

### Migration Pattern: Create and Swap

For complex schema changes:

```sql
-- 1. Create new table with desired schema
CREATE TABLE transcripts_v2 (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    -- ... new schema
);

-- 2. Copy data with transformations
INSERT INTO transcripts_v2 (id, user_id, ...)
SELECT id, user_id, ...
FROM transcripts;

-- 3. Recreate indexes on new table
CREATE INDEX idx_transcripts_v2_session ON transcripts_v2(user_id, session_id);

-- 4. Swap tables
DROP TABLE transcripts;
ALTER TABLE transcripts_v2 RENAME TO transcripts;
```

### Views for Schema Stability

Use views to insulate application code:

```sql
-- Original table has v1 schema
CREATE TABLE transcripts_raw (...);

-- Application uses view
CREATE VIEW transcripts AS
SELECT 
    id, user_id, session_id, sequence, turn, role, content, ts,
    -- Computed/default values for new fields
    COALESCE(embedding_model, 'unknown') AS embedding_model
FROM transcripts_raw;

-- When schema changes, update view definition
CREATE OR REPLACE VIEW transcripts AS
SELECT ... FROM transcripts_raw_v2;
```

---

## Concurrency and Connection Patterns

### DuckDB Concurrency Model

- **Single writer, multiple readers**
- Readers don't block writer; writer doesn't block readers
- Only one write transaction at a time
- Long read transactions can block writes from committing

### Connection Management

```python
import duckdb
from contextlib import contextmanager
from threading import Lock

class DuckDBConnectionManager:
    """Manage DuckDB connections with proper concurrency."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._write_lock = Lock()
        
    @contextmanager
    def read_connection(self):
        """Get a read-only connection (can have many concurrent)."""
        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            yield conn
        finally:
            conn.close()
    
    @contextmanager
    def write_connection(self):
        """Get a write connection (only one at a time)."""
        with self._write_lock:
            conn = duckdb.connect(self.db_path, read_only=False)
            try:
                yield conn
            finally:
                conn.close()
```

### Best Practices

1. **Keep transactions short** - Don't hold connections across user interactions
2. **Use read-only connections** for queries that don't modify data
3. **Serialize writes** with application-level locking
4. **Handle SQLITE_BUSY equivalent** - DuckDB returns errors on write conflicts

```python
import time

def execute_with_retry(conn, sql, params, max_retries=3):
    """Execute with retry for transient conflicts."""
    for attempt in range(max_retries):
        try:
            return conn.execute(sql, params).fetchall()
        except duckdb.TransactionException:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))
            else:
                raise
```

---

## Implementation Recommendations

### 1. Query Layer with View Abstraction

```python
class DuckDBQueryLayer:
    """Query layer with automatic vector exclusion."""
    
    SETUP_VIEWS_SQL = """
    CREATE OR REPLACE VIEW transcript_messages AS
    SELECT * EXCLUDE (
        user_query_vector,
        assistant_response_vector,
        assistant_thinking_vector,
        tool_output_vector
    )
    FROM transcripts;
    
    CREATE OR REPLACE VIEW session_list AS
    SELECT * FROM sessions;
    """
    
    def __init__(self, conn):
        self.conn = conn
        self._setup_views()
    
    def _setup_views(self):
        self.conn.execute(self.SETUP_VIEWS_SQL)
    
    def get_session_messages(self, user_id: str, session_id: str) -> list:
        """Get messages without vectors."""
        return self.conn.execute("""
            SELECT * FROM transcript_messages
            WHERE user_id = ? AND session_id = ?
            ORDER BY sequence
        """, [user_id, session_id]).fetchall()
```

### 2. Vector Search Helper

```python
def vector_search(
    conn,
    query_vector: list[float],
    user_id: str,
    limit: int = 20,
    include_vectors: bool = False
) -> list[dict]:
    """Search transcripts by vector similarity."""
    
    # Build column list
    if include_vectors:
        columns = "*"
    else:
        columns = """* EXCLUDE (
            user_query_vector,
            assistant_response_vector,
            assistant_thinking_vector,
            tool_output_vector
        )"""
    
    sql = f"""
        SELECT {columns},
            GREATEST(
                COALESCE(array_cosine_similarity(user_query_vector, ?), 0),
                COALESCE(array_cosine_similarity(assistant_response_vector, ?), 0)
            ) AS similarity
        FROM transcripts
        WHERE user_id = ?
          AND (user_query_vector IS NOT NULL 
               OR assistant_response_vector IS NOT NULL)
        ORDER BY similarity DESC
        LIMIT ?
    """
    
    results = conn.execute(
        sql, 
        [query_vector, query_vector, user_id, limit]
    ).fetchall()
    
    return [dict(r) for r in results]
```

### 3. Initialization Script

```python
def initialize_duckdb(db_path: str) -> duckdb.DuckDBPyConnection:
    """Initialize DuckDB with extensions and views."""
    conn = duckdb.connect(db_path)
    
    # Load extensions
    conn.execute("INSTALL vss; LOAD vss;")
    
    # Create tables if needed
    conn.execute(SCHEMA_SQL)
    
    # Create abstraction views
    conn.execute("""
        CREATE OR REPLACE VIEW transcript_messages AS
        SELECT * EXCLUDE (
            user_query_vector,
            assistant_response_vector,
            assistant_thinking_vector,
            tool_output_vector
        )
        FROM transcripts;
    """)
    
    return conn
```

### 4. Best Practices Checklist

- [ ] Use `SELECT * EXCLUDE (...)` or explicit columns (never bare `SELECT *`)
- [ ] Create views as abstraction layer for application queries
- [ ] Use read-only connections for queries
- [ ] Serialize write operations with application locking
- [ ] Keep transactions short
- [ ] Understand HNSW persistence caveats before production use
- [ ] Profile queries with `EXPLAIN ANALYZE`
- [ ] Use create-and-swap pattern for complex migrations
- [ ] Configure `ef_search` based on recall/latency requirements

---

## References

- [DuckDB SELECT Clause](https://duckdb.org/docs/stable/sql/query_syntax/select)
- [DuckDB VSS Extension](https://duckdb.org/2024/05/03/vector-similarity-search-vss.html)
- [DuckDB ALTER TABLE](https://duckdb.org/docs/stable/sql/statements/alter_table)
- [DuckDB Concurrency](https://duckdb.org/docs/stable/connect/concurrency)
- [DuckDB Indexing Guide](https://duckdb.org/docs/guides/performance/indexing.html)
- [GitHub: DuckDB VSS](https://github.com/duckdb/duckdb-vss)
