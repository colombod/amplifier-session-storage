# Changes Summary

## 1. Metadata Field Preservation

**Problem:** The `metadata` field was stored but excluded from query projections, causing data loss on read.

**Solution:** Added `c.metadata` to `TRANSCRIPT_PROJECTION` in all backends.

**Impact:**
- ✅ Metadata fields now round-trip correctly
- ✅ Supports multi-device sync scenarios  
- ✅ Enables future server-side metadata features
- 💰 ~10-15% RU increase in Cosmos DB read costs (acceptable)

## 2. Active Sessions Query Method

**Added:** `get_active_sessions()` convenience method for common session queries.

**Features:**
- Date range filtering (start_date, end_date)
- Project filtering
- Activity filtering (min_turn_count)
- Efficient indexed queries

**Use cases:**
- "Sessions from last week"
- "Active sessions in this project"
- "High-activity sessions (>10 turns)"

## Files Modified

**Core changes:**
- `amplifier_session_storage/backends/base.py` (+28 lines) - Abstract method
- `amplifier_session_storage/backends/cosmos.py` (+70 lines) - Cosmos impl + projection fix
- `amplifier_session_storage/backends/duckdb.py` (+53 lines) - DuckDB impl
- `amplifier_session_storage/backends/sqlite.py` (+48 lines) - SQLite impl

**Documentation:**
- `docs/SCHEMA.md` - Documented metadata field
- `docs/API_DESIGN.md` - Documented get_active_sessions()

**Tests:**
- `tests/test_quick_metadata_validation.py` - Logic validation
- `tests/test_metadata_preservation.py` - Cosmos round-trip tests
- `tests/test_cosmos_real_connection.py` - Real connection script
- `tests/test_cosmos_metadata.sh` - Test environment setup

## Backward Compatibility

✅ No breaking changes
- Old format (no metadata) continues to work
- Existing queries just return more data
- All backends updated consistently

## Next Steps

**After this PR merges:**
1. amplifier-session-sync server will automatically get changes via dependency update
2. Open issue in session-sync repo to validate metadata preservation in production
3. Consider adding metadata-based features (search by source, confidence filtering, etc.)
