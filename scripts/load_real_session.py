"""Load a real Amplifier session from disk and sync to Cosmos DB.

Proves the full pipeline works with real data: session metadata, transcript sync
with externalized vector documents, and all three search modes.

Usage:
    uv run python scripts/load_real_session.py [session_dir]

If no session_dir given, picks a recent parent session automatically.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import socket
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Default Cosmos environment (same as integration test .env)
# ---------------------------------------------------------------------------
os.environ.setdefault(
    "AMPLIFIER_COSMOS_ENDPOINT",
    "https://amplifier-session-storage-test.documents.azure.com:443/",
)
os.environ.setdefault("AMPLIFIER_COSMOS_DATABASE", "amplifier-test-db")
os.environ.setdefault("AMPLIFIER_COSMOS_AUTH_METHOD", "default_credential")
os.environ.setdefault("AMPLIFIER_COSMOS_ENABLE_VECTOR", "true")

from amplifier_session_storage.backends import SearchFilters, TranscriptSearchOptions  # noqa: E402
from amplifier_session_storage.backends.cosmos import CosmosBackend, CosmosConfig  # noqa: E402
from amplifier_session_storage.embeddings import EmbeddingProvider  # noqa: E402

# Suppress Azure SDK HTTP noise - only show our own output
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
for noisy in (
    "azure",
    "azure.core",
    "azure.identity",
    "azure.cosmos",
    "azure.core.pipeline",
    "azure.core.pipeline.policies",
    "msal",
    "urllib3",
):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# UUID pattern for parent sessions (no underscore in dirname)
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_MAX_TRANSCRIPT_BYTES = 500 * 1024  # 500 KB
_MIN_TURNS = 3


# ---------------------------------------------------------------------------
# Mock embedding provider (same as tests - deterministic 3072-dim vectors)
# ---------------------------------------------------------------------------
class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic 3072-dim embeddings derived from text hash."""

    @property
    def model_name(self) -> str:
        return "mock-load-script"

    @property
    def dimensions(self) -> int:
        return 3072

    async def embed_text(self, text: str) -> list[float]:
        hash_val = hash(text) % 1000
        return [float(hash_val) / 1000.0] * 3072

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_text(t) for t in texts]

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_user_id() -> str:
    """Read user_id from sync-daemon config, default to 'dicolomb'."""
    cfg_path = Path.home() / ".amplifier" / "sync-daemon.yaml"
    if cfg_path.exists():
        try:
            import yaml

            with open(cfg_path) as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and data.get("user_id"):
                return str(data["user_id"])
        except Exception as exc:
            logger.warning("Could not read sync-daemon.yaml: %s", exc)
    return "dicolomb"


def _find_recent_session() -> Path:
    """Scan ~/.amplifier/projects/*/sessions/ for a suitable parent session."""
    projects_root = Path.home() / ".amplifier" / "projects"
    if not projects_root.exists():
        raise FileNotFoundError(f"No projects directory at {projects_root}")

    candidates: list[tuple[float, Path]] = []

    for project_dir in projects_root.iterdir():
        if not project_dir.is_dir():
            continue
        sessions_dir = project_dir / "sessions"
        if not sessions_dir.is_dir():
            continue
        for sess_dir in sessions_dir.iterdir():
            if not sess_dir.is_dir():
                continue
            # Parent sessions have UUID names (no underscore)
            if not _UUID_RE.match(sess_dir.name):
                continue

            transcript = sess_dir / "transcript.jsonl"
            metadata_file = sess_dir / "metadata.json"
            if not transcript.exists() or not metadata_file.exists():
                continue

            # Size check
            if transcript.stat().st_size > _MAX_TRANSCRIPT_BYTES:
                continue

            # Turn count check
            try:
                with open(metadata_file) as f:
                    meta = json.load(f)
                turn_count = meta.get("turn_count", 0)
                if turn_count < _MIN_TURNS:
                    continue
            except Exception:
                continue

            # Use metadata file mtime as recency proxy
            mtime = metadata_file.stat().st_mtime
            candidates.append((mtime, sess_dir))

    if not candidates:
        raise FileNotFoundError(
            f"No suitable parent sessions found under {projects_root} "
            f"(need >{_MIN_TURNS} turns, transcript <500KB)"
        )

    # Most recent first
    candidates.sort(key=lambda x: x[0], reverse=True)
    chosen = candidates[0][1]
    logger.info("Auto-selected session: %s", chosen)
    return chosen


def _extract_first_user_keyword(lines: list[dict[str, Any]]) -> str:
    """Pick a keyword from the first user message for search testing."""
    for line in lines:
        if line.get("role") == "user":
            content = line.get("content", "")
            if isinstance(content, str) and content.strip():
                # Pick the longest word >= 4 chars
                words = [w for w in content.split() if len(w) >= 4 and w.isalpha()]
                if words:
                    return max(words, key=len)
                # Fallback: first 20 chars
                return content[:20]
    return "test"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def run(session_dir: Path) -> None:
    results: dict[str, str] = {}  # step -> PASS/FAIL

    # ---- 1. Read session data ----
    metadata_file = session_dir / "metadata.json"
    transcript_file = session_dir / "transcript.jsonl"

    with open(metadata_file) as f:
        session_metadata: dict[str, Any] = json.load(f)

    session_id = session_metadata["session_id"]
    # Use the raw encoded directory name as project_slug
    project_slug = session_dir.parent.parent.name
    session_metadata.setdefault("project_slug", project_slug)

    with open(transcript_file) as f:
        lines = [json.loads(line) for line in f if line.strip()]

    user_id = _read_user_id()
    host_id = socket.gethostname()

    print()
    print("=" * 70)
    print("LOAD REAL SESSION TO COSMOS")
    print("=" * 70)
    print(f"  session_id   : {session_id}")
    print(f"  project_slug : {project_slug}")
    print(f"  user_id      : {user_id}")
    print(f"  host_id      : {host_id}")
    print(f"  turns        : {session_metadata.get('turn_count', '?')}")
    print(f"  messages     : {len(lines)}")
    print(f"  transcript   : {transcript_file.stat().st_size / 1024:.1f} KB")
    print("=" * 70)

    # ---- 2. Connect to Cosmos ----
    config = CosmosConfig.from_env()
    embeddings = MockEmbeddingProvider()
    storage = await CosmosBackend.create(config=config, embedding_provider=embeddings)
    print(f"\nConnected to Cosmos: {config.endpoint}")
    print(f"  database: {config.database_name}")
    print(f"  vector:   {config.enable_vector_search}")
    results["connect"] = "PASS"

    try:
        # ---- 3. Sync session metadata ----
        print("\n--- Syncing session metadata ---")
        await storage.upsert_session_metadata(
            user_id=user_id, host_id=host_id, metadata=session_metadata
        )
        results["upsert_metadata"] = "PASS"
        print("  OK")

        # ---- 4. Sync transcript lines ----
        print("\n--- Syncing transcript lines ---")
        synced = await storage.sync_transcript_lines(
            user_id=user_id,
            host_id=host_id,
            project_slug=project_slug,
            session_id=session_id,
            lines=lines,
            start_sequence=0,
        )
        print(f"  synced {synced} transcript documents")
        results["sync_transcripts"] = "PASS" if synced == len(lines) else "FAIL"

        # ---- 5. Verify document counts ----
        print("\n--- Verifying documents in Cosmos ---")
        container = storage._get_container("session_data")
        pk = storage.make_partition_key(user_id, project_slug, session_id)

        doc_counts: dict[str, int] = {}
        for doc_type in ("session", "transcript", "transcript_vector"):
            count = 0
            async for _ in container.query_items(
                query="SELECT VALUE COUNT(1) FROM c WHERE c.partition_key = @pk AND c.type = @t",
                parameters=[
                    {"name": "@pk", "value": pk},
                    {"name": "@t", "value": doc_type},
                ],
            ):
                count = _
            doc_counts[doc_type] = count
            print(f"  {doc_type:25s}: {count}")

        has_vectors = doc_counts.get("transcript_vector", 0) > 0
        results["vector_docs_created"] = "PASS" if has_vectors else "FAIL"

        # ---- 6. Test all search modes ----
        keyword = _extract_first_user_keyword(lines)
        print(f"\n--- Testing search (keyword: '{keyword}') ---")

        # Semantic search
        try:
            sem_results = await storage.search_transcripts(
                user_id=user_id,
                options=TranscriptSearchOptions(
                    query=keyword,
                    search_type="semantic",
                    filters=SearchFilters(project_slug=project_slug),
                ),
                limit=5,
            )
            print(f"  semantic   : {len(sem_results)} results")
            if sem_results:
                print(f"               top score={sem_results[0].score:.4f}")
            results["search_semantic"] = "PASS" if sem_results else "FAIL"
        except Exception as exc:
            print(f"  semantic   : FAILED - {exc}")
            results["search_semantic"] = "FAIL"

        # Full-text search
        try:
            ft_results = await storage.search_transcripts(
                user_id=user_id,
                options=TranscriptSearchOptions(
                    query=keyword,
                    search_type="full_text",
                    filters=SearchFilters(project_slug=project_slug),
                ),
                limit=5,
            )
            print(f"  full_text  : {len(ft_results)} results")
            results["search_full_text"] = "PASS" if ft_results else "FAIL"
        except Exception as exc:
            print(f"  full_text  : FAILED - {exc}")
            results["search_full_text"] = "FAIL"

        # Hybrid search
        try:
            hyb_results = await storage.search_transcripts(
                user_id=user_id,
                options=TranscriptSearchOptions(
                    query=keyword,
                    search_type="hybrid",
                    filters=SearchFilters(project_slug=project_slug),
                ),
                limit=5,
            )
            print(f"  hybrid     : {len(hyb_results)} results")
            results["search_hybrid"] = "PASS" if hyb_results else "FAIL"
        except Exception as exc:
            print(f"  hybrid     : FAILED - {exc}")
            results["search_hybrid"] = "FAIL"

    finally:
        await storage.close()

    # ---- 7. Summary ----
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for step, status in results.items():
        marker = "PASS" if status == "PASS" else "FAIL"
        if status != "PASS":
            all_pass = False
        print(f"  {step:25s}: {marker}")
    print("=" * 70)

    if all_pass:
        print("\nAll checks passed. Data left in Cosmos for portal inspection.")
    else:
        print("\nSome checks failed. See above for details.")
        sys.exit(1)


def main() -> None:
    if len(sys.argv) > 1:
        session_dir = Path(sys.argv[1]).expanduser().resolve()
        if not session_dir.is_dir():
            print(f"Error: {session_dir} is not a directory", file=sys.stderr)
            sys.exit(1)
    else:
        session_dir = _find_recent_session()

    asyncio.run(run(session_dir))


if __name__ == "__main__":
    main()
