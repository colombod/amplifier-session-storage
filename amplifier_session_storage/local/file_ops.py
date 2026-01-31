"""
JSONL file operations for local storage.

Provides atomic read/write operations for JSONL files with:
- Atomic writes using temp file + rename
- Line-by-line reading for memory efficiency
- Backup creation before destructive operations
"""

import json
import os
import shutil
import tempfile
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os

from ..exceptions import StorageIOError


async def ensure_directory(path: Path) -> None:
    """Ensure directory exists, creating if necessary.

    Args:
        path: Directory path to ensure exists
    """
    try:
        await aiofiles.os.makedirs(path, exist_ok=True)
    except OSError as e:
        raise StorageIOError("create_directory", str(path), e) from e


async def read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON data or None if file doesn't exist
    """
    try:
        if not await aiofiles.os.path.exists(path):
            return None
        async with aiofiles.open(path, encoding="utf-8") as f:
            content = await f.read()
            return json.loads(content) if content.strip() else None
    except json.JSONDecodeError as e:
        raise StorageIOError("parse_json", str(path), e) from e
    except OSError as e:
        raise StorageIOError("read_json", str(path), e) from e


async def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write JSON file atomically using temp file + rename.

    Args:
        path: Target path for JSON file
        data: Data to serialize as JSON
    """
    await ensure_directory(path.parent)

    # Write to temp file first
    fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=".tmp_",
        suffix=".json",
    )
    try:
        os.close(fd)
        async with aiofiles.open(temp_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, indent=2, default=_json_serializer))
            await f.flush()
            os.fsync(f.fileno())

        # Atomic rename
        await aiofiles.os.rename(temp_path, path)
    except Exception as e:
        # Clean up temp file on error
        try:
            await aiofiles.os.remove(temp_path)
        except OSError:
            pass
        raise StorageIOError("write_json", str(path), e) from e


async def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all lines from a JSONL file.

    Args:
        path: Path to JSONL file

    Returns:
        List of parsed JSON objects
    """
    try:
        if not await aiofiles.os.path.exists(path):
            return []

        results = []
        async with aiofiles.open(path, encoding="utf-8") as f:
            async for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results
    except json.JSONDecodeError as e:
        raise StorageIOError("parse_jsonl", str(path), e) from e
    except OSError as e:
        raise StorageIOError("read_jsonl", str(path), e) from e


async def iter_jsonl(path: Path) -> AsyncIterator[dict[str, Any]]:
    """Iterate over lines in a JSONL file without loading all into memory.

    Args:
        path: Path to JSONL file

    Yields:
        Parsed JSON objects one at a time
    """
    try:
        if not await aiofiles.os.path.exists(path):
            return

        async with aiofiles.open(path, encoding="utf-8") as f:
            async for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    except json.JSONDecodeError as e:
        raise StorageIOError("parse_jsonl", str(path), e) from e
    except OSError as e:
        raise StorageIOError("read_jsonl", str(path), e) from e


async def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    """Append a single JSON object to a JSONL file.

    Args:
        path: Path to JSONL file
        data: Data to append
    """
    await ensure_directory(path.parent)

    try:
        async with aiofiles.open(path, "a", encoding="utf-8") as f:
            await f.write(json.dumps(data, default=_json_serializer) + "\n")
            await f.flush()
            os.fsync(f.fileno())
    except OSError as e:
        raise StorageIOError("append_jsonl", str(path), e) from e


async def write_jsonl_atomic(path: Path, data: list[dict[str, Any]]) -> None:
    """Write entire JSONL file atomically.

    Args:
        path: Target path for JSONL file
        data: List of objects to write
    """
    await ensure_directory(path.parent)

    fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=".tmp_",
        suffix=".jsonl",
    )
    try:
        os.close(fd)
        async with aiofiles.open(temp_path, "w", encoding="utf-8") as f:
            for item in data:
                await f.write(json.dumps(item, default=_json_serializer) + "\n")
            await f.flush()
            os.fsync(f.fileno())

        await aiofiles.os.rename(temp_path, path)
    except Exception as e:
        try:
            await aiofiles.os.remove(temp_path)
        except OSError:
            pass
        raise StorageIOError("write_jsonl", str(path), e) from e


async def truncate_jsonl(
    path: Path,
    keep_count: int | None = None,
    keep_predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> int:
    """Truncate a JSONL file, keeping only matching entries.

    Args:
        path: Path to JSONL file
        keep_count: Keep first N entries (if provided)
        keep_predicate: Function returning True for entries to keep

    Returns:
        Number of entries removed
    """
    if not await aiofiles.os.path.exists(path):
        return 0

    original = await read_jsonl(path)
    original_count = len(original)

    if keep_count is not None:
        kept = original[:keep_count]
    elif keep_predicate is not None:
        kept = [item for item in original if keep_predicate(item)]
    else:
        kept = original

    await write_jsonl_atomic(path, kept)
    return original_count - len(kept)


async def create_backup(path: Path, backup_dir: Path | None = None) -> Path:
    """Create a backup of a file.

    Args:
        path: Path to file to backup
        backup_dir: Directory for backup (defaults to same dir with .backup suffix)

    Returns:
        Path to backup file
    """
    if not await aiofiles.os.path.exists(path):
        raise StorageIOError("backup", str(path), FileNotFoundError(f"File not found: {path}"))

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if backup_dir is None:
        backup_path = path.with_suffix(f".{timestamp}.backup{path.suffix}")
    else:
        await ensure_directory(backup_dir)
        backup_path = backup_dir / f"{path.stem}.{timestamp}.backup{path.suffix}"

    try:
        # Use sync shutil.copy2 wrapped - aiofiles doesn't have copy
        await aiofiles.os.wrap(shutil.copy2)(path, backup_path)
        return backup_path
    except OSError as e:
        raise StorageIOError("backup", str(path), e) from e


async def file_exists(path: Path) -> bool:
    """Check if a file exists.

    Args:
        path: Path to check

    Returns:
        True if file exists
    """
    try:
        return await aiofiles.os.path.exists(path)
    except OSError:
        return False


async def remove_file(path: Path) -> bool:
    """Remove a file if it exists.

    Args:
        path: Path to remove

    Returns:
        True if file was removed, False if it didn't exist
    """
    try:
        if await aiofiles.os.path.exists(path):
            await aiofiles.os.remove(path)
            return True
        return False
    except OSError as e:
        raise StorageIOError("remove", str(path), e) from e


async def remove_directory(path: Path) -> bool:
    """Remove a directory and all contents.

    Args:
        path: Directory to remove

    Returns:
        True if removed, False if didn't exist
    """
    try:
        if await aiofiles.os.path.exists(path):
            await aiofiles.os.wrap(shutil.rmtree)(path)
            return True
        return False
    except OSError as e:
        raise StorageIOError("remove_directory", str(path), e) from e


async def list_directories(path: Path) -> list[str]:
    """List subdirectories in a directory.

    Args:
        path: Directory to list

    Returns:
        List of subdirectory names
    """
    try:
        if not await aiofiles.os.path.exists(path):
            return []

        entries = await aiofiles.os.listdir(path)
        dirs = []
        for entry in entries:
            entry_path = path / entry
            if await aiofiles.os.path.isdir(entry_path):
                dirs.append(entry)
        return dirs
    except OSError as e:
        raise StorageIOError("list_directories", str(path), e) from e


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for types not handled by default.

    Args:
        obj: Object to serialize

    Returns:
        JSON-serializable representation
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
