"""Core index operations: build, update, query, resolve, and scope management.

Orchestrates the full index lifecycle—ensuring the index exists, diffing
file hashes for incremental updates, and looking up blocks by ID with
automatic stale-entry refresh.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import file_sha256, normalize_path_query, normalize_relpath
from .constants import INDEX_PATH, INDEX_VERSION, SUPPORTED_SUFFIXES
from .parsers import index_file
from .scanner import collect_target_files
from .storage import load_index, save_index


def _scope_roots(root: Path, include_paths: list[str] | None) -> list[str] | None:
    """Normalise include_paths into deduplicated scope root prefixes."""
    if not include_paths:
        return None

    roots: list[str] = []
    seen: set[str] = set()
    for raw_path in include_paths:
        normalized = normalize_path_query(root, raw_path).strip().rstrip("/")
        if not normalized:
            continue
        if normalized == ".":
            return None
        if normalized in seen:
            continue
        roots.append(normalized)
        seen.add(normalized)
    return roots


def _relpath_in_scope(relpath: str, scope_roots: list[str] | None) -> bool:
    """Check whether a relative path falls under any of the scope roots."""
    if scope_roots is None:
        return True
    for scope in scope_roots:
        if relpath == scope or relpath.startswith(scope + "/"):
            return True
    return False


def ensure_index(
    root: Path,
    auto_build: bool = True,
    paths: list[str] | None = None,
) -> None:
    """Verify the runtime index exists and is current, auto-building if allowed."""
    index_path = root / INDEX_PATH
    if index_path.exists():
        index = load_index(root)
        if int(index.get("version", 0)) == INDEX_VERSION:
            return
        if not auto_build:
            raise FileNotFoundError(
                f"Runtime index is outdated (found v{index.get('version')}, "
                f"expected v{INDEX_VERSION}): {index_path}"
            )
        update_index(
            root=root,
            include_paths=paths,
            changed_only=False,
            remove_missing=False,
        )
        return

    if not auto_build:
        raise FileNotFoundError(f"Runtime index missing: {index_path}")
    update_index(
        root=root,
        include_paths=paths,
        changed_only=False,
        remove_missing=False,
    )


def update_index(
    root: Path,
    include_paths: list[str] | None = None,
    changed_only: bool = False,
    remove_missing: bool = False,
) -> dict[str, int]:
    """Scan files and update the runtime index, optionally diffing by SHA-256."""
    index = load_index(root)
    if int(index.get("version", 0)) != INDEX_VERSION:
        # Schema changed: force a full rebuild so existing entries gain new fields.
        changed_only = False
        index["files"] = {}

    files = collect_target_files(root, include_paths)
    files_by_rel = {normalize_relpath(path, root): path for path in files}
    indexed_files = index["files"]
    scope_roots = _scope_roots(root, include_paths)

    if include_paths is None:
        # Also keep previously indexed ad-hoc files (e.g. tools/* queried via --path).
        for relpath in list(indexed_files.keys()):
            path = root / relpath
            if not path.exists() or not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            files_by_rel.setdefault(relpath, path)

    to_index: list[tuple[str, Path]] = []
    for relpath, path in sorted(files_by_rel.items()):
        if not changed_only:
            to_index.append((relpath, path))
            continue
        existing = indexed_files.get(relpath)
        if existing is None:
            to_index.append((relpath, path))
            continue
        current_sha = file_sha256(path)
        if current_sha != existing.get("sha256"):
            to_index.append((relpath, path))

    updated = 0
    for relpath, path in to_index:
        indexed_files[relpath] = index_file(root, path)
        updated += 1

    removed = 0
    if remove_missing:
        current_relpaths = set(files_by_rel)
        for relpath in list(indexed_files.keys()):
            if not _relpath_in_scope(relpath, scope_roots):
                continue
            if relpath not in current_relpaths:
                del indexed_files[relpath]
                removed += 1

    save_index(root, index)
    return {
        "scanned": len(files_by_rel),
        "updated": updated,
        "removed": removed,
        "total_indexed": len(indexed_files),
    }


def validate_index_scope(
    root: Path,
    include_paths: list[str] | None = None,
) -> dict[str, int]:
    """Refresh the index using file-level hashes within the requested scope."""
    return update_index(
        root=root,
        include_paths=include_paths,
        changed_only=True,
        remove_missing=True,
    )


def iter_blocks(index: dict[str, Any]):
    """Yield (relpath, block) pairs for every block across all indexed files."""
    for relpath, entry in index.get("files", {}).items():
        for block in entry.get("blocks", []):
            yield relpath, block


def find_block(index: dict[str, Any], block_id: str) -> tuple[str, dict[str, Any]] | None:
    """Linear scan for a block by its ID, returning (relpath, block) or None."""
    for relpath, entry in index.get("files", {}).items():
        for block in entry.get("blocks", []):
            if block.get("id") == block_id:
                return relpath, block
    return None


def guess_relpath_from_block_id(block_id: str) -> str | None:
    """Extract the file relpath from a block ID like 'py:path/to/file.py:name'."""
    parts = block_id.split(":", 2)
    if len(parts) < 3:
        return None
    if parts[0] not in {"py", "md"}:
        return None
    return parts[1]


def refresh_file_entry_if_stale(
    root: Path,
    index: dict[str, Any],
    relpath: str,
) -> tuple[dict[str, Any] | None, bool]:
    """Re-index a single file if its SHA-256 has changed; remove if deleted."""
    path = root / relpath
    if not path.exists():
        if relpath in index["files"]:
            del index["files"][relpath]
        return None, True

    current_sha = file_sha256(path)
    existing = index["files"].get(relpath)
    if existing is not None and existing.get("sha256") == current_sha:
        return existing, False

    index["files"][relpath] = index_file(root, path)
    return index["files"][relpath], True


def resolve_block(
    root: Path,
    index: dict[str, Any],
    block_id: str,
    refresh_if_stale: bool = True,
) -> tuple[str, dict[str, Any], bool]:
    """Look up a block by ID, refreshing the owning file if stale.

    Falls back to guessing the file path from the block ID structure
    when the block is not found in the current index.
    """
    refreshed = False
    found = find_block(index, block_id)
    if found is not None:
        relpath, block = found
        if refresh_if_stale:
            _, did_refresh = refresh_file_entry_if_stale(root, index, relpath)
            refreshed = refreshed or did_refresh
            if did_refresh:
                found = find_block(index, block_id)
                if found is None:
                    raise KeyError(f"Block not found after refresh: {block_id}")
                relpath, block = found
        return relpath, block, refreshed

    guessed_relpath = guess_relpath_from_block_id(block_id)
    if guessed_relpath and refresh_if_stale:
        _, did_refresh = refresh_file_entry_if_stale(root, index, guessed_relpath)
        refreshed = refreshed or did_refresh
        found = find_block(index, block_id)
        if found is not None:
            relpath, block = found
            return relpath, block, refreshed

    raise KeyError(f"Block id not found: {block_id}")


def snapshot_stats(root: Path, include_paths: list[str] | None = None) -> dict[str, tuple[int, int]]:
    """Capture mtime and size for all target files, used by the watch loop to detect changes."""
    stats: dict[str, tuple[int, int]] = {}
    for path in collect_target_files(root, include_paths):
        relpath = normalize_relpath(path, root)
        stat = path.stat()
        stats[relpath] = (stat.st_mtime_ns, stat.st_size)
    return stats
