"""JSON persistence for the runtime index file.

Handles atomic writes via a temp-file swap to prevent corruption on
interruption, and initialises a fresh index skeleton when none exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import utc_now
from .constants import INDEX_PATH, INDEX_VERSION, RUNTIME_DIR


def load_index(root: Path) -> dict[str, Any]:
    """Load the runtime index from disk, returning a fresh skeleton if absent or corrupt."""
    index_path = root / INDEX_PATH
    if not index_path.exists():
        return {
            "version": INDEX_VERSION,
            "root": str(root.resolve()),
            "generated_at": utc_now(),
            "files": {},
        }
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "version": INDEX_VERSION,
            "root": str(root.resolve()),
            "generated_at": utc_now(),
            "files": {},
        }
    if "files" not in data or not isinstance(data["files"], dict):
        data["files"] = {}
    if "version" not in data:
        data["version"] = INDEX_VERSION
    if "root" not in data:
        data["root"] = str(root.resolve())
    return data


def save_index(root: Path, index: dict[str, Any]) -> None:
    """Atomically write the index to disk, updating version and timestamp."""
    runtime_dir = root / RUNTIME_DIR
    runtime_dir.mkdir(parents=True, exist_ok=True)
    index["version"] = INDEX_VERSION
    index["root"] = str(root.resolve())
    index["generated_at"] = utc_now()
    index_path = root / INDEX_PATH
    tmp_path = index_path.with_name(index_path.name + ".tmp")
    tmp_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(index_path)
