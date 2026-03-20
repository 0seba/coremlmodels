"""Filesystem scanner that discovers indexable files (.py, .md) under configured directories."""

from __future__ import annotations

import os
from pathlib import Path

from .common import normalize_relpath
from .constants import DEFAULT_SCAN_DIRS, DEFAULT_SCAN_FILES, IGNORED_DIRS, SUPPORTED_SUFFIXES


def is_supported_file(path: Path) -> bool:
    """Return True if the path is a regular file with a supported suffix."""
    return path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES


def iter_files_under_dir(dir_path: Path) -> list[Path]:
    """Walk a directory tree and collect all supported files, skipping ignored dirs."""
    files: list[Path] = []
    if not dir_path.exists() or not dir_path.is_dir():
        return files

    for current_root, dirs, current_files in os.walk(dir_path):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        current_root_path = Path(current_root)
        for name in current_files:
            path = current_root_path / name
            if is_supported_file(path):
                files.append(path)
    return files


def collect_target_files(root: Path, raw_paths: list[str] | None = None) -> list[Path]:
    """Collect deduplicated, sorted list of indexable files from explicit paths or defaults."""
    files: list[Path] = []

    if raw_paths:
        seen: set[str] = set()
        for raw_path in raw_paths:
            path = (root / raw_path).resolve()
            if not path.exists():
                continue
            if path.is_file():
                if is_supported_file(path):
                    rel = normalize_relpath(path, root)
                    if rel not in seen:
                        files.append(path)
                        seen.add(rel)
                continue
            for file_path in iter_files_under_dir(path):
                rel = normalize_relpath(file_path, root)
                if rel in seen:
                    continue
                files.append(file_path)
                seen.add(rel)
        return sorted(files)

    for dirname in DEFAULT_SCAN_DIRS:
        files.extend(iter_files_under_dir(root / dirname))

    for filename in DEFAULT_SCAN_FILES:
        file_path = root / filename
        if is_supported_file(file_path):
            files.append(file_path)

    unique: dict[str, Path] = {}
    for file_path in files:
        unique[normalize_relpath(file_path, root)] = file_path
    return [unique[key] for key in sorted(unique)]
