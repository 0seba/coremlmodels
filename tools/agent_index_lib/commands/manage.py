"""Index management commands: build/refresh, check staleness, and filesystem watcher."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

from ..common import file_sha256
from ..ops import snapshot_stats, update_index
from ..parsers import index_file
from ..storage import load_index, save_index


def cmd_update(args: argparse.Namespace) -> int:
    """Build or refresh the runtime index, printing a summary of changes."""
    root = Path(args.root).resolve()
    summary = update_index(
        root=root,
        include_paths=args.paths,
        changed_only=args.changed_only,
        remove_missing=args.remove_missing,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            f"Updated runtime index: scanned={summary['scanned']} "
            f"updated={summary['updated']} removed={summary['removed']} "
            f"total={summary['total_indexed']}"
        )
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Report stale or missing files in the index, optionally auto-fixing them."""
    root = Path(args.root).resolve()
    index = load_index(root)
    files = index.get("files", {})
    stale: list[str] = []
    missing: list[str] = []

    for relpath, entry in sorted(files.items()):
        path = root / relpath
        if not path.exists():
            missing.append(relpath)
            continue
        current_sha = file_sha256(path)
        if current_sha != entry.get("sha256"):
            stale.append(relpath)

    if args.fix and (stale or missing):
        for relpath in stale:
            path = root / relpath
            if path.exists():
                index["files"][relpath] = index_file(root, path)
        for relpath in missing:
            index["files"].pop(relpath, None)
        save_index(root, index)

    if stale:
        print("Stale files:")
        for relpath in stale:
            print(f"  - {relpath}")
    if missing:
        print("Missing files:")
        for relpath in missing:
            print(f"  - {relpath}")
    if not stale and not missing:
        print("Index is fresh.")

    if args.fix and (stale or missing):
        print("Applied fixes to runtime index.")
        return 0
    return 1 if (stale or missing) else 0


def cmd_watch(args: argparse.Namespace) -> int:
    """Poll for file changes and incrementally update the index in a loop."""
    root = Path(args.root).resolve()
    include_paths = args.paths
    if args.bootstrap:
        summary = update_index(
            root=root,
            include_paths=include_paths,
            changed_only=False,
            remove_missing=False,
        )
        print(
            f"Bootstrap index: scanned={summary['scanned']} "
            f"updated={summary['updated']} total={summary['total_indexed']}"
        )

    previous_stats = snapshot_stats(root, include_paths)
    print(
        f"Watching for changes every {args.interval:.2f}s "
        f"(tracked_files={len(previous_stats)}). Press Ctrl-C to stop."
    )

    try:
        while True:
            time.sleep(args.interval)
            current_stats = snapshot_stats(root, include_paths)

            changed_relpaths = [
                rel
                for rel, stats in current_stats.items()
                if rel not in previous_stats or previous_stats[rel] != stats
            ]
            deleted_relpaths = [rel for rel in previous_stats if rel not in current_stats]

            if not changed_relpaths and not deleted_relpaths:
                previous_stats = current_stats
                continue

            index = load_index(root)
            reindexed = 0
            removed = 0

            for relpath in sorted(changed_relpaths):
                path = root / relpath
                if not path.exists():
                    continue
                index["files"][relpath] = index_file(root, path)
                reindexed += 1

            for relpath in deleted_relpaths:
                if relpath in index["files"]:
                    del index["files"][relpath]
                    removed += 1

            if reindexed or removed:
                save_index(root, index)
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"updated={reindexed} removed={removed} total={len(index['files'])}"
                )

            previous_stats = current_stats
    except KeyboardInterrupt:
        print("\nStopped watcher.")
        return 0
