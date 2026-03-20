"""Block-level commands: resolve a block ID to line ranges, or read its source text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ..common import format_lines_with_numbers, read_file_text
from ..ops import ensure_index, guess_relpath_from_block_id, resolve_block, validate_index_scope
from ..storage import load_index


def cmd_resolve(args: argparse.Namespace) -> int:
    """Resolve a block ID to its file path, line span, and description metadata."""
    root = Path(args.root).resolve()
    try:
        ensure_index(root, auto_build=args.auto_build, paths=args.paths)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    guessed_relpath = guess_relpath_from_block_id(args.id)
    validation_scope = [guessed_relpath] if guessed_relpath else args.paths
    validation_summary = validate_index_scope(root=root, include_paths=validation_scope)
    index = load_index(root)
    try:
        relpath, block, _ = resolve_block(
            root=root,
            index=index,
            block_id=args.id,
            refresh_if_stale=False,
        )
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    refreshed = bool(validation_summary["updated"] or validation_summary["removed"])

    output = {
        "id": block["id"],
        "path": relpath,
        "kind": block.get("kind"),
        "name": block.get("name"),
        "parent_id": block.get("parent_id"),
        "start_line": block.get("start_line"),
        "end_line": block.get("end_line"),
        "desc_start_line": block.get("desc_start_line"),
        "desc_end_line": block.get("desc_end_line"),
        "desc_short": block.get("desc_short"),
        "refreshed": refreshed,
    }

    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print(f"id: {output['id']}")
        print(f"path: {output['path']}")
        print(f"kind: {output['kind']}")
        print(f"span: {output['start_line']}..{output['end_line']}")
        if output["desc_start_line"] and output["desc_end_line"]:
            print(f"description_span: {output['desc_start_line']}..{output['desc_end_line']}")
        if output["desc_short"]:
            print(f"description: {output['desc_short']}")
        print(f"refreshed: {output['refreshed']}")
    return 0


def cmd_read(args: argparse.Namespace) -> int:
    """Read and print the source text for a block (or its description span)."""
    root = Path(args.root).resolve()
    try:
        ensure_index(root, auto_build=args.auto_build, paths=args.paths)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    guessed_relpath = guess_relpath_from_block_id(args.id)
    validation_scope = [guessed_relpath] if guessed_relpath else args.paths
    validation_summary = validate_index_scope(root=root, include_paths=validation_scope)
    index = load_index(root)
    try:
        relpath, block, _ = resolve_block(
            root=root,
            index=index,
            block_id=args.id,
            refresh_if_stale=False,
        )
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    refreshed = bool(validation_summary["updated"] or validation_summary["removed"])

    path = root / relpath
    text = read_file_text(path)
    lines = text.splitlines()

    if args.description:
        start_line = block.get("desc_start_line") or block.get("start_line")
        end_line = block.get("desc_end_line") or block.get("end_line")
    else:
        start_line = block.get("start_line")
        end_line = block.get("end_line")

    start_line = int(start_line)
    end_line = int(end_line)

    if args.max_lines is not None and args.max_lines > 0:
        end_line = min(end_line, start_line + args.max_lines - 1)

    snippet = lines[start_line - 1 : end_line]
    snippet_text = (
        format_lines_with_numbers(snippet, start_line=start_line)
        if args.line_numbers
        else "\n".join(snippet)
    )

    if args.json:
        output = {
            "id": block["id"],
            "path": relpath,
            "kind": block.get("kind"),
            "start_line": start_line,
            "end_line": end_line,
            "refreshed": refreshed,
            "text": snippet_text,
        }
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        mode = "description" if args.description else "block"
        print(f"id: {block['id']}")
        print(f"path: {relpath}")
        print(f"mode: {mode}")
        print(f"span: {start_line}..{end_line}")
        print("---")
        print(snippet_text)
    return 0
