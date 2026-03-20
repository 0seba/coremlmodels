"""Discovery commands: find blocks by query, list file contents, and render repository trees."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

from ..common import compact_text, normalize_path_query
from ..constants import SUPPORTED_SUFFIXES
from ..dependencies import collect_dependency_rows
from ..ops import ensure_index, iter_blocks, validate_index_scope
from ..parsers import index_file
from ..storage import load_index, save_index


def _query_validation_scope(
    root: Path,
    query_path: str | None,
    fallback_paths: list[str] | None,
    require_existing: bool = False,
) -> list[str] | None:
    if query_path:
        normalized = normalize_path_query(root, query_path)
        if require_existing and not (root / normalized).exists():
            return fallback_paths
        return [normalized]
    return fallback_paths


def cmd_find(args: argparse.Namespace) -> int:
    """Search indexed blocks by substring query, path, and kind filters."""
    root = Path(args.root).resolve()
    try:
        ensure_index(root, auto_build=args.auto_build, paths=args.paths)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    validate_index_scope(
        root=root,
        include_paths=_query_validation_scope(
            root,
            args.path,
            args.paths,
            require_existing=True,
        ),
    )
    index = load_index(root)

    query = (args.query or "").strip().lower()
    path_filter = normalize_path_query(root, args.path).lower() if args.path else ""
    kind_filter = args.kind.lower() if args.kind else None

    matches: list[dict[str, Any]] = []
    for relpath, block in iter_blocks(index):
        if path_filter and path_filter not in relpath.lower():
            continue
        kind = str(block.get("kind", "")).lower()
        if kind_filter and kind != kind_filter:
            continue

        haystack_parts = [
            relpath,
            str(block.get("id", "")),
            str(block.get("name", "")),
            str(block.get("qualname", "")),
            str(block.get("desc_short", "")),
        ]
        haystack = " ".join(haystack_parts).lower()
        if query and query not in haystack:
            continue

        score = 0
        if query:
            if query in str(block.get("id", "")).lower():
                score += 5
            if query in str(block.get("qualname", "")).lower():
                score += 4
            if query in str(block.get("name", "")).lower():
                score += 3
            if query in str(block.get("desc_short", "")).lower():
                score += 2
            if query in relpath.lower():
                score += 1

        matches.append(
            {
                "id": block.get("id"),
                "path": relpath,
                "kind": block.get("kind"),
                "name": block.get("name"),
                "qualname": block.get("qualname"),
                "start_line": block.get("start_line"),
                "end_line": block.get("end_line"),
                "desc_short": block.get("desc_short"),
                "_score": score,
            }
        )

    matches.sort(
        key=lambda row: (
            -int(row.get("_score", 0)),
            str(row.get("path", "")),
            int(row.get("start_line", 0)),
            str(row.get("id", "")),
        )
    )

    limit = max(1, int(args.limit))
    limited = matches[:limit]

    if args.json:
        for row in limited:
            row.pop("_score", None)
        print(json.dumps(limited, indent=2, sort_keys=True))
        return 0

    if not limited:
        print("No matching blocks found.")
        return 0

    print(f"Found {len(matches)} matching blocks (showing {len(limited)}):")
    for row in limited:
        print(f"- id: {row['id']}")
        print(f"  path: {row['path']}:{row['start_line']}")
        print(f"  kind: {row['kind']}  name: {row['name']}")
        if row.get("desc_short"):
            print(f"  description: {row['desc_short']}")
    print("Use `resolve --id <id>` or `read --id <id>` for exact spans/content.")
    return 0


def cmd_ls(args: argparse.Namespace) -> int:
    """List all indexed files, or list blocks within a specific file."""
    root = Path(args.root).resolve()
    try:
        ensure_index(root, auto_build=args.auto_build, paths=args.paths)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    validation_scope = [normalize_path_query(root, args.file)] if args.file else args.paths
    validate_index_scope(root=root, include_paths=validation_scope)
    index = load_index(root)

    if not args.file:
        rows = []
        for relpath, entry in sorted(index.get("files", {}).items()):
            rows.append(
                {
                    "path": relpath,
                    "kind": entry.get("kind"),
                    "blocks": len(entry.get("blocks", [])),
                    "indexed_at": entry.get("indexed_at"),
                }
            )
        if args.json:
            print(json.dumps(rows, indent=2, sort_keys=True))
        else:
            print(f"Indexed files: {len(rows)}")
            for row in rows[: max(1, int(args.limit))]:
                print(f"- {row['path']} ({row['kind']}, blocks={row['blocks']})")
        return 0

    relpath = normalize_path_query(root, args.file)
    if relpath not in index.get("files", {}):
        target = root / relpath
        if target.exists() and target.is_file() and target.suffix.lower() in SUPPORTED_SUFFIXES:
            index["files"][relpath] = index_file(root, target)
            save_index(root, index)
        else:
            print(f"File not found in index: {relpath}", file=sys.stderr)
            return 1

    entry = index["files"][relpath]
    blocks = sorted(
        entry.get("blocks", []),
        key=lambda block: (int(block.get("start_line", 0)), str(block.get("id", ""))),
    )
    limit = max(1, int(args.limit))
    limited = blocks[:limit]

    if args.json:
        payload = {
            "path": relpath,
            "kind": entry.get("kind"),
            "sha256": entry.get("sha256"),
            "blocks": limited,
        }
        if args.show_deps != "none":
            payload["dependencies"] = collect_dependency_rows(
                entry=entry,
                deps_mode=args.show_deps,
                limit=max(1, int(args.deps_limit)),
            )
        print(
            json.dumps(payload, indent=2, sort_keys=True)
        )
        return 0

    print(f"path: {relpath}")
    print(f"kind: {entry.get('kind')}")
    print(f"blocks: {len(blocks)} (showing {len(limited)})")
    for block in limited:
        print(
            f"- {block.get('id')} "
            f"[{block.get('kind')}] "
            f"{block.get('start_line')}..{block.get('end_line')}"
        )

    if args.show_deps != "none":
        deps = collect_dependency_rows(
            entry=entry,
            deps_mode=args.show_deps,
            limit=max(1, int(args.deps_limit)),
        )
        print(f"dependencies ({args.show_deps}): {len(deps)} shown")
        for dep in deps:
            if dep.get("resolved_path"):
                print(f"- {dep['kind']} {dep['module']} -> {dep['resolved_path']}")
            else:
                unresolved = " (unresolved)" if dep.get("unresolved") else ""
                print(f"- {dep['kind']} {dep['module']}{unresolved}")
    return 0


def _block_visible_for_detail(kind: str, detail: str) -> bool:
    """Determine whether a block kind should be shown for the given detail level."""
    if detail == "files":
        return False
    if detail == "symbols":
        return kind in {"class", "function", "method", "heading"}
    if detail == "all":
        return kind not in {"module", "document"}
    return False


def _iter_visible_blocks_for_file(
    entry: dict[str, Any],
    detail: str,
    max_block_depth: int,
) -> list[tuple[int, dict[str, Any]]]:
    """Walk the block parent-child tree and collect visible blocks with their depth."""
    blocks = entry.get("blocks", [])
    if not blocks or detail == "files":
        return []

    blocks_by_id: dict[str, dict[str, Any]] = {}
    children: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        block_id = block.get("id")
        if not block_id:
            continue
        blocks_by_id[block_id] = block

    for block in blocks:
        block_id = block.get("id")
        if not block_id:
            continue
        parent_id = block.get("parent_id")
        if parent_id not in blocks_by_id:
            parent_id = None
        children[parent_id].append(block)

    for parent_id in list(children.keys()):
        children[parent_id].sort(
            key=lambda item: (
                int(item.get("start_line", 0)),
                int(item.get("end_line", 0)),
                str(item.get("id", "")),
            )
        )

    rendered: list[tuple[int, dict[str, Any]]] = []

    def visit(node: dict[str, Any], depth: int) -> None:
        kind = str(node.get("kind", ""))
        show = _block_visible_for_detail(kind, detail)
        child_depth = depth

        if show:
            if depth > max_block_depth:
                return
            rendered.append((depth, node))
            child_depth = depth + 1

        node_id = node.get("id")
        for child in children.get(node_id, []):
            visit(child, child_depth)

    for root_block in children.get(None, []):
        visit(root_block, 0)
    return rendered


def _build_directory_tree(files: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a nested dict representing the directory hierarchy from file rows."""
    tree: dict[str, Any] = {"dirs": {}, "files": []}
    for file_row in files:
        display_rel = file_row["display_rel"]
        parts = [part for part in display_rel.split("/") if part]
        if not parts:
            continue
        node = tree
        for part in parts[:-1]:
            node = node["dirs"].setdefault(part, {"dirs": {}, "files": []})
        node["files"].append(file_row)

    def sort_node(node: dict[str, Any]) -> None:
        node["files"].sort(key=lambda row: row["display_rel"])
        for child in node["dirs"].values():
            sort_node(child)

    sort_node(tree)
    return tree


def _render_directory_tree_lines(
    node: dict[str, Any],
    prefix: str,
    depth: int,
    max_dir_depth: int,
    out_lines: list[str],
    detail: str,
    max_block_depth: int,
    show_desc: str,
    show_signatures: bool,
    show_deps: str,
    deps_limit: int,
) -> None:
    """Recursively render a directory tree node into indented text lines."""
    dir_names = sorted(node["dirs"].keys())
    files = node["files"]

    entries: list[tuple[str, Any]] = [("dir", name) for name in dir_names] + [
        ("file", file_row) for file_row in files
    ]

    for idx, (entry_type, payload) in enumerate(entries):
        is_last = idx == len(entries) - 1
        branch = "└── " if is_last else "├── "
        child_prefix = prefix + ("    " if is_last else "│   ")

        if entry_type == "dir":
            dirname = payload
            out_lines.append(f"{prefix}{branch}{dirname}/")
            if depth < max_dir_depth:
                _render_directory_tree_lines(
                    node=node["dirs"][dirname],
                    prefix=child_prefix,
                    depth=depth + 1,
                    max_dir_depth=max_dir_depth,
                    out_lines=out_lines,
                    detail=detail,
                    max_block_depth=max_block_depth,
                    show_desc=show_desc,
                    show_signatures=show_signatures,
                    show_deps=show_deps,
                    deps_limit=deps_limit,
                )
            continue

        file_row = payload
        filename = Path(file_row["display_rel"]).name
        file_kind = file_row["kind"]
        out_lines.append(f"{prefix}{branch}{filename} [{file_kind}]")

        if detail == "files":
            continue

        visible_blocks = _iter_visible_blocks_for_file(
            entry=file_row["entry"],
            detail=detail,
            max_block_depth=max_block_depth,
        )
        for block_idx, (block_depth, block) in enumerate(visible_blocks):
            block_is_last = block_idx == len(visible_blocks) - 1
            block_branch = "└── " if block_is_last else "├── "
            block_prefix = child_prefix + ("    " * block_depth)

            block_kind = str(block.get("kind", ""))
            block_name = str(block.get("name", ""))
            start = block.get("start_line")
            end = block.get("end_line")
            sig = block.get("signature") or "" if show_signatures else ""
            out_lines.append(
                f"{block_prefix}{block_branch}{block_kind} {block_name}{sig} ({start}..{end})"
            )
            desc_lines = _description_lines_for_block(block, show_desc)
            if desc_lines:
                cont_prefix = block_prefix + ("    " if block_is_last else "│   ")
                for desc_line in desc_lines:
                    out_lines.append(f"{cont_prefix}- {desc_line}")

        if show_deps != "none":
            deps = collect_dependency_rows(
                entry=file_row["entry"],
                deps_mode=show_deps,
                limit=deps_limit,
            )
            for dep_idx, dep in enumerate(deps):
                dep_is_last = dep_idx == len(deps) - 1
                dep_branch = "└── " if dep_is_last else "├── "
                dep_prefix = child_prefix
                if dep.get("resolved_path"):
                    text = f"dep {dep['kind']} {dep['module']} -> {dep['resolved_path']}"
                else:
                    unresolved = " (unresolved)" if dep.get("unresolved") else ""
                    text = f"dep {dep['kind']} {dep['module']}{unresolved}"
                out_lines.append(f"{dep_prefix}{dep_branch}{text}")


def _description_lines_for_block(block: dict[str, Any], show_desc: str) -> list[str]:
    """Return formatted description lines for a block based on the show_desc mode."""
    if show_desc == "none":
        return []
    if show_desc == "short":
        value = block.get("desc_short")
        if not value:
            return []
        return [compact_text(str(value), max_chars=240)]

    value = block.get("desc_long") or block.get("desc_short")
    if not value:
        return []
    lines = [compact_text(part.strip(), max_chars=240) for part in str(value).splitlines() if part.strip()]
    return lines[:5]


def _description_value_for_block(block: dict[str, Any], show_desc: str) -> str | None:
    """Return a single description string for JSON output, or None if hidden."""
    if show_desc == "none":
        return None
    if show_desc == "short":
        value = block.get("desc_short")
        return compact_text(str(value), max_chars=240) if value else None

    value = block.get("desc_long") or block.get("desc_short")
    if not value:
        return None
    lines = [compact_text(part.strip(), max_chars=240) for part in str(value).splitlines() if part.strip()]
    return "\n".join(lines[:5]) if lines else None


def cmd_tree(args: argparse.Namespace) -> int:
    """Print the repository structure as a tree with optional block and description detail."""
    root = Path(args.root).resolve()
    try:
        ensure_index(root, auto_build=args.auto_build, paths=args.paths)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    validation_summary = validate_index_scope(
        root=root,
        include_paths=_query_validation_scope(root, args.path, args.paths),
    )
    refreshed_any = bool(validation_summary["updated"] or validation_summary["removed"])
    index = load_index(root)

    base_path = normalize_path_query(root, args.path) if args.path else ""
    is_single_file = False

    selected_relpaths: list[str] = []
    if base_path:
        if base_path in index.get("files", {}):
            selected_relpaths = [base_path]
            is_single_file = True
        else:
            target = root / base_path
            if target.exists() and target.is_file() and target.suffix.lower() in SUPPORTED_SUFFIXES:
                index["files"][base_path] = index_file(root, target)
                save_index(root, index)
                selected_relpaths = [base_path]
                is_single_file = True
            else:
                prefix = base_path.rstrip("/") + "/"
                selected_relpaths = sorted(
                    relpath
                    for relpath in index.get("files", {})
                    if relpath.startswith(prefix)
                )
    else:
        selected_relpaths = sorted(index.get("files", {}).keys())

    if not selected_relpaths:
        print("No indexed files matched the requested path/filter.")
        return 1

    file_rows: list[dict[str, Any]] = []
    for relpath in selected_relpaths:
        entry = index["files"].get(relpath)
        if entry is None:
            continue

        display_rel = relpath
        if base_path and not is_single_file:
            prefix = base_path.rstrip("/") + "/"
            display_rel = relpath[len(prefix) :] if relpath.startswith(prefix) else relpath
        file_rows.append(
            {
                "relpath": relpath,
                "display_rel": display_rel,
                "kind": entry.get("kind"),
                "entry": entry,
            }
        )

    file_kind = getattr(args, "file_kind", "both")
    if file_kind != "both":
        allowed_kind = "python" if file_kind == "scripts" else "markdown"
        file_rows = [row for row in file_rows if row["kind"] == allowed_kind]

    if args.json:
        payload_files = []
        for row in sorted(file_rows, key=lambda item: item["display_rel"]):
            blocks = _iter_visible_blocks_for_file(
                entry=row["entry"],
                detail=args.detail,
                max_block_depth=max(0, int(args.block_depth)),
            )

            payload_blocks = []
            for depth, block in blocks:
                block_payload = {
                    "depth": depth,
                    "id": block.get("id"),
                    "kind": block.get("kind"),
                    "name": block.get("name"),
                    "start_line": block.get("start_line"),
                    "end_line": block.get("end_line"),
                }
                if args.show_signatures and block.get("signature"):
                    block_payload["signature"] = block["signature"]
                description = _description_value_for_block(block, args.show_desc)
                if description:
                    block_payload["description"] = description
                payload_blocks.append(block_payload)

            payload_files.append(
                {
                    "path": row["relpath"],
                    "display_path": row["display_rel"],
                    "kind": row["kind"],
                    "blocks": payload_blocks,
                    "dependencies": collect_dependency_rows(
                        entry=row["entry"],
                        deps_mode=args.show_deps,
                        limit=max(1, int(args.deps_limit)),
                    )
                    if args.show_deps != "none"
                    else [],
                }
            )
        print(
            json.dumps(
                {
                    "root": str(root),
                    "base_path": base_path or ".",
                    "detail": args.detail,
                    "dir_depth": int(args.dir_depth),
                    "block_depth": int(args.block_depth),
                    "show_desc": args.show_desc,
                    "show_deps": args.show_deps,
                    "file_kind": file_kind,
                    "refreshed": refreshed_any,
                    "files": payload_files,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    header_base = base_path or "."
    print(
        f"Structure for {header_base} "
        f"(detail={args.detail}, file_kind={file_kind}, "
        f"dir_depth={args.dir_depth}, block_depth={args.block_depth}, "
        f"show_desc={args.show_desc}, show_deps={args.show_deps}, "
        f"files={len(file_rows)}, refreshed={refreshed_any})"
    )

    if is_single_file and len(file_rows) == 1:
        row = file_rows[0]
        print(f"└── {row['display_rel']} [{row['kind']}]")
        if args.detail != "files":
            blocks = _iter_visible_blocks_for_file(
                entry=row["entry"],
                detail=args.detail,
                max_block_depth=max(0, int(args.block_depth)),
            )
            for idx, (depth, block) in enumerate(blocks):
                branch = "└── " if idx == len(blocks) - 1 else "├── "
                indent = "    " * (depth + 1)
                sig = block.get("signature") or "" if args.show_signatures else ""
                print(
                    f"{indent}{branch}{block.get('kind')} {block.get('name')}{sig} "
                    f"({block.get('start_line')}..{block.get('end_line')})"
                )
                for desc_line in _description_lines_for_block(block, args.show_desc):
                    print(f"{indent}    - {desc_line}")
        if args.show_deps != "none":
            deps = collect_dependency_rows(
                entry=row["entry"],
                deps_mode=args.show_deps,
                limit=max(1, int(args.deps_limit)),
            )
            for dep in deps:
                if dep.get("resolved_path"):
                    print(f"    - dep {dep['kind']} {dep['module']} -> {dep['resolved_path']}")
                else:
                    unresolved = " (unresolved)" if dep.get("unresolved") else ""
                    print(f"    - dep {dep['kind']} {dep['module']}{unresolved}")
        return 0

    tree = _build_directory_tree(file_rows)
    out_lines: list[str] = ["."]
    _render_directory_tree_lines(
        node=tree,
        prefix="",
        depth=0,
        max_dir_depth=max(0, int(args.dir_depth)),
        out_lines=out_lines,
        detail=args.detail,
        max_block_depth=max(0, int(args.block_depth)),
        show_desc=args.show_desc,
        show_signatures=args.show_signatures,
        show_deps=args.show_deps,
        deps_limit=max(1, int(args.deps_limit)),
    )
    print("\n".join(out_lines))
    return 0
