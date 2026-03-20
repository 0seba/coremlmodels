"""Dependency tree command: build and render import graphs rooted at a file.

Supports forward (imports) and reverse (imported-by) traversal with
cycle detection, depth limits, and optional file descriptions.
"""

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
from ..ops import ensure_index, validate_index_scope
from ..parsers import index_file
from ..storage import load_index, save_index

_FILE_PREFIX = "file:"
_EXTERNAL_PREFIX = "external:"
_UNRESOLVED_INTERNAL_PREFIX = "internal-unresolved:"


def _file_key(relpath: str) -> str:
    return f"{_FILE_PREFIX}{relpath}"


def _external_key(module: str) -> str:
    return f"{_EXTERNAL_PREFIX}{module}"


def _unresolved_internal_key(module: str) -> str:
    return f"{_UNRESOLVED_INTERNAL_PREFIX}{module}"


def _key_type(node_key: str) -> str:
    if node_key.startswith(_FILE_PREFIX):
        return "file"
    if node_key.startswith(_EXTERNAL_PREFIX):
        return "external"
    if node_key.startswith(_UNRESOLVED_INTERNAL_PREFIX):
        return "internal_unresolved"
    return "unknown"


def _key_value(node_key: str) -> str:
    return node_key.split(":", 1)[1] if ":" in node_key else node_key


def _description_lines_for_entry(entry: dict[str, Any], show_desc: str) -> list[str]:
    """Extract module-level description lines from a file entry for tree display."""
    if show_desc == "none":
        return []

    blocks = entry.get("blocks", [])
    if not blocks:
        return []
    module_like = next(
        (
            block
            for block in blocks
            if block.get("parent_id") is None and block.get("kind") in {"module", "document"}
        ),
        blocks[0],
    )
    if show_desc == "short":
        value = module_like.get("desc_short")
        if not value:
            return []
        return [compact_text(str(value), max_chars=240)]

    value = module_like.get("desc_long") or module_like.get("desc_short")
    if not value:
        return []
    lines = [compact_text(part.strip(), max_chars=240) for part in str(value).splitlines() if part.strip()]
    return lines[:5]


def _description_value_for_entry(entry: dict[str, Any], show_desc: str) -> str | None:
    lines = _description_lines_for_entry(entry, show_desc)
    if not lines:
        return None
    if show_desc == "short":
        return lines[0]
    return "\n".join(lines)


def _build_dependency_adjacency(
    index: dict[str, Any],
    deps_mode: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """Build forward and reverse adjacency lists from all indexed file imports."""
    forward: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reverse: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for source_relpath, entry in sorted(index.get("files", {}).items()):
        source_key = _file_key(source_relpath)
        deps = collect_dependency_rows(entry=entry, deps_mode=deps_mode, limit=100000)
        for dep in deps:
            dep_kind = str(dep.get("kind", ""))
            module = str(dep.get("module", ""))
            resolved_path = dep.get("resolved_path")
            unresolved = bool(dep.get("unresolved", False))

            if dep_kind == "internal" and resolved_path:
                target_key = _file_key(str(resolved_path))
                target_type = "file"
            elif dep_kind == "internal":
                target_key = _unresolved_internal_key(module)
                target_type = "internal_unresolved"
            else:
                target_key = _external_key(module)
                target_type = "external"

            edge = {
                "from_key": source_key,
                "from_relpath": source_relpath,
                "to_key": target_key,
                "to_relpath": str(resolved_path) if resolved_path else None,
                "to_type": target_type,
                "dep_kind": dep_kind,
                "module": module,
                "unresolved": unresolved,
            }
            forward[source_key].append(edge)
            reverse[target_key].append(edge)

    for key in list(forward.keys()):
        forward[key].sort(
            key=lambda edge: (
                str(edge.get("dep_kind", "")),
                str(edge.get("to_relpath") or edge.get("module") or edge.get("to_key")),
            )
        )
    for key in list(reverse.keys()):
        reverse[key].sort(
            key=lambda edge: (
                str(edge.get("dep_kind", "")),
                str(edge.get("from_relpath", "")),
                str(edge.get("module", "")),
            )
        )
    return forward, reverse


def _node_display_label(node_key: str, index: dict[str, Any]) -> str:
    ntype = _key_type(node_key)
    value = _key_value(node_key)
    if ntype == "file":
        entry = index.get("files", {}).get(value)
        kind = entry.get("kind") if entry else "file"
        return f"{value} [{kind}]"
    if ntype == "external":
        return f"{value} [external]"
    if ntype == "internal_unresolved":
        return f"{value} [internal unresolved]"
    return value


def _edge_label(edge: dict[str, Any], direction: str) -> str:
    module = str(edge.get("module", ""))
    dep_kind = str(edge.get("dep_kind", ""))
    if direction == "imports":
        if edge.get("to_type") == "file":
            return f"[{dep_kind}] {edge.get('to_relpath')} (via {module})"
        if edge.get("to_type") == "internal_unresolved":
            unresolved = " (unresolved)" if edge.get("unresolved") else ""
            return f"[internal] {module}{unresolved}"
        return f"[external] {module}"
    return f"[imported-by] {edge.get('from_relpath')} (via {module})"


def _child_key_for_edge(edge: dict[str, Any], direction: str) -> str:
    if direction == "imports":
        return str(edge.get("to_key"))
    return str(edge.get("from_key"))


def _build_dep_tree_json(
    root_key: str,
    index: dict[str, Any],
    adjacency: dict[str, list[dict[str, Any]]],
    direction: str,
    depth: int,
    max_children: int,
    show_desc: str,
) -> dict[str, Any]:
    """Recursively build a JSON-serialisable dependency tree with cycle detection."""
    def visit(node_key: str, level: int, ancestors: set[str]) -> dict[str, Any]:
        node_type = _key_type(node_key)
        node_value = _key_value(node_key)
        entry = index.get("files", {}).get(node_value) if node_type == "file" else None
        node_payload: dict[str, Any] = {
            "key": node_key,
            "type": node_type,
            "label": _node_display_label(node_key, index),
        }
        if node_type == "file":
            node_payload["path"] = node_value
            node_payload["kind"] = entry.get("kind") if entry else "file"
            description = _description_value_for_entry(entry or {}, show_desc)
            if description:
                node_payload["description"] = description
        else:
            node_payload["name"] = node_value

        if level >= depth:
            node_payload["children"] = []
            node_payload["truncated_by_depth"] = True
            return node_payload

        edges = list(adjacency.get(node_key, []))
        total_children = len(edges)
        edges = edges[: max_children]
        children_payload = []
        for edge in edges:
            child_key = _child_key_for_edge(edge, direction)
            edge_payload: dict[str, Any] = {
                "edge_label": _edge_label(edge, direction),
                "dep_kind": edge.get("dep_kind"),
                "module": edge.get("module"),
                "node": {
                    "key": child_key,
                    "label": _node_display_label(child_key, index),
                    "type": _key_type(child_key),
                },
            }
            if child_key in ancestors:
                edge_payload["cycle"] = True
                children_payload.append(edge_payload)
                continue
            edge_payload["node"] = visit(child_key, level + 1, ancestors | {child_key})
            children_payload.append(edge_payload)

        node_payload["children"] = children_payload
        if total_children > len(edges):
            node_payload["truncated_children"] = total_children - len(edges)
        return node_payload

    return visit(root_key, level=0, ancestors={root_key})


def _render_dep_tree_text(
    root_key: str,
    index: dict[str, Any],
    adjacency: dict[str, list[dict[str, Any]]],
    direction: str,
    depth: int,
    max_children: int,
    show_desc: str,
) -> list[str]:
    """Render the dependency tree as indented text lines with box-drawing characters."""
    out_lines: list[str] = [f"└── {_node_display_label(root_key, index)}"]
    root_type = _key_type(root_key)
    root_value = _key_value(root_key)
    if root_type == "file":
        root_entry = index.get("files", {}).get(root_value, {})
        for desc_line in _description_lines_for_entry(root_entry, show_desc):
            out_lines.append(f"    - {desc_line}")

    def visit(node_key: str, prefix: str, level: int, ancestors: set[str]) -> None:
        if level >= depth:
            return

        edges = list(adjacency.get(node_key, []))
        total_children = len(edges)
        edges = edges[:max_children]
        for idx, edge in enumerate(edges):
            is_last = idx == len(edges) - 1
            branch = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")
            child_key = _child_key_for_edge(edge, direction)

            out_lines.append(f"{prefix}{branch}{_edge_label(edge, direction)}")

            child_type = _key_type(child_key)
            if child_type == "file":
                child_relpath = _key_value(child_key)
                child_entry = index.get("files", {}).get(child_relpath, {})
                for desc_line in _description_lines_for_entry(child_entry, show_desc):
                    out_lines.append(f"{child_prefix}- {desc_line}")

            if child_key in ancestors:
                out_lines.append(f"{child_prefix}↺ cycle: {_node_display_label(child_key, index)}")
                continue

            visit(child_key, child_prefix, level + 1, ancestors | {child_key})

        if total_children > len(edges):
            out_lines.append(f"{prefix}... ({total_children - len(edges)} more dependencies not shown)")

    visit(root_key, prefix="    ", level=0, ancestors={root_key})
    return out_lines


def cmd_dep_tree(args: argparse.Namespace) -> int:
    """Print an import dependency graph rooted at a given file."""
    root = Path(args.root).resolve()
    try:
        ensure_index(root, auto_build=args.auto_build, paths=args.paths)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    start_relpath = normalize_path_query(root, args.path)
    validation_scope = args.paths if args.paths is not None else [start_relpath]
    validate_index_scope(root=root, include_paths=validation_scope)
    index = load_index(root)

    if start_relpath not in index.get("files", {}):
        target = root / start_relpath
        if target.exists() and target.is_file() and target.suffix.lower() in SUPPORTED_SUFFIXES:
            index["files"][start_relpath] = index_file(root, target)
            save_index(root, index)
        else:
            print(f"File not found in index: {start_relpath}", file=sys.stderr)
            return 1

    forward, reverse = _build_dependency_adjacency(index=index, deps_mode=args.deps)
    direction = args.direction
    adjacency = forward if direction == "imports" else reverse

    root_key = _file_key(start_relpath)
    depth = max(0, int(args.depth))
    max_children = max(1, int(args.max_children))

    if args.json:
        payload = {
            "root": str(root),
            "path": start_relpath,
            "direction": direction,
            "deps": args.deps,
            "depth": depth,
            "max_children": max_children,
            "show_desc": args.show_desc,
            "tree": _build_dep_tree_json(
                root_key=root_key,
                index=index,
                adjacency=adjacency,
                direction=direction,
                depth=depth,
                max_children=max_children,
                show_desc=args.show_desc,
            ),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(
        f"Dependency tree for {start_relpath} "
        f"(direction={direction}, deps={args.deps}, depth={depth}, "
        f"max_children={max_children}, show_desc={args.show_desc})"
    )
    lines = _render_dep_tree_text(
        root_key=root_key,
        index=index,
        adjacency=adjacency,
        direction=direction,
        depth=depth,
        max_children=max_children,
        show_desc=args.show_desc,
    )
    print("\n".join(lines))
    return 0
