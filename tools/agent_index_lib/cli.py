"""CLI argument parser and entry point for the agent-index tool.

Defines subcommands for building, querying, and navigating the runtime
block index: update, check, find, ls, tree, dep-tree, resolve, read, watch.
"""

from __future__ import annotations

import argparse

from .commands.blocks import cmd_read, cmd_resolve
from .commands.dependency_tree import cmd_dep_tree
from .commands.discover import cmd_find, cmd_ls, cmd_tree
from .commands.manage import cmd_check, cmd_update, cmd_watch


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser with all agent-index subcommands."""
    parser = argparse.ArgumentParser(
        description="Runtime indexing for agent-oriented code/doc navigation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    update_parser = subparsers.add_parser("update", help="Build or refresh runtime index.")
    update_parser.add_argument("--root", default=".", help="Repository root.")
    update_parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="Optional file/dir paths to restrict indexing scope.",
    )
    update_parser.add_argument(
        "--changed-only",
        action="store_true",
        help="Only reindex files whose sha256 changed.",
    )
    update_parser.add_argument(
        "--remove-missing",
        action="store_true",
        help="Remove missing files from index.",
    )
    update_parser.add_argument("--json", action="store_true", help="Output JSON summary.")
    update_parser.set_defaults(func=cmd_update)

    check_parser = subparsers.add_parser("check", help="Check for stale or missing indexed files.")
    check_parser.add_argument("--root", default=".", help="Repository root.")
    check_parser.add_argument(
        "--fix",
        action="store_true",
        help="Reindex stale files and remove missing files.",
    )
    check_parser.set_defaults(func=cmd_check)

    find_parser = subparsers.add_parser(
        "find",
        help="Search indexed blocks and return block IDs.",
    )
    find_parser.add_argument("--root", default=".", help="Repository root.")
    find_parser.add_argument(
        "--query",
        default="",
        help="Substring query over id/name/path/description.",
    )
    find_parser.add_argument(
        "--path",
        default=None,
        help="Optional path substring filter (e.g., src/coremlmodels/vision_model_wrapper.py).",
    )
    find_parser.add_argument(
        "--kind",
        default=None,
        help="Optional block kind filter (class/function/method/section/heading/etc).",
    )
    find_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum matches to print.",
    )
    find_parser.add_argument(
        "--auto-build",
        action="store_true",
        default=True,
        help="Build index if missing (default: enabled).",
    )
    find_parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="Optional scope used only when auto-building an index.",
    )
    find_parser.add_argument("--json", action="store_true", help="Output JSON.")
    find_parser.set_defaults(func=cmd_find)

    ls_parser = subparsers.add_parser(
        "ls",
        help="List indexed files, or list blocks for a specific file.",
    )
    ls_parser.add_argument("--root", default=".", help="Repository root.")
    ls_parser.add_argument(
        "--file",
        default=None,
        help="List blocks for this file path. If omitted, lists indexed files.",
    )
    ls_parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum rows/blocks to print.",
    )
    ls_parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Legacy compatibility flag; scoped file-hash validation still runs.",
    )
    ls_parser.add_argument(
        "--auto-build",
        action="store_true",
        default=True,
        help="Build index if missing (default: enabled).",
    )
    ls_parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="Optional scope used only when auto-building an index.",
    )
    ls_parser.add_argument("--json", action="store_true", help="Output JSON.")
    ls_parser.add_argument(
        "--show-deps",
        choices=("none", "internal", "external", "all"),
        default="none",
        help="Show import dependencies (when --file is used).",
    )
    ls_parser.add_argument(
        "--deps-limit",
        type=int,
        default=30,
        help="Maximum dependencies to print when --show-deps is enabled.",
    )
    ls_parser.set_defaults(func=cmd_ls)

    tree_parser = subparsers.add_parser(
        "tree",
        help="Print repository structure (dirs/files) with optional block granularity.",
    )
    tree_parser.add_argument("--root", default=".", help="Repository root.")
    tree_parser.add_argument(
        "--path",
        default=None,
        help="Optional file or directory prefix to scope output.",
    )
    tree_parser.add_argument(
        "--file-kind",
        choices=("scripts", "docs", "both"),
        default="both",
        help="Show only scripts (.py), only docs (.md), or both.",
    )
    tree_parser.add_argument(
        "--detail",
        choices=("files", "symbols", "all"),
        default="symbols",
        help="Granularity for per-file blocks.",
    )
    tree_parser.add_argument(
        "--dir-depth",
        type=int,
        default=6,
        help="Maximum directory tree depth to print.",
    )
    tree_parser.add_argument(
        "--block-depth",
        type=int,
        default=6,
        help="Maximum nested block depth to print inside each file.",
    )
    tree_parser.add_argument(
        "--show-desc",
        choices=("none", "short", "long"),
        default="none",
        help="Show block descriptions in tree output.",
    )
    tree_parser.add_argument(
        "--show-signatures",
        action="store_true",
        default=False,
        help="Show function/method/class signatures inline.",
    )
    tree_parser.add_argument(
        "--show-deps",
        choices=("none", "internal", "external", "all"),
        default="none",
        help="Show per-file import dependencies in tree output.",
    )
    tree_parser.add_argument(
        "--deps-limit",
        type=int,
        default=30,
        help="Maximum dependencies to print per file when --show-deps is enabled.",
    )
    tree_parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Legacy compatibility flag; scoped file-hash validation still runs.",
    )
    tree_parser.add_argument(
        "--auto-build",
        action="store_true",
        default=True,
        help="Build index if missing (default: enabled).",
    )
    tree_parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="Optional scope used only when auto-building an index.",
    )
    tree_parser.add_argument("--json", action="store_true", help="Output JSON.")
    tree_parser.set_defaults(func=cmd_tree)

    dep_tree_parser = subparsers.add_parser(
        "dep-tree",
        help="Print import dependency graph rooted at a file.",
    )
    dep_tree_parser.add_argument("--root", default=".", help="Repository root.")
    dep_tree_parser.add_argument(
        "--path",
        required=True,
        help="Root file path for dependency exploration.",
    )
    dep_tree_parser.add_argument(
        "--direction",
        choices=("imports", "imported-by"),
        default="imports",
        help="Traverse imports from file or reverse imported-by relationships.",
    )
    dep_tree_parser.add_argument(
        "--deps",
        choices=("internal", "external", "all"),
        default="all",
        help="Which dependency kinds to include.",
    )
    dep_tree_parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Maximum traversal depth.",
    )
    dep_tree_parser.add_argument(
        "--max-children",
        type=int,
        default=40,
        help="Maximum dependencies shown per node.",
    )
    dep_tree_parser.add_argument(
        "--show-desc",
        choices=("none", "short", "long"),
        default="none",
        help="Show indexed file descriptions for file nodes.",
    )
    dep_tree_parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Legacy compatibility flag; scoped file-hash validation still runs.",
    )
    dep_tree_parser.add_argument(
        "--auto-build",
        action="store_true",
        default=True,
        help="Build index if missing (default: enabled).",
    )
    dep_tree_parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="Optional scope used only when auto-building an index.",
    )
    dep_tree_parser.add_argument("--json", action="store_true", help="Output JSON.")
    dep_tree_parser.set_defaults(func=cmd_dep_tree)

    resolve_parser = subparsers.add_parser("resolve", help="Resolve block ID to current line ranges.")
    resolve_parser.add_argument("--root", default=".", help="Repository root.")
    resolve_parser.add_argument("--id", required=True, help="Block ID.")
    resolve_parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Legacy compatibility flag; scoped file-hash validation still runs.",
    )
    resolve_parser.add_argument(
        "--auto-build",
        action="store_true",
        default=True,
        help="Build index if missing (default: enabled).",
    )
    resolve_parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="Optional scope used only when auto-building an index.",
    )
    resolve_parser.add_argument("--json", action="store_true", help="Output JSON.")
    resolve_parser.set_defaults(func=cmd_resolve)

    read_parser = subparsers.add_parser("read", help="Read source by block ID.")
    read_parser.add_argument("--root", default=".", help="Repository root.")
    read_parser.add_argument("--id", required=True, help="Block ID.")
    read_parser.add_argument(
        "--description",
        action="store_true",
        help="Read description span instead of full block span.",
    )
    read_parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Maximum number of lines to print.",
    )
    read_parser.add_argument(
        "--line-numbers",
        action="store_true",
        default=True,
        help="Print line numbers (default: enabled).",
    )
    read_parser.add_argument(
        "--no-line-numbers",
        action="store_false",
        dest="line_numbers",
        help="Disable line numbers.",
    )
    read_parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Legacy compatibility flag; scoped file-hash validation still runs.",
    )
    read_parser.add_argument(
        "--auto-build",
        action="store_true",
        default=True,
        help="Build index if missing (default: enabled).",
    )
    read_parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="Optional scope used only when auto-building an index.",
    )
    read_parser.add_argument("--json", action="store_true", help="Output JSON.")
    read_parser.set_defaults(func=cmd_read)

    watch_parser = subparsers.add_parser(
        "watch",
        help="Poll filesystem changes and keep runtime index fresh.",
    )
    watch_parser.add_argument("--root", default=".", help="Repository root.")
    watch_parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="Optional file/dir paths to restrict watch scope.",
    )
    watch_parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds.",
    )
    watch_parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Run full update once before watching.",
    )
    watch_parser.set_defaults(func=cmd_watch)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
