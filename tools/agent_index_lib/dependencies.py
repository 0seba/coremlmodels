"""Dependency row collection for displaying import relationships in index output."""

from __future__ import annotations

from typing import Any


def collect_dependency_rows(
    entry: dict[str, Any],
    deps_mode: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Collect deduplicated dependency rows from an indexed file entry.

    Args:
        entry: Indexed file entry.
        deps_mode: One of `internal`, `external`, or `all`.
        limit: Maximum rows returned.
    """
    imports = entry.get("imports") or {}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def add_rows(kind: str, items: list[dict[str, Any]]) -> None:
        for item in items:
            module = str(item.get("module", ""))
            resolved_path = str(item.get("resolved_path") or "")
            key = (kind, module, resolved_path)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "kind": kind,
                    "module": module,
                    "resolved_path": resolved_path or None,
                    "unresolved": bool(item.get("unresolved", False)),
                }
            )
            if len(rows) >= limit:
                return

    if deps_mode in {"all", "internal"}:
        add_rows("internal", list(imports.get("internal", [])))
    if len(rows) >= limit:
        return rows[:limit]
    if deps_mode in {"all", "external"}:
        add_rows("external", list(imports.get("external", [])))
    return rows[:limit]
