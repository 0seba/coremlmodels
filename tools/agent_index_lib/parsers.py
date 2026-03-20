from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import (
    compact_text,
    first_paragraph_lines,
    first_sentence,
    line_indent,
    normalize_relpath,
    read_file_text,
    sha256_bytes,
    slugify,
    utc_now,
)
from .constants import ABSOLUTE_IMPORT_SEARCH_PREFIXES, MD_HEADING_RE, PY_SECTION_RE


@dataclass
class ContainerInfo:
    block_id: str
    qualname: str
    kind: str
    start_line: int
    end_line: int
    indent: int


def _format_arg(arg: ast.arg) -> str:
    name = arg.arg
    if arg.annotation is not None:
        return f"{name}: {ast.unparse(arg.annotation)}"
    return name


def _extract_signature(node: ast.AST) -> str | None:
    if isinstance(node, ast.ClassDef):
        if not node.bases:
            return None
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        return f"({bases})"

    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None

    args = node.args
    parts: list[str] = []

    # positional-only params
    for i, arg in enumerate(args.posonlyargs):
        default_offset = len(args.posonlyargs) - len(args.defaults)
        if i >= default_offset and i - default_offset < len(args.defaults):
            parts.append(f"{_format_arg(arg)}={ast.unparse(args.defaults[i - default_offset])}")
        else:
            parts.append(_format_arg(arg))
    if args.posonlyargs:
        parts.append("/")

    # regular positional params
    num_regular = len(args.args)
    num_defaults = len(args.defaults)
    defaults_start = len(args.posonlyargs) + num_regular - num_defaults
    for i, arg in enumerate(args.args):
        idx_global = len(args.posonlyargs) + i
        if idx_global >= defaults_start:
            d = args.defaults[idx_global - defaults_start]
            parts.append(f"{_format_arg(arg)}={ast.unparse(d)}")
        else:
            parts.append(_format_arg(arg))

    # *args
    if args.vararg:
        parts.append(f"*{_format_arg(args.vararg)}")
    elif args.kwonlyargs:
        parts.append("*")

    # keyword-only params
    for i, arg in enumerate(args.kwonlyargs):
        default = args.kw_defaults[i] if i < len(args.kw_defaults) else None
        if default is not None:
            parts.append(f"{_format_arg(arg)}={ast.unparse(default)}")
        else:
            parts.append(_format_arg(arg))

    # **kwargs
    if args.kwarg:
        parts.append(f"**{_format_arg(args.kwarg)}")

    sig = f"({', '.join(parts)})"
    if node.returns is not None:
        sig += f" -> {ast.unparse(node.returns)}"
    return sig


def _docstring_meta(node: ast.AST) -> tuple[str, int | None, int | None]:
    body = getattr(node, "body", None)
    if not body:
        return "", None, None

    first_stmt = body[0]
    if not isinstance(first_stmt, ast.Expr):
        return "", None, None

    value = first_stmt.value
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        doc = value.value
        return doc, getattr(first_stmt, "lineno", None), getattr(first_stmt, "end_lineno", None)
    return "", None, None


def _summarize_docstring(docstring: str) -> tuple[str, str]:
    if not docstring.strip():
        return "", ""
    short = first_sentence(docstring)
    paragraph = first_paragraph_lines(docstring, max_lines=5)
    long = "\n".join(paragraph)
    return short, long


def _summarize_python_section(
    lines: list[str],
    start_line: int,
    end_line: int,
) -> tuple[str, str, int | None, int | None]:
    if start_line > end_line or start_line > len(lines):
        return "", "", None, None

    comment_lines: list[tuple[int, str]] = []
    code_line: tuple[int, str] | None = None
    line_no = start_line
    while line_no <= end_line and line_no <= len(lines):
        raw = lines[line_no - 1]
        stripped = raw.strip()
        if not stripped:
            if comment_lines:
                break
            line_no += 1
            continue
        if stripped.startswith("#"):
            text = stripped.lstrip("#").strip()
            if text:
                comment_lines.append((line_no, text))
                line_no += 1
                continue
            if comment_lines:
                break
            line_no += 1
            continue
        code_line = (line_no, stripped)
        break

    if comment_lines:
        desc_start = comment_lines[0][0]
        desc_end = comment_lines[min(len(comment_lines), 5) - 1][0]
        long = "\n".join(line for _, line in comment_lines[:5])
        short = compact_text(" ".join(line for _, line in comment_lines[:2]))
        return short, long, desc_start, desc_end

    if code_line is not None:
        text = f"Code starts with: {code_line[1]}"
        short = compact_text(text)
        return short, "", code_line[0], code_line[0]

    return "", "", None, None


def _resolve_module_parts_with_prefixes(root: Path, module_parts: list[str]) -> str | None:
    if not module_parts:
        return None

    for prefix in ABSOLUTE_IMPORT_SEARCH_PREFIXES:
        base_path = root.joinpath(*prefix, *module_parts)
        as_file = base_path.with_suffix(".py")
        if as_file.exists() and as_file.is_file():
            return normalize_relpath(as_file, root)
        as_package = base_path / "__init__.py"
        if as_package.exists() and as_package.is_file():
            return normalize_relpath(as_package, root)
    return None


def _resolve_absolute_module(root: Path, module_name: str | None) -> str | None:
    if not module_name:
        return None
    return _resolve_module_parts_with_prefixes(root, module_name.split("."))


def _resolve_relative_module(
    root: Path,
    current_relpath: str,
    module_name: str | None,
    level: int,
) -> str | None:
    module_parts = module_name.split(".") if module_name else []

    current_module_parts = list(Path(current_relpath).with_suffix("").parts)
    current_package_parts = current_module_parts[:-1]
    up_count = max(level - 1, 0)
    if up_count > len(current_package_parts):
        return None
    prefix_parts = current_package_parts[: len(current_package_parts) - up_count]
    full_parts = prefix_parts + module_parts
    if not full_parts:
        return None

    base_path = root.joinpath(*full_parts)
    as_file = base_path.with_suffix(".py")
    if as_file.exists() and as_file.is_file():
        return normalize_relpath(as_file, root)
    as_package = base_path / "__init__.py"
    if as_package.exists() and as_package.is_file():
        return normalize_relpath(as_package, root)
    return None


def _looks_internal_absolute_module(root: Path, module_name: str | None) -> bool:
    if not module_name:
        return False
    top_level = module_name.split(".", 1)[0]
    if (root / top_level).exists():
        return True
    if (root / "src" / top_level).exists():
        return True
    if (root / "tools" / top_level).exists():
        return True
    return False


def _extract_python_imports(
    root: Path,
    relpath: str,
    tree: ast.AST,
) -> dict[str, list[dict[str, Any]]]:
    internal: list[dict[str, Any]] = []
    external: list[dict[str, Any]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                resolved_relpath = _resolve_absolute_module(root, module_name)
                entry = {
                    "import_type": "import",
                    "module": module_name,
                    "line": int(getattr(node, "lineno", 0)),
                    "alias": alias.asname,
                    "resolved_path": resolved_relpath,
                }
                if resolved_relpath is not None:
                    internal.append(entry)
                elif _looks_internal_absolute_module(root, module_name):
                    entry["unresolved"] = True
                    internal.append(entry)
                else:
                    external.append(entry)
            continue

        if isinstance(node, ast.ImportFrom):
            module_name = node.module
            level = int(getattr(node, "level", 0) or 0)
            if level > 0:
                resolved_relpath = _resolve_relative_module(
                    root=root,
                    current_relpath=relpath,
                    module_name=module_name,
                    level=level,
                )
                is_internal = True
            else:
                resolved_relpath = _resolve_absolute_module(root, module_name)
                is_internal = resolved_relpath is not None or _looks_internal_absolute_module(root, module_name)

            displayed_module = f"{'.' * level}{module_name or ''}" if level > 0 else (module_name or "")
            entry = {
                "import_type": "from",
                "module": displayed_module or "*",
                "line": int(getattr(node, "lineno", 0)),
                "names": [alias.name for alias in node.names],
                "aliases": {alias.name: alias.asname for alias in node.names if alias.asname},
                "resolved_path": resolved_relpath,
            }
            if is_internal:
                if resolved_relpath is None:
                    entry["unresolved"] = True
                internal.append(entry)
            else:
                external.append(entry)

    internal.sort(key=lambda item: (int(item.get("line", 0)), str(item.get("module", ""))))
    external.sort(key=lambda item: (int(item.get("line", 0)), str(item.get("module", ""))))
    return {"internal": internal, "external": external}


def _build_python_entry(root: Path, path: Path) -> dict[str, Any]:
    relpath = normalize_relpath(path, root)
    text = read_file_text(path)
    lines = text.splitlines()
    data = text.encode("utf-8", errors="replace")
    sha = sha256_bytes(data)

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return {
            "path": relpath,
            "kind": "python",
            "sha256": sha,
            "size_bytes": len(data),
            "indexed_at": utc_now(),
            "parse_error": f"{exc.__class__.__name__}: {exc}",
            "imports": {"internal": [], "external": []},
            "blocks": [],
        }

    blocks: list[dict[str, Any]] = []
    containers: list[ContainerInfo] = []
    id_counts: defaultdict[str, int] = defaultdict(int)

    def unique_id(base_id: str) -> str:
        id_counts[base_id] += 1
        if id_counts[base_id] == 1:
            return base_id
        return f"{base_id}#{id_counts[base_id]}"

    module_doc, module_desc_start, module_desc_end = _docstring_meta(tree)
    module_short, module_long = _summarize_docstring(module_doc)
    module_id = unique_id(f"py:{relpath}:module")
    module_block = {
        "id": module_id,
        "kind": "module",
        "name": "module",
        "qualname": "module",
        "parent_id": None,
        "start_line": 1,
        "end_line": max(len(lines), 1),
        "desc_short": module_short,
        "desc_long": module_long,
        "desc_start_line": module_desc_start,
        "desc_end_line": module_desc_end,
    }
    blocks.append(module_block)
    containers.append(
        ContainerInfo(
            block_id=module_id,
            qualname="module",
            kind="module",
            start_line=1,
            end_line=max(len(lines), 1),
            indent=0,
        )
    )

    def walk_nodes(
        body: list[ast.stmt],
        parent_qualname: str,
        parent_id: str,
        parent_kind: str,
    ) -> None:
        for node in body:
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            name = node.name
            qualname = f"{parent_qualname}.{name}" if parent_qualname else name
            start_line = int(getattr(node, "lineno", 1))
            end_line = int(getattr(node, "end_lineno", start_line))
            source_line = lines[start_line - 1] if 1 <= start_line <= len(lines) else ""
            indent = line_indent(source_line)

            if isinstance(node, ast.ClassDef):
                kind = "class"
            elif parent_kind == "class":
                kind = "method"
            elif parent_kind in {"function", "method"}:
                kind = "inner_function"
            else:
                kind = "function"

            base_id = f"py:{relpath}:{qualname}"
            block_id = unique_id(base_id)

            doc, desc_start, desc_end = _docstring_meta(node)
            desc_short, desc_long = _summarize_docstring(doc)
            # No fallback — blocks without docstrings get no description.
            # Fallback text like "Function `main`." just restates the name.

            signature = _extract_signature(node)

            block: dict[str, Any] = {
                "id": block_id,
                "kind": kind,
                "name": name,
                "qualname": qualname,
                "parent_id": parent_id,
                "start_line": start_line,
                "end_line": end_line,
                "signature": signature,
                "desc_short": desc_short,
                "desc_long": desc_long,
                "desc_start_line": desc_start,
                "desc_end_line": desc_end,
            }

            blocks.append(block)
            containers.append(
                ContainerInfo(
                    block_id=block_id,
                    qualname=qualname,
                    kind=kind,
                    start_line=start_line,
                    end_line=end_line,
                    indent=indent,
                )
            )

            walk_nodes(node.body, qualname, block_id, "class" if isinstance(node, ast.ClassDef) else "function")

    walk_nodes(tree.body, "", module_id, "module")

    parent_by_id = {container.block_id: container for container in containers}
    section_hits: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, start=1):
        match = PY_SECTION_RE.match(line)
        if not match:
            continue
        title = match.group(1).strip()
        indent = line_indent(line)
        candidates = [
            c
            for c in containers
            if c.start_line <= line_no <= c.end_line
            and indent >= c.indent
            and c.kind in {"module", "class", "function", "method", "inner_function"}
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda c: ((c.end_line - c.start_line), -len(c.qualname)))
        parent = candidates[0]
        section_hits.append(
            {
                "line_no": line_no,
                "title": title,
                "indent": indent,
                "parent_id": parent.block_id,
                "parent_qualname": parent.qualname,
            }
        )

    section_counts: defaultdict[str, int] = defaultdict(int)
    hits_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hit in section_hits:
        hits_by_parent[hit["parent_id"]].append(hit)

    for parent_id, hits in hits_by_parent.items():
        hits.sort(key=lambda item: item["line_no"])
        parent = parent_by_id[parent_id]
        for idx, hit in enumerate(hits):
            start_line = hit["line_no"]
            if idx + 1 < len(hits):
                end_line = max(start_line, hits[idx + 1]["line_no"] - 1)
            else:
                end_line = parent.end_line

            section_slug = slugify(hit["title"])
            base_id = f"py:{relpath}:{hit['parent_qualname']}:section:{section_slug}"
            section_counts[base_id] += 1
            block_id = base_id if section_counts[base_id] == 1 else f"{base_id}#{section_counts[base_id]}"

            desc_short, desc_long, desc_start, desc_end = _summarize_python_section(
                lines,
                start_line + 1,
                end_line,
            )
            # No fallback — sections without a leading comment get no description.

            blocks.append(
                {
                    "id": block_id,
                    "kind": "section",
                    "name": hit["title"],
                    "qualname": f"{hit['parent_qualname']}:section:{section_slug}",
                    "parent_id": parent_id,
                    "start_line": start_line,
                    "end_line": end_line,
                    "desc_short": desc_short,
                    "desc_long": desc_long,
                    "desc_start_line": desc_start,
                    "desc_end_line": desc_end,
                }
            )

    blocks.sort(key=lambda block: (block["start_line"], block["end_line"] - block["start_line"], block["id"]))
    imports = _extract_python_imports(root=root, relpath=relpath, tree=tree)
    return {
        "path": relpath,
        "kind": "python",
        "sha256": sha,
        "size_bytes": len(data),
        "indexed_at": utc_now(),
        "imports": imports,
        "blocks": blocks,
    }


def _summarize_markdown_section(
    lines: list[str],
    start_line: int,
    end_line: int,
) -> tuple[str, str, int | None, int | None]:
    if start_line > end_line:
        return "", "", None, None

    paragraph: list[tuple[int, str]] = []
    line_no = start_line
    while line_no <= end_line and line_no <= len(lines):
        stripped = lines[line_no - 1].strip()
        if not stripped:
            if paragraph:
                break
            line_no += 1
            continue
        if MD_HEADING_RE.match(stripped):
            if paragraph:
                break
            line_no += 1
            continue
        paragraph.append((line_no, stripped))
        line_no += 1
        if len(paragraph) >= 5:
            break

    if not paragraph:
        return "", "", None, None

    desc_start = paragraph[0][0]
    desc_end = paragraph[-1][0]
    short = compact_text(" ".join(text for _, text in paragraph[:2]))
    long = "\n".join(text for _, text in paragraph)
    return short, long, desc_start, desc_end


def _build_markdown_entry(root: Path, path: Path) -> dict[str, Any]:
    relpath = normalize_relpath(path, root)
    text = read_file_text(path)
    lines = text.splitlines()
    data = text.encode("utf-8", errors="replace")
    sha = sha256_bytes(data)

    headings: list[dict[str, Any]] = []
    in_fenced_code = False
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fenced_code = not in_fenced_code
            continue
        if in_fenced_code:
            continue

        match = MD_HEADING_RE.match(line)
        if not match:
            continue
        level = len(match.group(1))
        title = match.group(2).strip()
        if not title:
            continue
        headings.append({"line_no": line_no, "level": level, "title": title})

    blocks: list[dict[str, Any]] = []
    id_counts: defaultdict[str, int] = defaultdict(int)

    def unique_id(base_id: str) -> str:
        id_counts[base_id] += 1
        if id_counts[base_id] == 1:
            return base_id
        return f"{base_id}#{id_counts[base_id]}"

    doc_id = unique_id(f"md:{relpath}:document")
    doc_short, doc_long, doc_desc_start, doc_desc_end = _summarize_markdown_section(
        lines,
        1,
        max(len(lines), 1),
    )
    blocks.append(
        {
            "id": doc_id,
            "kind": "document",
            "name": "document",
            "qualname": "document",
            "parent_id": None,
            "start_line": 1,
            "end_line": max(len(lines), 1),
            "desc_short": doc_short,
            "desc_long": doc_long,
            "desc_start_line": doc_desc_start,
            "desc_end_line": doc_desc_end,
        }
    )

    stack: list[dict[str, Any]] = []
    for idx, heading in enumerate(headings):
        level = heading["level"]
        title = heading["title"]
        start_line = heading["line_no"]

        while stack and stack[-1]["level"] >= level:
            stack.pop()
        parent_id = stack[-1]["id"] if stack else doc_id

        end_line = len(lines)
        for nxt in headings[idx + 1 :]:
            if nxt["level"] <= level:
                end_line = nxt["line_no"] - 1
                break
        end_line = max(start_line, end_line)

        slug = slugify(title)
        base_id = f"md:{relpath}:h{level}:{slug}"
        block_id = unique_id(base_id)

        desc_short, desc_long, desc_start, desc_end = _summarize_markdown_section(
            lines,
            start_line + 1,
            end_line,
        )
        # No fallback — headings without body text get no description.

        block = {
            "id": block_id,
            "kind": "heading",
            "name": title,
            "qualname": title,
            "level": level,
            "parent_id": parent_id,
            "start_line": start_line,
            "end_line": end_line,
            "desc_short": desc_short,
            "desc_long": desc_long,
            "desc_start_line": desc_start,
            "desc_end_line": desc_end,
        }
        blocks.append(block)
        stack.append({"level": level, "id": block_id})

    blocks.sort(key=lambda block: (block["start_line"], block["end_line"] - block["start_line"], block["id"]))
    return {
        "path": relpath,
        "kind": "markdown",
        "sha256": sha,
        "size_bytes": len(data),
        "indexed_at": utc_now(),
        "imports": {"internal": [], "external": []},
        "blocks": blocks,
    }


def index_file(root: Path, path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return _build_python_entry(root, path)
    if suffix == ".md":
        return _build_markdown_entry(root, path)
    raise ValueError(f"Unsupported file type: {path}")
