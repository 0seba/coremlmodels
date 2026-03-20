"""Shared utility functions: hashing, path normalisation, text summarisation, and formatting."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def normalize_relpath(path: Path, root: Path) -> str:
    """Resolve a path and return its POSIX-style relative path from root."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def line_indent(line: str) -> int:
    """Count leading spaces in a line."""
    return len(line) - len(line.lstrip(" "))


def slugify(text: str) -> str:
    """Convert text to a lowercase dash-separated slug for use in block IDs."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or "block"


def compact_text(text: str, max_chars: int = 220) -> str:
    """Collapse multi-line text into a single line, truncating with ellipsis if needed."""
    compact = " ".join(part.strip() for part in text.splitlines() if part.strip())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def first_sentence(text: str, max_chars: int = 220) -> str:
    """Extract the first sentence from text, used for desc_short."""
    compact = compact_text(text, max_chars=max_chars * 4)
    if not compact:
        return ""
    match = re.search(r"[.!?](?:\s|$)", compact)
    if match:
        sentence = compact[: match.end()].strip()
    else:
        sentence = compact
    if len(sentence) <= max_chars:
        return sentence
    return sentence[: max_chars - 3].rstrip() + "..."


def first_paragraph_lines(text: str, max_lines: int = 5) -> list[str]:
    """Extract up to max_lines from the first non-empty paragraph, used for desc_long."""
    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    paragraph: list[str] = []
    for ln in lines:
        if not ln.strip():
            if paragraph:
                break
            continue
        paragraph.append(ln.strip())
        if len(paragraph) >= max_lines:
            break
    return paragraph


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    """Compute the hex SHA-256 digest of a file, reading in 1 MB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_file_text(path: Path) -> str:
    """Read a file as UTF-8 text, replacing malformed bytes."""
    return path.read_text(encoding="utf-8", errors="replace")


def normalize_path_query(root: Path, raw_path: str) -> str:
    """Normalise a user-supplied path query to a POSIX relative path from root."""
    path = Path(raw_path)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except Exception:
            return raw_path.replace("\\", "/")
    return raw_path.replace("\\", "/")


def format_lines_with_numbers(lines: list[str], start_line: int) -> str:
    """Format source lines with right-aligned line numbers and a pipe separator."""
    width = len(str(start_line + len(lines)))
    out_lines = []
    for idx, line in enumerate(lines):
        line_no = start_line + idx
        out_lines.append(f"{line_no:>{width}} | {line}")
    return "\n".join(out_lines)
