"""Shared constants for the agent-index runtime: paths, version, file filters, and regex patterns."""

from pathlib import Path
import re

INDEX_VERSION = 3
RUNTIME_DIR = Path(".agent-index/runtime")
INDEX_PATH = RUNTIME_DIR / "index.json"
SUPPORTED_SUFFIXES = {".py", ".md"}
ABSOLUTE_IMPORT_SEARCH_PREFIXES = ((), ("src",), ("tools",))

# Keep default scan scope focused and cheap.
DEFAULT_SCAN_DIRS = ("src", "examples", "docs")
DEFAULT_SCAN_FILES = ("README.md",)

IGNORED_DIRS = {
    ".agent-index",
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "wheels",
    "huggingface_models",
    "reference",
}

PY_SECTION_RE = re.compile(r"^\s*#\s*---\s*(.+?)\s*---\s*$")
MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
