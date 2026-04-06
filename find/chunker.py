"""Text chunker — splits file content into overlapping token windows.

Code files get split on top-level def/class boundaries first, then by
token window. Prose files split on paragraph boundaries.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Generator

from .config import MAX_CHUNK_TOKENS, CHUNK_OVERLAP

# Rough chars-per-token estimate (avoids loading a tokenizer just for chunking)
_CHARS_PER_TOKEN = 4

_CODE_SPLIT_RE = re.compile(
    r"(?m)^(?=(?:def |class |async def |fn |func |function |pub fn |impl ))",
)

_CODE_EXTS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".mjs",
    ".go", ".rs", ".c", ".cpp", ".h", ".hpp",
    ".java", ".kt", ".swift", ".rb", ".php",
}


def _token_approx(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


def _window_split(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Split text into overlapping token windows."""
    max_chars = max_tokens * _CHARS_PER_TOKEN
    overlap_chars = overlap * _CHARS_PER_TOKEN
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap_chars
    return chunks


def chunk_file(path: Path, content: str) -> Generator[tuple[str, str], None, None]:
    """
    Yield (chunk_text, label) pairs.
    label is a human-readable location hint like "path.py:def foo".
    """
    suffix = path.suffix.lower()

    if suffix in _CODE_EXTS:
        # Split on top-level definitions
        sections = _CODE_SPLIT_RE.split(content)
    else:
        # Split on blank lines (paragraphs)
        sections = re.split(r"\n\s*\n", content)

    for section in sections:
        section = section.strip()
        if not section:
            continue
        # Extract a label from first line
        first_line = section.splitlines()[0].strip()[:80]
        label = f"{path.name}: {first_line}" if first_line else path.name

        if _token_approx(section) <= MAX_CHUNK_TOKENS:
            yield section, label
        else:
            for sub in _window_split(section, MAX_CHUNK_TOKENS, CHUNK_OVERLAP):
                yield sub, label
