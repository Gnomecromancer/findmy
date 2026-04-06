"""Filesystem crawler — yields (path, mtime, size) for indexable files."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Iterable

from .config import INDEXABLE_EXTENSIONS, DEFAULT_EXCLUDES, MAX_FILE_SIZE_MB


def _should_skip_dir(name: str, extra_excludes: set[str]) -> bool:
    if name.startswith(".") and name not in {".claude"}:
        return True
    return name in (DEFAULT_EXCLUDES | extra_excludes)


def crawl(
    root: Path,
    extra_excludes: Iterable[str] = (),
) -> Generator[tuple[Path, float, int], None, None]:
    """
    Yield (path, mtime, size_bytes) for each indexable file under root.
    Skips hidden dirs, excluded dirs, files > MAX_FILE_SIZE_MB.
    """
    excludes = set(extra_excludes)
    max_bytes = int(MAX_FILE_SIZE_MB * 1024 * 1024)

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Prune dirs in-place so os.walk doesn't descend into them
        dirnames[:] = [
            d for d in dirnames
            if not _should_skip_dir(d, excludes)
        ]

        for fname in filenames:
            p = Path(dirpath) / fname
            if p.suffix.lower() not in INDEXABLE_EXTENSIONS:
                continue
            try:
                stat = p.stat()
            except OSError:
                continue
            if stat.st_size > max_bytes or stat.st_size == 0:
                continue
            yield p, stat.st_mtime, stat.st_size
