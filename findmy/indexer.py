"""Indexer — orchestrates crawling, chunking, embedding, storing."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

from .chunker import chunk_file
from .config import BATCH_SIZE
from .crawler import crawl
from .embedder import Embedder
from .store import Store

_SAVE_INTERVAL_SEC = 60
_FLUSH_THRESHOLD = BATCH_SIZE * 8   # 512 chunks per GPU batch
_READ_WORKERS = 8


def _read_file(args: tuple[Path, float, int]) -> tuple[Path, float, int, str] | None:
    p, mt, sz = args
    try:
        return p, mt, sz, p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def index(
    root: Path,
    extra_excludes: list[str] = (),
    force: bool = False,
    on_progress: Optional[Callable[[str, int, int], None]] = None,
) -> dict:
    """
    Index all files under root.

    on_progress(path_str, current, total) called for each file processed.
    Returns stats dict.
    """
    embedder = Embedder()
    # Warm up model before the main loop so first batch pays no load cost
    embedder.embed(["warmup"])

    with Store() as store:
        # Stream the crawl — filter on the fly using the in-memory mtime cache
        to_index = [
            (p, mt, sz)
            for p, mt, sz in crawl(root, extra_excludes)
            if force or store.file_needs_index(p, mt)
        ]

        total = len(to_index)
        processed = 0
        skipped = 0  # tracked separately after filter
        total_chunks = 0
        last_save = time.monotonic()

        chunk_buffer: list[tuple[Path, float, int, list[tuple[str, str]]]] = []
        chunk_texts: list[str] = []

        def _flush(force_save: bool = False) -> None:
            nonlocal total_chunks, last_save
            if not chunk_buffer:
                return
            vecs = embedder.embed(chunk_texts)
            # One transaction for all files in this batch
            store.begin_batch()
            try:
                offset = 0
                for p, mt, sz, chunks in chunk_buffer:
                    n = len(chunks)
                    store.add_file(p, mt, sz, chunks, vecs[offset:offset + n], in_batch=True)
                    offset += n
                    total_chunks += n
                store.end_batch()
            except Exception:
                store._conn.execute("ROLLBACK")
                raise
            chunk_buffer.clear()
            chunk_texts.clear()
            now = time.monotonic()
            if force_save or (now - last_save) >= _SAVE_INTERVAL_SEC:
                store.save()
                last_save = now

        try:
            with ThreadPoolExecutor(max_workers=_READ_WORKERS) as pool:
                futures = {pool.submit(_read_file, args): args for args in to_index}
                for future in as_completed(futures):
                    result = future.result()
                    if result is None:
                        skipped += 1
                        continue
                    p, mt, sz, content = result

                    chunks = list(chunk_file(p, content))
                    if not chunks:
                        continue

                    chunk_buffer.append((p, mt, sz, chunks))
                    chunk_texts.extend(text for text, _ in chunks)
                    processed += 1

                    if on_progress:
                        on_progress(str(p), processed, total)

                    if len(chunk_texts) >= _FLUSH_THRESHOLD:
                        _flush()

            _flush(force_save=True)

        except KeyboardInterrupt:
            _flush(force_save=True)
            raise

    return {
        "processed": processed,
        "skipped": skipped,
        "chunks": total_chunks,
    }
