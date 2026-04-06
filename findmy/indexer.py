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

# Save index to disk at most once every N seconds (not every flush)
_SAVE_INTERVAL_SEC = 60
# Chunk buffer threshold before triggering a GPU embed+store flush
_FLUSH_THRESHOLD = BATCH_SIZE * 8   # 512 chunks per GPU batch
# How many files to read concurrently via thread pool
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

    with Store() as store:
        # Streaming crawl — filter on the fly instead of list() upfront
        all_crawled = list(crawl(root, extra_excludes))
        to_index = [
            (p, mt, sz)
            for p, mt, sz in all_crawled
            if force or store.file_needs_index(p, mt)
        ]

        total = len(to_index)
        skipped = len(all_crawled) - total
        processed = 0
        total_chunks = 0
        last_save = time.monotonic()

        chunk_buffer: list[tuple[Path, float, int, list[tuple[str, str]]]] = []
        chunk_texts: list[str] = []

        def _flush(force_save: bool = False):
            nonlocal total_chunks, last_save
            if not chunk_buffer:
                return
            vecs = embedder.embed(chunk_texts)
            offset = 0
            for p, mt, sz, chunks in chunk_buffer:
                n = len(chunks)
                store.add_file(p, mt, sz, chunks, vecs[offset : offset + n])
                offset += n
                total_chunks += n
            chunk_buffer.clear()
            chunk_texts.clear()
            # Only write FAISS to disk periodically — it's the main bottleneck
            now = time.monotonic()
            if force_save or (now - last_save) >= _SAVE_INTERVAL_SEC:
                store.save()
                last_save = now

        try:
            # Read files concurrently to hide I/O latency
            with ThreadPoolExecutor(max_workers=_READ_WORKERS) as pool:
                futures = {pool.submit(_read_file, args): args for args in to_index}
                for future in as_completed(futures):
                    result = future.result()
                    if result is None:
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
