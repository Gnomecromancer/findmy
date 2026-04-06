"""Indexer — orchestrates crawling, chunking, embedding, storing."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from .chunker import chunk_file
from .config import BATCH_SIZE
from .crawler import crawl
from .embedder import Embedder
from .store import Store


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
        # Collect files to process
        all_files = list(crawl(root, extra_excludes))
        to_index = [
            (p, mt, sz)
            for p, mt, sz in all_files
            if force or store.file_needs_index(p, mt)
        ]

        total = len(to_index)
        skipped = len(all_files) - total
        processed = 0
        total_chunks = 0

        # Process in file-level batches, accumulate chunks for GPU batching
        chunk_buffer: list[tuple[Path, float, int, list[tuple[str, str]]]] = []
        chunk_texts: list[str] = []

        def _flush():
            nonlocal total_chunks
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
            store.save()

        try:
            for p, mt, sz in to_index:
                try:
                    content = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue

                chunks = list(chunk_file(p, content))
                if not chunks:
                    continue

                chunk_buffer.append((p, mt, sz, chunks))
                chunk_texts.extend(text for text, _ in chunks)
                processed += 1

                if on_progress:
                    on_progress(str(p), processed, total)

                # Flush when we have enough chunks for a GPU batch
                if len(chunk_texts) >= BATCH_SIZE * 4:
                    _flush()

            _flush()
        except KeyboardInterrupt:
            _flush()  # save whatever is in the buffer
            raise

    return {
        "processed": processed,
        "skipped": skipped,
        "chunks": total_chunks,
    }
