"""SQLite metadata store + FAISS vector index management."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .config import FIND_DIR, INDEX_PATH, META_PATH, EMBED_DIM


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS files (
            id      INTEGER PRIMARY KEY,
            path    TEXT UNIQUE NOT NULL,
            mtime   REAL NOT NULL,
            size    INTEGER NOT NULL,
            indexed REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id      INTEGER PRIMARY KEY,
            file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            faiss_id INTEGER NOT NULL,
            label   TEXT NOT NULL,
            text    TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_faiss ON chunks(faiss_id);
        CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);
    """)
    conn.commit()


class Store:
    """Combines FAISS index + SQLite metadata."""

    def __init__(self) -> None:
        import faiss  # type: ignore

        FIND_DIR.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(META_PATH), timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        _init_db(self._conn)
        self._conn.execute("PRAGMA foreign_keys = ON")

        if INDEX_PATH.exists():
            self._index = faiss.read_index(str(INDEX_PATH))
        else:
            self._index = faiss.IndexFlatIP(EMBED_DIM)  # inner product (cosine after normalize)

        self._faiss = faiss

    # ------------------------------------------------------------------ #
    # File tracking                                                        #
    # ------------------------------------------------------------------ #

    def file_needs_index(self, path: Path, mtime: float) -> bool:
        row = self._conn.execute(
            "SELECT mtime FROM files WHERE path = ?", (str(path),)
        ).fetchone()
        if row is None:
            return True
        return float(row["mtime"]) < mtime

    def delete_file(self, path: Path) -> None:
        """Remove all chunks for a file (FAISS ids are not freed, just orphaned)."""
        self._conn.execute("DELETE FROM files WHERE path = ?", (str(path),))
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Adding chunks                                                        #
    # ------------------------------------------------------------------ #

    def add_file(
        self,
        path: Path,
        mtime: float,
        size: int,
        chunks: list[tuple[str, str]],   # (text, label)
        embeddings: np.ndarray,           # shape (len(chunks), EMBED_DIM)
    ) -> None:
        """Upsert a file's chunks into FAISS + SQLite."""
        # Normalize for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = (embeddings / norms).astype("float32")

        # Assign FAISS ids = current index size + offset
        start_id = self._index.ntotal
        self._index.add(normed)

        # Delete old record if present
        self.delete_file(path)

        # Insert file record
        cur = self._conn.execute(
            "INSERT INTO files (path, mtime, size, indexed) VALUES (?, ?, ?, ?)",
            (str(path), mtime, size, time.time()),
        )
        file_id = cur.lastrowid

        # Insert chunk records
        self._conn.executemany(
            "INSERT INTO chunks (file_id, faiss_id, label, text) VALUES (?, ?, ?, ?)",
            [
                (file_id, start_id + i, label, text)
                for i, (text, label) in enumerate(chunks)
            ],
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Searching                                                            #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query_vec: np.ndarray,   # shape (EMBED_DIM,)
        top_k: int = 10,
        ext_filter: Optional[set[str]] = None,
    ) -> list[dict]:
        """Return top_k results as list of dicts."""
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        q = query_vec.reshape(1, -1).astype("float32")

        fetch_k = max(top_k * 5, 50)  # oversample, then filter
        scores, ids = self._index.search(q, fetch_k)

        results = []
        for score, fid in zip(scores[0], ids[0]):
            if fid < 0:
                continue
            row = self._conn.execute(
                """
                SELECT c.text, c.label, f.path, ? as score
                FROM chunks c JOIN files f ON c.file_id = f.id
                WHERE c.faiss_id = ?
                """,
                (float(score), int(fid)),
            ).fetchone()
            if row is None:
                continue
            if ext_filter:
                if Path(row["path"]).suffix.lower() not in ext_filter:
                    continue
            results.append(dict(row))
            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------ #
    # Stats                                                               #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        files = self._conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {
            "files": files,
            "chunks": chunks,
            "vectors": self._index.ntotal,
            "index_size_mb": (INDEX_PATH.stat().st_size / 1e6) if INDEX_PATH.exists() else 0.0,
        }

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self) -> None:
        self._faiss.write_index(self._index, str(INDEX_PATH))

    def close(self) -> None:
        self.save()
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *_) -> None:
        self.close()
