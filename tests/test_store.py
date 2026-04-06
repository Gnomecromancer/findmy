"""Tests for the Store (FAISS + SQLite)."""
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

# Patch FIND_DIR in both config and store (store imports at module level)
@pytest.fixture(autouse=True)
def tmp_store_dir(tmp_path, monkeypatch):
    import findmy.config as cfg
    import findmy.store as st
    idx = tmp_path / "index.faiss"
    meta = tmp_path / "meta.db"
    monkeypatch.setattr(cfg, "FIND_DIR", tmp_path)
    monkeypatch.setattr(cfg, "INDEX_PATH", idx)
    monkeypatch.setattr(cfg, "META_PATH", meta)
    monkeypatch.setattr(st, "FIND_DIR", tmp_path)
    monkeypatch.setattr(st, "INDEX_PATH", idx)
    monkeypatch.setattr(st, "META_PATH", meta)


def _rand_vecs(n: int, dim: int = 384) -> np.ndarray:
    rng = np.random.default_rng(42)
    v = rng.standard_normal((n, dim)).astype("float32")
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms


def test_add_and_search():
    from findmy.store import Store
    chunks = [("hello world code", "foo.py: def hello"), ("goodbye world", "foo.py: def bye")]
    vecs = _rand_vecs(2)

    with Store() as store:
        store.add_file(Path("/fake/foo.py"), 1000.0, 500, chunks, vecs)
        # Search with the first vector (should be its own top result)
        results = store.search(vecs[0])

    assert len(results) > 0
    assert results[0]["path"] == str(Path("/fake/foo.py"))


def test_stats_after_add():
    from findmy.store import Store
    chunks = [("chunk one", "a.py: def a"), ("chunk two", "a.py: def b")]
    vecs = _rand_vecs(2)

    with Store() as store:
        store.add_file(Path("/a.py"), 1000.0, 100, chunks, vecs)
        s = store.stats()

    assert s["files"] == 1
    assert s["chunks"] == 2
    assert s["vectors"] == 2


def test_file_needs_index_new_file():
    from findmy.store import Store
    with Store() as store:
        assert store.file_needs_index(Path("/new.py"), 9999.0) is True


def test_file_needs_index_unchanged():
    from findmy.store import Store
    vecs = _rand_vecs(1)
    with Store() as store:
        store.add_file(Path("/a.py"), 1000.0, 100, [("x", "a.py")], vecs)
        assert store.file_needs_index(Path("/a.py"), 1000.0) is False


def test_file_needs_index_newer_mtime():
    from findmy.store import Store
    vecs = _rand_vecs(1)
    with Store() as store:
        store.add_file(Path("/a.py"), 1000.0, 100, [("x", "a.py")], vecs)
        assert store.file_needs_index(Path("/a.py"), 2000.0) is True


def test_delete_file():
    from findmy.store import Store
    vecs = _rand_vecs(1)
    with Store() as store:
        store.add_file(Path("/a.py"), 1000.0, 100, [("x", "a.py")], vecs)
        store.delete_file(Path("/a.py"))
        s = store.stats()
    assert s["files"] == 0
    assert s["chunks"] == 0


def test_ext_filter():
    from findmy.store import Store
    py_vec = _rand_vecs(1)
    md_vec = _rand_vecs(1)
    with Store() as store:
        store.add_file(Path("/a.py"), 1000.0, 100, [("python code", "a.py")], py_vec)
        store.add_file(Path("/b.md"), 1001.0, 100, [("markdown notes", "b.md")], md_vec)
        # Search near py_vec, filter to .md only → should not return a.py
        results = store.search(py_vec[0], ext_filter={".md"})

    assert all(r["path"].endswith(".md") for r in results)
