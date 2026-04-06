"""Tests for the filesystem crawler."""
import tempfile
from pathlib import Path

from findmy.crawler import crawl


def _make_tree(root: Path, files: dict) -> None:
    """Create files from a {relative_path: content} dict."""
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def test_finds_py_files():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_tree(root, {"a.py": "x=1", "b.js": "let x=1;"})
        found = {p.name for p, _, _ in crawl(root)}
        assert "a.py" in found
        assert "b.js" in found


def test_skips_node_modules():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_tree(root, {
            "src/main.py": "x=1",
            "node_modules/lodash/index.js": "x=1",
        })
        found = {p for p, _, _ in crawl(root)}
        paths_str = [str(p) for p in found]
        assert not any("node_modules" in s for s in paths_str)
        assert any("main.py" in s for s in paths_str)


def test_skips_git_dir():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_tree(root, {
            "main.py": "x=1",
            ".git/COMMIT_EDITMSG": "initial",
        })
        found = [str(p) for p, _, _ in crawl(root)]
        assert not any(".git" in s for s in found)


def test_skips_non_indexable_extensions():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_tree(root, {
            "image.png": "fakebinary",
            "data.db": "fakebinary",
            "main.py": "x=1",
        })
        found = {p.name for p, _, _ in crawl(root)}
        assert "main.py" in found
        assert "image.png" not in found
        assert "data.db" not in found


def test_skips_empty_files():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_tree(root, {"empty.py": "", "full.py": "x=1"})
        found = {p.name for p, _, _ in crawl(root)}
        assert "full.py" in found
        assert "empty.py" not in found


def test_extra_excludes():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_tree(root, {
            "src/main.py": "x=1",
            "vendor/lib.py": "x=1",
        })
        found = [str(p) for p, _, _ in crawl(root, extra_excludes=["vendor"])]
        assert not any("vendor" in s for s in found)
        assert any("main.py" in s for s in found)


def test_returns_mtime_and_size():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_tree(root, {"a.py": "x = 1\n"})
        results = list(crawl(root))
        assert len(results) == 1
        p, mtime, size = results[0]
        assert mtime > 0
        assert size > 0
