"""Tests for the chunker module."""
import pytest
from pathlib import Path
from find.chunker import chunk_file, _token_approx


def test_short_file_single_chunk():
    path = Path("test.py")
    content = "def hello():\n    return 'world'\n"
    chunks = list(chunk_file(path, content))
    assert len(chunks) >= 1
    assert "hello" in chunks[0][0]


def test_empty_content_yields_nothing():
    path = Path("test.py")
    chunks = list(chunk_file(path, ""))
    assert chunks == []


def test_multiple_defs_split():
    path = Path("test.py")
    content = "\n".join([
        "def foo():",
        "    pass",
        "",
        "def bar():",
        "    pass",
        "",
        "def baz():",
        "    pass",
    ])
    chunks = list(chunk_file(path, content))
    # Should have at least 3 chunks (one per def)
    assert len(chunks) >= 3


def test_prose_splits_on_paragraphs():
    path = Path("notes.md")
    content = "First paragraph with some text.\n\nSecond paragraph here.\n\nThird paragraph."
    chunks = list(chunk_file(path, content))
    assert len(chunks) == 3


def test_long_section_windowed():
    path = Path("bigfile.py")
    # Create a section that's way over the token limit
    big_content = "x = 1\n" * 1000  # ~6000 tokens
    chunks = list(chunk_file(path, big_content))
    # Should produce multiple chunks
    assert len(chunks) > 1


def test_label_contains_filename():
    path = Path("mymodule.py")
    content = "def my_function():\n    pass\n"
    chunks = list(chunk_file(path, content))
    assert all("mymodule.py" in label for _, label in chunks)


def test_token_approx_reasonable():
    text = "hello world " * 100  # 1200 chars → ~300 tokens
    approx = _token_approx(text)
    assert 200 <= approx <= 400
