"""CLI smoke tests using Click's test runner."""
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from click.testing import CliRunner

from find.cli import main


@pytest.fixture(autouse=True)
def tmp_store_dir(tmp_path, monkeypatch):
    import find.config as cfg
    import find.store as st
    idx = tmp_path / "index.faiss"
    meta = tmp_path / "meta.db"
    monkeypatch.setattr(cfg, "FIND_DIR", tmp_path)
    monkeypatch.setattr(cfg, "INDEX_PATH", idx)
    monkeypatch.setattr(cfg, "META_PATH", meta)
    monkeypatch.setattr(st, "FIND_DIR", tmp_path)
    monkeypatch.setattr(st, "INDEX_PATH", idx)
    monkeypatch.setattr(st, "META_PATH", meta)


def test_status_no_index():
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 1
    assert "No index found" in result.output


def test_index_empty_dir():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        result = runner.invoke(main, ["index", tmp])
    assert result.exit_code == 0


def test_search_no_index():
    runner = CliRunner()
    result = runner.invoke(main, ["search", "hello world"])
    assert result.exit_code == 1
    assert "No results" in result.output or "No index" in result.output


def test_index_and_search():
    """Full round-trip: index a temp dir with a .py file, then search it."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "hello.py"
        p.write_text("def greet():\n    return 'hello world'\n")

        result = runner.invoke(main, ["index", tmp, "--quiet"])
        assert result.exit_code == 0, result.output

        result = runner.invoke(main, ["search", "greet function"])
        assert result.exit_code == 0, result.output
        assert "hello.py" in result.output


def test_status_after_index():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "a.py"
        p.write_text("x = 1\n")
        runner.invoke(main, ["index", tmp, "--quiet"])
        result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    assert "Files indexed" in result.output


def test_search_with_show_text():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "readme.md"
        p.write_text("This is a README with important notes.\n")
        runner.invoke(main, ["index", tmp, "--quiet"])
        result = runner.invoke(main, ["search", "important notes", "--show-text"])
    assert result.exit_code == 0
    assert "readme.md" in result.output
