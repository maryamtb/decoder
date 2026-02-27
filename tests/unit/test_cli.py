"""Tests for CLI commands."""

import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from decoder.cli import app

runner = CliRunner()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def indexed_dir(temp_dir: Path):
    """Create a temp directory with a Python file and index it."""
    py_file = temp_dir / "sample.py"
    py_file.write_text("def hello():\n    pass\n\ndef world():\n    hello()\n")
    runner.invoke(app, ["index", str(temp_dir)])
    return temp_dir


class TestIndexCommand:
    """Tests for the index command."""

    def test_index_happy_path(self, temp_dir: Path) -> None:
        """Test that indexing a valid directory succeeds."""
        py_file = temp_dir / "sample.py"
        py_file.write_text("def foo():\n    pass\n")

        result = runner.invoke(app, ["index", str(temp_dir)])

        assert result.exit_code == 0
        assert "Done" in result.output

    def test_index_invalid_path(self) -> None:
        """Test that indexing a nonexistent path indexes zero files."""
        result = runner.invoke(app, ["index", "/nonexistent/path/xyz"])

        assert result.exit_code == 0
        assert "Files indexed: 0" in result.output


class TestFindCommand:
    """Tests for the find command."""

    def test_find_happy_path(self, indexed_dir: Path) -> None:
        """Test that finding an existing symbol returns results."""
        result = runner.invoke(app, ["find", "hello"])

        assert result.exit_code == 0
        assert "hello" in result.output

    def test_find_no_matches(self, indexed_dir: Path) -> None:
        """Test that searching for nonexistent symbol shows no matches."""
        result = runner.invoke(app, ["find", "nonexistentsymbolxyz"])

        assert result.exit_code == 0
        assert "No matches" in result.output


class TestTraceCommand:
    """Tests for the trace command."""

    def test_trace_happy_path(self, indexed_dir: Path) -> None:
        """Test that tracing an existing symbol succeeds."""
        result = runner.invoke(app, ["trace", "world"])

        assert result.exit_code == 0
        assert "world" in result.output

    def test_trace_no_matches(self, indexed_dir: Path) -> None:
        """Test that tracing a nonexistent symbol shows no matches."""
        result = runner.invoke(app, ["trace", "nonexistentsymbolxyz"])

        assert result.exit_code == 0
        assert "No matches" in result.output