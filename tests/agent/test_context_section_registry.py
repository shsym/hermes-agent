"""Tests for agent/context_section_registry.py — Phase 2.1 (Idea D)."""

from __future__ import annotations

import urllib.error
from unittest.mock import patch

import pytest

from agent.context_section_registry import (
    discover_context_files,
    register_context_files,
    render_context_files_index,
)
from agent.context_summary import ContextSection, summarize_context_file
from tools.read_context_tool import (
    clear_cache,
    get_cached_section_ids,
)


@pytest.fixture(autouse=True)
def _reset_cache_and_env(monkeypatch):
    """Body cache + HERMES_INFERLET_URL env var reset around each test."""
    clear_cache()
    monkeypatch.delenv("HERMES_INFERLET_URL", raising=False)
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# discover_context_files
# ---------------------------------------------------------------------------


def test_discover_context_files_finds_agents_md(tmp_path):
    (tmp_path / "AGENTS.md").write_text("# A\n\nbody A\n", encoding="utf-8")
    out = discover_context_files(str(tmp_path))
    assert len(out) == 1
    assert out[0][0] == "AGENTS.md"
    assert "body A" in out[0][1]


def test_discover_context_files_returns_empty_when_none(tmp_path):
    out = discover_context_files(str(tmp_path))
    assert out == []


def test_discover_context_files_finds_multiple_filenames(tmp_path):
    (tmp_path / "AGENTS.md").write_text("# A\n", encoding="utf-8")
    (tmp_path / ".cursorrules").write_text("# C\n", encoding="utf-8")
    out = discover_context_files(str(tmp_path))
    names = [n for n, _ in out]
    # _CONTEXT_FILENAMES order: AGENTS.md before .cursorrules
    assert names == ["AGENTS.md", ".cursorrules"]


def test_discover_context_files_skips_unreadable(tmp_path):
    p = tmp_path / "AGENTS.md"
    p.write_text("# A\n", encoding="utf-8")

    real_read_text = type(p).read_text

    def fake_read_text(self, *args, **kwargs):
        if self.name == "AGENTS.md":
            raise OSError("permission denied")
        return real_read_text(self, *args, **kwargs)

    with patch.object(type(p), "read_text", fake_read_text):
        out = discover_context_files(str(tmp_path))
    assert out == []


def test_discover_context_files_ignores_directories_named_like_context_files(tmp_path):
    (tmp_path / "AGENTS.md").mkdir()  # a directory, not a file
    out = discover_context_files(str(tmp_path))
    assert out == []


# ---------------------------------------------------------------------------
# register_context_files
# ---------------------------------------------------------------------------


def _agents_md_two_sections() -> str:
    return (
        "# Coding style\n\n"
        "Use 4 spaces.\n\n"
        "# Testing\n\n"
        "Always log to test-YYYYMMDD-HHMMSS.log.\n"
    )


def test_register_populates_body_cache(tmp_path):
    (tmp_path / "AGENTS.md").write_text(_agents_md_two_sections(), encoding="utf-8")
    sections = register_context_files(cwd=str(tmp_path))
    ids = get_cached_section_ids()
    assert ids == sorted(s.id for s in sections)
    assert "agents.md#coding-style" in ids
    assert "agents.md#testing" in ids


def test_register_returns_sections_in_document_order(tmp_path):
    (tmp_path / "AGENTS.md").write_text(_agents_md_two_sections(), encoding="utf-8")
    sections = register_context_files(cwd=str(tmp_path))
    assert [s.title for s in sections] == ["Coding style", "Testing"]


def test_register_returns_empty_when_no_context_files(tmp_path):
    sections = register_context_files(cwd=str(tmp_path))
    assert sections == []
    assert get_cached_section_ids() == []


def test_register_skips_inferlet_when_url_unset(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text(_agents_md_two_sections(), encoding="utf-8")
    monkeypatch.delenv("HERMES_INFERLET_URL", raising=False)
    with patch("agent.context_section_registry._post_register") as mock_post:
        sections = register_context_files(cwd=str(tmp_path), inferlet_url=None)
    assert mock_post.call_count == 0
    assert len(sections) == 2


def test_register_skips_inferlet_when_url_empty_string(tmp_path):
    (tmp_path / "AGENTS.md").write_text(_agents_md_two_sections(), encoding="utf-8")
    with patch("agent.context_section_registry._post_register") as mock_post:
        register_context_files(cwd=str(tmp_path), inferlet_url="")
    assert mock_post.call_count == 0


def test_register_swallows_inferlet_post_failures(tmp_path):
    """A flaky/missing inferlet must not break the agent boot path."""
    (tmp_path / "AGENTS.md").write_text(_agents_md_two_sections(), encoding="utf-8")

    def boom(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    with patch("urllib.request.urlopen", side_effect=boom):
        sections = register_context_files(
            cwd=str(tmp_path),
            inferlet_url="http://localhost:65535",
        )

    # Cache still populated; sections still returned.
    assert len(sections) == 2
    assert "agents.md#coding-style" in get_cached_section_ids()


def test_register_reads_inferlet_url_env_fallback(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text(_agents_md_two_sections(), encoding="utf-8")
    monkeypatch.setenv("HERMES_INFERLET_URL", "http://example.invalid:9090")
    with patch("agent.context_section_registry._post_register", return_value=True) as mock_post:
        register_context_files(cwd=str(tmp_path))
    assert mock_post.call_count == 2
    # Verify URL passed through (rstrip handled inside _post_register, not here)
    assert mock_post.call_args_list[0].args[0] == "http://example.invalid:9090"


def test_register_explicit_url_overrides_env(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text(_agents_md_two_sections(), encoding="utf-8")
    monkeypatch.setenv("HERMES_INFERLET_URL", "http://from-env:1")
    with patch("agent.context_section_registry._post_register", return_value=True) as mock_post:
        register_context_files(
            cwd=str(tmp_path), inferlet_url="http://explicit:2",
        )
    assert mock_post.call_args_list[0].args[0] == "http://explicit:2"


def test_register_inferlet_post_handles_non_200_gracefully(tmp_path):
    """Non-2xx response must be treated as failure but not raise."""
    (tmp_path / "AGENTS.md").write_text(_agents_md_two_sections(), encoding="utf-8")

    class FakeResp:
        status = 500

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    with patch("urllib.request.urlopen", return_value=FakeResp()):
        sections = register_context_files(
            cwd=str(tmp_path),
            inferlet_url="http://localhost:65535",
        )
    # Still populates cache, still returns sections.
    assert len(sections) == 2


def test_register_inferlet_post_handles_timeout(tmp_path):
    (tmp_path / "AGENTS.md").write_text(_agents_md_two_sections(), encoding="utf-8")

    def timeout(*args, **kwargs):
        raise TimeoutError("timed out")

    with patch("urllib.request.urlopen", side_effect=timeout):
        sections = register_context_files(
            cwd=str(tmp_path),
            inferlet_url="http://localhost:65535",
        )
    assert len(sections) == 2
    assert "agents.md#coding-style" in get_cached_section_ids()


# ---------------------------------------------------------------------------
# render_context_files_index
# ---------------------------------------------------------------------------


def test_render_empty_sections_returns_empty_string():
    assert render_context_files_index([]) == ""


def test_render_one_line_per_section():
    sections = [
        ContextSection(
            id="agents.md#a",
            title="Title A",
            summary="Summary A",
            body="body A",
            body_hash="aaaa",
        ),
        ContextSection(
            id="agents.md#b",
            title="Title B",
            summary="Summary B",
            body="body B",
            body_hash="bbbb",
        ),
    ]
    out = render_context_files_index(sections)
    lines = out.splitlines()
    assert lines[0].startswith("# Project context files (index")
    assert "read_context" in lines[0]
    assert lines[1] == "- agents.md#a [Title A] — Summary A"
    assert lines[2] == "- agents.md#b [Title B] — Summary B"
    assert len(lines) == 3


def test_render_is_deterministic(tmp_path):
    """Same content -> same index output (load-bearing for prefix cache)."""
    (tmp_path / "AGENTS.md").write_text(_agents_md_two_sections(), encoding="utf-8")
    sections1 = summarize_context_file("AGENTS.md", _agents_md_two_sections())
    sections2 = summarize_context_file("AGENTS.md", _agents_md_two_sections())
    assert render_context_files_index(sections1) == render_context_files_index(sections2)
