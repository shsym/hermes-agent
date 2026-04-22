"""Tests for Phase 2.1 (Idea D) prompt-builder index emission.

Covers ``build_context_files_prompt(sections=...)`` — the optional
parameter introduced by Task 2D that swaps verbatim AGENTS.md content
for a compact index.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.context_section_registry import register_context_files
from agent.context_summary import ContextSection
from agent.prompt_builder import build_context_files_prompt
from tools.read_context_tool import clear_cache


@pytest.fixture(autouse=True)
def _no_soul(monkeypatch):
    """Force load_soul_md to return None so SOUL.md from a real HERMES_HOME
    can't leak into prompt body assertions."""
    with patch("agent.prompt_builder.load_soul_md", return_value=None):
        yield


@pytest.fixture(autouse=True)
def _reset_cache():
    clear_cache()
    yield
    clear_cache()


_SAMPLE_AGENTS_MD = (
    "# Coding style\n\n"
    "Use 4 spaces. Never tabs.\n\n"
    "# Testing\n\n"
    "Always write tests before merging.\n"
)


def _make_sections() -> list[ContextSection]:
    return [
        ContextSection(
            id="agents.md#coding-style",
            title="Coding style",
            summary="Use 4 spaces. Never tabs.",
            body="Use 4 spaces. Never tabs.",
            body_hash="aaaa1111",
        ),
        ContextSection(
            id="agents.md#testing",
            title="Testing",
            summary="Always write tests before merging.",
            body="Always write tests before merging.",
            body_hash="bbbb2222",
        ),
    ]


# ---------------------------------------------------------------------------
# Index path
# ---------------------------------------------------------------------------


def test_index_path_contains_section_ids_not_bodies(tmp_path):
    (tmp_path / "AGENTS.md").write_text(_SAMPLE_AGENTS_MD, encoding="utf-8")
    sections = _make_sections()
    out = build_context_files_prompt(
        cwd=str(tmp_path), skip_soul=True, sections=sections,
    )
    # Index has the IDs.
    assert "agents.md#coding-style" in out
    assert "agents.md#testing" in out
    # Bodies are NOT inlined in the index path.
    assert "Use 4 spaces. Never tabs." in out  # this is the summary, fine
    assert "Always write tests before merging." in out  # also a summary line
    # Index is the ONLY content (no "# Project Context" wrapper or body markdown).
    assert "# Project Context" not in out
    # The literal heading lines from the source file must NOT appear.
    assert "# Coding style" not in out
    assert "# Testing" not in out


def test_index_path_uses_compact_index_format(tmp_path):
    (tmp_path / "AGENTS.md").write_text(_SAMPLE_AGENTS_MD, encoding="utf-8")
    sections = _make_sections()
    out = build_context_files_prompt(
        cwd=str(tmp_path), skip_soul=True, sections=sections,
    )
    lines = out.splitlines()
    assert lines[0].startswith("# Project context files (index")
    assert "read_context" in lines[0]
    assert lines[1] == "- agents.md#coding-style [Coding style] — Use 4 spaces. Never tabs."
    assert lines[2] == "- agents.md#testing [Testing] — Always write tests before merging."


# ---------------------------------------------------------------------------
# Back-compat: sections=None / sections=[] preserves legacy output
# ---------------------------------------------------------------------------


def test_back_compat_sections_none_yields_full_body(tmp_path):
    (tmp_path / "AGENTS.md").write_text(_SAMPLE_AGENTS_MD, encoding="utf-8")
    legacy = build_context_files_prompt(cwd=str(tmp_path), skip_soul=True)
    assert legacy.startswith("# Project Context")
    # Full body present — including the section heading from the file.
    assert "# Coding style" in legacy
    assert "Always write tests before merging." in legacy


def test_back_compat_sections_none_default_arg(tmp_path):
    """Calling without the new sections kwarg should yield the same output
    as explicitly passing sections=None."""
    (tmp_path / "AGENTS.md").write_text(_SAMPLE_AGENTS_MD, encoding="utf-8")
    a = build_context_files_prompt(cwd=str(tmp_path), skip_soul=True)
    b = build_context_files_prompt(cwd=str(tmp_path), skip_soul=True, sections=None)
    assert a == b


def test_back_compat_empty_sections_treated_as_legacy(tmp_path):
    """sections=[] (e.g. opt-in but no context files discovered) must
    fall through to legacy behavior — never emit a stray empty index
    block, never silently swallow the file content."""
    (tmp_path / "AGENTS.md").write_text(_SAMPLE_AGENTS_MD, encoding="utf-8")
    legacy = build_context_files_prompt(cwd=str(tmp_path), skip_soul=True)
    empty_idx = build_context_files_prompt(
        cwd=str(tmp_path), skip_soul=True, sections=[],
    )
    assert empty_idx == legacy


# ---------------------------------------------------------------------------
# No content at all
# ---------------------------------------------------------------------------


def test_index_path_with_no_files_and_no_sections_returns_empty(tmp_path):
    out = build_context_files_prompt(
        cwd=str(tmp_path), skip_soul=True, sections=[],
    )
    assert out == ""


def test_legacy_path_with_no_files_returns_empty(tmp_path):
    out = build_context_files_prompt(cwd=str(tmp_path), skip_soul=True)
    assert out == ""


# ---------------------------------------------------------------------------
# End-to-end: registrar + builder
# ---------------------------------------------------------------------------


def test_e2e_registrar_then_index_build(tmp_path):
    (tmp_path / "AGENTS.md").write_text(_SAMPLE_AGENTS_MD, encoding="utf-8")
    sections = register_context_files(cwd=str(tmp_path), inferlet_url=None)
    out = build_context_files_prompt(
        cwd=str(tmp_path), skip_soul=True, sections=sections,
    )
    # The produced index lists exactly what was registered.
    for s in sections:
        assert s.id in out
    # Verbatim section headings absent → token saving achieved.
    assert "# Coding style" not in out
    assert "# Testing" not in out
