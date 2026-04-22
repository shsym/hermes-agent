"""Tests for the offline context-files summarizer (Phase 2.1, Idea D).

Determinism is the critical invariant: the same input must produce the
same id, summary, and body_hash on every run. If this regresses, the
inferlet's cached system prefix gets invalidated on every cold boot.
"""

from __future__ import annotations

import hashlib

from agent.context_summary import (
    ContextSection,
    slugify_title,
    summarize_context_file,
)


# -----------------------------------------------------------------------------
# slugify_title — pure function, tight unit coverage
# -----------------------------------------------------------------------------

def test_slugify_title_lowercases_and_hyphenates():
    assert slugify_title("Coding Style") == "coding-style"


def test_slugify_title_collapses_punctuation_runs():
    assert slugify_title("Test/Build & Deploy") == "test-build-deploy"


def test_slugify_title_strips_leading_trailing_hyphens():
    assert slugify_title("- Trim me - ") == "trim-me"


def test_slugify_title_empty_input_returns_untitled():
    assert slugify_title("") == "untitled"
    assert slugify_title("    ") == "untitled"
    assert slugify_title("!!!") == "untitled"


# -----------------------------------------------------------------------------
# summarize_context_file — section extraction, ids, hashes, summaries
# -----------------------------------------------------------------------------

def test_returns_empty_list_for_no_headings():
    assert summarize_context_file("AGENTS.md", "Just some prose with no headings.\n") == []
    assert summarize_context_file("AGENTS.md", "") == []


def test_extracts_sections_with_stable_ids():
    md = """# Project rules
## Coding style
Use 2-space indents; prefer explicit imports.

## Test policy
All PRs must include tests. Use pytest.
"""
    sections = summarize_context_file("AGENTS.md", md)
    assert len(sections) == 3  # H1 + two H2s

    ids = [s.id for s in sections]
    assert ids == ["agents.md#project-rules", "agents.md#coding-style", "agents.md#test-policy"]


def test_section_body_excludes_heading_and_trailing_blank_lines():
    md = "## A\nbody A\n\n## B\nbody B\n\n\n"
    [a, b] = summarize_context_file("AGENTS.md", md)
    assert a.body == "body A"
    assert b.body == "body B"


def test_summary_is_first_nonblank_line_truncated_at_80():
    md = "## X\n\n\n   Use 2-space indents; prefer explicit imports.\n"
    [s] = summarize_context_file("AGENTS.md", md)
    assert s.summary == "Use 2-space indents; prefer explicit imports."


def test_summary_truncates_long_lines_at_word_boundary_with_ellipsis():
    long = "a " * 60  # ~120 chars of repeated 'a '
    md = f"## X\n{long.rstrip()}\n"
    [s] = summarize_context_file("AGENTS.md", md)
    assert len(s.summary) <= 81   # 80 chars + ellipsis
    assert s.summary.endswith("…")


def test_summary_strips_markdown_bullet_prefix():
    md = "## X\n- the actual rule is here\n"
    [s] = summarize_context_file("AGENTS.md", md)
    assert s.summary == "the actual rule is here"


def test_summary_falls_back_to_title_when_body_empty():
    md = "## Empty Section\n\n## Next\nbody\n"
    [empty, next_] = summarize_context_file("AGENTS.md", md)
    assert empty.summary == "Empty Section"


def test_body_hash_is_stable_for_same_input():
    md = "## X\nthe body of X\n"
    [a] = summarize_context_file("AGENTS.md", md)
    [b] = summarize_context_file("AGENTS.md", md)
    assert a.body_hash == b.body_hash
    assert a.body_hash == hashlib.sha256(b"the body of X").hexdigest()[:16]


def test_body_hash_changes_when_body_changes_but_id_stays_same():
    md1 = "## Coding style\nuse 2 spaces\n"
    md2 = "## Coding style\nuse 4 spaces\n"
    [s1] = summarize_context_file("AGENTS.md", md1)
    [s2] = summarize_context_file("AGENTS.md", md2)
    assert s1.id == s2.id
    assert s1.body_hash != s2.body_hash


def test_filename_case_normalized_to_lowercase_in_id():
    [s] = summarize_context_file("AGENTS.md", "## X\nbody\n")
    assert s.id.startswith("agents.md#")


def test_pre_first_heading_content_is_dropped():
    md = "preamble that should not appear\n\n## X\nbody\n"
    [s] = summarize_context_file("AGENTS.md", md)
    assert "preamble" not in s.body


def test_each_heading_creates_its_own_section_regardless_of_depth():
    md = "# H1\nbody1\n## H2\nbody2\n### H3\nbody3\n"
    sections = summarize_context_file("AGENTS.md", md)
    assert [s.title for s in sections] == ["H1", "H2", "H3"]


def test_determinism_across_runs():
    """The same input must yield byte-identical sections on every call.
    This is the load-bearing invariant for prefix-cache stability."""
    md = "## A\nbody a\n## B\nbody b\n"
    runs = [summarize_context_file("AGENTS.md", md) for _ in range(3)]
    assert all(r == runs[0] for r in runs)


# -----------------------------------------------------------------------------
# Edge cases discovered during self-review
# -----------------------------------------------------------------------------

def test_blank_only_content_returns_empty_list():
    """Pure whitespace/newlines must not crash and must yield no sections."""
    assert summarize_context_file("AGENTS.md", "\n\n\n   \n") == []


def test_heading_with_no_body_yields_empty_body_and_title_summary():
    """A trailing heading with nothing after it should hash the empty string
    and fall back to the title for its summary."""
    md = "## Last Heading\n"
    [s] = summarize_context_file("AGENTS.md", md)
    assert s.body == ""
    assert s.summary == "Last Heading"
    # sha256("") prefix — locks the empty-body hash so callers can detect it.
    assert s.body_hash == hashlib.sha256(b"").hexdigest()[:16]


def test_heading_with_only_whitespace_body_falls_back_to_title():
    md = "## Only Spaces\n   \n\t\n"
    [s] = summarize_context_file("AGENTS.md", md)
    assert s.body == ""
    assert s.summary == "Only Spaces"


def test_title_with_only_punctuation_uses_untitled_slug():
    md = "## !!!\nbody\n"
    [s] = summarize_context_file("AGENTS.md", md)
    assert s.id == "agents.md#untitled"
    assert s.title == "!!!"


def test_filename_without_extension_preserved_as_prefix():
    [s] = summarize_context_file("CURSORRULES", "## X\nbody\n")
    assert s.id == "cursorrules#x"


def test_dotfile_filename_preserved_as_prefix():
    [s] = summarize_context_file(".cursorrules", "## X\nbody\n")
    assert s.id == ".cursorrules#x"


def test_long_title_slugifies_without_truncation():
    """slugify doesn't impose a length cap — the section id stays unique
    even for verbose titles."""
    title = "A Very Long Section Title That Just Keeps Going And Going"
    md = f"## {title}\nbody\n"
    [s] = summarize_context_file("AGENTS.md", md)
    assert s.id == "agents.md#a-very-long-section-title-that-just-keeps-going-and-going"


def test_atx_headings_require_space_after_hashes():
    """`#foo` (no space) is not a real ATX heading and must not split."""
    md = "#nope this is not a heading\n## real\nbody\n"
    sections = summarize_context_file("AGENTS.md", md)
    assert [s.title for s in sections] == ["real"]


def test_section_equality_uses_dataclass_fields():
    """frozen=True implies eq=True — equality must be value-based,
    which the determinism test relies on."""
    md = "## X\nbody\n"
    [a] = summarize_context_file("AGENTS.md", md)
    [b] = summarize_context_file("AGENTS.md", md)
    assert a == b
    assert isinstance(a, ContextSection)
