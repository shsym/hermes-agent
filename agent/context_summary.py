"""Heuristic markdown-section summarizer for context files.

Used by Phase 2.1 (Idea D) to replace verbatim AGENTS.md / .cursorrules
content in the system prompt with a compact index. Each section is keyed
by content hash so re-edits produce a new handle on the inferlet side
and the old one becomes GC-eligible.

Determinism is load-bearing: this code is called every time the agent
boots, and the rendered index becomes part of the cached system prefix.
LLM-based summarization would change the prefix on every cold start and
defeat the cache. The heuristic here uses only the section title + the
first non-blank line of the body — slug, hash, and summary are all pure
functions of input bytes.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


__all__ = ["ContextSection", "summarize_context_file", "slugify_title"]


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


@dataclass(frozen=True)
class ContextSection:
    """One section of a context file.

    Fields:
        id          Stable identifier of the form "<filename-lower>#<slug>".
                    Stable across runs as long as filename and title are
                    unchanged.
        title       The H1/H2/H3+ heading text, trimmed.
        summary     The first non-blank line of the body, truncated to ~80
                    chars at a word boundary if longer. Falls back to title
                    when body is empty.
        body        Full markdown of the section between this heading and
                    the next heading of any depth (heading text NOT included).
                    Trailing whitespace stripped; leading blank lines removed.
        body_hash   16-char hex prefix of sha256(body). Used by the inferlet
                    side as the cache-handle suffix so a re-edited section
                    gets a new handle.
    """

    id: str
    title: str
    summary: str
    body: str
    body_hash: str


def slugify_title(title: str) -> str:
    """Turn 'Coding Style' into 'coding-style'. Pure function.

    Lowercases, strips non-alphanumeric runs into single hyphens, trims
    leading/trailing hyphens. Empty / whitespace-only titles become 'untitled'.
    """
    s = title.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s or "untitled"


def _summary_for(body: str, fallback: str) -> str:
    """First non-blank body line, truncated at 80 chars on a word boundary."""
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading markdown noise (bullets, blockquotes) so the summary
        # reads as a sentence.
        cleaned = re.sub(r"^([>*\-+]|\d+\.)\s+", "", line)
        if len(cleaned) <= 80:
            return cleaned
        # Truncate on the last whitespace before the 80th column.
        cut = cleaned.rfind(" ", 0, 80)
        return (cleaned[:cut] if cut > 0 else cleaned[:80]) + "…"
    return fallback


def summarize_context_file(filename: str, content: str) -> list[ContextSection]:
    """Split `content` into ContextSection records keyed by `filename`.

    Sections are demarcated by markdown ATX headings (# .. ######). Content
    BEFORE the first heading is dropped (callers typically inject a synthetic
    H1 if they want to preserve a preamble). Each heading at depth >= 2 is
    treated as a section under the previous depth-1 heading; for indexing
    purposes, only the LEAF heading determines the section's slug (so
    "## Coding style" inside "# Project rules" yields id "<filename>#coding-style",
    not "<filename>#project-rules-coding-style"). This matches how a model
    would reference a section by name.

    Args:
        filename: Display filename used as the id prefix (case-insensitive).
            Typically "AGENTS.md", ".cursorrules", "SOUL.md".
        content: Full markdown text.

    Returns:
        Empty list when no headings are present.
    """
    file_prefix = filename.lower()
    sections: list[ContextSection] = []

    # First pass: locate every heading line.
    lines = content.splitlines()
    heading_idxs: list[tuple[int, int, str]] = []  # (line_index, depth, title)
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m:
            heading_idxs.append((i, len(m.group(1)), m.group(2)))

    if not heading_idxs:
        return []

    # Second pass: for each heading, the body runs from the next line up to
    # (exclusive) the next heading line of any depth.
    for k, (lineno, _depth, title) in enumerate(heading_idxs):
        body_start = lineno + 1
        body_end = heading_idxs[k + 1][0] if k + 1 < len(heading_idxs) else len(lines)
        body = "\n".join(lines[body_start:body_end])
        # Strip leading blank lines + trailing whitespace; preserves internal layout.
        body = body.lstrip("\n").rstrip()
        slug = slugify_title(title)
        section_id = f"{file_prefix}#{slug}"
        body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
        sections.append(ContextSection(
            id=section_id,
            title=title,
            summary=_summary_for(body, fallback=title),
            body=body,
            body_hash=body_hash,
        ))

    return sections
