"""Boot-time registrar for context-file sections (Phase 2.1, Idea D).

Discovers AGENTS.md and friends in the agent's cwd, splits each into
sections via `agent.context_summary.summarize_context_file`, populates
the in-process body cache used by `tools.read_context_tool.read_context`,
and OPTIONALLY POSTs each section's body to the inferlet's
`POST /v1/pie/context-section/register` route so a future inferlet-side
optimization can hit the KV-prefix cache for tool results.

The inferlet register call is groundwork only: failure (no URL, no
network, non-200, timeout) is logged at DEBUG and does not prevent the
agent from booting. The token-savings goal of Idea D is realized purely
by the in-process cache + index emission.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from agent.context_summary import ContextSection, summarize_context_file
from tools.read_context_tool import set_body_cache


__all__ = [
    "register_context_files",
    "discover_context_files",
    "render_context_files_index",
]


logger = logging.getLogger(__name__)


# Files we treat as context files. Order is intentional — we scan all of
# them (no priority dedup) since each is independently meaningful as a
# section index. Keep in sync with the discovery rules in
# `agent/prompt_builder.py:_load_*` helpers.
_CONTEXT_FILENAMES = [
    "AGENTS.md",
    "agents.md",
    ".cursorrules",
    "SOUL.md",
    "CLAUDE.md",
    "claude.md",
]


def discover_context_files(cwd: str) -> list[tuple[str, str]]:
    """Return [(filename, content), ...] for context files present in cwd.

    The returned filename preserves the on-disk casing the summarizer
    sees (it lowercases internally for the id prefix). On case-insensitive
    filesystems (macOS default, Windows), ``AGENTS.md`` and ``agents.md``
    refer to the same inode; we dedupe by resolved path so the same file
    is not summarized twice.
    """
    cwd_path = Path(cwd).resolve()
    out: list[tuple[str, str]] = []
    # Dedupe by (device, inode) so case-insensitive filesystems don't
    # double-count "AGENTS.md" and "agents.md" (same inode on macOS/Windows
    # default mounts; distinct inodes on case-sensitive Linux).
    seen_inodes: set[tuple[int, int]] = set()
    for name in _CONTEXT_FILENAMES:
        p = cwd_path / name
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except OSError as e:
            logger.debug("could not stat %s: %s", p, e)
            continue
        key = (st.st_dev, st.st_ino)
        if key in seen_inodes:
            continue
        seen_inodes.add(key)
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.debug("could not read %s: %s", p, e)
            continue
        out.append((name, content))
    return out


def _post_register(
    inferlet_url: str,
    section: ContextSection,
    timeout_s: float = 2.0,
) -> bool:
    """POST one section to the inferlet register route.

    Returns True on 2xx, False on any failure. Failures are logged at
    DEBUG only — this is groundwork; the agent functions correctly when
    the inferlet doesn't have the route.
    """
    payload = json.dumps({
        "section_id": section.id,
        "body_hash":  section.body_hash,
        "body_text":  section.body,
    }).encode("utf-8")
    req = urllib.request.Request(
        url=inferlet_url.rstrip("/") + "/v1/pie/context-section/register",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            ok = 200 <= resp.status < 300
            if not ok:
                logger.debug(
                    "context-section register non-2xx for %s: %s",
                    section.id, resp.status,
                )
            return ok
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        logger.debug("context-section register failed for %s: %s", section.id, e)
        return False
    except Exception as e:  # pragma: no cover — last-resort defensive
        logger.debug(
            "context-section register unexpected error for %s: %s",
            section.id, e,
        )
        return False


def register_context_files(
    cwd: Optional[str] = None,
    inferlet_url: Optional[str] = None,
) -> list[ContextSection]:
    """Discover context files in cwd, summarize each into sections, populate
    the in-process body cache, and (optionally) POST bodies to the inferlet.

    Args:
        cwd: Directory to scan. Default: current working directory.
        inferlet_url: Base URL of the hermes-openai-compat inferlet
            (e.g. "http://localhost:9090"). When None or empty, the
            inferlet POST is skipped entirely. The HERMES_INFERLET_URL
            env var is consulted as a fallback.

    Returns:
        The full list of sections discovered (across all context files).
        Caller can use the list to render the index in the system prompt.
        Empty list when no context files are present.
    """
    if cwd is None:
        cwd = os.getcwd()
    if inferlet_url is None:
        inferlet_url = os.environ.get("HERMES_INFERLET_URL", "").strip()

    all_sections: list[ContextSection] = []
    body_cache: dict[str, str] = {}
    for filename, content in discover_context_files(cwd):
        sections = summarize_context_file(filename, content)
        for s in sections:
            all_sections.append(s)
            body_cache[s.id] = s.body

    set_body_cache(body_cache)

    if inferlet_url and all_sections:
        ok_count = sum(
            1 for s in all_sections if _post_register(inferlet_url, s)
        )
        logger.debug(
            "context-section register: %d/%d sections accepted by inferlet at %s",
            ok_count, len(all_sections), inferlet_url,
        )

    return all_sections


def render_context_files_index(sections: list[ContextSection]) -> str:
    """Render an index block suitable for system-prompt insertion.

    Format kept compact — every byte costs prompt tokens. Each line:
        - <id> [<title>] — <summary>
    """
    if not sections:
        return ""
    lines = [
        "# Project context files (index — load full body via read_context(section_id))"
    ]
    for s in sections:
        lines.append(f"- {s.id} [{s.title}] — {s.summary}")
    return "\n".join(lines)
