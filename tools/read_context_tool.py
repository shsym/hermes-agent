"""read_context tool: surface a context-file section body on demand.

Phase 2.1 (Idea D) — replaces verbatim AGENTS.md content in the system
prompt with a compact index. The model calls read_context(section_id)
when it actually needs a section body. Bodies live in a module-level
cache populated at agent boot by `agent.context_section_registry`.

Tool resolution is purely client-side: the inferlet's KV-prefix-handle
storage (Phase 2.1 Task 2B / VENDOR_SOURCE.md #7) is not consulted here.
That storage is groundwork for a later inferlet-side optimization where
the same body text appearing in a tool result triggers a KV import
instead of a fresh prefill.
"""

from __future__ import annotations

import json
from typing import Optional

from tools.registry import registry


__all__ = ["read_context", "set_body_cache", "get_cached_section_ids", "clear_cache"]


# Module-level cache: section_id -> body_text. Populated by
# `agent.context_section_registry.register_context_files`.
_BODY_CACHE: dict[str, str] = {}


def set_body_cache(cache: dict[str, str]) -> None:
    """Replace the body cache. Called by the registrar at agent boot.

    Idempotent: subsequent calls overwrite. Empty dict clears.
    """
    global _BODY_CACHE
    _BODY_CACHE = dict(cache)


def get_cached_section_ids() -> list[str]:
    """Return the set of section_ids the model can validly request.

    Useful for diagnostics + the tool description (so the model knows
    what it can ask for without trial-and-error)."""
    return sorted(_BODY_CACHE.keys())


def clear_cache() -> None:
    """Drop all cached bodies. Test affordance."""
    global _BODY_CACHE
    _BODY_CACHE = {}


def read_context(section_id: str, task_id: Optional[str] = None) -> str:
    """Return the body text for a registered context-file section.

    Returns a JSON envelope so the model can parse success vs error
    consistently (matches the convention used by other handlers like
    skills_tool, todo_tool).
    """
    if not isinstance(section_id, str) or not section_id.strip():
        return json.dumps({"error": "section_id must be a non-empty string"})
    sid = section_id.strip()
    body = _BODY_CACHE.get(sid)
    if body is None:
        return json.dumps({
            "error": f"unknown section_id: {sid!r}",
            "available": sorted(_BODY_CACHE.keys()),
        })
    return json.dumps({"section_id": sid, "body": body})


READ_CONTEXT_SCHEMA = {
    "name": "read_context",
    "description": (
        "Load the full body of a project context-file section "
        "(AGENTS.md / .cursorrules / SOUL.md). Use ONLY when the "
        "compact index in the system prompt indicates the relevant "
        "rule lives in a section you have not yet loaded — do NOT "
        "call reflexively for unrelated questions. Returns "
        '{"section_id": "...", "body": "..."} on success or '
        '{"error": "...", "available": [...]} on unknown id.'
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "section_id": {
                "type": "string",
                "description": (
                    "Section identifier of the form '<filename-lower>#<slug>' "
                    "as listed in the system prompt's context-files index, "
                    "e.g. 'agents.md#coding-style'."
                ),
            },
        },
        "required": ["section_id"],
        "additionalProperties": False,
    },
}


def _check_read_context_requirements() -> bool:
    """Always available — no external dependencies."""
    return True


registry.register(
    name="read_context",
    toolset="context_files",
    schema=READ_CONTEXT_SCHEMA,
    handler=lambda args, **kw: read_context(
        args.get("section_id", ""), task_id=kw.get("task_id")
    ),
    check_fn=_check_read_context_requirements,
    emoji="📖",
)
